"""Interfaz Streamlit para segmentación y conteo de células en fotos y videos.

El video exportado conserva el video original y superpone las máscaras de
segmentación, sin cajas delimitadoras ni etiquetas.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import gc
import hashlib
from html import escape
import json
import math
import os
import shutil
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable
from urllib.parse import quote, urlparse

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
    WEBRTC_IMPORT_ERROR = ""
except ImportError as error:
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = f"{type(error).__name__}: {error}"


APP_DIR = Path(__file__).resolve().parent
LOCAL_MODEL = APP_DIR / "models" / "best.pt"
MODEL_URL = os.getenv(
    "CELL_MODEL_URL",
    "https://github.com/sarayconsuegrav-prog/cell-health-streamlit/releases/download/v1.0.0/best.pt",
)
MODEL_BACKEND = os.getenv("CELL_MODEL_BACKEND", "openvino").strip().lower()
OPENVINO_MODEL_DIR = APP_DIR / "models" / "best_openvino_model"
OPENVINO_MODEL_URL = os.getenv("CELL_OPENVINO_MODEL_URL", "").strip()
CACHED_MODEL = Path(tempfile.gettempdir()) / "cell-health-streamlit" / "best.pt"
CACHED_OPENVINO_ROOT = Path(tempfile.gettempdir()) / "cell-health-streamlit" / "openvino"
CACHED_OPENVINO_ARCHIVE = CACHED_OPENVINO_ROOT / "best_openvino_model.zip"
PHOTO_RENDER_VERSION = "measurement-diameter-perimeter-class-masks-v11"
# La foto de referencia tiene una barra de escala negra cuyo rótulo indica 50 µm.
SCALE_BAR_LENGTH_UM = 50.0
# Referencia observada en la fotografía entregada con barra: 144 px de barra en 859 px de ancho.
# Se usa únicamente cuando una foto de 40× no conserva la barra visible.
REFERENCE_SCALE_BAR_PX = 144.0
REFERENCE_IMAGE_WIDTH_PX = 859.0
# Las fotos se analizan con más detalle; para cámara/video se usa un tamaño menor
# para que el procesamiento sea ágil en una Mac sin GPU dedicada.
PHOTO_IMAGE_SIZE = 1280
LIVE_IMAGE_SIZE = 640
VIDEO_IMAGE_SIZE = 512
LIVE_INFERENCE_EVERY_N_FRAMES = 3
VIDEO_INFERENCE_STRIDE = 2
VIDEO_OUTPUT_MAX_WIDTH = 1280
LIVE_INFERENCE_RETRY_SECONDS = 2.0
LIVE_RECORD_EVERY_N_FRAMES = 2
# Las mediciones se usan para promedios; conservar cada observación de cada
# fotograma puede hacer crecer la RAM sin mejorar de forma apreciable el reporte.
MAX_MEASUREMENT_OBSERVATIONS = 2000
LIVE_RESOLUTIONS = {
    "640 × 480 (480p)": {
        "width": 640,
        "height": 480,
        "frame_rate": 24,
        "inference_size": 512,
        "inference_every": 3,
    },
    "1280 × 720 (HD)": {
        "width": 1280,
        "height": 720,
        "frame_rate": 20,
        "inference_size": 640,
        "inference_every": 4,
    },
    "1920 × 1080 (Full HD · 2 MP)": {
        "width": 1920,
        "height": 1080,
        "frame_rate": 15,
        "inference_size": 640,
        "inference_every": 5,
    },
}
MAX_LIVE_SAMPLES = 4
POOL_QUERY_KEY = "cell_pools"
NEW_POOL_OPTION = "➕ Registrar nueva piscina"
# OpenVINO puede no exponer `model.names` en algunas versiones de Ultralytics.
# Estos son los nombres incluidos en `models/best_openvino_model/metadata.yaml`.
DEFAULT_MODEL_NAMES = {
    0: "celula_enferma",
    1: "celula_sana",
}

# Colores BGR de las capas de segmentación.
COLORS_BGR = {
    "sana": (72, 184, 72),       # Verde
    "enferma": (60, 60, 235),    # Rojo
    "otra": (220, 165, 30),      # Azul
}
LIVE_INFERENCE_LOCK = Lock()
DEFAULT_ICE_SERVERS = [
    {"urls": [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
        "stun:stun2.l.google.com:19302",
    ]},
    {"urls": ["stun:stun.relay.metered.ca:80"]},
]


def runtime_setting(name: str) -> str:
    """Lee una configuración opcional desde Secrets de Streamlit o variables de entorno."""
    environment_value = os.getenv(name, "").strip()
    try:
        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None
    if secret_value is None:
        return environment_value
    return str(secret_value).strip() or environment_value


@st.cache_data(ttl=300, show_spinner=False)
def fetch_metered_ice_servers(app_name: str, api_key: str) -> list[dict[str, Any]]:
    """Obtiene credenciales TURN temporales de Metered sin exponer la API key."""
    endpoint = (
        f"https://{quote(app_name.strip(), safe='')}.metered.live/"
        f"api/v1/turn/credentials?apiKey={quote(api_key.strip(), safe='')}"
    )
    try:
        with urllib.request.urlopen(endpoint, timeout=8) as response:
            payload = json.load(response)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []

    if isinstance(payload, dict):
        payload = payload.get("iceServers", payload.get("ice_servers", []))
    if not isinstance(payload, list):
        return []
    return [server for server in payload if isinstance(server, dict)]


def normalize_metered_app_name(value: str) -> str:
    """Acepta tanto el slug de Metered como su dominio completo."""
    raw_value = value.strip()
    parsed = urlparse(
        raw_value if "://" in raw_value else f"https://{raw_value}"
    )
    host = (parsed.netloc or parsed.path.split("/", 1)[0]).strip().lower()
    suffix = ".metered.live"
    if host.endswith(suffix):
        host = host[: -len(suffix)]
    return host.strip(".")


def rtc_configuration() -> dict[str, Any]:
    """Construye ICE servers y añade TURN solo cuando el despliegue lo configura."""
    ice_servers = [dict(server) for server in DEFAULT_ICE_SERVERS]
    raw_ice_servers = runtime_setting("RTC_ICE_SERVERS_JSON")
    if raw_ice_servers:
        try:
            parsed = json.loads(raw_ice_servers)
            if isinstance(parsed, dict):
                parsed = parsed.get("iceServers", [])
            if isinstance(parsed, list):
                ice_servers.extend(server for server in parsed if isinstance(server, dict))
        except (TypeError, ValueError, json.JSONDecodeError):
            st.warning("RTC_ICE_SERVERS_JSON no tiene un formato JSON válido; se usará STUN.")
        return {"iceServers": ice_servers}

    metered_app_name = normalize_metered_app_name(runtime_setting("METERED_APP_NAME"))
    metered_api_key = runtime_setting("METERED_API_KEY")
    if metered_app_name and metered_api_key:
        if metered_api_key.startswith("pk_live_"):
            st.warning(
                "La clave de Metered parece ser una clave Publishable de Realtime. "
                "Usa la API key de TURN REST en METERED_API_KEY."
            )
        else:
            metered_servers = fetch_metered_ice_servers(metered_app_name, metered_api_key)
            if metered_servers:
                ice_servers.extend(metered_servers)
            else:
                st.warning(
                    "Metered no devolvió servidores TURN. Verifica METERED_APP_NAME "
                    "y que METERED_API_KEY sea la API key de TURN REST."
                )

    turn_urls = [
        url.strip()
        for url in runtime_setting("RTC_TURN_URLS").split(",")
        if url.strip()
    ]
    turn_username = runtime_setting("RTC_TURN_USERNAME")
    turn_credential = runtime_setting("RTC_TURN_CREDENTIAL")
    if turn_urls and turn_username and turn_credential:
        ice_servers.append(
            {
                "urls": turn_urls,
                "username": turn_username,
                "credential": turn_credential,
            }
        )
    return {"iceServers": ice_servers}


def is_openvino_model_dir(path: Path) -> bool:
    """Comprueba que una carpeta contiene un modelo OpenVINO completo."""
    return path.is_dir() and any(path.glob("*.xml")) and any(path.glob("*.bin"))


def cached_openvino_model_path() -> Path | None:
    """Obtiene el modelo OpenVINO local o lo descarga desde un ZIP configurado."""
    configured_path = os.getenv("CELL_OPENVINO_MODEL_DIR", "").strip()
    candidates = []
    if configured_path:
        candidates.append(Path(configured_path).expanduser())
    candidates.extend(
        [
            OPENVINO_MODEL_DIR,
            CACHED_OPENVINO_ROOT / "best_openvino_model",
        ]
    )
    for candidate in candidates:
        if is_openvino_model_dir(candidate):
            return candidate

    if not OPENVINO_MODEL_URL:
        return None

    try:
        CACHED_OPENVINO_ROOT.mkdir(parents=True, exist_ok=True)
        if not CACHED_OPENVINO_ARCHIVE.exists():
            with urllib.request.urlopen(OPENVINO_MODEL_URL, timeout=180) as source, CACHED_OPENVINO_ARCHIVE.open("wb") as target:
                shutil.copyfileobj(source, target)

        extract_root = CACHED_OPENVINO_ROOT / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(CACHED_OPENVINO_ARCHIVE) as archive:
            root = extract_root.resolve()
            for member in archive.infolist():
                destination = (extract_root / member.filename).resolve()
                if root not in destination.parents and destination != root:
                    raise ValueError("El ZIP del modelo contiene una ruta no válida.")
            archive.extractall(extract_root)

        for candidate in [extract_root, *extract_root.rglob("*")]:
            if is_openvino_model_dir(candidate):
                return candidate
    except Exception:
        CACHED_OPENVINO_ARCHIVE.unlink(missing_ok=True)
    return None


def default_pt_model_path() -> Path:
    """Usa el modelo local o descarga la copia pública de la release."""
    if LOCAL_MODEL.exists():
        return LOCAL_MODEL
    if CACHED_MODEL.exists():
        return CACHED_MODEL

    try:
        CACHED_MODEL.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MODEL_URL, timeout=180) as source, CACHED_MODEL.open("wb") as target:
            shutil.copyfileobj(source, target)
        return CACHED_MODEL
    except Exception:
        # La interfaz mostrará el mensaje existente de modelo no encontrado.
        CACHED_MODEL.unlink(missing_ok=True)
        return LOCAL_MODEL


def export_openvino_model(model_path: Path) -> Path | None:
    """Convierte el checkpoint PyTorch a OpenVINO y conserva el resultado en caché."""
    digest = hashlib.sha256()
    with model_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    model_revision = digest.hexdigest()[:16]
    CACHED_OPENVINO_ROOT.mkdir(parents=True, exist_ok=True)
    cached_source = CACHED_OPENVINO_ROOT / f"{model_path.stem}_{model_revision}.pt"
    exported_dir = CACHED_OPENVINO_ROOT / f"{model_path.stem}_{model_revision}_openvino_model"
    if is_openvino_model_dir(exported_dir):
        return exported_dir

    try:
        if not cached_source.exists():
            shutil.copy2(model_path, cached_source)
        export_model = YOLO(str(cached_source))
        exported_path = export_model.export(
            format="openvino",
            imgsz=LIVE_IMAGE_SIZE,
            dynamic=True,
            nms=False,
            device="cpu",
        )
        exported_path = Path(exported_path)
        if not is_openvino_model_dir(exported_path):
            return None
        if exported_path != exported_dir:
            exported_path.rename(exported_dir)
        return exported_dir
    except Exception:
        return None


def default_model_path() -> Path | None:
    """Usa OpenVINO si fue seleccionado y conserva PyTorch como respaldo."""
    if MODEL_BACKEND in {"openvino", "ov"}:
        if st.session_state.get("openvino_fallback"):
            return default_pt_model_path()
        openvino_path = cached_openvino_model_path()
        if openvino_path is not None:
            return openvino_path
        return export_openvino_model(default_pt_model_path())
    return default_pt_model_path()


def normalize_class_name(name: str) -> str:
    """Agrupa los nombres del modelo en las categorías usadas por el tablero."""
    normalized = name.lower().strip().replace("á", "a").replace("é", "e")
    if "enferm" in normalized:
        return "enferma"
    if "sana" in normalized or "salud" in normalized or "healthy" in normalized:
        return "sana"
    return "otra"


def load_model(model_path: str | Path) -> YOLO:
    """Carga el modelo de YOLO y declara segmentación para carpetas OpenVINO."""
    path = Path(model_path)
    if path.is_dir():
        return YOLO(str(path), task="segment")
    return YOLO(str(path))


def load_live_model(model_path: str | Path) -> YOLO:
    """Carga el modelo de cámara sin depender del estado de Streamlit.

    La cámara se mantiene renderizada mientras este trabajo ocurre en segundo
    plano. Esto evita que el primer clic de «Iniciar detección» bloquee el
    componente WebRTC mientras OpenVINO compila el modelo.
    """
    try:
        loaded_model = load_model(model_path)
    except Exception as error:
        if MODEL_BACKEND in {"openvino", "ov"} and Path(model_path).is_dir():
            fallback_path = default_pt_model_path()
            try:
                loaded_model = load_model(fallback_path)
            except Exception as fallback_error:
                raise RuntimeError(
                    "No fue posible cargar OpenVINO ni el respaldo `.pt`: "
                    f"{fallback_error}"
                ) from error
        else:
            raise

    if loaded_model.task != "segment":
        raise RuntimeError(
            f"El modelo cargado es de tipo `{loaded_model.task}`. "
            "Esta aplicación requiere un modelo de segmentación."
        )
    return loaded_model


def get_model_names(model: YOLO | None) -> dict[int, str]:
    """Obtiene las clases tanto de YOLO/PyTorch como de ciertos backends OpenVINO.

    Algunas versiones de Ultralytics cargan el modelo OpenVINO sin publicar
    `names` a través de `YOLO.__getattr__`. En ese caso usamos las clases
    declaradas en el modelo convertido para que la cámara y los videos sigan
    clasificando las máscaras correctamente.
    """
    if model is None:
        return DEFAULT_MODEL_NAMES.copy()

    candidates: list[Any] = []
    try:
        candidates.append(model.names)
    except Exception:
        # Algunos backends OpenVINO inicializan el predictor al consultar
        # ``names`` y pueden fallar al compilar el modelo en ese momento.
        # La metadata incluida con el modelo ya contiene las clases correctas;
        # no debemos dejar que esa consulta derribe toda la vista en vivo.
        pass

    try:
        backend_model = model.model
    except Exception:
        backend_model = None
    if backend_model is not None:
        try:
            candidates.append(backend_model.names)
        except Exception:
            pass

    for names in candidates:
        if isinstance(names, dict):
            normalized = {}
            for key, value in names.items():
                try:
                    normalized[int(key)] = str(value)
                except (TypeError, ValueError):
                    continue
            if normalized:
                return normalized
        elif isinstance(names, (list, tuple)) and names:
            return {index: str(value) for index, value in enumerate(names)}

    return DEFAULT_MODEL_NAMES.copy()


def get_session_model(model_path: Path) -> YOLO:
    """Conserva el modelo entre reruns sin compartir el tracker entre usuarios."""
    model_key = f"{MODEL_BACKEND}:{model_path}"
    cached_model = st.session_state.get("cell_model")
    cached_model_path = st.session_state.get("cell_model_path")
    if cached_model is None or cached_model_path != model_key:
        cached_model = load_model(model_path)
        st.session_state["cell_model"] = cached_model
        st.session_state["cell_model_path"] = model_key
    return cached_model


def reset_trackers(model: YOLO) -> None:
    """Evita que IDs de un análisis anterior pasen al siguiente video."""
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []) or []:
        if hasattr(tracker, "reset"):
            tracker.reset()


def mask_frame(
    result: Any,
    frame_shape: tuple[int, ...],
    model_names: dict[int, str],
    included_categories: set[str] | None = None,
) -> np.ndarray:
    """Construye un cuadro negro con las máscaras de las clases indicadas."""
    height, width = frame_shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if result.masks is None or result.boxes is None:
        return canvas

    polygons = result.masks.xy
    class_ids = result.boxes.cls.int().cpu().tolist()
    for polygon, class_id in zip(polygons, class_ids):
        polygon = np.asarray(polygon, dtype=np.int32)
        if polygon.shape[0] < 3:
            continue
        class_name = model_names.get(int(class_id), str(class_id))
        category = normalize_class_name(class_name)
        if included_categories is not None and category not in included_categories:
            continue
        color = COLORS_BGR[category]
        cv2.fillPoly(canvas, [polygon.reshape((-1, 1, 2))], color)

    return canvas


def overlay_masks(image: np.ndarray, masks: np.ndarray, opacity: float) -> np.ndarray:
    """Superpone las máscaras coloreadas solo donde existe una segmentación."""
    output = image.copy()
    masked_pixels = np.any(masks != 0, axis=2)
    output[masked_pixels] = (
        image[masked_pixels] * (1.0 - opacity) + masks[masked_pixels] * opacity
    ).astype(np.uint8)
    return output


def draw_live_status_overlay(
    image: np.ndarray,
    detection_active: bool,
) -> np.ndarray:
    """Dibuja una señal visible sobre el video para indicar el estado actual."""
    output = image.copy()
    height, width = output.shape[:2]
    if not detection_active:
        return output
    label = "EN VIVO"
    badge_color = (40, 40, 220)  # rojo en BGR

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.48, min(0.82, width / 1500))
    thickness = max(1, round(font_scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    left = max(12, round(width * 0.018))
    top = max(12, round(height * 0.028))
    padding_x = max(10, round(width * 0.012))
    padding_y = max(8, round(height * 0.012))
    badge_height = text_height + baseline + padding_y * 2
    badge_width = text_width + padding_x * 3 + round(badge_height * 0.35)
    right = min(width - 8, left + badge_width)
    bottom = min(height - 8, top + badge_height)

    background = output.copy()
    cv2.rectangle(background, (left, top), (right, bottom), (5, 24, 47), -1)
    output = cv2.addWeighted(background, 0.82, output, 0.18, 0)
    center = (left + padding_x + round(badge_height * 0.17), top + badge_height // 2)
    radius = max(5, round(badge_height * 0.16))
    cv2.circle(output, center, radius, badge_color, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        output,
        label,
        (left + padding_x * 2 + round(badge_height * 0.24), top + padding_y + text_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.rectangle(output, (left, top), (right, bottom), badge_color, 1, lineType=cv2.LINE_AA)
    return output


def update_track_votes(result: Any, votes: dict[int, Counter], model_names: dict[int, str]) -> int:
    """Registra la clase observada para cada ID único producido por el tracker."""
    if result.boxes is None or result.boxes.id is None:
        return 0

    track_ids = result.boxes.id.int().cpu().tolist()
    class_ids = result.boxes.cls.int().cpu().tolist()
    for track_id, class_id in zip(track_ids, class_ids):
        class_name = model_names.get(int(class_id), str(class_id))
        votes[int(track_id)][normalize_class_name(class_name)] += 1
    return len(track_ids)


def summarize_tracks(votes: dict[int, Counter]) -> dict[str, int]:
    """Da a cada célula la clase que recibió en más fotogramas."""
    counts = Counter()
    for class_votes in votes.values():
        category, _ = class_votes.most_common(1)[0]
        counts[category] += 1
    return {
        "sana": counts["sana"],
        "enferma": counts["enferma"],
        "otra": counts["otra"],
        "total": sum(counts.values()),
    }


def summarize_instances(result: Any, model_names: dict[int, str]) -> dict[str, int]:
    """Cuenta las instancias detectadas en una sola fotografía."""
    counts = Counter()
    if result.boxes is not None:
        for class_id in result.boxes.cls.int().cpu().tolist():
            class_name = model_names.get(int(class_id), str(class_id))
            counts[normalize_class_name(class_name)] += 1
    return {
        "sana": counts["sana"],
        "enferma": counts["enferma"],
        "otra": counts["otra"],
        "total": sum(counts.values()),
    }


def detect_scale_bar_length_px(image: np.ndarray) -> float | None:
    """Encuentra la barra negra horizontal de escala en la parte superior izquierda."""
    height, width = image.shape[:2]
    if height < 40 or width < 100:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # La barra de referencia proporcionada está arriba a la izquierda. La apertura
    # horizontal elimina el texto y conserva únicamente segmentos largos y rectos.
    search_height = max(1, int(height * 0.30))
    search_width = max(1, int(width * 0.40))
    search_area = gray[:search_height, :search_width]
    dark_pixels = cv2.threshold(search_area, 100, 255, cv2.THRESH_BINARY_INV)[1]
    kernel_width = max(20, int(round(width * 0.02)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    horizontal_lines = cv2.morphologyEx(dark_pixels, cv2.MORPH_OPEN, horizontal_kernel)
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    minimum_length = max(40, int(round(width * 0.04)))
    candidates: list[tuple[float, float]] = []
    for contour in contours:
        x, y, line_width, line_height = cv2.boundingRect(contour)
        if line_width < minimum_length or line_height > 8:
            continue
        aspect_ratio = line_width / max(line_height, 1)
        if aspect_ratio < 12:
            continue
        candidates.append((float(line_width), aspect_ratio))

    if not candidates:
        return None
    # La barra es el segmento horizontal más largo que cumple las restricciones.
    return max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[0]


def calibration_from_image(image: np.ndarray) -> dict[str, Any]:
    """Obtiene la calibración directa o estimada de una imagen a 40×."""
    scale_bar_length_px = detect_scale_bar_length_px(image)
    if scale_bar_length_px is None:
        # Algunas fotos del conjunto fueron exportadas sin la barra visible. Al
        # conservar 40× y el mismo campo de visión, la escala se ajusta a su ancho.
        scale_bar_length_px = image.shape[1] * REFERENCE_SCALE_BAR_PX / REFERENCE_IMAGE_WIDTH_PX
        calibration_method = "estimada a 40×"
    else:
        calibration_method = "barra de escala detectada"
    micrometers_per_pixel = SCALE_BAR_LENGTH_UM / scale_bar_length_px
    return {
        "micrometers_per_pixel": micrometers_per_pixel,
        "scale_bar_length_px": scale_bar_length_px,
        "calibration_method": calibration_method,
    }


def new_measurement_samples() -> dict[str, dict[str, list[float]]]:
    """Prepara las listas de métricas de las máscaras segmentadas."""
    return {
        "areas_px2": {"sana": [], "enferma": []},
        "perimeters_px": {"sana": [], "enferma": []},
        "diameters_px": {"sana": [], "enferma": []},
    }


@dataclass
class LiveSessionState:
    """Estado compartido entre el callback WebRTC y la interfaz Streamlit."""

    lock: Lock = field(default_factory=Lock)
    detection_active: bool = False
    show_healthy_masks: bool = True
    show_sick_masks: bool = True
    frame_number: int = 0
    camera_frames: int = 0
    captured_frames: int = 0
    processed_frames: int = 0
    recording_writer: Any | None = None
    recording_path: Path | None = None
    recording_size: tuple[int, int] | None = None
    recorded_frames: int = 0
    track_votes: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    measurement_samples: dict[str, dict[str, list[float]]] = field(default_factory=new_measurement_samples)
    calibration: dict[str, Any] | None = None
    cached_shape: tuple[int, ...] | None = None
    cached_healthy_masks: np.ndarray | None = None
    cached_sick_masks: np.ndarray | None = None
    inference_busy: bool = False
    inference_generation: int = 0
    last_error: str = ""
    inference_retry_at: float = 0.0
    model_loading: bool = False
    inference_model: YOLO | None = None
    inference_model_names: dict[int, str] = field(default_factory=lambda: DEFAULT_MODEL_NAMES.copy())
    completed_samples: list[dict[str, Any]] = field(default_factory=list)
    pending_sample: dict[str, Any] | None = None
    sample_ready: bool = False
    latest_frame: np.ndarray | None = None

    def _close_recording_locked(self, delete_file: bool = False) -> None:
        """Cierra el archivo temporal actual sin borrar un video ya detenido."""
        if self.recording_writer is not None:
            self.recording_writer.release()
            self.recording_writer = None
        if delete_file and self.recording_path is not None:
            try:
                self.recording_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.recording_path = None
        self.recording_size = None

    def record_frame(self, frame: np.ndarray, fps: float) -> None:
        """Escribe el fotograma mostrado en un MP4 temporal de baja carga."""
        with self.lock:
            if self.recording_writer is None:
                recording_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                self.recording_path = Path(recording_file.name)
                recording_file.close()
                output_width = min(int(frame.shape[1]), VIDEO_OUTPUT_MAX_WIDTH)
                output_height = max(1, round(frame.shape[0] * output_width / frame.shape[1]))
                self.recording_size = (output_width, output_height)
                self.recording_writer = cv2.VideoWriter(
                    str(self.recording_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(float(fps), 1.0),
                    self.recording_size,
                )
                if not self.recording_writer.isOpened():
                    self._close_recording_locked(delete_file=True)
                    self.last_error = "No se pudo crear el video temporal de la muestra."
                    return

            output_frame = frame
            if output_frame.shape[1] != self.recording_size[0] or output_frame.shape[0] != self.recording_size[1]:
                output_frame = cv2.resize(
                    output_frame,
                    self.recording_size,
                    interpolation=cv2.INTER_AREA,
                )
            try:
                self.recording_writer.write(output_frame)
                self.recorded_frames += 1
            except Exception as error:
                self.last_error = f"No se pudo grabar un fotograma: {error}"

    def reset_metrics(self) -> None:
        """Limpia el conteo al iniciar una nueva captura."""
        with self.lock:
            self._close_recording_locked(delete_file=True)
            self.frame_number = 0
            self.camera_frames = 0
            self.captured_frames = 0
            self.processed_frames = 0
            self.recorded_frames = 0
            self.track_votes = defaultdict(Counter)
            self.measurement_samples = new_measurement_samples()
            self.calibration = None
            self.cached_shape = None
            self.cached_healthy_masks = None
            self.cached_sick_masks = None
            self.inference_generation += 1
            self.inference_busy = False
            self.last_error = ""
            self.inference_retry_at = 0.0
            self.pending_sample = None
            self.sample_ready = False

    def _freeze_current_sample_locked(self) -> None:
        """Congela el resultado al detener para que el guardado no dependa del callback."""
        if self.captured_frames <= 0:
            self.pending_sample = None
            self.sample_ready = False
            return

        measurements = (
            summarize_measurement_samples(self.measurement_samples, self.calibration)
            if self.calibration is not None
            else None
        )
        self.pending_sample = {
            "captured_frames": self.captured_frames,
            "processed_frames": self.processed_frames,
            "recorded_frames": self.recorded_frames,
            "recording_path": str(self.recording_path) if self.recording_path else "",
            "counts": dict(summarize_tracks(self.track_votes)),
            "measurements": dict(measurements) if measurements is not None else None,
        }
        self.sample_ready = True

    def set_detection_active(self, active: bool) -> None:
        with self.lock:
            self.detection_active = active
            if not active:
                self.inference_busy = False
                self._close_recording_locked()
                self._freeze_current_sample_locked()

    def clear_completed_samples(self) -> None:
        with self.lock:
            for sample in self.completed_samples:
                recording_path = sample.get("recording_path")
                if recording_path:
                    try:
                        Path(recording_path).unlink(missing_ok=True)
                    except OSError:
                        pass
            self.completed_samples = []

    def save_current_sample(self, sample_code: str, lot_name: str) -> bool:
        """Guarda las métricas acumuladas como una de las cuatro muestras."""
        with self.lock:
            if len(self.completed_samples) >= MAX_LIVE_SAMPLES:
                return False

            if self.pending_sample is None:
                self._freeze_current_sample_locked()
            if self.pending_sample is None:
                return False

            pending_sample = self.pending_sample
            sample_number = len(self.completed_samples) + 1
            self.completed_samples.append(
                {
                    "sample_number": sample_number,
                    "label": f"Camarón muestra {sample_number}",
                    "code": sample_code.strip() or f"Muestra-{sample_number}",
                    "lot_name": lot_name.strip() or "Lote sin nombre",
                    "captured_frames": pending_sample["captured_frames"],
                    "processed_frames": pending_sample["processed_frames"],
                    "recorded_frames": pending_sample["recorded_frames"],
                    "recording_path": pending_sample["recording_path"],
                    "counts": dict(pending_sample["counts"]),
                    "measurements": (
                        dict(pending_sample["measurements"])
                        if pending_sample.get("measurements") is not None
                        else None
                    ),
                }
            )
            # El archivo pasa a ser propiedad de la muestra guardada. El
            # siguiente análisis solo podrá borrar su propio archivo temporal.
            self.recording_path = None
            self.recording_size = None
            self.frame_number = 0
            self.captured_frames = 0
            self.processed_frames = 0
            self.recorded_frames = 0
            self.track_votes = defaultdict(Counter)
            self.measurement_samples = new_measurement_samples()
            self.calibration = None
            self.cached_shape = None
            self.cached_healthy_masks = None
            self.cached_sick_masks = None
            self.inference_generation += 1
            self.inference_busy = False
            self.last_error = ""
            self.inference_retry_at = 0.0
            self.pending_sample = None
            self.sample_ready = False
            return True

    def set_latest_frame(self, frame: np.ndarray) -> None:
        """Conserva solo el último cuadro para el visor Streamlit."""
        with self.lock:
            self.latest_frame = np.ascontiguousarray(frame).copy()

    def latest_frame_snapshot(self) -> np.ndarray | None:
        """Devuelve una copia segura del cuadro más reciente."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def completed_samples_snapshot(self) -> list[dict[str, Any]]:
        """Devuelve una copia de las muestras finalizadas para renderizar y descargar."""
        with self.lock:
            return [
                {
                    **sample,
                    "counts": dict(sample["counts"]),
                    "measurements": (
                        dict(sample["measurements"])
                        if sample.get("measurements") is not None
                        else None
                    ),
                }
                for sample in self.completed_samples
            ]

    def update_completed_sample_code(self, sample_number: int, code: str) -> bool:
        """Corrige el código de una muestra ya guardada."""
        normalized_code = code.strip() or f"Muestra-{sample_number}"
        with self.lock:
            for sample in self.completed_samples:
                if sample.get("sample_number") == sample_number:
                    sample["code"] = normalized_code
                    return True
        return False

    def snapshot(self) -> dict[str, Any]:
        """Obtiene una copia coherente de las métricas para el panel lateral."""
        with self.lock:
            counts = summarize_tracks(self.track_votes)
            measurements = (
                summarize_measurement_samples(self.measurement_samples, self.calibration)
                if self.calibration is not None
                else None
            )
            return {
                "detection_active": self.detection_active,
                "counts": counts,
                "measurements": measurements,
                "camera_frames": self.camera_frames,
                "captured_frames": self.captured_frames,
                "processed_frames": self.processed_frames,
                "recorded_frames": self.recorded_frames,
                "recording_ready": bool(self.recording_path),
                "sample_ready": self.sample_ready,
                "inference_busy": self.inference_busy,
                "model_loading": self.model_loading,
                "last_error": self.last_error,
                "completed_samples": len(self.completed_samples),
            }


def get_live_session_state() -> LiveSessionState:
    """Crea un estado independiente para cada sesión del navegador."""
    state = st.session_state.get("live_session_state")
    if not isinstance(state, LiveSessionState):
        state = LiveSessionState()
        persisted_samples = st.session_state.get("live_completed_samples")
        if isinstance(persisted_samples, list):
            state.completed_samples = deepcopy(persisted_samples)
        st.session_state["live_session_state"] = state
    return state


def persist_completed_live_samples(state: LiveSessionState) -> None:
    """Conserva una copia serializable para que un rerun no borre resultados."""
    st.session_state["live_completed_samples"] = state.completed_samples_snapshot()


def normalize_pool_name(value: str) -> str:
    """Normaliza el nombre de una piscina para evitar duplicados accidentales."""
    return " ".join(str(value).strip().split())


def _unique_pool_names(values: list[Any]) -> list[str]:
    """Conserva los nombres de piscina únicos respetando el orden de registro."""
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = normalize_pool_name(str(value))
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


def get_saved_pools() -> list[str]:
    """Obtiene las piscinas guardadas en la sesión y en la URL del navegador.

    Streamlit Community Cloud no ofrece un disco persistente para datos de usuario.
    La lista se conserva en ``st.session_state`` y también en un parámetro JSON de
    la URL, de modo que vuelve a aparecer al recargar este enlace en el mismo
    navegador. Más adelante puede sustituirse por una base de datos corporativa.
    """
    pools = st.session_state.get("saved_pools")
    if isinstance(pools, list):
        return pools

    raw_value: Any = ""
    try:
        raw_value = st.query_params.get(POOL_QUERY_KEY, "")
    except Exception:
        raw_value = ""
    if isinstance(raw_value, list):
        raw_value = raw_value[-1] if raw_value else ""

    loaded: list[Any] = []
    if raw_value:
        try:
            decoded = json.loads(str(raw_value))
            if isinstance(decoded, list):
                loaded = decoded
        except (TypeError, ValueError, json.JSONDecodeError):
            loaded = []

    pools = _unique_pool_names(loaded)
    st.session_state["saved_pools"] = pools
    return pools


def persist_saved_pools(pools: list[str]) -> None:
    """Guarda la lista de piscinas en la URL cuando el navegador lo permite."""
    try:
        if pools:
            st.query_params[POOL_QUERY_KEY] = json.dumps(pools, ensure_ascii=False)
        elif POOL_QUERY_KEY in st.query_params:
            del st.query_params[POOL_QUERY_KEY]
    except Exception:
        # La lista de la sesión sigue funcionando aunque una versión antigua de
        # Streamlit no exponga query_params.
        pass


def remember_pool(name: str) -> str:
    """Registra una piscina nueva y devuelve el nombre normalizado."""
    normalized = normalize_pool_name(name)
    if not normalized:
        return ""

    pools = get_saved_pools()
    for existing in pools:
        if existing.casefold() == normalized.casefold():
            return existing

    pools.append(normalized)
    st.session_state["saved_pools"] = pools
    persist_saved_pools(pools)
    return normalized


def collect_measurement_samples(
    result: Any, model_names: dict[int, str], samples: dict[str, dict[str, list[float]]]
) -> None:
    """Añade las métricas de cada máscara sana o enferma al acumulador."""
    if result.masks is None or result.boxes is None:
        return
    class_ids = result.boxes.cls.int().cpu().tolist()

    for polygon, class_id in zip(result.masks.xy, class_ids):
        class_name = model_names.get(int(class_id), str(class_id))
        category = normalize_class_name(class_name)
        if category not in {"sana", "enferma"}:
            continue
        contour = np.asarray(polygon, dtype=np.float32)
        if contour.shape[0] < 3:
            continue
        area_px2 = float(cv2.contourArea(contour))
        if area_px2 <= 0:
            continue
        area_values = samples["areas_px2"][category]
        perimeter_values = samples["perimeters_px"][category]
        diameter_values = samples["diameters_px"][category]
        if len(area_values) >= MAX_MEASUREMENT_OBSERVATIONS:
            continue
        area_values.append(area_px2)
        perimeter_values.append(float(cv2.arcLength(contour, True)))
        diameter_values.append(math.sqrt(4.0 * area_px2 / math.pi))


def summarize_measurement_samples(
    samples: dict[str, dict[str, list[float]]], calibration: dict[str, Any]
) -> dict[str, Any] | None:
    """Convierte las métricas en píxeles a promedios en micras."""
    areas_px2 = samples["areas_px2"]
    perimeters_px = samples["perimeters_px"]
    diameters_px = samples["diameters_px"]
    measured_cells = sum(len(areas) for areas in areas_px2.values())
    if not measured_cells:
        return None

    micrometers_per_pixel = float(calibration["micrometers_per_pixel"])
    measurements: dict[str, Any] = {**calibration, "measured_cells": measured_cells}
    for category in ("sana", "enferma"):
        count = len(areas_px2[category])
        measurements[f"{category}_measured_cells"] = count
        measurements[f"{category}_mean_area_um2"] = (
            float(np.mean(areas_px2[category])) * micrometers_per_pixel**2 if count else None
        )
        measurements[f"{category}_mean_perimeter_um"] = (
            float(np.mean(perimeters_px[category])) * micrometers_per_pixel if count else None
        )
        measurements[f"{category}_mean_equivalent_diameter_um"] = (
            float(np.mean(diameters_px[category])) * micrometers_per_pixel if count else None
        )
    return measurements


def measure_cells_from_scale_bar(
    result: Any, model_names: dict[int, str], image: np.ndarray
) -> dict[str, Any] | None:
    """Calcula área y diámetro con barra directa o calibración estimada a 40×."""
    samples = new_measurement_samples()
    collect_measurement_samples(result, model_names, samples)
    return summarize_measurement_samples(samples, calibration_from_image(image))


def grade_from_percentage(
    affected_percentage: float, low_limit: float, medium_limit: float, high_limit: float
) -> tuple[str, str]:
    """Escala provisional configurable hasta contar con la rúbrica validada."""
    if affected_percentage <= 0:
        return "Grado 0", "Sin afectación"
    if affected_percentage <= low_limit:
        return "Grado 1", "Afectación baja"
    if affected_percentage <= medium_limit:
        return "Grado 2", "Afectación moderada"
    if affected_percentage <= high_limit:
        return "Grado 3", "Afectación alta"
    return "Grado 4", "Afectación muy alta"


def format_class_measurement(measurements: dict[str, Any], category: str) -> str:
    """Forma una lectura breve del promedio de una clase de células."""
    count = int(measurements[f"{category}_measured_cells"])
    if count == 0:
        return "No detectadas"
    diameter = float(measurements[f"{category}_mean_equivalent_diameter_um"])
    perimeter = float(measurements[f"{category}_mean_perimeter_um"])
    area = float(measurements[f"{category}_mean_area_um2"])
    return f"Diámetro: {diameter:.2f} µm · Perímetro: {perimeter:.2f} µm · Área: {area:.2f} µm²"


def build_measurement_card(measurements: dict[str, Any], category: str, label: str) -> str:
    """Construye una ficha compacta de perímetro y área para una clase."""
    count = int(measurements[f"{category}_measured_cells"])
    if count == 0:
        details = '<div class="measurement-empty">No detectadas</div>'
    else:
        diameter = float(measurements[f"{category}_mean_equivalent_diameter_um"])
        perimeter = float(measurements[f"{category}_mean_perimeter_um"])
        area = float(measurements[f"{category}_mean_area_um2"])
        details = (
            f'<div class="measurement-row"><span>Diámetro promedio</span><strong>{diameter:.2f} µm</strong></div>'
            f'<div class="measurement-row"><span>Perímetro promedio</span><strong>{perimeter:.2f} µm</strong></div>'
            f'<div class="measurement-row"><span>Área promedio</span><strong>{area:.2f} µm²</strong></div>'
        )
    return (
        f'<article class="result-card measurement-card">'
        f'<span class="measurement-heading">Medidas · {label}</span>'
        f'{details}'
        f'</article>'
    )


def make_report_csv(
    counts: dict[str, int],
    affected_percentage: float,
    grade: str,
    description: str,
    total_label: str,
    measurements: dict[str, Any] | None = None,
) -> bytes:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Métrica", "Valor"])
    writer.writerow(["Células sanas", counts["sana"]])
    writer.writerow(["Células enfermas", counts["enferma"]])
    writer.writerow(["Células de otras clases", counts["otra"]])
    writer.writerow([total_label, counts["total"]])
    writer.writerow(["Porcentaje de células enfermas", f"{affected_percentage:.2f}%"])
    writer.writerow(["Calificación", grade])
    writer.writerow(["Descripción", description])
    if measurements:
        writer.writerow(["Magnificación de la calibración", "40×"])
        writer.writerow(["Método de calibración", measurements["calibration_method"]])
        writer.writerow(["Referencia de escala", f"{SCALE_BAR_LENGTH_UM:.1f} µm"])
        writer.writerow(["Longitud de referencia", f"{measurements['scale_bar_length_px']:.0f} píxeles"])
        writer.writerow(["Escala", f"{measurements['micrometers_per_pixel']:.6f} µm/píxel"])
        writer.writerow(["Células usadas para medición", measurements["measured_cells"]])
        for category, label in (("sana", "Células sanas"), ("enferma", "Células enfermas")):
            count = int(measurements[f"{category}_measured_cells"])
            writer.writerow([f"{label} usadas para medición", count])
            if count:
                mean_area = float(measurements[f"{category}_mean_area_um2"])
                mean_perimeter = float(measurements[f"{category}_mean_perimeter_um"])
                mean_diameter = float(measurements[f"{category}_mean_equivalent_diameter_um"])
                writer.writerow([f"{label}: área promedio", f"{mean_area:.2f} µm²"])
                writer.writerow(
                    [f"{label}: perímetro promedio", f"{mean_perimeter:.2f} µm"]
                )
                writer.writerow(
                    [f"{label}: diámetro promedio equivalente", f"{mean_diameter:.2f} µm"]
                )
    return output.getvalue().encode("utf-8-sig")


def aggregate_live_sample_metrics(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, Any] | None]:
    """Suma los conteos y calcula promedios ponderados para todo el lote."""
    counts = {"sana": 0, "enferma": 0, "otra": 0, "total": 0}
    measurement_totals: dict[str, dict[str, float]] = {
        category: {"count": 0.0, "area": 0.0, "perimeter": 0.0, "diameter": 0.0}
        for category in ("sana", "enferma")
    }

    for sample in samples:
        sample_counts = sample["counts"]
        for key in counts:
            counts[key] += int(sample_counts.get(key, 0))

        measurements = sample.get("measurements")
        if not measurements:
            continue
        for category in ("sana", "enferma"):
            measured_count = int(measurements.get(f"{category}_measured_cells", 0) or 0)
            if measured_count <= 0:
                continue
            bucket = measurement_totals[category]
            bucket["count"] += measured_count
            bucket["area"] += float(measurements[f"{category}_mean_area_um2"]) * measured_count
            bucket["perimeter"] += float(measurements[f"{category}_mean_perimeter_um"]) * measured_count
            bucket["diameter"] += float(
                measurements[f"{category}_mean_equivalent_diameter_um"]
            ) * measured_count

    measured_cells = int(sum(bucket["count"] for bucket in measurement_totals.values()))
    if not measured_cells:
        return counts, None

    aggregate_measurements: dict[str, Any] = {
        "calibration_method": "Promedio ponderado de las muestras",
        "measured_cells": measured_cells,
    }
    for category, bucket in measurement_totals.items():
        measured_count = int(bucket["count"])
        aggregate_measurements[f"{category}_measured_cells"] = measured_count
        aggregate_measurements[f"{category}_mean_area_um2"] = (
            bucket["area"] / measured_count if measured_count else None
        )
        aggregate_measurements[f"{category}_mean_perimeter_um"] = (
            bucket["perimeter"] / measured_count if measured_count else None
        )
        aggregate_measurements[f"{category}_mean_equivalent_diameter_um"] = (
            bucket["diameter"] / measured_count if measured_count else None
        )
    return counts, aggregate_measurements


def sample_result_grade(
    counts: dict[str, int], low_limit: float, medium_limit: float, high_limit: float
) -> tuple[float, str, str]:
    """Obtiene afectación y calificación para una muestra o para el lote."""
    denominator = counts["sana"] + counts["enferma"]
    affected_percentage = counts["enferma"] / denominator * 100 if denominator else 0.0
    if denominator:
        grade, description = grade_from_percentage(
            affected_percentage, low_limit, medium_limit, high_limit
        )
    else:
        grade, description = "Sin datos", "No se detectaron células clasificadas"
    return affected_percentage, grade, description


def make_live_samples_report_csv(
    lot_name: str,
    samples: list[dict[str, Any]],
    low_limit: float,
    medium_limit: float,
    high_limit: float,
) -> bytes:
    """Crea un reporte CSV con las cuatro muestras y el total del lote."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Nombre del lote / piscina", lot_name])
    writer.writerow(["Muestras analizadas", f"{len(samples)} de {MAX_LIVE_SAMPLES}"])
    writer.writerow([])
    writer.writerow(
        [
            "Muestra",
            "Código",
            "Fotogramas analizados",
            "Células sanas",
            "Células enfermas",
            "Total contabilizado",
            "Afectación",
            "Calificación",
            "Descripción",
        ]
    )

    for sample in samples:
        counts = sample["counts"]
        affected_percentage, grade, description = sample_result_grade(
            counts, low_limit, medium_limit, high_limit
        )
        writer.writerow(
            [
                sample["label"],
                sample["code"],
                sample["processed_frames"],
                counts["sana"],
                counts["enferma"],
                counts["total"],
                f"{affected_percentage:.2f}%",
                grade,
                description,
            ]
        )

    aggregate_counts, aggregate_measurements = aggregate_live_sample_metrics(samples)
    aggregate_percentage, aggregate_grade, aggregate_description = sample_result_grade(
        aggregate_counts, low_limit, medium_limit, high_limit
    )
    writer.writerow([])
    writer.writerow(["MÉTRICA GENERAL DEL LOTE"])
    writer.writerow(["Células sanas", aggregate_counts["sana"]])
    writer.writerow(["Células enfermas", aggregate_counts["enferma"]])
    writer.writerow(["Total contabilizado", aggregate_counts["total"]])
    writer.writerow(["Afectación general", f"{aggregate_percentage:.2f}%"])
    writer.writerow(["Calificación general", aggregate_grade])
    writer.writerow(["Descripción general", aggregate_description])
    if aggregate_measurements:
        writer.writerow(["Células usadas para medición", aggregate_measurements["measured_cells"]])
        for category, label in (("sana", "Sanas"), ("enferma", "Enfermas")):
            count = int(aggregate_measurements[f"{category}_measured_cells"])
            writer.writerow([f"{label} usadas para medición", count])
            if count:
                writer.writerow(
                    [
                        f"{label}: diámetro promedio",
                        f"{aggregate_measurements[f'{category}_mean_equivalent_diameter_um']:.2f} µm",
                    ]
                )
                writer.writerow(
                    [
                        f"{label}: perímetro promedio",
                        f"{aggregate_measurements[f'{category}_mean_perimeter_um']:.2f} µm",
                    ]
                )
                writer.writerow(
                    [
                        f"{label}: área promedio",
                        f"{aggregate_measurements[f'{category}_mean_area_um2']:.2f} µm²",
                    ]
                )
    return output.getvalue().encode("utf-8-sig")


def live_report_filename(lot_name: str) -> str:
    """Genera un nombre de archivo seguro y legible para el reporte del lote."""
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in lot_name.strip()
    ).strip("_")
    return f"reporte_{safe_name or 'lote'}.csv"


def show_summary(
    counts: dict[str, int],
    low_limit: float,
    medium_limit: float,
    high_limit: float,
    total_label: str,
    sidebar: bool = False,
    measurements: dict[str, Any] | None = None,
) -> tuple[float, str, str]:
    """Muestra un panel compacto y legible de resultados para foto y video."""
    target = st.sidebar if sidebar else st
    denominator = counts["sana"] + counts["enferma"]
    affected_percentage = (counts["enferma"] / denominator * 100) if denominator else 0.0
    if denominator:
        grade, description = grade_from_percentage(affected_percentage, low_limit, medium_limit, high_limit)
        result_note = (
            f"<strong>{escape(description)}.</strong> Se detectaron {counts['enferma']} células enfermas "
            f"de {denominator} células clasificadas."
        )
    else:
        grade, description = "Sin datos", "No se detectaron células clasificadas"
        result_note = "<strong>No hay datos para calificar.</strong> Analiza una foto donde se detecten células."
    measurement_cards = ""
    if measurements:
        measurement_cards = (
            build_measurement_card(measurements, "sana", "células sanas")
            + build_measurement_card(measurements, "enferma", "células enfermas")
        )

    target.markdown(
        f"""<section class="results-panel">
<div class="results-heading">Resultados</div>
<div class="results-grid">
<article class="result-card grade-card grade-featured"><span class="result-label">Calificación</span><strong class="grade-value">{escape(grade)}</strong><span class="grade-detail">{affected_percentage:.1f}% enfermas</span></article>
<article class="result-card"><span class="result-label">Células sanas</span><strong class="result-value">{counts['sana']}</strong></article>
<article class="result-card"><span class="result-label">Células enfermas</span><strong class="result-value">{counts['enferma']}</strong></article>
<article class="result-card"><span class="result-label">{escape(total_label)}</span><strong class="result-value">{counts['total']}</strong></article>
{measurement_cards}
</div>
<p class="results-note">{result_note}</p>
</section>""",
        unsafe_allow_html=True,
    )
    if counts["otra"]:
        target.caption(f"Además se detectaron {counts['otra']} células de otras clases.")
    return affected_percentage, grade, description


def process_image(
    model: YOLO, image_bytes: bytes, confidence: float, image_size: int, mask_opacity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, Any] | None]:
    """Segmenta una fotografía sin seguimiento: una inferencia y un conteo directo."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No fue posible leer la imagen. Usa PNG, JPG, JPEG o TIFF.")

    result = model.predict(image, conf=confidence, imgsz=image_size, verbose=False)[0]
    model_names = get_model_names(model)
    healthy_masks = mask_frame(result, image.shape, model_names, {"sana"})
    sick_masks = mask_frame(result, image.shape, model_names, {"enferma"})
    measurements = measure_cells_from_scale_bar(result, model_names, image)
    return image, healthy_masks, sick_masks, summarize_instances(result, model_names), measurements


def process_video(
    model: YOLO,
    uploaded_video: Any,
    confidence: float,
    image_size: int,
    mask_opacity: float,
    frame_stride: int,
    show_masks: bool,
    progress_bar: Any,
    status_text: Any,
    preview_placeholder: Any | None = None,
) -> tuple[bytes, dict[str, int], int, dict[str, Any] | None]:
    """Procesa el video, actualiza una previsualización y cuenta células únicas."""
    suffix = Path(uploaded_video.name).suffix or ".mp4"
    source_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    source_path = Path(source_file.name)
    output_path = Path(output_file.name)
    source_file.close()
    output_file.close()

    capture: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    results: Any | None = None
    track_votes: dict[int, Counter] | None = None
    video_measurement_samples: dict[str, dict[str, list[float]]] | None = None
    try:
        uploaded_video.seek(0)
        with source_path.open("wb") as target:
            shutil.copyfileobj(uploaded_video, target)

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise ValueError("No fue posible abrir el video. Usa un archivo MP4, AVI o MOV válido.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise ValueError("El video no tiene dimensiones válidas.")

        # Solo se usa OpenCV para conocer las propiedades del video; YOLO recibe
        # el archivo completo en una única llamada, no un fotograma por llamada.
        capture.release()
        capture = None

        # Se guardan los fotogramas analizados a una tasa proporcional para que
        # el video final conserve aproximadamente la misma duración.
        output_fps = max(fps / frame_stride, 1.0)
        output_width = min(width, VIDEO_OUTPUT_MAX_WIDTH)
        output_height = max(1, round(height * output_width / width))
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (output_width, output_height),
        )
        if not writer.isOpened():
            raise RuntimeError("No se pudo crear el video de salida en formato MP4.")

        reset_trackers(model)
        model_names = get_model_names(model)
        track_votes = defaultdict(Counter)
        tracked_detections = 0
        processed_frames = 0
        last_preview_update = 0.0
        video_measurement_samples = new_measurement_samples()
        video_calibration: dict[str, Any] | None = None

        results = model.track(
            source=str(source_path),
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            imgsz=image_size,
            vid_stride=frame_stride,
            verbose=False,
        )
        for result in results:
            if video_calibration is None:
                video_calibration = calibration_from_image(result.orig_img)
            collect_measurement_samples(result, model_names, video_measurement_samples)
            masks_frame = mask_frame(result, result.orig_img.shape, model_names)
            output_frame = (
                overlay_masks(result.orig_img, masks_frame, mask_opacity)
                if show_masks
                else result.orig_img
            )
            if output_frame.shape[1] != output_width or output_frame.shape[0] != output_height:
                output_frame = cv2.resize(
                    output_frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
            writer.write(output_frame)
            tracked_detections += update_track_votes(result, track_votes, model_names)
            processed_frames += 1
            source_frame = (
                min(processed_frames * frame_stride, frame_count)
                if frame_count > 0
                else processed_frames * frame_stride
            )

            if frame_count > 0:
                progress_bar.progress(min(source_frame / frame_count, 1.0))
            status_text.caption(f"Procesando fotograma {source_frame:,} de {frame_count:,}…")

            # Se muestra el video durante el análisis, pero se limita a cuatro
            # actualizaciones por segundo para no ralentizar YOLO.
            now = time.monotonic()
            if preview_placeholder is not None and (
                processed_frames == 1 or now - last_preview_update >= 0.25
            ):
                preview_frame = output_frame
                preview_width = preview_frame.shape[1]
                if preview_width > 960:
                    scale = 960 / preview_width
                    preview_frame = cv2.resize(
                        preview_frame,
                        (960, max(1, round(preview_frame.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                preview_placeholder.image(
                    preview_frame,
                    channels="BGR",
                    caption=(
                        "Video original con máscaras · detección en curso"
                        if show_masks
                        else "Video original · detección en curso"
                    ),
                    use_container_width=True,
                )
                last_preview_update = now

        writer.release()
        writer = None
        progress_bar.progress(1.0)

        if processed_frames == 0:
            raise ValueError("No se encontraron fotogramas para procesar.")

        video_measurements = (
            summarize_measurement_samples(video_measurement_samples, video_calibration)
            if video_calibration is not None
            else None
        )
        return output_path.read_bytes(), summarize_tracks(track_votes), tracked_detections, video_measurements
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        for temporary_path in (source_path, output_path):
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        results = None
        if track_votes is not None:
            track_votes.clear()
        if video_measurement_samples is not None:
            for values_by_category in video_measurement_samples.values():
                for values in values_by_category.values():
                    values.clear()
        gc.collect()


def render_photo_mode(
    model: YOLO | None,
    confidence: float,
    image_size: int,
    mask_opacity: float,
    low_limit: float,
    medium_limit: float,
    high_limit: float,
    model_loader: Callable[[], YOLO] | None = None,
) -> None:
    """Renderiza el flujo rápido de una sola fotografía."""
    st.markdown(
        """
        <div class="upload-heading">
          <span class="upload-step">1</span>
          <div><strong>Carga una fotografía</strong><small>PNG, JPG o TIFF</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    active_photo = st.session_state.get("active_photo")
    if active_photo is None:
        upload_revision = st.session_state.get("photo_upload_revision", 0)
        uploaded_image = st.file_uploader(
            "Seleccionar fotografía",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            label_visibility="collapsed",
            key=f"photo_upload_{upload_revision}",
        )
        if uploaded_image is not None:
            st.session_state["active_photo"] = {
                "name": uploaded_image.name,
                "bytes": uploaded_image.getvalue(),
            }
            st.rerun()

        st.button("Analizar foto", type="primary", use_container_width=True, disabled=True)
        st.info("Selecciona una foto para habilitar el botón de análisis.")
        return

    file_name_column, change_photo_column = st.columns([5, 1])
    with file_name_column:
        st.markdown(
            f'<div class="photo-file-row"><span>Foto cargada</span><strong>{escape(active_photo["name"])}</strong></div>',
            unsafe_allow_html=True,
        )
    with change_photo_column:
        change_photo = st.button("Cambiar", key="change_photo", use_container_width=True)
    if change_photo:
        st.session_state.pop("active_photo", None)
        st.session_state.pop("photo_analysis", None)
        st.session_state["photo_upload_revision"] = st.session_state.get("photo_upload_revision", 0) + 1
        st.rerun()

    analyze_photo = st.button("Analizar foto", type="primary", use_container_width=True)
    image_bytes = active_photo["bytes"]
    image_id = hashlib.sha256(image_bytes).hexdigest()
    saved_analysis = st.session_state.get("photo_analysis")
    if saved_analysis and (
        saved_analysis["image_id"] != image_id or saved_analysis.get("render_version") != PHOTO_RENDER_VERSION
    ):
        st.session_state.pop("photo_analysis", None)
        saved_analysis = None

    if not analyze_photo:
        if saved_analysis is None:
            return
    else:
        with st.spinner("Analizando la foto…"):
            try:
                analysis_model = model
                if analysis_model is None:
                    if model_loader is None:
                        raise RuntimeError("El modelo de inferencia aún no está disponible.")
                    analysis_model = model_loader()
                original_image, healthy_masks, sick_masks, counts, measurements = process_image(
                    analysis_model, image_bytes, confidence, image_size, mask_opacity
                )
            except Exception as error:
                st.exception(error)
                return
        st.session_state["photo_analysis"] = {
            "image_id": image_id,
            "render_version": PHOTO_RENDER_VERSION,
            "original_image": original_image,
            "healthy_masks": healthy_masks,
            "sick_masks": sick_masks,
            "counts": counts,
            "measurements": measurements,
        }
        saved_analysis = st.session_state["photo_analysis"]

    affected_percentage, grade, description = show_summary(
        saved_analysis["counts"],
        low_limit,
        medium_limit,
        high_limit,
        "Total de células detectadas",
        sidebar=True,
        measurements=saved_analysis["measurements"],
    )
    visible_photo_masks = st.pills(
        "Controles de máscaras",
        ["Sanas", "Enfermas"],
        selection_mode="multi",
        default=["Sanas", "Enfermas"],
        key=f"photo_visible_masks_{image_id}",
        label_visibility="collapsed",
    )
    show_healthy_masks = "Sanas" in visible_photo_masks
    show_sick_masks = "Enfermas" in visible_photo_masks

    selected_masks = np.zeros_like(saved_analysis["original_image"])
    if show_healthy_masks:
        selected_masks = cv2.bitwise_or(selected_masks, saved_analysis["healthy_masks"])
    if show_sick_masks:
        selected_masks = cv2.bitwise_or(selected_masks, saved_analysis["sick_masks"])
    has_selected_masks = show_healthy_masks or show_sick_masks
    image_to_display = (
        overlay_masks(saved_analysis["original_image"], selected_masks, mask_opacity)
        if has_selected_masks
        else saved_analysis["original_image"]
    )
    legend_items: list[str] = []
    if show_healthy_masks:
        legend_items.append('<span><i class="legend-dot legend-healthy"></i>Célula sana</span>')
    if show_sick_masks:
        legend_items.append('<span><i class="legend-dot legend-sick"></i>Célula enferma</span>')
    if legend_items:
        st.markdown(
            '<div class="mask-legend">' + "".join(legend_items) + "</div>",
            unsafe_allow_html=True,
        )
    st.image(image_to_display, channels="BGR")
    if saved_analysis["measurements"]:
        measurements = saved_analysis["measurements"]
        scale = measurements["micrometers_per_pixel"]
        bar_pixels = measurements["scale_bar_length_px"]
        healthy_measurement = format_class_measurement(measurements, "sana")
        sick_measurement = format_class_measurement(measurements, "enferma")
        st.markdown(
            f"""
            <section class="measurement-inline">
              <span class="measurement-inline-title">Estimación promedio por clase</span>
              <strong>Sanas: {healthy_measurement}</strong>
              <strong>Enfermas: {sick_measurement}</strong>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"Calibración {measurements['calibration_method']}: referencia de 50 µm · "
            f"{bar_pixels:.0f} px · escala {scale:.4f} µm/píxel."
        )
    else:
        st.info("No se segmentaron células; por eso no es posible calcular un promedio de tamaño.")

    encoded, image_png = cv2.imencode(".png", image_to_display)
    if encoded:
        download_image, download_report = st.columns(2)
        with download_image:
            st.download_button(
                "Descargar imagen PNG",
                data=image_png.tobytes(),
                file_name="celulas_segmentadas.png" if has_selected_masks else "celulas_original.png",
                mime="image/png",
                use_container_width=True,
            )
        with download_report:
            report = make_report_csv(
                saved_analysis["counts"],
                affected_percentage,
                grade,
                description,
                "Total de células detectadas",
                measurements=saved_analysis["measurements"],
            )
            st.download_button(
                "Descargar reporte CSV",
                data=report,
                file_name="reporte_celular.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.caption(
        "Cada célula segmentada en la fotografía se cuenta una sola vez. "
        "La calificación debe validarse con tu criterio científico o clínico."
    )


def render_live_metrics_panel(
    state: LiveSessionState,
    lot_name: str,
    current_sample_code: str,
    low_limit: float,
    medium_limit: float,
    high_limit: float,
) -> None:
    """Muestra las cuatro fichas de camarones y el acumulado del lote."""

    def metric_card(
        label: str,
        code: str,
        counts: dict[str, int],
        measurements: dict[str, Any] | None,
        low_limit: float,
        medium_limit: float,
        high_limit: float,
        card_class: str = "",
    ) -> str:
        affected_percentage, grade, _ = sample_result_grade(
            counts, low_limit, medium_limit, high_limit
        )
        measurement_text = ""
        if measurements:
            measurement_text = (
                f'<div class="live-card-measurements">'
                f"{escape(format_class_measurement(measurements, 'sana'))}<br>"
                f"{escape(format_class_measurement(measurements, 'enferma'))}"
                f"</div>"
            )
        return f"""
        <article class="live-sample-card {card_class}">
          <div class="live-sample-card-heading">
            <strong>{escape(label)}</strong>
            <span>Código: {escape(code)}</span>
          </div>
          <div class="live-card-grid">
            <div><span>Sanas</span><strong>{counts['sana']}</strong></div>
            <div><span>Enfermas</span><strong>{counts['enferma']}</strong></div>
            <div><span>Total</span><strong>{counts['total']}</strong></div>
            <div><span>Afectación</span><strong>{affected_percentage:.1f}%</strong></div>
          </div>
          <div class="live-card-grade"><span>Calificación</span><strong>{escape(grade)}</strong></div>
          {measurement_text}
        </article>
        """

    def render_metrics() -> None:
        snapshot = state.snapshot()
        completed_samples = state.completed_samples_snapshot()
        detection_requested = bool(st.session_state.get("live_detection_requested", False))
        counts = snapshot["counts"]
        sample_number = min(len(completed_samples) + 1, MAX_LIVE_SAMPLES)
        current_card = ""
        if (
            detection_requested
            or snapshot["detection_active"]
            or snapshot["sample_ready"]
            or snapshot["captured_frames"] > 0
        ) and len(completed_samples) < MAX_LIVE_SAMPLES:
            current_card = metric_card(
                f"Camarón muestra {sample_number}",
                current_sample_code or "Código pendiente",
                counts,
                snapshot["measurements"],
                low_limit,
                medium_limit,
                high_limit,
                "live-current-card",
            )

        sample_cards = "".join(
            metric_card(
                sample["label"],
                sample["code"],
                sample["counts"],
                sample.get("measurements"),
                low_limit,
                medium_limit,
                high_limit,
            )
            for sample in completed_samples
        )
        aggregate_counts, aggregate_measurements = aggregate_live_sample_metrics(completed_samples)
        aggregate_card = ""
        if completed_samples:
            aggregate_card = metric_card(
                f"Métrica general del lote ({len(completed_samples)}/{MAX_LIVE_SAMPLES} muestras)",
                "Acumulado",
                aggregate_counts,
                aggregate_measurements,
                low_limit,
                medium_limit,
                high_limit,
                "live-aggregate-card",
            )

        cards = current_card + sample_cards + aggregate_card
        if not cards:
            st.caption("Inicia la detección para comenzar a capturar la muestra.")
        else:
            st.markdown(
                f"""
                <div class="live-sample-cards">{cards}</div>
                """,
                unsafe_allow_html=True,
            )

        if (
            snapshot["captured_frames"] > 0
            or snapshot["sample_ready"]
            or snapshot["detection_active"]
            or detection_requested
        ):
            status = (
                "Grabando muestra"
                if snapshot["detection_active"] or detection_requested
                else "Muestra detenida"
            )
            st.caption(
                f"{status} · {snapshot['captured_frames']:,} fotogramas capturados · "
                f"{snapshot['processed_frames']:,} inferidos"
            )

        if not snapshot["detection_active"] and snapshot["camera_frames"] > 0:
            st.caption(f"Cámara activa · {snapshot['camera_frames']:,} fotogramas recibidos")

        if detection_requested and snapshot["camera_frames"] == 0:
            st.warning(
                "La cámara todavía no entrega fotogramas. Autoriza el acceso en el navegador "
                "y pulsa Iniciar cámara fuera del recuadro de video."
            )

        if snapshot["model_loading"]:
            st.caption("Preparando el modelo… el video continúa grabándose.")

        if snapshot["last_error"]:
            st.warning(f"La cámara continúa, pero una inferencia dio error: {snapshot['last_error']}")

    # El fragmento refresca solamente el panel, sin reiniciar la cámara completa.
    if hasattr(st, "fragment"):
        @st.fragment(run_every="1s")
        def live_metrics_fragment() -> None:
            render_metrics()

        live_metrics_fragment()
    else:
        render_metrics()


def render_live_sample_results(
    state: LiveSessionState,
    lot_name: str,
    camera_is_playing: bool,
    low_limit: float,
    medium_limit: float,
    high_limit: float,
) -> None:
    """Muestra la tabla final y habilita el reporte al completar cuatro muestras."""
    samples = state.completed_samples_snapshot()
    if not samples:
        return

    rows: list[dict[str, Any]] = []
    for sample in samples:
        affected_percentage, grade, _ = sample_result_grade(
            sample["counts"], low_limit, medium_limit, high_limit
        )
        rows.append(
            {
                "Muestra": sample["label"],
                "Código": sample["code"],
                "Sanas": sample["counts"]["sana"],
                "Enfermas": sample["counts"]["enferma"],
                "Total": sample["counts"]["total"],
                "Afectación": f"{affected_percentage:.1f}%",
                "Calificación": grade,
            }
        )

    st.subheader(f"Muestras de {lot_name or 'lote sin nombre'}")
    st.dataframe(rows, hide_index=True, use_container_width=True)

    aggregate_counts, aggregate_measurements = aggregate_live_sample_metrics(samples)
    aggregate_percentage, aggregate_grade, aggregate_description = sample_result_grade(
        aggregate_counts, low_limit, medium_limit, high_limit
    )
    st.markdown(
        f"""
        <section class="live-overall-summary">
          <strong>Métrica general del lote</strong>
          <span>{aggregate_counts['sana']} sanas · {aggregate_counts['enferma']} enfermas · {aggregate_counts['total']} células en total</span>
          <span>{aggregate_percentage:.1f}% de afectación · {escape(aggregate_grade)} · {escape(aggregate_description)}</span>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Editar códigos guardados", expanded=False):
        edited_codes: dict[int, str] = {}
        with st.form("live_edit_sample_codes_form", clear_on_submit=False):
            edit_columns = st.columns(2, gap="small")
            for index, sample in enumerate(samples):
                sample_number = int(sample["sample_number"])
                field_key = f"live_edit_sample_code_{sample_number}"
                if field_key not in st.session_state:
                    st.session_state[field_key] = sample["code"]
                with edit_columns[index % 2]:
                    edited_codes[sample_number] = st.text_input(
                        sample["label"],
                        key=field_key,
                    )
            save_code_edits = st.form_submit_button(
                "Guardar cambios",
                use_container_width=True,
            )
        if save_code_edits:
            for sample_number, code in edited_codes.items():
                state.update_completed_sample_code(sample_number, code)
            persist_completed_live_samples(state)
            st.rerun()

    if len(samples) == MAX_LIVE_SAMPLES:
        report = make_live_samples_report_csv(
            lot_name or "Lote sin nombre",
            samples,
            low_limit,
            medium_limit,
            high_limit,
        )
        st.download_button(
            "Descargar reporte del lote (4 muestras)",
            data=report,
            file_name=live_report_filename(lot_name),
            mime="text/csv",
            use_container_width=True,
            key="live_samples_report",
        )
    if st.button(
        "Nuevo lote / análisis",
        key="live_new_analysis",
        use_container_width=True,
        disabled=camera_is_playing or state.snapshot()["detection_active"],
    ):
        state.set_detection_active(False)
        st.session_state["live_detection_requested"] = False
        state.reset_metrics()
        state.clear_completed_samples()
        st.session_state.pop("live_completed_samples", None)
        st.session_state.pop("live_lot_name", None)
        for key in list(st.session_state):
            if key.startswith(("live_sample_code_", "live_edit_sample_code_")):
                st.session_state.pop(key, None)
        st.session_state["live_camera_playing"] = False
        st.rerun()


def render_live_camera(
    model: YOLO | None,
    confidence: float,
    mask_opacity: float,
    model_loader: Callable[[], YOLO] | None = None,
) -> None:
    """Mantiene la cámara activa y ejecuta el tracker solo cuando el usuario lo inicia."""
    if not WEBRTC_AVAILABLE:
        st.error("No se pudo cargar el componente de cámara en vivo.")
        st.caption(
            "Detalle de la dependencia: "
            f"{WEBRTC_IMPORT_ERROR or 'error de importación desconocido'}"
        )
        return

    state = get_live_session_state()
    save_feedback = st.session_state.pop("live_save_feedback", "")
    if save_feedback:
        st.success(save_feedback)
    model_names = get_model_names(model) if model is not None else None
    with state.lock:
        # El procesador WebRTC puede sobrevivir a un rerun de Streamlit; guardar
        # el modelo en el estado permite que detecte aun si el callback anterior
        # se creó antes de que el usuario pulsara "Iniciar detección".
        if model is not None:
            state.inference_model = model
            state.inference_model_names = model_names or DEFAULT_MODEL_NAMES.copy()
    if "live_camera_playing" not in st.session_state:
        st.session_state["live_camera_playing"] = False
    if "live_camera_requested" not in st.session_state:
        st.session_state["live_camera_requested"] = False
    if "live_detection_requested" not in st.session_state:
        st.session_state["live_detection_requested"] = state.snapshot()["detection_active"]
    if st.session_state["live_detection_requested"] and not state.snapshot()["detection_active"]:
        # El componente WebRTC puede provocar un rerun mientras el callback
        # del botón todavía termina. Recuperar aquí la bandera evita que el
        # primer fotograma vea la detección como detenida.
        state.set_detection_active(True)

    camera_is_playing = bool(st.session_state.get("live_camera_playing", False))
    camera_requested = bool(st.session_state.get("live_camera_requested", False))
    completed_count = len(state.completed_samples_snapshot())
    saved_pools = get_saved_pools()
    if "live_pool_selector_pending" in st.session_state:
        st.session_state["live_pool_selector"] = st.session_state.pop("live_pool_selector_pending")
    if "live_pool_selector" not in st.session_state:
        st.session_state["live_pool_selector"] = (
            saved_pools[0] if saved_pools else NEW_POOL_OPTION
        )
    pool_options = [*saved_pools, NEW_POOL_OPTION]

    pool_column, sample_column = st.columns([1.0, 1.0], gap="large")
    with pool_column:
        selected_pool = st.selectbox(
            "Piscina registrada",
            pool_options,
            key="live_pool_selector",
            disabled=completed_count > 0,
        )
        if selected_pool == NEW_POOL_OPTION:
            new_pool_name = st.text_input(
                "Nombre o código de la nueva piscina",
                key="live_new_pool_name",
                placeholder="Ej.: Piscina Norte 01",
                disabled=completed_count > 0,
            )
            if st.button(
                "Guardar piscina en la lista",
                key="live_save_pool",
                use_container_width=True,
                disabled=completed_count > 0,
            ):
                registered_pool = remember_pool(new_pool_name)
                if registered_pool:
                    st.session_state["live_pool_selector_pending"] = registered_pool
                    st.rerun()
                st.warning("Escribe un nombre o código para guardar la piscina.")
            lot_name = normalize_pool_name(new_pool_name)
        else:
            lot_name = normalize_pool_name(selected_pool)
        st.session_state["live_lot_name"] = lot_name

    current_sample_number = min(completed_count + 1, MAX_LIVE_SAMPLES)
    sample_code_key = f"live_sample_code_{current_sample_number}"
    if sample_code_key not in st.session_state:
        st.session_state[sample_code_key] = f"CAM-{current_sample_number:02d}"
    with sample_column:
        sample_code = st.text_input(
            f"Código del camarón · muestra {current_sample_number} de {MAX_LIVE_SAMPLES}",
            key=sample_code_key,
        ).strip()
    resolution_label = st.selectbox(
        "Resolución de captura de video",
        list(LIVE_RESOLUTIONS),
        index=0,
        key="live_resolution_label",
        disabled=camera_is_playing or camera_requested,
    )
    resolution = LIVE_RESOLUTIONS[resolution_label]

    visible_masks = st.pills(
        "Controles de máscaras",
        ["Sanas", "Enfermas"],
        selection_mode="multi",
        default=["Sanas", "Enfermas"],
        key="live_visible_masks",
        label_visibility="collapsed",
    )
    show_healthy_masks = "Sanas" in visible_masks
    show_sick_masks = "Enfermas" in visible_masks
    with state.lock:
        state.show_healthy_masks = show_healthy_masks
        state.show_sick_masks = show_sick_masks

    # Los tres controles viven fuera del video. Se usan las columnas directamente
    # en lugar de `st.empty()`: los placeholders podían conservar un contenedor
    # vacío después de un rerun de WebRTC y aparentar un segundo control blanco.
    camera_action, detection_action, save_action = st.columns(
        [1.05, 1.25, 1.15], gap="small"
    )

    def toggle_live_camera() -> None:
        """Solicita iniciar o detener la cámara desde el botón exterior."""
        requested = bool(st.session_state.get("live_camera_requested", False))
        if requested:
            # Si se detiene la cámara durante una muestra, la muestra queda
            # cerrada pero sus métricas siguen disponibles para guardarlas.
            if state.snapshot()["detection_active"]:
                state.set_detection_active(False)
                st.session_state["live_detection_requested"] = False
            st.session_state["live_camera_requested"] = False
            st.session_state["live_camera_playing"] = False
            return
        st.session_state["live_camera_requested"] = True
        st.session_state.pop("live_camera_start_required", None)

    def toggle_live_detection() -> None:
        """Cambia la detección antes de que Streamlit vuelva a dibujar la interfaz."""
        if state.snapshot()["detection_active"] or st.session_state.get(
            "live_detection_requested", False
        ):
            state.set_detection_active(False)
            st.session_state["live_detection_requested"] = False
            return

        state.reset_metrics()
        with state.lock:
            active_model = state.inference_model
        if active_model is not None:
            with LIVE_INFERENCE_LOCK:
                reset_trackers(active_model)
        if lot_name:
            remember_pool(lot_name)
        if not bool(st.session_state.get("live_camera_requested", False)):
            st.session_state["live_detection_requested"] = False
            st.session_state["live_camera_start_required"] = True
            return
        st.session_state["live_detection_requested"] = True
        state.set_detection_active(True)

    def ensure_live_model() -> None:
        """Prepara YOLO en segundo plano sin detener el flujo de la cámara."""
        if model_loader is None:
            return
        with state.lock:
            if state.inference_model is not None or state.model_loading:
                return
            state.model_loading = True

        def load_in_background() -> None:
            try:
                loaded_model = model_loader()
                loaded_names = get_model_names(loaded_model)
                with state.lock:
                    state.inference_model = loaded_model
                    state.inference_model_names = loaded_names
                    state.model_loading = False
                    state.inference_retry_at = 0.0
                    state.last_error = ""
            except Exception as error:
                with state.lock:
                    state.model_loading = False
                    state.last_error = f"{type(error).__name__}: {error}"
                    state.inference_retry_at = time.monotonic() + LIVE_INFERENCE_RETRY_SECONDS

        Thread(target=load_in_background, daemon=True).start()

    def run_live_inference(image: np.ndarray, generation: int) -> None:
        """Ejecuta YOLO fuera del callback para no congelar el video al comenzar."""
        try:
            with state.lock:
                active_model = state.inference_model
                model_names = dict(state.inference_model_names)
            if active_model is None:
                raise RuntimeError("El modelo aún no está listo; vuelve a iniciar la detección.")
            with LIVE_INFERENCE_LOCK:
                results = active_model.track(
                    source=image,
                    conf=confidence,
                    imgsz=resolution["inference_size"],
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            if not results:
                empty_mask = np.zeros_like(image)
                with state.lock:
                    if state.inference_generation == generation and state.detection_active:
                        state.cached_shape = image.shape
                        state.cached_healthy_masks = empty_mask
                        state.cached_sick_masks = empty_mask.copy()
                        if state.calibration is None:
                            state.calibration = calibration_from_image(image)
                        state.processed_frames += 1
                        state.last_error = ""
                        state.inference_retry_at = 0.0
                return

            result = results[0]
            healthy_masks = mask_frame(result, image.shape, model_names, {"sana"})
            sick_masks = mask_frame(result, image.shape, model_names, {"enferma"})
            with state.lock:
                # Si el usuario detuvo o reinició la muestra mientras YOLO
                # trabajaba, el resultado viejo no debe contaminar la siguiente.
                if state.inference_generation != generation or not state.detection_active:
                    return
                state.cached_shape = image.shape
                state.cached_healthy_masks = healthy_masks
                state.cached_sick_masks = sick_masks
                if state.calibration is None:
                    state.calibration = calibration_from_image(image)
                update_track_votes(result, state.track_votes, model_names)
                collect_measurement_samples(result, model_names, state.measurement_samples)
                state.processed_frames += 1
                state.last_error = ""
                state.inference_retry_at = 0.0
        except Exception as error:
            with state.lock:
                if state.inference_generation == generation:
                    state.last_error = f"{type(error).__name__}: {error}"
                    # Evita crear un hilo nuevo por cada fotograma si el backend
                    # devuelve un error rápido o tarda en preparar el predictor.
                    state.inference_retry_at = time.monotonic() + LIVE_INFERENCE_RETRY_SECONDS
        finally:
            with state.lock:
                if state.inference_generation == generation:
                    state.inference_busy = False

    def process_live_frame(frame: Any) -> Any:
        image: np.ndarray | None = None

        def make_output_frame(output: np.ndarray) -> av.VideoFrame:
            """Devuelve un cuadro reproducible y conserva su reloj WebRTC."""
            output_frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(output),
                format="bgr24",
            )
            # aiortc necesita conservar PTS/time_base para que Chrome pueda
            # reproducir el track de salida de forma continua.
            output_frame.pts = getattr(frame, "pts", None)
            output_frame.time_base = getattr(frame, "time_base", None)
            return output_frame

        try:
            image = frame.to_ndarray(format="bgr24")
            with state.lock:
                state.frame_number += 1
                state.camera_frames += 1
                frame_number = state.frame_number
                detection_active = state.detection_active
                cached_shape = state.cached_shape
                cached_healthy_masks = state.cached_healthy_masks
                cached_sick_masks = state.cached_sick_masks
                inference_busy = state.inference_busy
                active_model_available = state.inference_model is not None
                inference_generation = state.inference_generation
                show_healthy = state.show_healthy_masks
                show_sick = state.show_sick_masks
                inference_retry_at = state.inference_retry_at

            if not detection_active:
                # No reutilizar el objeto recibido por aiortc: el componente
                # necesita un cuadro nuevo para entregar el video al navegador.
                state.set_latest_frame(image)
                return make_output_frame(image)

            # La captura se contabiliza antes de esperar a OpenVINO. Así,
            # detener la sesión siempre deja una muestra guardable, incluso
            # si una inferencia puntual tarda o devuelve cero detecciones.
            with state.lock:
                state.captured_frames += 1

            needs_inference = (
                cached_healthy_masks is None
                or cached_sick_masks is None
                or cached_shape != image.shape
                or frame_number % resolution["inference_every"] == 1
            )
            if (
                needs_inference
                and active_model_available
                and not inference_busy
                and time.monotonic() >= inference_retry_at
            ):
                # Solo se agenda un fotograma a la vez. El callback sigue
                # entregando video mientras OpenVINO termina el fotograma.
                with state.lock:
                    if (
                        state.detection_active
                        and not state.inference_busy
                        and state.inference_generation == inference_generation
                    ):
                        state.inference_busy = True
                        inference_busy = True
                        Thread(
                            target=run_live_inference,
                            args=(image.copy(), inference_generation),
                            daemon=True,
                        ).start()

            masks = np.zeros_like(image)
            if show_healthy and cached_healthy_masks is not None:
                masks = cv2.bitwise_or(masks, cached_healthy_masks)
            if show_sick and cached_sick_masks is not None:
                masks = cv2.bitwise_or(masks, cached_sick_masks)
            output = overlay_masks(image, masks, mask_opacity) if (show_healthy or show_sick) else image
            output = draw_live_status_overlay(
                output,
                True,
            )
            if frame_number % LIVE_RECORD_EVERY_N_FRAMES == 1:
                state.record_frame(
                    output,
                    resolution["frame_rate"] / LIVE_RECORD_EVERY_N_FRAMES,
                )
            state.set_latest_frame(output)
            return make_output_frame(output)
        except Exception as error:
            with state.lock:
                state.last_error = f"{type(error).__name__}: {error}"
            # Si un cuadro puntual falla, se conserva el video en lugar de cerrar la cámara.
            if image is not None:
                state.set_latest_frame(image)
                return make_output_frame(image)
            return frame

    with camera_action:
        st.button(
            "Detener cámara" if camera_requested else "Iniciar cámara",
            type="secondary" if camera_requested else "primary",
            key="live_camera_toggle",
            use_container_width=True,
            on_click=toggle_live_camera,
        )

    camera_column, status_column = st.columns([1.55, 1.45], gap="large")
    with camera_column:
        # WebRTC se encarga de transportar y pintar el video a la frecuencia
        # de la cámara. Streamlit no debe reconstruir una imagen JPEG en cada
        # rerun: hacerlo por el WebSocket de la interfaz causa saltos visibles.
        camera_playing = False
        ice_state = ""
        if camera_requested:
            webrtc_options: dict[str, Any] = {
                "key": "cell_live_camera",
                "mode": WebRtcMode.SENDRECV,
                "desired_playing_state": True,
                "rtc_configuration": rtc_configuration(),
                "media_stream_constraints": {
                    "video": {
                        "width": {"ideal": resolution["width"]},
                        "height": {"ideal": resolution["height"]},
                        "frameRate": {"ideal": resolution["frame_rate"]},
                    },
                    "audio": False,
                },
                "video_frame_callback": process_live_frame,
                "async_processing": False,
                "sendback_video": True,
                "sendback_audio": False,
                "translations": {
                    "device_ask_permission": "Autoriza el acceso a la cámara para comenzar.",
                    "device_not_available": "No se encontró una cámara disponible.",
                    "device_access_denied": "Se denegó el acceso a la cámara.",
                },
            }
            camera_context = webrtc_streamer(**webrtc_options)
            camera_state = getattr(camera_context, "state", None)
            camera_playing = bool(getattr(camera_state, "playing", False))
            ice_state = str(getattr(camera_state, "ice_connection_state", ""))
            st.session_state["live_camera_playing"] = camera_playing
            if camera_playing:
                st.session_state.pop("live_camera_start_required", None)
                st.caption("Cámara activa · usa el botón superior para detenerla.")
            else:
                st.info("Conectando cámara… si Chrome solicita permiso, selecciona Permitir.")
        else:
            st.session_state["live_camera_playing"] = False
            if st.session_state.pop("live_camera_start_required", False):
                st.warning("Pulsa **Iniciar cámara** para mostrar el video.")
            else:
                st.caption("Pulsa **Iniciar cámara** para mostrar el video.")

        camera_snapshot = state.snapshot()
        if camera_playing and camera_snapshot["camera_frames"] == 0:
            st.warning(
                "La cámara está encendida, pero todavía no llegan fotogramas. "
                "Si permanece en blanco durante varios segundos, permite la cámara "
                "en Chrome y verifica TURN en Manage app → Settings → Secrets."
            )
        if ice_state.lower() in {"failed", "disconnected", "closed"}:
            st.warning(
                "El navegador no pudo conectar el video. Revisa el permiso de cámara o "
                "prueba otra red; algunas redes requieren un servidor TURN."
            )
        with status_column:
            render_live_metrics_panel(
                state,
                lot_name,
                sample_code,
                10.0,
                30.0,
                60.0,
            )

    detection_is_active = state.snapshot()["detection_active"] or bool(
        st.session_state.get("live_detection_requested", False)
    )
    with detection_action:
        detection_label = "Detener detección" if detection_is_active else "Iniciar detección"
        st.button(
            detection_label,
            type="primary",
            key="live_detection_toggle",
            use_container_width=True,
            disabled=(
                completed_count >= MAX_LIVE_SAMPLES
                or not camera_requested
            ),
            on_click=toggle_live_detection,
        )
    if state.snapshot()["detection_active"] or st.session_state.get(
        "live_detection_requested", False
    ):
        ensure_live_model()
    with save_action:
        snapshot_after_action = state.snapshot()
        if st.button(
            "Guardar muestra",
            key="live_sample_save",
            use_container_width=True,
            disabled=(
                detection_is_active
                or completed_count >= MAX_LIVE_SAMPLES
                or not snapshot_after_action["sample_ready"]
            ),
        ):
            state.set_detection_active(False)
            st.session_state["live_detection_requested"] = False
            if lot_name:
                remember_pool(lot_name)
            if state.save_current_sample(sample_code, lot_name):
                persist_completed_live_samples(state)
                st.session_state["live_save_feedback"] = "Muestra guardada correctamente."
                st.rerun()
            else:
                st.warning("No hay una captura detenida lista para guardar.")

    render_live_sample_results(
        state,
        lot_name,
        camera_playing,
        10.0,
        30.0,
        60.0,
    )


def main() -> None:
    st.set_page_config(page_title="Análisis celular", layout="wide")
    st.markdown(
        """
        <style>
            .stApp { background: #061a33; color: #ffffff; }
            [data-testid="stHeader"] { background: rgba(0, 0, 0, 0); }
            [data-testid="stSidebar"] { background: #0a2745; }
            h1, h2, h3, h4, p, label, .stMarkdown, .stCaption,
            [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
            [data-testid="stMetricDelta"] { color: #ffffff !important; }
            [data-testid="stMetric"] {
                background: #103858;
                border: 1px solid #41d8cc;
                border-radius: 10px;
                padding: 0.75rem;
            }
            [data-testid="stFileUploader"] {
                background: #0d3153;
                border: 1px solid #41d8cc;
                border-radius: 10px;
                padding: 0.2rem 0.35rem;
            }
            [data-testid="stFileUploaderDropzone"] {
                min-height: 3.1rem !important;
                padding: 0.25rem 0.55rem !important;
                background: #0d3153 !important;
                border: 1px dashed #6be7da !important;
                border-radius: 7px !important;
            }
            /* Una vez cargada la foto, solo queda su nombre en una fila compacta. */
            [data-testid="stFileUploader"]:has([data-testid="stFileUploaderFile"])
            [data-testid="stFileUploaderDropzone"] {
                display: none !important;
            }
            [data-testid="stFileUploaderFile"] {
                min-height: 2.2rem !important;
                margin: 0 !important;
                padding: 0.25rem 0.4rem !important;
                font-size: 0.85rem !important;
            }
            [data-testid="stFileUploaderFile"] button {
                min-height: 1.8rem !important;
                min-width: 1.8rem !important;
                padding: 0.15rem !important;
            }
            [data-testid="stFileUploaderDropzone"] svg {
                width: 1.45rem !important;
                height: 1.45rem !important;
                color: #41d8cc !important;
            }
            [data-testid="stFileUploaderDropzoneInstructions"] {
                margin: 0 !important;
            }
            [data-testid="stFileUploaderDropzoneInstructions"] div {
                font-size: 0.75rem !important;
                color: #ffffff !important;
                opacity: 1 !important;
            }
            [data-testid="stFileUploaderDropzoneInstructions"] *,
            [data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] * {
                color: #ffffff !important;
                opacity: 1 !important;
            }
            [data-testid="stFileUploaderDropzone"] button {
                background: #12b8c2 !important;
                color: #ffffff !important;
                border: none !important;
                font-weight: 700 !important;
                min-height: 2rem !important;
                padding: 0.25rem 0.75rem !important;
            }
            [data-testid="stFileUploaderDropzone"] button * {
                color: #ffffff !important;
            }
            .upload-heading {
                display: flex;
                align-items: center;
                gap: 0.65rem;
                margin: 0.15rem 0 0.45rem;
                color: #ffffff;
            }
            .upload-heading strong { display: block; font-size: 1rem; }
            .upload-heading small { display: block; color: #a6d8eb; font-size: 0.78rem; }
            .upload-step {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 1.65rem;
                height: 1.65rem;
                border-radius: 50%;
                background: #12b8c2;
                color: #ffffff;
                font-weight: 800;
            }
            .photo-file-row {
                box-sizing: border-box;
                min-height: 2.75rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin: 0 !important;
                padding: 0.45rem 0.75rem;
                background: #0d3153;
                border: 1px solid #41d8cc;
                border-radius: 8px;
                color: #ffffff;
                overflow: hidden;
            }
            .photo-file-row span {
                flex: 0 0 auto;
                color: #a6d8eb;
                font-size: 0.78rem;
            }
            .photo-file-row strong {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                font-size: 0.9rem;
            }
            [data-testid="stAlert"] { color: #ffffff; }
            .stButton > button, .stDownloadButton > button,
            button[kind="primary"], [data-testid="stBaseButton-primary"] {
                background: #12b8c2 !important;
                color: #ffffff !important;
                border: 1px solid #8ef4e8 !important;
                font-weight: 750 !important;
                min-height: 2.75rem;
            }
            button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
                background: #31546b !important;
                color: #ffffff !important;
                border: 1px solid #6bbfd1 !important;
                font-weight: 750 !important;
                min-height: 2.75rem;
            }
            .stButton > button *, .stDownloadButton > button *,
            button[kind="primary"] *, [data-testid="stBaseButton-primary"] * {
                color: #ffffff !important;
            }
            button[kind="secondary"] *, [data-testid="stBaseButton-secondary"] * {
                color: #ffffff !important;
            }
            .stButton > button:disabled, button[kind="primary"]:disabled {
                background: #31546b !important;
                border-color: #31546b !important;
                color: #b6c8d5 !important;
                opacity: 1 !important;
            }
            input, textarea { color: #ffffff !important; background: #0d3153 !important; }
            /* Streamlit/BaseWeb vuelve a pintar estos controles con fondo blanco
               durante los reruns de la cámara. Mantener contraste en todo momento. */
            [data-testid="stTextInput"] > div > div,
            [data-testid="stTextInput"] input {
                background: #0d3153 !important;
                color: #ffffff !important;
                border-color: #6bbfd1 !important;
                caret-color: #ffffff !important;
            }
            [data-testid="stTextInput"] [data-baseweb="input"],
            [data-testid="stTextInput"] [data-baseweb="base-input"],
            [data-testid="stTextInput"] [data-baseweb="input"] > div,
            [data-testid="stTextInput"] [data-baseweb="base-input"] > div,
            [data-testid="stSelectbox"] [data-baseweb="select"],
            [data-testid="stSelectbox"] [data-baseweb="select"] > div {
                background-color: #0d3153 !important;
                border-color: #6bbfd1 !important;
            }
            [data-testid="stTextInput"] input::placeholder {
                color: #a6d8eb !important;
                opacity: 1 !important;
            }
            [data-baseweb="input"],
            [data-baseweb="input"] > div,
            [data-baseweb="base-input"],
            [data-baseweb="base-input"] > div {
                background: #0d3153 !important;
                border-color: #6bbfd1 !important;
            }
            [data-baseweb="input"] input,
            [data-baseweb="base-input"] input {
                background: transparent !important;
                color: #ffffff !important;
                caret-color: #ffffff !important;
            }
            [data-testid="stSelectbox"] [data-baseweb="select"] > div {
                background: #0d3153 !important;
                border-color: #6bbfd1 !important;
                color: #ffffff !important;
            }
            [data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
                background: #0d3153 !important;
            }
            [data-testid="stSelectbox"] [data-baseweb="select"] div,
            [data-testid="stSelectbox"] [data-baseweb="select"] span,
            [data-testid="stSelectbox"] [data-baseweb="select"] input {
                color: #ffffff !important;
            }
            [data-testid="stSelectbox"] [data-baseweb="select"] svg {
                fill: #c4fffa !important;
            }
            [data-baseweb="popover"] [role="listbox"],
            [data-baseweb="popover"] [role="option"] {
                background: #0d3153 !important;
                color: #ffffff !important;
            }
            [data-baseweb="popover"] [role="option"]:hover,
            [data-baseweb="popover"] [aria-selected="true"] {
                background: #176b8a !important;
                color: #ffffff !important;
            }
            [data-testid="stRadio"] div[role="radiogroup"] { gap: 0.5rem; }
            [data-testid="stRadio"] label {
                background: #0d3153;
                border: 1px solid #2a7895;
                border-radius: 8px;
                color: #ffffff !important;
                padding: 0.4rem 1rem;
            }
            [data-testid="stRadio"] label:has(input:checked) {
                background: #176b8a;
                border-color: #6bbfd1;
                color: #ffffff !important;
            }
            [data-testid="stRadio"] label:has(input:checked) p { color: #ffffff !important; }
            [data-testid="stRadio"] label:has(input:checked) > div:first-child,
            [data-testid="stToggle"] label:has(input:checked) > div:first-child {
                background: #12b8c2 !important;
                border-color: #12b8c2 !important;
            }
            [data-testid="stRadio"] label:has(input:checked) > div:first-child > div,
            [data-testid="stToggle"] label:has(input:checked) > div:first-child > div {
                background: #ffffff !important;
            }
            [data-testid="stRadio"] label p, [data-testid="stToggle"] label,
            [data-testid="stToggle"] p { color: #ffffff !important; }
            button[data-testid="stBaseButton-pills"],
            button[data-testid="stBaseButton-pillsActive"],
            [data-testid="stPills"] button {
                background: #0d3153 !important;
                border: 1px solid #41d8cc !important;
                color: #ffffff !important;
                border-radius: 999px !important;
            }
            button[data-testid="stBaseButton-pillsActive"],
            [data-testid="stPills"] button[aria-pressed="true"] {
                background: #176b8a !important;
                border-color: #6bbfd1 !important;
                color: #ffffff !important;
            }
            [data-testid="stPills"] button * { color: inherit !important; }
            /* El video debe permanecer en WebRTC: así conserva la frecuencia
               de la cámara y no depende de reruns del WebSocket de Streamlit. */
            [data-testid="stCustomComponentV1"] {
                width: 100% !important;
                min-height: 24rem !important;
                height: auto !important;
                overflow: visible !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            [data-testid="stCustomComponentV1"] iframe {
                display: block !important;
                position: relative !important;
                width: 100% !important;
                min-height: 24rem !important;
                height: 30rem !important;
                border: 0 !important;
            }
            .live-preview {
                max-width: 42rem;
                margin: 0.25rem 0 0.8rem;
                padding: 0.55rem 0.7rem;
                background: #0a2745;
                border: 1px solid #41d8cc;
                border-radius: 10px;
            }
            .live-preview-header {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                color: #ffffff;
                font-size: 0.9rem;
            }
            .live-preview-dot {
                width: 0.55rem;
                height: 0.55rem;
                border-radius: 50%;
                background: #12b8c2;
                box-shadow: 0 0 0 3px rgba(18, 184, 194, 0.18);
            }
            .live-preview-state {
                margin-left: auto;
                color: #a6f5f0;
                font-size: 0.72rem;
                font-weight: 700;
            }
            .live-preview-screen {
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 3.8rem;
                margin-top: 0.5rem;
                padding: 0.45rem;
                background: #061a33;
                border: 1px dashed #2c8396;
                border-radius: 7px;
                color: #a6d8eb;
                font-size: 0.82rem;
                text-align: center;
            }
            .live-video-placeholder {
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 24rem;
                padding: 1rem;
                background: #061a33;
                border: 1px solid #2c8396;
                border-radius: 12px;
                color: #a6d8eb;
                text-align: center;
            }
            .live-preview-footer {
                margin-top: 0.35rem;
                color: #a6d8eb;
                font-size: 0.72rem;
            }
            .live-measurements-heading {
                margin: 1rem 0 0.45rem;
                color: #a6f5f0;
                font-size: 0.88rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .live-measurement-list {
                display: grid;
                gap: 0.55rem;
                margin-bottom: 0.75rem;
            }
            .live-measurement-list div {
                display: flex;
                flex-direction: column;
                gap: 0.15rem;
                padding: 0.55rem 0.7rem;
                background: #0d3153;
                border: 1px solid rgba(107, 191, 209, 0.58);
                border-radius: 8px;
            }
            .live-measurement-list strong { color: #ffffff; font-size: 0.82rem; }
            .live-measurement-list span { color: #e8ffff; font-size: 0.78rem; line-height: 1.35; }
            .live-sample-cards {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.65rem;
                margin-top: 0.7rem;
            }
            .live-sample-card {
                min-width: 0;
                padding: 0.75rem;
                background: #0d3153;
                border: 1px solid rgba(107, 191, 209, 0.62);
                border-radius: 10px;
                color: #ffffff;
            }
            .live-current-card {
                border-color: #c4fffa;
                box-shadow: 0 0 0 1px rgba(196, 255, 250, 0.18);
            }
            .live-aggregate-card {
                background: #176b8a;
                border: 2px solid #c4fffa;
            }
            .live-sample-card-heading {
                display: flex;
                flex-direction: column;
                gap: 0.15rem;
                margin-bottom: 0.55rem;
            }
            .live-sample-card-heading strong {
                color: #ffffff;
                font-size: 0.86rem;
                line-height: 1.2;
            }
            .live-sample-card-heading span {
                color: #a6f5f0;
                font-size: 0.7rem;
                overflow-wrap: anywhere;
            }
            .live-card-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.38rem;
            }
            .live-card-grid div {
                display: flex;
                flex-direction: column;
                gap: 0.1rem;
                padding: 0.35rem;
                background: rgba(6, 26, 51, 0.35);
                border-radius: 6px;
            }
            .live-card-grid span, .live-card-grade span {
                color: #a6d8eb;
                font-size: 0.66rem;
            }
            .live-card-grid strong {
                color: #ffffff;
                font-size: 0.9rem;
            }
            .live-card-grade {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.4rem;
                margin-top: 0.45rem;
                padding-top: 0.45rem;
                border-top: 1px solid rgba(107, 231, 218, 0.22);
            }
            .live-card-grade strong {
                color: #c4fffa;
                font-size: 0.8rem;
                text-align: right;
            }
            .live-card-measurements {
                margin-top: 0.45rem;
                color: #e8ffff;
                font-size: 0.65rem;
                line-height: 1.35;
            }
            .live-empty-metrics {
                margin: 0.7rem 0 0;
                color: #a6d8eb;
                font-size: 0.82rem;
            }
            .live-overall-summary {
                display: flex;
                flex-direction: column;
                gap: 0.25rem;
                margin: 0.85rem 0;
                padding: 0.9rem 1rem;
                background: #176b8a;
                border: 2px solid #c4fffa;
                border-radius: 12px;
                color: #ffffff;
            }
            .live-overall-summary strong { font-size: 1.05rem; }
            .live-overall-summary span { color: #e8ffff; font-size: 0.88rem; }
            @media (max-width: 1100px) {
                .live-sample-cards { grid-template-columns: 1fr; }
            }
            .mask-legend {
                display: inline-flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 0.45rem 1.1rem;
                margin: 0.5rem 0 0.75rem;
                padding: 0.48rem 0.85rem;
                background: linear-gradient(135deg, rgba(23, 107, 138, 0.58), rgba(10, 39, 69, 0.78));
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid rgba(107, 191, 209, 0.62);
                border-radius: 9px;
                color: #ffffff;
                font-size: 0.85rem;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.16);
            }
            .mask-legend span {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                color: #ffffff;
                font-weight: 650;
            }
            .legend-dot {
                display: inline-block;
                width: 0.65rem;
                height: 0.65rem;
                border-radius: 50%;
            }
            .legend-healthy { background: #48b848; }
            .legend-sick { background: #eb3c3c; }
            .results-panel {
                background: #0a2745;
                border: 1px solid #41d8cc;
                border-radius: 16px;
                padding: 1rem;
                margin-top: 0.5rem;
            }
            .results-heading {
                color: #ffffff;
                font-size: 1.7rem;
                font-weight: 750;
                margin-bottom: 1.1rem;
            }
            .results-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 1rem;
            }
            .result-card {
                min-width: 0;
                background: #176b8a;
                border: 1px solid #6bbfd1;
                border-radius: 12px;
                min-height: 4.9rem;
                padding: 1.1rem 1.15rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.75rem;
            }
            .result-label {
                display: block;
                color: #ffffff;
                font-size: 1.05rem;
                font-weight: 650;
                line-height: 1.2;
                min-height: 0;
                overflow-wrap: anywhere;
                text-shadow: 0 2px 4px rgba(6, 26, 51, 0.62);
            }
            .result-value {
                display: block;
                color: #ffffff;
                font-size: 2.5rem;
                line-height: 1;
                font-weight: 700;
                text-align: right;
                text-shadow: 0 2px 4px rgba(6, 26, 51, 0.62);
            }
            .measurement-card {
                min-height: 0;
                display: block;
                padding: 1rem 1.15rem;
            }
            .measurement-heading {
                display: block;
                color: #ffffff;
                font-size: 1.05rem;
                font-weight: 750;
                text-shadow: 0 2px 4px rgba(6, 26, 51, 0.62);
            }
            .measurement-row {
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                gap: 0.75rem;
                margin-top: 0.5rem;
                padding-top: 0.5rem;
                border-top: 1px solid rgba(255, 255, 255, 0.45);
            }
            .measurement-row span {
                color: #e8ffff;
                font-size: 0.82rem;
                font-weight: 650;
            }
            .measurement-row strong {
                color: #ffffff;
                font-size: 1.05rem;
                font-weight: 800;
                text-align: right;
                text-shadow: 0 2px 4px rgba(6, 26, 51, 0.62);
            }
            .measurement-empty {
                margin-top: 0.5rem;
                color: #e8ffff;
                font-size: 0.9rem;
            }
            .measurement-inline {
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 0.4rem 1rem;
                margin: 0.7rem 0 0.25rem;
                padding: 0.65rem 0.85rem;
                background: #0d3153;
                border: 1px solid #41d8cc;
                border-radius: 9px;
                color: #ffffff;
            }
            .measurement-inline-title {
                color: #a6f5f0;
                font-size: 0.85rem;
                font-weight: 700;
            }
            .measurement-inline strong {
                color: #ffffff;
                font-size: 1rem;
                text-shadow: 0 2px 4px rgba(6, 26, 51, 0.62);
            }
            .grade-card { background: #1b7894; flex-wrap: wrap; }
            .grade-featured {
                min-height: 8.1rem;
                padding: 1.3rem 1.25rem;
                border: 2px solid #c4fffa;
                box-shadow: 0 7px 18px rgba(0, 0, 0, 0.18);
            }
            .grade-featured .result-label { font-size: 1.2rem; }
            .grade-featured .grade-value { font-size: 2.45rem; }
            .grade-featured .grade-detail { font-size: 1.05rem; margin-top: 0.15rem; }
            .grade-value {
                display: block;
                color: #ffffff;
                font-size: 1.45rem;
                font-weight: 750;
                line-height: 1.2;
                margin-left: auto;
                text-shadow: 0 2px 4px rgba(6, 26, 51, 0.62);
            }
            .grade-detail {
                display: inline-block;
                color: #ffffff;
                font-size: 0.9rem;
                font-weight: 600;
                flex-basis: 100%;
                margin-top: -0.25rem;
                text-shadow: 0 2px 4px rgba(6, 26, 51, 0.62);
            }
            .results-note {
                color: #e7f2ff !important;
                font-size: 0.9rem;
                line-height: 1.45;
                margin: 3.25rem 0 0 !important;
                padding-top: 0.25rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Monitoreo celular en tiempo real")

    # Parámetros internos: se ocultan para mantener la interfaz enfocada en el análisis.
    confidence = 0.35
    image_size = PHOTO_IMAGE_SIZE
    mask_opacity = 0.50
    low_limit, medium_limit, high_limit = 10.0, 30.0, 60.0

    model_path = default_model_path()
    if model_path is None and MODEL_BACKEND in {"openvino", "ov"}:
        st.session_state["openvino_fallback"] = True
        model_path = default_pt_model_path()
    if model_path is None or not (model_path.is_file() or is_openvino_model_dir(model_path)):
        if MODEL_BACKEND in {"openvino", "ov"}:
            st.error(
                "Se seleccionó OpenVINO, pero no se encontró el modelo convertido. "
                "Configura `CELL_OPENVINO_MODEL_DIR` o `CELL_OPENVINO_MODEL_URL`."
            )
        else:
            st.error("No se encontró el archivo del modelo `best.pt`.")
        st.stop()

    def get_app_model() -> YOLO:
        """Carga el modelo solo cuando el modo elegido realmente lo necesita."""
        nonlocal model_path
        try:
            loaded_model = get_session_model(model_path)
        except Exception as error:
            if MODEL_BACKEND in {"openvino", "ov"} and model_path.is_dir():
                st.session_state["openvino_fallback"] = True
                fallback_path = default_pt_model_path()
                try:
                    loaded_model = get_session_model(fallback_path)
                    st.warning(
                        "OpenVINO no pudo compilarse en este CPU; se está usando `.pt` como respaldo. "
                        "La app continúa funcionando, pero este equipo requiere otra optimización."
                    )
                    model_path = fallback_path
                except Exception as fallback_error:
                    st.error(
                        "No fue posible cargar el modelo OpenVINO ni el respaldo `.pt`: "
                        f"{fallback_error}"
                    )
                    st.stop()
            else:
                st.error(f"No fue posible cargar el modelo: {error}")
                st.stop()

        if loaded_model.task != "segment":
            st.error(
                f"El modelo cargado es de tipo `{loaded_model.task}`. "
                "Esta aplicación requiere un modelo de segmentación."
            )
            st.stop()
        return loaded_model

    analysis_mode = st.radio(
        "Modo de análisis",
        ["Foto", "Video"],
        horizontal=True,
        key="analysis_mode",
    )
    if analysis_mode == "Foto":
        render_photo_mode(
            None,
            confidence,
            image_size,
            mask_opacity,
            low_limit,
            medium_limit,
            high_limit,
            model_loader=get_app_model,
        )
        return

    video_section = st.radio(
        "Sección de video", ["Cámara en vivo", "Subir video"], horizontal=True, key="video_section"
    )
    if video_section == "Cámara en vivo":
        # La carga/compilación de OpenVINO se inicia en segundo plano después
        # de pulsar «Iniciar detección». La vista WebRTC debe seguir dibujándose
        # para que capture fotogramas desde el primer clic.
        model = st.session_state.get("cell_model")

        def load_live_model_for_session() -> YOLO:
            return load_live_model(model_path)

        render_live_camera(
            model,
            confidence,
            mask_opacity,
            model_loader=load_live_model_for_session,
        )
        return

    uploaded_video = st.file_uploader("Carga un video para analizar", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is None:
        st.info("Cuando tengas el video, súbelo aquí y presiona **Analizar y reproducir detección**.")
        return

    video_display = st.selectbox(
        "Visualización del video",
        ["Con máscaras", "Sin máscaras"],
        key="video_display",
    )
    show_video_masks = video_display == "Con máscaras"
    if not st.button("Analizar y reproducir detección", type="primary", use_container_width=True):
        return

    model = get_app_model()

    st.subheader("Detección en curso")
    st.caption(
        "El video original se muestra con las máscaras superpuestas mientras se analiza."
        if show_video_masks
        else "El video original se muestra mientras se analiza y se realiza el conteo."
    )
    preview_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    try:
        video_masks, counts, tracked_detections, video_measurements = process_video(
            model,
            uploaded_video,
            confidence,
            VIDEO_IMAGE_SIZE,
            mask_opacity,
            VIDEO_INFERENCE_STRIDE,
            show_video_masks,
            progress_bar,
            status_text,
            preview_placeholder,
        )
    except Exception as error:
        progress_bar.empty()
        status_text.empty()
        gc.collect()
        st.exception(error)
        return

    # Al terminar se retira la previsualización temporal: queda un solo video final.
    preview_placeholder.empty()
    status_text.success("Análisis terminado.")
    affected_percentage, grade, description = show_summary(
        counts,
        low_limit,
        medium_limit,
        high_limit,
        "Total de células únicas",
        sidebar=True,
        measurements=video_measurements,
    )
    if tracked_detections == 0:
        st.warning("No se obtuvieron identificadores de seguimiento; revisa la confianza, el video y el modelo.")

    st.subheader("Video original con máscaras" if show_video_masks else "Video original")
    st.video(video_masks, format="video/mp4")
    download_video, download_report = st.columns(2)
    with download_video:
        st.download_button(
            "Descargar video con máscaras" if show_video_masks else "Descargar video original",
            data=video_masks,
            file_name="celulas_con_mascaras.mp4" if show_video_masks else "video_original.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    with download_report:
        report = make_report_csv(
            counts,
            affected_percentage,
            grade,
            description,
            "Total de células únicas rastreadas",
            measurements=video_measurements,
        )
        st.download_button(
            "Descargar reporte CSV",
            data=report,
            file_name="reporte_celular.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        "El conteo corresponde a IDs únicos seguidos a lo largo del video. "
        "La calificación debe validarse con tu criterio científico o clínico."
    )
    if video_measurements:
        st.caption(
            "Las medidas promedio se calcularon con las máscaras de los fotogramas analizados "
            f"({video_measurements['calibration_method']})."
        )
    del video_masks
    gc.collect()


if __name__ == "__main__":
    main()

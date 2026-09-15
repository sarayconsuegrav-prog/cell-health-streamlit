"""Interfaz Streamlit para segmentación y conteo de células en fotos y videos.

El video exportado conserva el video original y superpone las máscaras de
segmentación, sin cajas delimitadoras ni etiquetas.
"""

from __future__ import annotations

import csv
import hashlib
from html import escape
import math
import os
import shutil
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from threading import Lock
from typing import Any

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
CACHED_MODEL = Path(tempfile.gettempdir()) / "cell-health-streamlit" / "best.pt"
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

# Colores BGR de las capas de segmentación.
COLORS_BGR = {
    "sana": (72, 184, 72),       # Verde
    "enferma": (60, 60, 235),    # Rojo
    "otra": (220, 165, 30),      # Azul
}
LIVE_INFERENCE_LOCK = Lock()
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
}


def default_model_path() -> Path:
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


def normalize_class_name(name: str) -> str:
    """Agrupa los nombres del modelo en las categorías usadas por el tablero."""
    normalized = name.lower().strip().replace("á", "a").replace("é", "e")
    if "enferm" in normalized:
        return "enferma"
    if "sana" in normalized or "salud" in normalized or "healthy" in normalized:
        return "sana"
    return "otra"


def load_model(model_path: str) -> YOLO:
    """Carga una instancia nueva para evitar reutilizar un modelo ya fusionado."""
    return YOLO(model_path)


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
        samples["areas_px2"][category].append(area_px2)
        samples["perimeters_px"][category].append(float(cv2.arcLength(contour, True)))
        samples["diameters_px"][category].append(math.sqrt(4.0 * area_px2 / math.pi))


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
    model_names = {int(key): str(value) for key, value in model.names.items()}
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
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("No se pudo crear el video de salida en formato MP4.")

        reset_trackers(model)
        model_names = {int(key): str(value) for key, value in model.names.items()}
        track_votes: dict[int, Counter] = defaultdict(Counter)
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


def render_photo_mode(
    model: YOLO,
    confidence: float,
    image_size: int,
    mask_opacity: float,
    low_limit: float,
    medium_limit: float,
    high_limit: float,
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
                original_image, healthy_masks, sick_masks, counts, measurements = process_image(
                    model, image_bytes, confidence, image_size, mask_opacity
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


def render_live_camera(model: YOLO, confidence: float, image_size: int, mask_opacity: float) -> None:
    """Mantiene la cámara encendida y devuelve los cuadros con máscaras en tiempo real."""
    if not WEBRTC_AVAILABLE:
        st.error("No se pudo cargar el componente de cámara en vivo.")
        st.caption(
            "Detalle de la dependencia: "
            f"{WEBRTC_IMPORT_ERROR or 'error de importación desconocido'}"
        )
        return

    st.markdown(
        """
        <section class="live-preview">
          <div class="live-preview-header">
            <span class="live-preview-dot"></span>
            <strong>Cámara en vivo</strong>
            <span class="live-preview-state">EN VIVO</span>
          </div>
          <div class="live-preview-footer">Pulsa <strong>INICIAR CÁMARA</strong>. Se mantendrá activa hasta que la detengas.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
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

    if "live_camera_requested" not in st.session_state:
        st.session_state["live_camera_requested"] = False
    camera_action, camera_hint = st.columns([1.2, 3.8])
    with camera_action:
        camera_is_requested = st.session_state["live_camera_requested"]
        action_label = "Detener cámara" if camera_is_requested else "Iniciar cámara"
        if st.button(action_label, type="primary", key="live_camera_action", use_container_width=True):
            st.session_state["live_camera_requested"] = not camera_is_requested
            st.rerun()
    with camera_hint:
        st.caption("Usa la cámara predeterminada del equipo. Puedes detenerla cuando quieras.")

    model_names = {int(key): str(value) for key, value in model.names.items()}
    frame_number = 0
    cached_shape: tuple[int, ...] | None = None
    cached_healthy_masks: np.ndarray | None = None
    cached_sick_masks: np.ndarray | None = None

    def process_live_frame(frame: Any) -> Any:
        nonlocal frame_number, cached_shape, cached_healthy_masks, cached_sick_masks
        image = frame.to_ndarray(format="bgr24")
        try:
            frame_number += 1
            needs_inference = (
                cached_healthy_masks is None
                or cached_sick_masks is None
                or cached_shape != image.shape
                or frame_number % LIVE_INFERENCE_EVERY_N_FRAMES == 1
            )
            if needs_inference:
                with LIVE_INFERENCE_LOCK:
                    result = model.predict(image, conf=confidence, imgsz=image_size, verbose=False)[0]
                cached_shape = image.shape
                cached_healthy_masks = mask_frame(result, image.shape, model_names, {"sana"})
                cached_sick_masks = mask_frame(result, image.shape, model_names, {"enferma"})

            masks = np.zeros_like(image)
            if show_healthy_masks:
                masks = cv2.bitwise_or(masks, cached_healthy_masks)
            if show_sick_masks:
                masks = cv2.bitwise_or(masks, cached_sick_masks)
            output = overlay_masks(image, masks, mask_opacity) if (show_healthy_masks or show_sick_masks) else image
            return av.VideoFrame.from_ndarray(output, format="bgr24")
        except Exception:
            # Si un cuadro puntual falla, se conserva el video en lugar de cerrar la cámara.
            return frame

    if st.session_state["live_camera_requested"]:
        camera_column, status_column = st.columns([1.2, 1.0], gap="large")
        with camera_column:
            webrtc_streamer(
                key="cell_live_camera",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 24, "max": 24},
                    },
                    "audio": False,
                },
                video_frame_callback=process_live_frame,
                desired_playing_state=True,
                video_html_attrs={
                    "autoPlay": True,
                    "controls": False,
                    "muted": True,
                    "playsInline": True,
                    "width": 640,
                    "height": 480,
                    "style": {"width": "100%", "height": "auto", "borderRadius": "12px"},
                },
                translations={
                    "start": "INICIAR CÁMARA",
                    "stop": "DETENER CÁMARA",
                    "select_device": "ELEGIR CÁMARA",
                    "device_ask_permission": "Autoriza el acceso a la cámara para comenzar.",
                    "device_not_available": "No se encontró una cámara disponible.",
                    "device_access_denied": "Se denegó el acceso a la cámara.",
                },
            )
        with status_column:
            st.markdown(
                """
                <aside class="live-status-panel">
                  <span class="live-status-badge">CÁMARA ACTIVA</span>
                  <h3>Monitoreo en vivo</h3>
                  <p>Las máscaras seleccionadas se superponen directamente sobre la imagen.</p>
                  <div class="live-status-row"><span>Modelo</span><strong>YOLO · segmentación</strong></div>
                  <div class="live-status-row"><span>Resolución</span><strong>640 × 480</strong></div>
                  <div class="live-status-row"><span>Actualización</span><strong>Continua</strong></div>
                </aside>
                """,
                unsafe_allow_html=True,
            )
        st.caption("La segmentación se actualiza continuamente para mantener una vista fluida.")
    else:
        st.markdown(
            '<div class="camera-idle">Pulsa <strong>Iniciar cámara</strong> para comenzar la detección en vivo.</div>',
            unsafe_allow_html=True,
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
            .stButton > button *, .stDownloadButton > button *,
            button[kind="primary"] *, [data-testid="stBaseButton-primary"] * {
                color: #ffffff !important;
            }
            .stButton > button:disabled, button[kind="primary"]:disabled {
                background: #31546b !important;
                border-color: #31546b !important;
                color: #b6c8d5 !important;
                opacity: 1 !important;
            }
            input, textarea { color: #ffffff !important; background: #0d3153 !important; }
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
            /* Se muestra solamente el área del video del componente de cámara;
               los controles se reemplazan por los botones turquesa de la app. */
            [data-testid="stCustomComponentV1"] {
                max-height: 34rem !important;
                overflow: hidden !important;
            }
            [data-testid="stCustomComponentV1"] iframe {
                height: 34rem !important;
                max-height: 34rem !important;
                border-radius: 12px !important;
                overflow: hidden !important;
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
            .live-preview-footer {
                margin-top: 0.35rem;
                color: #a6d8eb;
                font-size: 0.72rem;
            }
            .camera-idle {
                max-width: 40rem;
                margin-top: 0.75rem;
                padding: 1.1rem 1.2rem;
                background: #0a2745;
                border: 1px dashed #41d8cc;
                border-radius: 12px;
                color: #a6d8eb;
                text-align: center;
            }
            .camera-idle strong { color: #ffffff; }
            .live-status-panel {
                box-sizing: border-box;
                min-height: 34rem;
                padding: 1.5rem;
                background: #0a2745;
                border: 1px solid #41d8cc;
                border-radius: 14px;
                color: #ffffff;
            }
            .live-status-badge {
                display: inline-block;
                padding: 0.3rem 0.55rem;
                border-radius: 999px;
                background: rgba(18, 184, 194, 0.18);
                color: #a6f5f0;
                font-size: 0.7rem;
                font-weight: 800;
                letter-spacing: 0.04em;
            }
            .live-status-panel h3 {
                margin: 1rem 0 0.45rem;
                color: #ffffff;
                font-size: 1.35rem;
            }
            .live-status-panel p {
                margin: 0 0 1.4rem;
                color: #a6d8eb !important;
                line-height: 1.45;
            }
            .live-status-row {
                display: flex;
                flex-direction: column;
                gap: 0.18rem;
                padding: 0.85rem 0;
                border-top: 1px solid rgba(107, 231, 218, 0.22);
            }
            .live-status-row span { color: #a6d8eb; font-size: 0.78rem; }
            .live-status-row strong { color: #ffffff; font-size: 0.95rem; }
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
    if not model_path.is_file():
        st.error("No se encontró el archivo del modelo `best.pt`.")
        st.stop()

    try:
        model = load_model(str(model_path))
    except Exception as error:
        st.error(f"No fue posible cargar el modelo: {error}")
        st.stop()

    if model.task != "segment":
        st.error(f"El modelo cargado es de tipo `{model.task}`. Esta aplicación requiere un modelo de segmentación.")
        st.stop()

    analysis_mode = st.radio("Modo de análisis", ["Foto", "Video"], horizontal=True)
    if analysis_mode == "Foto":
        render_photo_mode(
            model, confidence, image_size, mask_opacity, low_limit, medium_limit, high_limit
        )
        return

    video_section = st.radio(
        "Sección de video", ["Cámara en vivo", "Subir video"], horizontal=True, key="video_section"
    )
    if video_section == "Cámara en vivo":
        render_live_camera(model, confidence, LIVE_IMAGE_SIZE, mask_opacity)
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


if __name__ == "__main__":
    main()

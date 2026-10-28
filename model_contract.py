"""Contract checks for the two-class shrimp-cell segmentation model."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path


DEFAULT_MODEL_NAMES = {
    0: "celula_enferma",
    1: "celula_sana",
}


def normalize_class_name(name: str) -> str:
    """Map common Spanish/English model labels to the app's two cell classes."""
    normalized = name.lower().strip().replace("á", "a").replace("é", "e")
    if any(
        token in normalized
        for token in ("enferm", "sick", "diseas", "infect", "unhealthy", "wssv")
    ):
        return "enferma"
    if any(token in normalized for token in ("sana", "salud", "healthy", "health")):
        return "sana"
    return "otra"


def build_class_id_map(names: Mapping[int, str] | Sequence[str]) -> dict[int, int]:
    """Map a model's two labels to the app's fixed sick/healthy IDs.

    The app uses ID 0 for sick cells and ID 1 for healthy cells. Unknown labels
    are rejected instead of silently being treated as healthy.
    """
    if isinstance(names, Mapping):
        source_names = {int(class_id): str(name) for class_id, name in names.items()}
    else:
        source_names = {class_id: str(name) for class_id, name in enumerate(names)}

    categories = {
        class_id: normalize_class_name(name)
        for class_id, name in source_names.items()
    }
    if set(categories) != {0, 1} or set(categories.values()) != {"enferma", "sana"}:
        rendered = ", ".join(
            f"{class_id}: {name}" for class_id, name in sorted(source_names.items())
        ) or "sin clases"
        raise ValueError(
            "El modelo debe declarar exactamente las clases célula enferma y célula sana "
            f"(IDs 0 y 1). Clases recibidas: {rendered}."
        )

    return {
        class_id: 0 if category == "enferma" else 1
        for class_id, category in categories.items()
    }


def select_mask2former_candidates(
    class_probabilities: Sequence[Sequence[float]],
    mask_confidences: Sequence[float],
    confidence_threshold: float,
) -> list[tuple[int, int, float]]:
    """Select at most one class prediction for each Mask2Former query.

    A Mask2Former query represents one candidate instance. Selecting every
    query/class score independently can return the same mask as both classes.
    """
    if len(class_probabilities) != len(mask_confidences):
        raise ValueError("Debe haber una confianza de máscara por cada consulta.")

    selected: list[tuple[int, int, float]] = []
    for query_id, (class_scores, mask_confidence) in enumerate(
        zip(class_probabilities, mask_confidences)
    ):
        if len(class_scores) == 0:
            continue
        class_id = max(range(len(class_scores)), key=lambda index: float(class_scores[index]))
        score = float(class_scores[class_id]) * float(mask_confidence)
        if score >= confidence_threshold:
            selected.append((query_id, class_id, score))
    return selected


def read_model_metadata_names(model_path: Path) -> dict[int, str]:
    """Read a model's YAML class map without guessing a fallback class order."""
    metadata_path = (
        model_path / "metadata.yaml"
        if model_path.is_dir()
        else model_path.with_suffix(".yaml")
    )
    try:
        metadata_text = metadata_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}
    names_match = re.search(
        r"(?ms)^names:\s*\n((?:^[ \t]+\d+:[^\n]*\n?)+)",
        metadata_text,
    )
    if not names_match:
        return {}
    return {
        int(class_id): label.strip().strip("\"'")
        for class_id, label in re.findall(
            r"(?m)^\s+(\d+):\s*(.*?)\s*$",
            names_match.group(1),
        )
    }


def validate_cell_model_contract(task: str | None, names: dict[int, str]) -> None:
    """Fail fast if the loaded artifact cannot produce the app's expected masks."""
    if str(task or "").lower() != "segment":
        raise ValueError(
            f"El modelo cargado es de tipo `{task or 'desconocido'}`; "
            "se requiere segmentación."
        )

    categories = {
        int(class_id): normalize_class_name(str(name))
        for class_id, name in names.items()
    }
    if set(categories) != {0, 1} or set(categories.values()) != {"enferma", "sana"}:
        rendered_names = ", ".join(
            f"{class_id}: {name}" for class_id, name in sorted(names.items())
        ) or "sin clases"
        raise ValueError(
            "El modelo debe declarar exactamente las clases célula enferma y célula sana "
            f"(IDs 0 y 1). Clases recibidas: {rendered_names}."
        )

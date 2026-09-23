"""Contract checks for the two-class shrimp-cell segmentation model."""

from __future__ import annotations

import re
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

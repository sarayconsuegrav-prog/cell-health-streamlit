"""Pinned metadata and integrity checks for the published WSSV v6 weights."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifact:
    kind: str
    display_name: str
    filename: str
    url: str
    size_bytes: int
    sha256: str


RELEASE_BASE_URL = (
    "https://github.com/sarayconsuegrav-prog/cell-health-streamlit/releases/download/"
    "wssv-v6-models"
)

MODEL_ARTIFACTS = {
    "mask2former": ModelArtifact(
        kind="mask2former",
        display_name="Mask2Former Swin-S · CEDIA WSSV v6",
        filename="mask2former-mejor.pth",
        url=f"{RELEASE_BASE_URL}/mask2former-mejor.pth",
        size_bytes=275_785_875,
        sha256="b379eb6e5262c4a0b325baa81c6bf7068e40493252c168258c495aaba8759d8c",
    ),
    "rfdetr": ModelArtifact(
        kind="rfdetr",
        display_name="RF-DETR-Seg 2XL · CEDIA WSSV v6",
        filename="rfdetr-checkpoint-best-total.pth",
        url=f"{RELEASE_BASE_URL}/rfdetr-checkpoint-best-total.pth",
        size_bytes=160_963_881,
        sha256="b681d916f8b16f1f5d0b69b82e1347d766c5ee71d19dd770e0cc7f1515fb2892",
    ),
}


def sha256_file(path: Path) -> str:
    """Return the full SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_artifact(path: Path, artifact: ModelArtifact) -> str:
    """Reject a cached or downloaded file unless its size and digest match."""
    actual_size = path.stat().st_size
    if actual_size != artifact.size_bytes:
        raise ValueError(
            f"Tamaño incorrecto para {artifact.filename}: "
            f"{actual_size:,} bytes; se esperaban {artifact.size_bytes:,}."
        )
    actual_digest = sha256_file(path)
    if actual_digest != artifact.sha256:
        raise ValueError(f"La huella SHA-256 no coincide para {artifact.filename}.")
    return actual_digest

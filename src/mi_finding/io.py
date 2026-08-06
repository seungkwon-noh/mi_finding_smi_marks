from __future__ import annotations

import base64
import binascii
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"cannot encode image as {suffix}")
    encoded.tofile(path)


def load_templates(directory: str | Path, prefix: str = "") -> list[np.ndarray]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"template directory does not exist: {directory}")
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.name.startswith(prefix)
    )
    return [read_image(path) for path in paths]


def decode_base64_image(value: str) -> np.ndarray:
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image is not valid base64") from exc
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("base64 payload is not a supported image")
    return image

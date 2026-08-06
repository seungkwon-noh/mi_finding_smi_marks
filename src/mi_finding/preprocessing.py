from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EdgeTemplate:
    image: np.ndarray
    edge: str
    ratio: float
    offset_x: int
    offset_y: int
    full_width: int
    full_height: int


def validate_bgr_image(image: np.ndarray, name: str = "image") -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if image.size == 0:
        raise ValueError(f"{name} is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be a BGR image with shape (height, width, 3)")


def remove_green_mark(image: np.ndarray) -> np.ndarray:
    """Remove bright green cursor/marker pixels with local inpainting."""

    validate_bgr_image(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 70, 50]), np.array([90, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    if cv2.countNonZero(mask) == 0:
        return image.copy()
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def apply_clahe(
    gray: np.ndarray, clip_limit: float = 2.0, grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size).apply(gray)


def generate_edge_templates(
    template: np.ndarray, ratio: float, min_size: int = 7
) -> list[EdgeTemplate]:
    validate_bgr_image(template, "template")
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")

    height, width = template.shape[:2]
    crop_height = max(1, round(height * ratio))
    crop_width = max(1, round(width * ratio))
    edges: list[EdgeTemplate] = []

    if crop_height >= min_size and width >= min_size:
        edges.extend(
            [
                EdgeTemplate(
                    template[:crop_height, :], "top", ratio, 0, 0, width, height
                ),
                EdgeTemplate(
                    template[height - crop_height :, :],
                    "bottom",
                    ratio,
                    0,
                    height - crop_height,
                    width,
                    height,
                ),
            ]
        )
    if crop_width >= min_size and height >= min_size:
        edges.extend(
            [
                EdgeTemplate(
                    template[:, :crop_width], "left", ratio, 0, 0, width, height
                ),
                EdgeTemplate(
                    template[:, width - crop_width :],
                    "right",
                    ratio,
                    width - crop_width,
                    0,
                    width,
                    height,
                ),
            ]
        )
    return edges

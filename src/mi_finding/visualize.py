from __future__ import annotations

import cv2
import numpy as np

from .models import FindingResult


def annotate_result(image: np.ndarray, result: FindingResult) -> np.ndarray:
    annotated = image.copy()
    candidate = result.candidate
    if candidate is None:
        cv2.putText(
            annotated,
            "FAIL: no candidate",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    color = (0, 190, 0) if result.success else (0, 0, 255)
    cv2.rectangle(annotated, candidate.top_left, candidate.bottom_right, color, 2)
    label = (
        f"{'PASS' if result.success else 'FAIL'} "
        f"{candidate.stage}/{candidate.method} score={candidate.score:.3f}"
    )
    text_y = max(22, candidate.top_left[1] - 8)
    cv2.putText(
        annotated,
        label,
        (candidate.top_left[0], text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated

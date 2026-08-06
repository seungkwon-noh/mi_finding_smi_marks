from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class MatchingConfig:
    """Thresholds recovered from the original mi_finding discussion."""

    full_min_score: float = 0.60
    full_direct_score: float = 0.70
    partial_min_score: float = 0.70
    partial_ratios: tuple[float, ...] = (0.70, 0.35)

    variance_ratio_min: float = 0.10
    variance_ratio_max: float = 7.0
    ssim_min: float = 0.44
    color_hist_min: float = 0.15
    nmi_min: float = 0.20

    partial_nmi_floor: float = 0.005
    partial_color_hist_floor: float = 0.0015
    partial_bhattacharyya_limit: float = 0.90
    partial_bhattacharyya_soft_limit: float = 0.83
    partial_average_color_limit: float = 150.0

    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    remove_green_marker: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.full_min_score <= self.full_direct_score <= 1:
            raise ValueError("full scores must satisfy 0 <= min <= direct <= 1")
        if not 0 <= self.partial_min_score <= 1:
            raise ValueError("partial_min_score must be between 0 and 1")
        if any(not 0 < ratio < 1 for ratio in self.partial_ratios):
            raise ValueError("partial ratios must be between 0 and 1")
        if (
            self.variance_ratio_min < 0
            or self.variance_ratio_max <= self.variance_ratio_min
        ):
            raise ValueError("invalid variance ratio range")


@dataclass(frozen=True)
class MatchMetrics:
    ssim: float
    variance_ratio: float
    color_hist_similarity: float
    nmi: float
    bhattacharyya_distance: float
    average_color_distance: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass
class MatchCandidate:
    score: float
    method: str
    stage: Literal["full", "partial"]
    ratio: float
    top_left: tuple[int, int]
    width: int
    height: int
    metrics: MatchMetrics
    template_index: int
    edge: str = "full"
    matched_top_left: tuple[int, int] | None = None
    roi: np.ndarray | None = field(default=None, repr=False)
    template: np.ndarray | None = field(default=None, repr=False)

    @property
    def center(self) -> tuple[int, int]:
        return self.top_left[0] + self.width // 2, self.top_left[1] + self.height // 2

    @property
    def bottom_right(self) -> tuple[int, int]:
        return self.top_left[0] + self.width, self.top_left[1] + self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "method": self.method,
            "stage": self.stage,
            "ratio": float(self.ratio),
            "edge": self.edge,
            "template_index": self.template_index,
            "top_left": list(self.top_left),
            "bottom_right": list(self.bottom_right),
            "center": list(self.center),
            "matched_top_left": list(self.matched_top_left or self.top_left),
            "width": self.width,
            "height": self.height,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class FindingResult:
    success: bool
    reason: str
    candidate: MatchCandidate | None = None
    attempts: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "reason": self.reason,
            "attempts": self.attempts,
            "match": self.candidate.to_dict() if self.candidate else None,
            "diagnostics": self.diagnostics,
        }

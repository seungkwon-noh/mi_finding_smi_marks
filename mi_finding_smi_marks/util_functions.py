from __future__ import annotations

import base64
import binascii
import math
from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np
from skimage.metrics import structural_similarity


@dataclass(frozen=True)
class MatchingConfig:
    """Thresholds agreed in the original mi_finding_smi_marks discussion."""

    full_min_score: float = 0.60
    full_direct_score: float = 0.70
    partial_min_score: float = 0.70
    partial_ratios: tuple[float, ...] = (0.70, 0.35)
    # Popup 대화에서 후보를 모으던 기준: max_val > 0.5
    popup_min_score: float = 0.50

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

    def __post_init__(self) -> None:
        if not 0 <= self.full_min_score <= self.full_direct_score <= 1:
            raise ValueError("full score thresholds are invalid")
        if not 0 <= self.partial_min_score <= 1:
            raise ValueError("partial_min_score must be between 0 and 1")
        if any(not 0 < ratio < 1 for ratio in self.partial_ratios):
            raise ValueError("partial ratios must be between 0 and 1")
        if not 0 <= self.popup_min_score <= 1:
            raise ValueError("popup_min_score must be between 0 and 1")
        if not 0 <= self.variance_ratio_min < self.variance_ratio_max:
            raise ValueError("variance ratio range is invalid")


@dataclass(frozen=True)
class MatchMetrics:
    ssim: float
    variance_ratio: float
    color_hist_similarity: float
    nmi: float
    bhattacharyya_distance: float
    average_color_distance: float

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class EdgeTemplate:
    image: np.ndarray
    edge: Literal["full", "top", "bottom", "left", "right"]
    ratio: float
    offset_x: int
    offset_y: int
    full_width: int
    full_height: int


@dataclass(frozen=True)
class MatchCandidate:
    score: float
    method: str
    stage: Literal["full", "partial"]
    ratio: float
    top_left: tuple[int, int]
    matched_top_left: tuple[int, int]
    width: int
    height: int
    metrics: MatchMetrics
    template_name: str
    edge: str = "full"

    @property
    def center(self) -> tuple[int, int]:
        return self.top_left[0] + self.width // 2, self.top_left[1] + self.height // 2

    def to_dict(self) -> dict[str, object]:
        return {
            "score": float(self.score),
            "method": self.method,
            "stage": self.stage,
            "ratio": float(self.ratio),
            "edge": self.edge,
            "template_name": self.template_name,
            "top_left": list(self.top_left),
            "matched_top_left": list(self.matched_top_left),
            "width": self.width,
            "height": self.height,
            "center": list(self.center),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class FindingResult:
    success: bool
    reason: str
    candidate: MatchCandidate | None = None
    attempts: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "reason": self.reason,
            "attempts": self.attempts,
            "match": self.candidate.to_dict() if self.candidate else None,
        }


def validate_bgr_image(image: np.ndarray, name: str = "image") -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if image.size == 0:
        raise ValueError(f"{name} is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be a BGR image")


def decode_base64_image(value: str) -> np.ndarray:
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image is not valid base64") from exc
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("image is not a supported encoded image")
    return image


def remove_green_mark(image: np.ndarray) -> np.ndarray:
    """Remove the bright-green SMI marker without changing image coordinates."""

    validate_bgr_image(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([80, 255, 255]))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    if cv2.countNonZero(mask) == 0:
        return image.copy()
    return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)


def process_image_with_detection(
    image: np.ndarray, x: int = 340, y: int = 340
) -> np.ndarray:
    """Remove the exact white/black center cross used by 680x680 review images."""

    validate_bgr_image(image)
    result = image.copy()
    height, width = result.shape[:2]
    if not (0 < x < width - 1 and 0 < y < height - 1):
        return result

    vertical = result[:, x]
    horizontal = result[y, :]
    vertical_white = np.all(vertical == 255, axis=1)
    vertical_black = np.all(vertical == 0, axis=1)
    horizontal_white = np.all(horizontal == 255, axis=1)
    horizontal_black = np.all(horizontal == 0, axis=1)

    is_cross = (np.all(vertical_white) and np.all(horizontal_white)) or (
        np.all(vertical_black) and np.all(horizontal_black)
    )
    if not is_cross:
        return result

    left = result[:, x - 1].astype(np.int32)
    right = result[:, x + 1].astype(np.int32)
    result[:, x] = ((left + right) // 2).astype(np.uint8)
    upper = result[y - 1, :].astype(np.int32)
    lower = result[y + 1, :].astype(np.int32)
    result[y, :] = ((upper + lower) // 2).astype(np.uint8)
    return result


def apply_clahe_image(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("CLAHE input must be a non-empty grayscale image")
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size).apply(gray)


# Familiar public name retained for notebook/local use.
apply_clahe = apply_clahe_image


def color_hist_score(first: np.ndarray, second: np.ndarray) -> float:
    """HSV histogram correlation; negative values intentionally stay negative."""

    first_hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    second_hsv = cv2.cvtColor(second, cv2.COLOR_BGR2HSV)
    first_hist = cv2.calcHist([first_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    second_hist = cv2.calcHist([second_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(first_hist, first_hist)
    cv2.normalize(second_hist, second_hist)
    return float(cv2.compareHist(first_hist, second_hist, cv2.HISTCMP_CORREL))


def color_ssim(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("SSIM images must have the same shape")
    if first.ndim not in {2, 3}:
        raise ValueError("SSIM images must be grayscale or color images")

    minimum_dimension = min(first.shape[:2])
    if minimum_dimension < 3:
        first_float = first.astype(np.float64)
        second_float = second.astype(np.float64)
        mse = float(np.mean((first_float - second_float) ** 2))
        return max(-1.0, 1.0 - mse / (255.0**2))

    # scikit-image의 기본 win_size=7을 사용하되 작은 template도 처리한다.
    win_size = min(7, minimum_dimension)
    if win_size % 2 == 0:
        win_size -= 1
    return float(
        structural_similarity(
            first,
            second,
            channel_axis=-1 if first.ndim == 3 else None,
            data_range=255,
            win_size=win_size,
        )
    )


def calculate_nmi_score(first: np.ndarray, second: np.ndarray, bins: int = 32) -> float:
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY) if first.ndim == 3 else first
    second_gray = (
        cv2.cvtColor(second, cv2.COLOR_BGR2GRAY) if second.ndim == 3 else second
    )
    if first_gray.shape != second_gray.shape:
        second_gray = cv2.resize(
            second_gray, (first_gray.shape[1], first_gray.shape[0])
        )

    joint = cv2.calcHist(
        [first_gray, second_gray], [0, 1], None, [bins, bins], [0, 256, 0, 256]
    ).astype(np.float64)
    total = float(joint.sum())
    if total <= 0:
        return 0.0
    probability = joint / total
    first_probability = probability.sum(axis=1, keepdims=True)
    second_probability = probability.sum(axis=0, keepdims=True)
    denominator = first_probability * second_probability
    valid = (probability > 0) & (denominator > 0)
    mutual_information = float(
        np.sum(probability[valid] * np.log(probability[valid] / denominator[valid]))
    )
    first_nonzero = first_probability > 0
    second_nonzero = second_probability > 0
    first_entropy = -float(
        np.sum(
            first_probability[first_nonzero] * np.log(first_probability[first_nonzero])
        )
    )
    second_entropy = -float(
        np.sum(
            second_probability[second_nonzero]
            * np.log(second_probability[second_nonzero])
        )
    )
    nmi = (2 * mutual_information) / (first_entropy + second_entropy + 1e-12)
    return float(np.clip(nmi, 0.0, 1.0))


def compare_color_features(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    first_hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    second_hsv = cv2.cvtColor(second, cv2.COLOR_BGR2HSV)
    first_value_hist = cv2.calcHist([first_hsv], [2], None, [32], [0, 256])
    second_value_hist = cv2.calcHist([second_hsv], [2], None, [32], [0, 256])
    cv2.normalize(first_value_hist, first_value_hist)
    cv2.normalize(second_value_hist, second_value_hist)
    bhattacharyya = float(
        cv2.compareHist(first_value_hist, second_value_hist, cv2.HISTCMP_BHATTACHARYYA)
    )
    first_mean = np.mean(first.reshape(-1, 3), axis=0)
    second_mean = np.mean(second.reshape(-1, 3), axis=0)
    average_distance = float(np.linalg.norm(first_mean - second_mean))
    return bhattacharyya, average_distance


def variance_ratio(roi: np.ndarray, template: np.ndarray) -> float:
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    roi_variance = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
    template_variance = float(cv2.Laplacian(template_gray, cv2.CV_64F).var())
    if template_variance <= 1e-12:
        return 1.0 if roi_variance <= 1e-12 else math.inf
    return roi_variance / template_variance


def calculate_metrics(roi: np.ndarray, template: np.ndarray) -> MatchMetrics:
    bhattacharyya, average_distance = compare_color_features(roi, template)
    return MatchMetrics(
        ssim=color_ssim(template, roi),
        variance_ratio=variance_ratio(roi, template),
        color_hist_similarity=color_hist_score(roi, template),
        nmi=calculate_nmi_score(roi, template),
        bhattacharyya_distance=bhattacharyya,
        average_color_distance=average_distance,
    )


def _empty_match() -> tuple[
    tuple[int, int], np.ndarray, float, float, float, float, float, float, float
]:
    return (
        (0, 0),
        np.empty((0, 0, 3), dtype=np.uint8),
        -1.0,
        0.0,
        math.inf,
        -1.0,
        0.0,
        1.0,
        math.inf,
    )


def match_template_logic(
    image: np.ndarray,
    image_gray: np.ndarray,
    template: np.ndarray,
    ratio: float = 1.0,
    *,
    apply_clahe: bool = False,
    config: MatchingConfig | None = None,
) -> tuple[
    tuple[int, int], np.ndarray, float, float, float, float, float, float, float
]:
    """Return the original handler-compatible nine template-match values."""

    del ratio  # Kept in the signature for compatibility with the original function.
    config = config or MatchingConfig()
    validate_bgr_image(image)
    validate_bgr_image(template, "template")
    if image_gray.ndim != 2 or image_gray.shape[:2] != image.shape[:2]:
        raise ValueError("image_gray must match image dimensions")

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    matching_gray = image_gray
    if apply_clahe:
        matching_gray = apply_clahe_image(
            image_gray, config.clahe_clip_limit, config.clahe_grid_size
        )
        template_gray = apply_clahe_image(
            template_gray, config.clahe_clip_limit, config.clahe_grid_size
        )

    image_height, image_width = matching_gray.shape
    template_height, template_width = template_gray.shape
    if template_height > image_height or template_width > image_width:
        return _empty_match()
    if float(template_gray.std()) <= 1e-8:
        return _empty_match()

    response = cv2.matchTemplate(matching_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, top_left = cv2.minMaxLoc(response)
    x, y = top_left
    roi = image[y : y + template_height, x : x + template_width]
    if roi.size == 0 or roi.shape != template.shape:
        return _empty_match()

    if score < 0.50:
        metrics = MatchMetrics(0.0, math.inf, -1.0, 0.0, 1.0, math.inf)
    else:
        metrics = calculate_metrics(roi, template)
    return (
        top_left,
        roi,
        float(score),
        metrics.ssim,
        metrics.variance_ratio,
        metrics.color_hist_similarity,
        metrics.nmi,
        metrics.bhattacharyya_distance,
        metrics.average_color_distance,
    )


def match_template_for_popup(
    image: np.ndarray, image_gray: np.ndarray, template: np.ndarray
) -> tuple[tuple[int, int], np.ndarray, float]:
    validate_bgr_image(image)
    validate_bgr_image(template, "template")
    if image_gray.ndim != 2 or image_gray.shape[:2] != image.shape[:2]:
        raise ValueError("image_gray must match image dimensions")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_height, image_width = image_gray.shape
    template_height, template_width = template_gray.shape
    if template_height > image_height or template_width > image_width:
        return (0, 0), np.empty((0, 0, 3), dtype=np.uint8), -1.0
    if float(template_gray.std()) <= 1e-8:
        return (0, 0), np.empty((0, 0, 3), dtype=np.uint8), -1.0
    response = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, top_left = cv2.minMaxLoc(response)
    x, y = top_left
    roi = image[y : y + template_height, x : x + template_width]
    if roi.size == 0 or roi.shape != template.shape:
        return (0, 0), np.empty((0, 0, 3), dtype=np.uint8), -1.0
    return top_left, roi, float(score)


def full_template(template: np.ndarray) -> EdgeTemplate:
    height, width = template.shape[:2]
    return EdgeTemplate(template, "full", 1.0, 0, 0, width, height)


def generate_edge_templates(
    template: np.ndarray, ratio: float, minimum_size: int = 7
) -> list[EdgeTemplate]:
    validate_bgr_image(template, "template")
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")

    height, width = template.shape[:2]
    crop_height = max(1, round(height * ratio))
    crop_width = max(1, round(width * ratio))
    templates: list[EdgeTemplate] = []
    if crop_height >= minimum_size and width >= minimum_size:
        templates.extend(
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
    if crop_width >= minimum_size and height >= minimum_size:
        templates.extend(
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
    return templates


def candidate_from_match(
    match: tuple[
        tuple[int, int],
        np.ndarray,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    edge_template: EdgeTemplate,
    *,
    image_shape: tuple[int, ...],
    method: str,
    template_name: str,
) -> MatchCandidate | None:
    top_left, roi, score, ssim, var_ratio_value, hist, nmi, bd, ad = match
    if score < 0 or roi.size == 0:
        return None
    matched_x, matched_y = top_left
    full_x = matched_x - edge_template.offset_x
    full_y = matched_y - edge_template.offset_y
    image_height, image_width = image_shape[:2]
    if (
        full_x < 0
        or full_y < 0
        or full_x + edge_template.full_width > image_width
        or full_y + edge_template.full_height > image_height
    ):
        return None
    return MatchCandidate(
        score=score,
        method=method,
        stage="full" if edge_template.edge == "full" else "partial",
        ratio=edge_template.ratio,
        top_left=(full_x, full_y),
        matched_top_left=top_left,
        width=edge_template.full_width,
        height=edge_template.full_height,
        metrics=MatchMetrics(ssim, var_ratio_value, hist, nmi, bd, ad),
        template_name=template_name,
        edge=edge_template.edge,
    )


def auxiliary_failures(
    candidate: MatchCandidate, config: MatchingConfig, *, partial: bool
) -> list[str]:
    metrics = candidate.metrics
    failures: list[str] = []
    if (
        not config.variance_ratio_min
        < metrics.variance_ratio
        < config.variance_ratio_max
    ):
        failures.append("variance_ratio")
    if (
        metrics.color_hist_similarity < config.color_hist_min
        and metrics.ssim < config.ssim_min
    ):
        failures.append("ssim_and_color_hist")
    if metrics.ssim < config.ssim_min and metrics.nmi < config.nmi_min:
        failures.append("ssim_and_nmi")
    if partial:
        if metrics.nmi < config.partial_nmi_floor:
            failures.append("partial_nmi_floor")
        if metrics.color_hist_similarity < config.partial_color_hist_floor:
            failures.append("partial_color_hist_floor")
        if metrics.bhattacharyya_distance > config.partial_bhattacharyya_limit:
            failures.append("partial_bhattacharyya")
        elif (
            metrics.bhattacharyya_distance > config.partial_bhattacharyya_soft_limit
            and metrics.average_color_distance > config.partial_average_color_limit
        ):
            failures.append("partial_color_distance")
    return failures


def select_full_candidate(
    candidates: list[MatchCandidate], config: MatchingConfig
) -> tuple[MatchCandidate | None, str]:
    # The recent handler flow builds ``best_candidates_per_template`` only from
    # candidates that already passed the auxiliary metrics. Apply that gate before
    # either score threshold so a high correlation score cannot bypass obviously
    # incompatible variance/SSIM/histogram/NMI values.
    valid = [
        item
        for item in candidates
        if not auxiliary_failures(item, config, partial=False)
    ]
    direct = [item for item in valid if item.score >= config.full_direct_score]
    if direct:
        return max(direct, key=lambda item: item.score), "full_direct"
    assisted = [item for item in valid if item.score >= config.full_min_score]
    if assisted:
        return max(assisted, key=lambda item: item.score), "full_assisted"
    return None, "full_rejected"


def select_partial_candidate(
    candidates: list[MatchCandidate], config: MatchingConfig
) -> MatchCandidate | None:
    eligible = [
        item
        for item in candidates
        if item.score >= config.partial_min_score
        and not auxiliary_failures(item, config, partial=True)
    ]
    return max(eligible, key=lambda item: item.score) if eligible else None

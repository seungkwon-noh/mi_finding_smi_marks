from __future__ import annotations

import math

import cv2
import numpy as np

from .models import MatchMetrics


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def variance_ratio(roi: np.ndarray, template: np.ndarray) -> float:
    roi_variance = float(cv2.Laplacian(_gray(roi), cv2.CV_64F).var())
    template_variance = float(cv2.Laplacian(_gray(template), cv2.CV_64F).var())
    if template_variance <= 1e-12:
        return 1.0 if roi_variance <= 1e-12 else math.inf
    return roi_variance / template_variance


def normalized_mutual_information(
    first: np.ndarray, second: np.ndarray, bins: int = 64
) -> float:
    first_gray = _gray(first).ravel()
    second_gray = _gray(second).ravel()
    joint, _, _ = np.histogram2d(
        first_gray, second_gray, bins=bins, range=((0, 256), (0, 256))
    )
    total = joint.sum()
    if total == 0:
        return 0.0

    joint_probability = joint / total
    first_probability = joint_probability.sum(axis=1)
    second_probability = joint_probability.sum(axis=0)

    nz_joint = joint_probability > 0
    denominator = first_probability[:, None] * second_probability[None, :]
    mutual_information = float(
        np.sum(
            joint_probability[nz_joint]
            * np.log(joint_probability[nz_joint] / denominator[nz_joint])
        )
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
    entropy_scale = math.sqrt(first_entropy * second_entropy)
    if entropy_scale <= 1e-12:
        return 1.0 if np.array_equal(first_gray, second_gray) else 0.0
    return float(np.clip(mutual_information / entropy_scale, 0.0, 1.0))


def color_histogram_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    second_hsv = cv2.cvtColor(second, cv2.COLOR_BGR2HSV)
    scores: list[float] = []
    for channel, bins, value_range in (
        (0, 36, (0, 180)),
        (1, 32, (0, 256)),
        (2, 32, (0, 256)),
    ):
        hist_first = cv2.calcHist(
            [first_hsv], [channel], None, [bins], list(value_range)
        )
        hist_second = cv2.calcHist(
            [second_hsv], [channel], None, [bins], list(value_range)
        )
        cv2.normalize(hist_first, hist_first)
        cv2.normalize(hist_second, hist_second)
        scores.append(
            float(cv2.compareHist(hist_first, hist_second, cv2.HISTCMP_CORREL))
        )
    # A negative correlation remains negative. It must not be made positive with abs().
    return float(np.mean(scores))


def bhattacharyya_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    second_hsv = cv2.cvtColor(second, cv2.COLOR_BGR2HSV)
    hist_first = cv2.calcHist([first_hsv], [0, 1], None, [36, 32], [0, 180, 0, 256])
    hist_second = cv2.calcHist([second_hsv], [0, 1], None, [36, 32], [0, 180, 0, 256])
    cv2.normalize(hist_first, hist_first)
    cv2.normalize(hist_second, hist_second)
    return float(cv2.compareHist(hist_first, hist_second, cv2.HISTCMP_BHATTACHARYYA))


def average_color_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_mean = np.mean(first.reshape(-1, 3), axis=0)
    second_mean = np.mean(second.reshape(-1, 3), axis=0)
    return float(np.linalg.norm(first_mean - second_mean))


def _single_channel_ssim(first: np.ndarray, second: np.ndarray) -> float:
    first_float = first.astype(np.float64)
    second_float = second.astype(np.float64)
    min_dimension = min(first.shape[:2])
    if min_dimension < 3:
        mse = float(np.mean((first_float - second_float) ** 2))
        return max(-1.0, 1.0 - mse / (255.0**2))

    kernel = min(11, min_dimension if min_dimension % 2 == 1 else min_dimension - 1)
    sigma = max(0.5, kernel / 7.333)
    mu_first = cv2.GaussianBlur(first_float, (kernel, kernel), sigma)
    mu_second = cv2.GaussianBlur(second_float, (kernel, kernel), sigma)
    mu_first_sq = mu_first * mu_first
    mu_second_sq = mu_second * mu_second
    mu_cross = mu_first * mu_second

    variance_first = (
        cv2.GaussianBlur(first_float * first_float, (kernel, kernel), sigma)
        - mu_first_sq
    )
    variance_second = (
        cv2.GaussianBlur(second_float * second_float, (kernel, kernel), sigma)
        - mu_second_sq
    )
    covariance = (
        cv2.GaussianBlur(first_float * second_float, (kernel, kernel), sigma) - mu_cross
    )

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mu_cross + c1) * (2 * covariance + c2)
    denominator = (mu_first_sq + mu_second_sq + c1) * (
        variance_first + variance_second + c2
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def color_ssim(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("SSIM images must have the same shape")
    if first.ndim == 2:
        return _single_channel_ssim(first, second)
    return float(
        np.mean(
            [_single_channel_ssim(first[:, :, i], second[:, :, i]) for i in range(3)]
        )
    )


def calculate_metrics(roi: np.ndarray, template: np.ndarray) -> MatchMetrics:
    return MatchMetrics(
        ssim=color_ssim(template, roi),
        variance_ratio=variance_ratio(roi, template),
        color_hist_similarity=color_histogram_similarity(roi, template),
        nmi=normalized_mutual_information(roi, template),
        bhattacharyya_distance=bhattacharyya_distance(roi, template),
        average_color_distance=average_color_distance(roi, template),
    )

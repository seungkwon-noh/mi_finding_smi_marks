from __future__ import annotations

from .models import MatchCandidate, MatchingConfig


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
        metrics.ssim < config.ssim_min
        and metrics.color_hist_similarity < config.color_hist_min
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
    direct = [item for item in candidates if item.score >= config.full_direct_score]
    if direct:
        return max(direct, key=lambda item: item.score), "full_direct"

    assisted = [
        item
        for item in candidates
        if item.score >= config.full_min_score
        and not auxiliary_failures(item, config, partial=False)
    ]
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

from __future__ import annotations

import numpy as np

from mi_finding_smi_marks.handler import TemplateAsset, _find_review
from mi_finding_smi_marks.util_functions import (
    MatchCandidate,
    MatchingConfig,
    MatchMetrics,
    select_full_candidate,
)


def candidate(score: float, *, good_metrics: bool) -> MatchCandidate:
    metrics = MatchMetrics(
        ssim=0.8 if good_metrics else 0.1,
        variance_ratio=1.0 if good_metrics else 0.01,
        color_hist_similarity=0.8 if good_metrics else -0.5,
        nmi=0.8 if good_metrics else 0.0,
        bhattacharyya_distance=0.1 if good_metrics else 0.95,
        average_color_distance=10.0 if good_metrics else 200.0,
    )
    return MatchCandidate(
        score=score,
        method="RAW",
        stage="full",
        ratio=1.0,
        top_left=(10, 20),
        matched_top_left=(10, 20),
        width=30,
        height=40,
        metrics=metrics,
        template_name="template.png",
    )


def test_full_score_at_least_point_seven_still_needs_valid_metrics() -> None:
    selected, reason = select_full_candidate(
        [candidate(0.71, good_metrics=False)], MatchingConfig()
    )
    assert selected is None
    assert reason == "full_rejected"


def test_full_score_at_least_point_seven_with_valid_metrics_is_direct_pass() -> None:
    selected, reason = select_full_candidate(
        [candidate(0.71, good_metrics=True)], MatchingConfig()
    )
    assert selected is not None
    assert reason == "full_direct"


def test_invalid_high_score_does_not_hide_a_valid_assisted_candidate() -> None:
    selected, reason = select_full_candidate(
        [candidate(0.85, good_metrics=False), candidate(0.64, good_metrics=True)],
        MatchingConfig(),
    )
    assert selected is not None
    assert selected.score == 0.64
    assert reason == "full_assisted"


def test_full_score_between_point_six_and_point_seven_needs_metrics() -> None:
    selected, reason = select_full_candidate(
        [candidate(0.65, good_metrics=False), candidate(0.64, good_metrics=True)],
        MatchingConfig(),
    )
    assert selected is not None
    assert selected.score == 0.64
    assert reason == "full_assisted"


def test_review_rejects_high_scores_when_auxiliary_metrics_fail(monkeypatch) -> None:
    bad_candidate = candidate(0.95, good_metrics=False)

    def always_bad_candidate(**_kwargs):
        return bad_candidate

    monkeypatch.setattr("mi_finding_smi_marks.handler._candidate", always_bad_candidate)
    result = _find_review(
        np.ones((60, 60, 3), dtype=np.uint8),
        [
            TemplateAsset(
                "MI/GA_TEMPLATE/P_01.png",
                np.ones((20, 20, 3), dtype=np.uint8),
            )
        ],
        MatchingConfig(),
    )

    assert result.success is False
    assert result.reason == "no_candidate_passed_thresholds"


def test_bottom_partial_recovers_the_original_template_center() -> None:
    rng = np.random.default_rng(19)
    image = rng.integers(0, 256, (260, 300, 3), dtype=np.uint8)
    template = rng.integers(0, 256, (100, 80, 3), dtype=np.uint8)
    target = rng.integers(0, 256, template.shape, dtype=np.uint8)
    bottom_height = round(template.shape[0] * 0.35)
    target[-bottom_height:, :] = template[-bottom_height:, :]
    x, y = 103, 81
    image[y : y + 100, x : x + 80] = target

    result = _find_review(
        image,
        [TemplateAsset("MI/GA_TEMPLATE/P_01.png", template)],
        MatchingConfig(),
    )

    assert result.success is True
    assert result.reason == "partial_pass"
    assert result.candidate is not None
    assert result.candidate.ratio == 0.35
    assert result.candidate.edge == "bottom"
    assert result.candidate.top_left == (x, y)
    assert result.candidate.center == (x + 40, y + 50)

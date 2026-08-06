from mi_finding.decision import select_full_candidate, select_partial_candidate
from mi_finding.models import MatchCandidate, MatchingConfig, MatchMetrics


def candidate(
    score: float, *, good_metrics: bool = True, stage: str = "full"
) -> MatchCandidate:
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
        stage=stage,  # type: ignore[arg-type]
        ratio=1.0 if stage == "full" else 0.7,
        top_left=(10, 20),
        width=30,
        height=40,
        metrics=metrics,
        template_index=0,
    )


def test_full_score_at_least_point_seven_is_direct_pass() -> None:
    selected, reason = select_full_candidate(
        [candidate(0.71, good_metrics=False)], MatchingConfig()
    )
    assert selected is not None
    assert reason == "full_direct"


def test_full_score_between_point_six_and_point_seven_needs_metrics() -> None:
    selected, reason = select_full_candidate(
        [candidate(0.65, good_metrics=False), candidate(0.64, good_metrics=True)],
        MatchingConfig(),
    )
    assert selected is not None
    assert selected.score == 0.64
    assert reason == "full_assisted"


def test_full_score_below_point_six_is_rejected() -> None:
    selected, reason = select_full_candidate([candidate(0.59)], MatchingConfig())
    assert selected is None
    assert reason == "full_rejected"


def test_partial_requires_point_seven_and_good_metrics() -> None:
    selected = select_partial_candidate(
        [
            candidate(0.69, stage="partial"),
            candidate(0.80, good_metrics=False, stage="partial"),
            candidate(0.72, stage="partial"),
        ],
        MatchingConfig(),
    )
    assert selected is not None
    assert selected.score == 0.72

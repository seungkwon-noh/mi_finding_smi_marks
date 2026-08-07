from __future__ import annotations

import numpy as np

from mi_finding_smi_marks.util_functions import (
    MatchingConfig,
    color_ssim,
    generate_edge_templates,
    match_template_logic,
)


def test_popup_candidate_threshold_matches_recent_handler_code() -> None:
    assert MatchingConfig().popup_min_score == 0.50


def test_small_ssim_does_not_require_a_seven_pixel_window() -> None:
    first = np.zeros((3, 3, 3), dtype=np.uint8)
    second = first.copy()
    assert color_ssim(first, second) == 1.0


def test_template_larger_than_image_is_safely_rejected() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    template = np.ones((30, 30, 3), dtype=np.uint8)
    result = match_template_logic(image, np.zeros((20, 20), dtype=np.uint8), template)
    assert result[2] == -1.0
    assert result[1].size == 0


def test_edge_templates_keep_offsets_for_coordinate_recovery() -> None:
    template = np.zeros((100, 80, 3), dtype=np.uint8)
    edges = {edge.edge: edge for edge in generate_edge_templates(template, 0.35)}
    assert edges["top"].offset_y == 0
    assert edges["bottom"].offset_y == 65
    assert edges["left"].offset_x == 0
    assert edges["right"].offset_x == 52
    assert all(
        edge.full_width == 80 and edge.full_height == 100 for edge in edges.values()
    )

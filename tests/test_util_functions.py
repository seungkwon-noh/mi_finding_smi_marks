from __future__ import annotations

import numpy as np

import mi_finding_smi_marks.handler as handler_module
import mi_finding_smi_marks.util_functions as util_module
from mi_finding_smi_marks.util_functions import (
    MatchingConfig,
    color_ssim,
    generate_edge_templates,
    match_template_logic,
)


def test_popup_candidate_threshold_matches_recent_handler_code() -> None:
    assert MatchingConfig().popup_min_score == 0.50


def test_default_partial_search_disables_point_three_five() -> None:
    assert MatchingConfig().partial_ratios == (0.70,)


def test_match_methods_calculates_image_clahe_once(monkeypatch) -> None:
    calls: list[tuple[int, ...]] = []

    def fake_clahe(gray, *_args, **_kwargs):
        calls.append(gray.shape)
        return gray.copy()

    monkeypatch.setattr(handler_module, "apply_clahe_image", fake_clahe)
    methods = handler_module._match_methods(np.ones((40, 50, 3), dtype=np.uint8))

    assert calls == [(40, 50)]
    assert methods[1].image_gray_is_clahe is True


def test_matching_reuses_precomputed_image_clahe(monkeypatch) -> None:
    calls: list[tuple[int, ...]] = []

    def fake_clahe(gray, *_args, **_kwargs):
        calls.append(gray.shape)
        return gray.copy()

    monkeypatch.setattr(util_module, "apply_clahe_image", fake_clahe)
    rng = np.random.default_rng(53)
    image = rng.integers(0, 256, (70, 80, 3), dtype=np.uint8)
    template = image[10:40, 20:55].copy()

    match_template_logic(
        image,
        np.zeros((70, 80), dtype=np.uint8),
        template,
        apply_clahe=True,
        image_gray_is_clahe=True,
    )

    assert calls == [template.shape[:2]]


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

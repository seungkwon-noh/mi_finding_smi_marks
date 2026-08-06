import numpy as np

from mi_finding import MatchingConfig, TemplateFinder


def test_exact_full_template_returns_original_center() -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (220, 260, 3), dtype=np.uint8)
    template = rng.integers(0, 256, (80, 60, 3), dtype=np.uint8)
    x, y = 91, 73
    image[y : y + 80, x : x + 60] = template

    result = TemplateFinder(MatchingConfig(remove_green_marker=False)).find(
        image, [template]
    )

    assert result.success is True
    assert result.reason == "full_direct"
    assert result.candidate is not None
    assert result.candidate.top_left == (x, y)
    assert result.candidate.center == (x + 30, y + 40)


def test_bottom_partial_match_is_mapped_back_to_full_template_box() -> None:
    rng = np.random.default_rng(19)
    image = rng.integers(0, 256, (260, 300, 3), dtype=np.uint8)
    template = rng.integers(0, 256, (100, 80, 3), dtype=np.uint8)
    target = rng.integers(0, 256, template.shape, dtype=np.uint8)
    bottom_height = round(template.shape[0] * 0.35)
    target[-bottom_height:, :] = template[-bottom_height:, :]
    x, y = 103, 81
    image[y : y + 100, x : x + 80] = target

    result = TemplateFinder(MatchingConfig(remove_green_marker=False)).find(
        image, [template]
    )

    assert result.success is True
    assert result.reason == "partial_pass"
    assert result.candidate is not None
    assert result.candidate.stage == "partial"
    assert result.candidate.ratio == 0.35
    assert result.candidate.edge == "bottom"
    assert result.candidate.top_left == (x, y)
    assert result.candidate.center == (x + 40, y + 50)

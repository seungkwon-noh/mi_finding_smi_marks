import numpy as np

from mi_finding.preprocessing import generate_edge_templates


def test_edge_templates_keep_offsets_for_full_coordinate_recovery() -> None:
    template = np.zeros((100, 80, 3), dtype=np.uint8)
    edges = {edge.edge: edge for edge in generate_edge_templates(template, 0.35)}

    assert edges["top"].offset_y == 0
    assert edges["bottom"].offset_y == 65
    assert edges["left"].offset_x == 0
    assert edges["right"].offset_x == 52
    assert all(
        edge.full_width == 80 and edge.full_height == 100 for edge in edges.values()
    )

"""Scanline crossings agree across scalar, row, and batch implementations."""

import numpy as np
import pytest

from core_pdf.impl._impl.render import paths


@pytest.mark.parametrize("limit", [0, 15, 1000])
@pytest.mark.parametrize("reverse_edges", [False, True])
@pytest.mark.parametrize("offset", [-100, 0, 50])
def test_scanline_crossings_preserve_half_open_bounds_and_edge_order(
    monkeypatch, limit, reverse_edges, offset
):
    monkeypatch.setattr(paths, "INTERNAL_CROSSING_MASK_CELL_LIMIT", limit)
    # Rising left edge, descending right edge, and an inactive horizontal edge.
    edges: list[tuple[float, float, float, float, float, float]] = [
        (2, 0, 2, 6, 0, 6),
        (8, 6, 8, 0, 0, 6),
        (2, 0, 8, 0, 0, 0),
    ]
    edges = [(x0 + offset, y0, x1 + offset, y1, low, high) for x0, y0, x1, y1, low, high in edges]
    expected_crossings = [(2 + offset, 1), (8 + offset, -1)]
    if reverse_edges:
        edges.reverse()
        expected_crossings.reverse()
    ys = [-1, 0, 3, 6, 7]
    expected = [[], expected_crossings, expected_crossings, [], []]
    array = np.array(edges, dtype=np.float64)
    assert (
        paths.internal_fill_path_sample_crossings_numpy(array, np.array(ys, dtype=np.float64))
        == expected
    )
    assert [paths.internal_fill_path_sample_crossings_row(array, y) for y in ys] == expected
    assert [paths.internal_fill_path_sample_crossings(edges, y) for y in ys] == expected


@pytest.mark.parametrize("limit", [0, 1000])
@pytest.mark.parametrize(("rows", "edges"), [(0, 0), (3, 0), (0, 2)])
def test_empty_crossing_inputs_preserve_row_cardinality(monkeypatch, limit, rows, edges):
    monkeypatch.setattr(paths, "INTERNAL_CROSSING_MASK_CELL_LIMIT", limit)
    result = paths.internal_fill_path_sample_crossings_numpy(
        np.zeros((edges, 6)), np.arange(rows, dtype=np.float64)
    )
    assert result == [[] for _ in range(rows)]


@pytest.mark.parametrize("limit", [0, 1000])
def test_diagonal_crossings_interpolate_positions_and_winding(monkeypatch, limit):
    monkeypatch.setattr(paths, "INTERNAL_CROSSING_MASK_CELL_LIMIT", limit)
    edges = np.array([(0, 0, 4, 4, 0, 4), (0, 4, 4, 0, 0, 4)], dtype=np.float64)
    assert paths.internal_fill_path_sample_crossings_numpy(edges, np.array([0, 1, 2, 3, 4])) == [
        [(0, 1), (4, -1)],
        [(1, 1), (3, -1)],
        [(2, 1), (2, -1)],
        [(3, 1), (1, -1)],
        [],
    ]


@pytest.mark.parametrize(
    ("crossings", "nonzero", "evenodd"),
    [
        ([], [], []),
        ([(0, 1), (10, -1)], list(range(10)), list(range(10))),
        ([(5, 1), (5, -1)], [], []),
        ([(0, 1), (2, -1), (8, 1), (10, -1)], [0, 1, 8, 9], [0, 1, 8, 9]),
        ([(0, 1), (2, 1), (8, -1), (10, -1)], list(range(10)), [0, 1, 8, 9]),
        ([(0, 1), (5, -1), (5, 1), (10, -1)], list(range(10)), list(range(10))),
        ([(0, 1), (0, -1), (10, 1), (10, -1)], [], []),
    ],
)
@pytest.mark.parametrize("rule", ["nonzero", "evenodd"])
@pytest.mark.parametrize("reverse", [False, True])
def test_crossing_spans_distinguish_holes_overlaps_and_shared_boundaries(
    crossings, nonzero, evenodd, rule, reverse
):
    values = list(reversed(crossings)) if reverse else list(crossings)
    spans = paths.internal_fill_path_crossing_spans(values, rule)
    occupied = [x for x in range(10) if any(start <= x + 0.5 < end for start, end in spans)]
    assert occupied == (nonzero if rule == "nonzero" else evenodd)
    assert all(start < end for start, end in spans)

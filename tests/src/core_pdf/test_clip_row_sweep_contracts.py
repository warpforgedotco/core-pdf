"""A clip's row spans, swept, are the spans of testing every edge on every row."""

import math
import random

import pytest

from core_pdf.impl.render_clipping import ClipState
from core_pdf.impl.render_paths import fill_path_crossing_spans
from tests.src.core_pdf.raster_support import make_target

Edge = tuple[float, float, float, float]


def every_edge_row(
    clip: ClipState, edges: list[Edge], py: int, fill_rule: str
) -> tuple[tuple[int, int], ...]:
    page_y = clip.crop_y1 - (py + 0.5) / clip.scale
    crossings: list[tuple[float, int]] = []
    for x0, y0, x1, y1 in edges:
        if y0 == y1:
            continue
        if min(y1, y0) <= page_y < max(y0, y1):
            offset = (page_y - y0) / (y1 - y0)
            crossings.append((x0 + offset * (x1 - x0), 1 if y1 > y0 else -1))
    spans: list[tuple[int, int]] = []
    for start_x, end_x in fill_path_crossing_spans(crossings, fill_rule):
        span = clip.page_x_to_pixel_span(start_x, end_x)
        if span is not None:
            spans.append(span)
    return tuple(spans)


def polygon_edges(rng: random.Random, count: int) -> list[Edge]:
    # Coordinates on a coarse grid, so rows land on vertices and edges share
    # tops, and horizontal edges turn up.
    points = [(rng.randint(0, 16) * 2.5, rng.randint(0, 16) * 2.5) for _ in range(count)]
    return [(*points[i], *points[(i + 1) % count]) for i in range(count)]


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("fill_rule", ["nonzero", "evenodd"])
def test_swept_rows_match_every_edge_on_every_row(seed: int, fill_rule: str) -> None:
    rng = random.Random(seed)
    clip = make_target(width=40, height=40).clip
    edges = polygon_edges(rng, rng.randint(3, 12)) + polygon_edges(rng, rng.randint(3, 6))
    swept = clip.path_rows_spans(tuple(edges), 0, 40, fill_rule)
    assert swept == [every_edge_row(clip, edges, py, fill_rule) for py in range(40)]


def test_an_edge_with_a_nan_end_crosses_no_row() -> None:
    clip = make_target(width=40, height=40).clip
    edges = [
        (0.0, math.nan, 10.0, 30.0),
        (0.0, 5.0, 10.0, math.nan),
        (5.0, 2.0, 5.0, 38.0),
        (30.0, 38.0, 30.0, 2.0),
    ]
    swept = clip.path_rows_spans(tuple(edges), 0, 40, "nonzero")
    assert swept == [every_edge_row(clip, edges, py, "nonzero") for py in range(40)]
    assert any(swept)

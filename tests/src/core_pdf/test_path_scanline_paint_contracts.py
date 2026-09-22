import numpy as np
import pytest

from core_pdf.impl.capture.records import CapturedPath, CapturedSubpath
from core_pdf.impl.render import path_fill_target
from core_pdf.impl.render.clipping import internal_ClipState
from core_pdf.impl.render.target import internal_RasterTarget


def internal_target(clip_kind):
    width, height = 48, 20
    pixels = bytearray(width * height * 4)
    view = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
    clip = internal_ClipState(crop_x0=0, crop_y1=height, scale=1, width=width, height=height)
    if clip_kind != "none":
        points: list[tuple[float, float]] = (
            [(2, 2), (40, 2), (40, 16), (2, 16)]
            if clip_kind == "rect"
            else [(2, 2), (43, 2), (2, 18)]
        )
        clip.push(CapturedPath([CapturedSubpath(points, closed=True)]), "nonzero")
    return internal_RasterTarget(
        pixels,
        None,
        clip=clip,
        width=width,
        height=height,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=height,
        page_view=view,
    ), view


def internal_in_clip(x, y, kind):
    if kind == "none":
        return True
    if kind == "rect":
        return 2 <= x < 40 and 2 <= y < 16
    return x >= 2 and y >= 2 and (x - 2) / 41 + (y - 2) / 16 < 1


@pytest.mark.parametrize("threshold", [0, 1_000_000])
@pytest.mark.parametrize("clip_kind", ["none", "rect", "triangle"])
@pytest.mark.parametrize("mode", [None, "Normal", "Multiply"])
@pytest.mark.parametrize("alpha", [0, 128, 255])
@pytest.mark.parametrize("fill_rule", ["nonzero", "evenodd"])
@pytest.mark.parametrize("opposite_inner", [False, True])
def test_scanline_holes_and_clipping_match_geometry(
    monkeypatch, threshold, clip_kind, mode, alpha, fill_rule, opposite_inner
):
    monkeypatch.setattr(path_fill_target, "RASTER_NUMPY_SPAN_MIN_PIXELS", threshold)
    target, actual = internal_target(clip_kind)
    edges = [(46, 1, 46, 18), (1, 18, 1, 1), (30, 5, 30, 14), (10, 14, 10, 5)]
    if opposite_inner:
        edges[2:] = [(x1, y1, x0, y0) for x0, y0, x1, y1 in edges[2:]]
    segments = [(x0, y0, x1, y1, min(y0, y1), max(y0, y1)) for x0, y0, x1, y1 in edges]
    color = (200, 30, 50, alpha)
    target.fill_path_scanlines(segments, (0, 0, 48, 20), color, mode, fill_rule)
    expected = np.zeros_like(actual)
    if alpha:
        for row in range(20):
            for column in range(48):
                x, y = column + 0.5, 19.5 - row
                outer = 1 <= x < 46 and 1 <= y < 18
                inner = 10 <= x < 30 and 5 <= y < 14
                hole = inner and (opposite_inner or fill_rule == "evenodd")
                if outer and not hole and internal_in_clip(x, y, clip_kind):
                    expected[row, column] = color
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fill_rule", ["nonzero", "evenodd"])
@pytest.mark.parametrize("extent", [(70, 80), (-20, -10), (5.1, 5.2), (4.5, 5.5)])
def test_scanline_spans_use_pixel_centers_and_crop_bounds(fill_rule, extent):
    target, actual = internal_target("none")
    left, right = extent
    segments = [(left, 0, left, 20, 0, 20), (right, 20, right, 0, 0, 20)]
    target.fill_path_scanlines(segments, (0, 0, 48, 20), (20, 30, 40, 255), None, fill_rule)
    expected = np.zeros_like(actual)
    for column in range(48):
        if left <= column + 0.5 < right:
            expected[:, column] = (20, 30, 40, 255)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("cached_edges", [False, True])
@pytest.mark.parametrize("clip_kind", ["none", "triangle"])
@pytest.mark.parametrize("mode", [None, "Multiply"])
@pytest.mark.parametrize("alpha", [0, 128, 255])
@pytest.mark.parametrize("fill_rule", ["nonzero", "evenodd"])
@pytest.mark.parametrize("opposite_inner", [False, True])
def test_sampled_and_analytic_fills_match_fractional_rectangle_geometry(
    cached_edges, clip_kind, mode, alpha, fill_rule, opposite_inner
):
    target, actual = internal_target(clip_kind)
    boxes = [(1.25, 1.25, 46.75, 18.75)] + [
        (left + 0.25, 5.25, left + 7.75, 14.75) for left in (5, 18, 31)
    ]
    subpaths = []
    for index, (left, bottom, right, top) in enumerate(boxes):
        points = [(left, bottom), (right, bottom), (right, top), (left, top)]
        if index and opposite_inner:
            points.reverse()
        subpaths.append(CapturedSubpath(points, closed=True))
    path = CapturedPath(subpaths)
    target.fill_path(
        path,
        (200, 30, 50, alpha),
        mode,
        fill_rule,
        edge_array=np.asarray(path.fill_edges()) if cached_edges else None,
    )
    expected = np.zeros_like(actual)
    for row in range(20):
        for column in range(48):
            if not internal_in_clip(column + 0.5, 19.5 - row, clip_kind):
                continue
            covered = 0
            for sy in range(4):
                for sx in range(4):
                    x, y = column + (sx + 0.5) / 4, 20 - row - (sy + 0.5) / 4
                    inside = [
                        left <= x < right and bottom <= y < top
                        for left, bottom, right, top in boxes
                    ]
                    hole = any(inside[1:]) and (opposite_inner or fill_rule == "evenodd")
                    covered += inside[0] and not hole
            opacity = round(alpha * covered / 16)
            if opacity:
                expected[row, column] = (200, 30, 50, opacity)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("cached_edges", [False, True])
@pytest.mark.parametrize("clip_kind", ["none", "rect", "triangle"])
@pytest.mark.parametrize("fill_rule", ["nonzero", "evenodd"])
@pytest.mark.parametrize("opposite_inner", [False, True])
def test_black_polygon_shortcut_preserves_integer_holes(
    cached_edges, clip_kind, fill_rule, opposite_inner
):
    target, actual = internal_target(clip_kind)
    outer: list[tuple[float, float]] = [(1, 1), (46, 1), (46, 19), (1, 19)]
    inner: list[tuple[float, float]] = [(10, 5), (30, 5), (30, 14), (10, 14)]
    if opposite_inner:
        inner.reverse()
    path = CapturedPath([CapturedSubpath(outer, closed=True), CapturedSubpath(inner, closed=True)])
    target.fill_path(
        path,
        (0, 0, 0, 255),
        fill_rule=fill_rule,
        edge_array=np.asarray(path.fill_edges()) if cached_edges else None,
    )
    expected = np.zeros_like(actual)
    for row in range(20):
        for column in range(48):
            x, y = column + 0.5, 19.5 - row
            outer_inside = 1 <= x < 46 and 1 <= y < 19
            hole = 10 <= x < 30 and 5 <= y < 14 and (opposite_inner or fill_rule == "evenodd")
            if outer_inside and not hole and internal_in_clip(x, y, clip_kind):
                expected[row, column] = (0, 0, 0, 255)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("alpha", [0, 128, 255])
@pytest.mark.parametrize("mode", [None, "Multiply"])
@pytest.mark.parametrize("isolated", [False, True])
def test_scanline_fill_tracks_group_coverage_without_changing_page_buffer(alpha, mode, isolated):
    target, page = internal_target("none")
    target.push_group(bytearray(page.nbytes), 1, None, isolated=isolated, knockout=False)
    segments = [(1, 1, 1, 19, 1, 19), (46, 19, 46, 1, 1, 19)]
    target.fill_path_scanlines(segments, (0, 0, 48, 20), (200, 30, 50, alpha), mode, "nonzero")
    expected = np.zeros_like(page)
    if alpha:
        expected[1:19, 1:46] = (200, 30, 50, alpha)
    np.testing.assert_array_equal(
        np.frombuffer(target.pixels, dtype=np.uint8).reshape(page.shape), expected
    )
    assert not page.any()
    coverage = np.zeros((20, 48), dtype=np.uint8)
    coverage[1:19, 1:46] = alpha
    if isolated:
        assert target.group_source_alpha is None
    else:
        np.testing.assert_allclose(target.group_source_alpha, coverage / 255, rtol=1e-7)
    target.composite_group(target.pop_group())
    np.testing.assert_array_equal(page, expected)

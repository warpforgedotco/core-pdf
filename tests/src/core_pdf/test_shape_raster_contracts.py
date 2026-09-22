import numpy as np
import pytest

from core_pdf.impl.capture.records import CapturedPath, CapturedSubpath
from core_pdf.impl.render import target as render_target
from core_pdf.impl.render.clipping import internal_ClipState
from core_pdf.impl.render.target import internal_RasterTarget


def internal_target(clip_kind):
    pixels = bytearray(12 * 12 * 4)
    view = np.frombuffer(pixels, dtype=np.uint8).reshape(12, 12, 4)
    clip = internal_ClipState(crop_x0=0, crop_y1=12, scale=1, width=12, height=12)
    if clip_kind != "none":
        points: list[tuple[float, float]] = (
            [(2, 2), (8, 2), (8, 10), (2, 10)]
            if clip_kind == "rect"
            else [(2, 2), (10, 2), (2, 10)]
        )
        clip.push(CapturedPath([CapturedSubpath(points, closed=True)]), "nonzero")
    return internal_RasterTarget(
        pixels,
        None,
        clip=clip,
        width=12,
        height=12,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=12,
        page_view=view,
    ), view


def internal_in_clip(x, y, kind):
    if kind == "none":
        return True
    if kind == "rect":
        return 2 <= x < 8 and 2 <= y < 10
    return x >= 2 and y >= 2 and x + y < 12


@pytest.mark.parametrize("threshold", [0, 1_000_000])
@pytest.mark.parametrize("clip_kind", ["none", "rect", "triangle"])
@pytest.mark.parametrize("alpha", [128, 255])
@pytest.mark.parametrize(
    ("cx", "cy", "radius"), [(6, 6, 3), (1, 1, 3), (20, 20, 1), (6, 6, 0), (5.25, 6.75, 2.25)]
)
def test_circle_routes_follow_pixel_center_geometry(
    monkeypatch, threshold, clip_kind, alpha, cx, cy, radius
):
    monkeypatch.setattr(render_target, "RASTER_CIRCLE_MIN_PIXEL_AREA", threshold)
    target, actual = internal_target(clip_kind)
    color = (200, 30, 50, alpha)
    target.fill_circle(cx, cy, radius, color)
    expected = np.zeros_like(actual)
    for row in range(12):
        for column in range(12):
            x, y = column + 0.5, 11.5 - row
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2 and internal_in_clip(x, y, clip_kind):
                expected[row, column] = color
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("clip_kind", ["none", "rect", "triangle"])
@pytest.mark.parametrize("mode", [None, "Normal", "Multiply", "Screen"])
@pytest.mark.parametrize("alpha", [0, 128, 255])
def test_integer_rectangles_respect_clips_and_blend_into_transparent_backdrop(
    clip_kind, mode, alpha
):
    target, actual = internal_target(clip_kind)
    color = (200, 30, 50, alpha)
    target.fill_rect((1, 1, 9, 9), color, mode)
    expected = np.zeros_like(actual)
    if alpha:
        for row in range(12):
            for column in range(12):
                x, y = column + 0.5, 11.5 - row
                if 1 <= x < 9 and 1 <= y < 9 and internal_in_clip(x, y, clip_kind):
                    expected[row, column] = color
    np.testing.assert_array_equal(actual, expected)

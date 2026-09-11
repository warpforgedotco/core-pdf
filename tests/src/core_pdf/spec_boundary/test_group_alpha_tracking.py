# SPDX-License-Identifier: AGPL-3.0-only
"""Group source alpha excludes the initial backdrop across every raster paint path."""

import numpy
import pytest

from core_pdf.impl._impl.capture.records import CapturedPath, CapturedSubpath
from core_pdf.impl._impl.render.clipping import internal_ClipState
from core_pdf.impl._impl.render.target import internal_RasterTarget
from core_pdf.impl._impl.runtime.array_views import uint8_image_view

internal_WIDTH = 60
internal_HEIGHT = 24


def internal_target(*, opaque: bool) -> internal_RasterTarget:
    pixels = bytearray((37, 89, 151, 255) if opaque else (0, 0, 0, 0)) * (
        internal_WIDTH * internal_HEIGHT
    )
    return internal_RasterTarget(
        pixels,
        None,
        clip=internal_ClipState(
            crop_x0=0,
            crop_y1=internal_HEIGHT,
            scale=1,
            width=internal_WIDTH,
            height=internal_HEIGHT,
        ),
        width=internal_WIDTH,
        height=internal_HEIGHT,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=internal_HEIGHT,
        page_view=uint8_image_view(pixels, (internal_HEIGHT, internal_WIDTH, 4)),
    )


def internal_triangle() -> CapturedPath:
    return CapturedPath([CapturedSubpath([(2, 2), (58, 2), (30, 23)], closed=True)])


def internal_paint(
    target: internal_RasterTarget,
    paint: str,
    rgba: tuple[int, int, int, int],
    blend_mode: str | None,
) -> None:
    if paint == "pixel":
        target.blend_px((8 * internal_WIDTH + 9) * 4, rgba, blend_mode)
    elif paint == "normal-pixel":
        target.blend_normal_pixel((8 * internal_WIDTH + 9) * 4, *rgba)
    elif paint in {"span-small", "span-large"}:
        target.blend_normal_solid_span(
            8 * internal_WIDTH * 4, 4, 12 if paint == "span-small" else 52, rgba
        )
    elif paint in {"rectangle", "rectangle-aa", "rectangle-clipped"}:
        if paint == "rectangle-clipped":
            target.clip.push(internal_triangle(), "nonzero")
        box = (4.2, 4.7, 53.4, 18.6) if paint == "rectangle-aa" else (4, 4, 54, 19)
        target.fill_rect(box, rgba, blend_mode)
    elif paint in {"path-analytic", "path-sampled", "path-clipped"}:
        if paint == "path-clipped":
            target.clip.push(internal_triangle(), "nonzero")
        target.fill_path(
            CapturedPath([CapturedSubpath([(4, 4), (53, 5), (51, 21), (9, 20)], closed=True)]),
            rgba,
            blend_mode,
            "evenodd" if paint == "path-sampled" else "nonzero",
        )
    elif paint in {"scanlines", "scanlines-clipped"}:
        if paint == "scanlines-clipped":
            target.clip.push(internal_triangle(), "nonzero")
        target.fill_path_scanlines(
            [(4, 4, 9, 20, 4, 20), (54, 4, 49, 20, 4, 20)],
            (4, 4, 54, 20),
            rgba,
            blend_mode,
            "evenodd",
        )
    elif paint == "glyph-bitmap":
        target.draw_glyph_bitmap(
            (6, 5, 22, 13), [0b10000001, 0b10100101, 0b11111111, 0], rgba, blend_mode, 8, 4
        )
    elif paint in {"circle-small", "circle-large"}:
        target.fill_circle(15, 12, 1.1 if paint == "circle-small" else 5.2, rgba, blend_mode)
    elif paint in {"stroke-small", "stroke-large"}:
        target.fill_line(
            4,
            4,
            7 if paint == "stroke-small" else 54,
            7 if paint == "stroke-small" else 20,
            1.5,
            rgba,
            blend_mode=blend_mode,
        )
    elif paint.startswith("image"):
        if paint == "image-rotated":
            quad = ((38.2, 5.8), (38.2, 20.2), (6.8, 5.8), (6.8, 20.2))
        elif paint == "image-edge":
            quad = ((6.8, 5.8), (54.2, 5.8), (6.8, 20.2), (54.2, 20.2))
        elif paint in {"image-affine", "image-masked", "image-clipped"}:
            quad = ((7, 5), (50, 7), (10, 20), (53, 22))
        else:
            quad = ((6.2, 5.3), (54.7, 5.3), (6.2, 20.8), (54.7, 20.8))
        if paint == "image-clipped":
            target.clip.push(internal_triangle(), "nonzero")
        source_alpha = (
            numpy.array([[255, 128], [0, 64]], dtype=numpy.uint8)
            if paint == "image-masked"
            else None
        )
        soft_mask = (
            numpy.array([[255, 128], [64, 0]], dtype=numpy.uint8)
            if paint == "image-masked"
            else None
        )
        assert target.blit_affine_image(
            quad,
            bytes((10, 50, 100, 200)),
            2,
            2,
            1,
            rgba[3] / 255 if rgba[3] != 255 else None,
            blend_mode,
            source_alpha=source_alpha,
            soft_mask=soft_mask,
        )
    else:
        raise AssertionError(paint)


@pytest.mark.parametrize(
    "paint",
    [
        "pixel",
        "normal-pixel",
        "span-small",
        "span-large",
        "rectangle",
        "rectangle-aa",
        "rectangle-clipped",
        "path-analytic",
        "path-sampled",
        "path-clipped",
        "scanlines",
        "scanlines-clipped",
        "glyph-bitmap",
        "circle-small",
        "circle-large",
        "stroke-small",
        "stroke-large",
        "image",
        "image-edge",
        "image-rotated",
        "image-affine",
        "image-masked",
        "image-clipped",
    ],
)
@pytest.mark.parametrize("alpha", [85, 255])
@pytest.mark.parametrize("blend_mode", [None, "Multiply"])
def test_source_alpha_matches_paint_over_transparent_backdrop(
    paint: str, alpha: int, blend_mode: str | None
) -> None:
    # ISO 32000-2 §11.4.5: a group's accumulated alpha excludes its initial backdrop.
    tracked = internal_target(opaque=True)
    tracked.push_group(bytearray(len(tracked.pixels)), None, None, isolated=False)
    transparent = internal_target(opaque=False)
    rgba = (149, 201, 57, alpha)
    internal_paint(tracked, paint, rgba, blend_mode)
    internal_paint(transparent, paint, rgba, blend_mode)
    assert tracked.group_source_alpha is not None
    expected = transparent.pixel_view(transparent.pixels)[..., 3]
    assert numpy.any(expected)
    assert numpy.any(expected == 0)
    assert numpy.all(tracked.pixel_view(tracked.pixels)[..., 3] == 255)
    numpy.testing.assert_allclose(tracked.group_source_alpha * 255, expected, atol=1)


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("blend_mode", [None, "Multiply"])
@pytest.mark.parametrize("child_alpha", [0.0, 0.5, 1.0])
def test_nested_group_records_its_alpha_once_in_parent(
    isolated: bool, blend_mode: str | None, child_alpha: float
) -> None:
    target = internal_target(opaque=True)
    target.push_group(bytearray(len(target.pixels)), None, None, isolated=False)
    target.fill_rect((4, 4, 20, 18), (211, 59, 139, 85))
    assert target.group_source_alpha is not None
    parent_alpha = target.group_source_alpha.copy()
    target.push_group(bytearray(len(target.pixels)), child_alpha, blend_mode, isolated=isolated)
    target.fill_rect((12, 4, 28, 18), (45, 219, 153, 153), "Screen")
    child = target.pop_group()
    numpy.testing.assert_array_equal(target.group_source_alpha, parent_alpha)
    target.composite_group(child)
    child_plane = numpy.zeros_like(parent_alpha)
    child_plane[internal_HEIGHT - 18 : internal_HEIGHT - 4, 12:28] = round(153 * child_alpha) / 255
    expected = parent_alpha + (1 - parent_alpha) * child_plane
    numpy.testing.assert_allclose(target.group_source_alpha, expected, atol=1 / 255)
    assert numpy.all(target.pixel_view(target.pixels)[..., 3] == 255)

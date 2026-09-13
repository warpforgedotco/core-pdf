# SPDX-License-Identifier: AGPL-3.0-only
"""Graphics-state Alpha masks modify source alpha and shape at paint boundaries."""

from dataclasses import replace

import numpy
import pytest

from core_pdf.impl._impl.capture.program import CapturedProgram
from core_pdf.impl._impl.capture.records import CapturedDrawing, CapturedPath, CapturedSoftMask
from core_pdf.impl._impl.render.clipping import internal_ClipState
from core_pdf.impl._impl.render.display import DisplayList
from core_pdf.impl._impl.render.model import ImagePaintItem, PathPaintItem
from core_pdf.impl._impl.render.soft_masks import (
    internal_graphics_soft_mask,
    internal_resolve_soft_mask,
)
from core_pdf.impl._impl.render.target import internal_RasterTarget
from core_pdf.impl._impl.runtime.array_views import uint8_image_view
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.image_spec import ImageSource, SoftMask
from core_pdf_spec.s_08_graphics.pdf_function import PdfFunctionEvaluator

internal_WIDTH = 12
internal_HEIGHT = 8
internal_BACKDROP = (40, 120, 200, 255)


def internal_path(x0: float = 0, width: float = internal_WIDTH) -> CapturedPath:
    path = CapturedPath()
    path.rect(x0, 0, width, internal_HEIGHT)
    return path


def internal_mask(
    opacity: float = 0.5,
    *,
    width: float = 6,
    transfer: PdfFunctionEvaluator | None = None,
    color: tuple[float, ...] = (1, 0, 0),
    offset: tuple[float, float] = (0, 0),
) -> CapturedSoftMask:
    return CapturedSoftMask(
        CapturedProgram(
            drawings=(CapturedDrawing(0, color, opacity, path=internal_path(width=width)),)
        ),
        transfer,
        offset,
    )


def internal_target(
    backdrop: tuple[int, int, int, int] = internal_BACKDROP,
) -> internal_RasterTarget:
    pixels = bytearray(backdrop) * (internal_WIDTH * internal_HEIGHT)
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


def internal_paint(
    mask: CapturedSoftMask | None,
    *,
    opacity: float = 1,
    blend: str | None = None,
    alpha_is_shape: bool = False,
    color: tuple[float, ...] = (1, 0, 0),
    kind: str = "fill",
) -> PathPaintItem:
    display = DisplayList(internal_WIDTH, internal_HEIGHT)
    display.append(
        kind,
        0,
        path=internal_path(),
        fill=color,
        stroke_color=color,
        fill_opacity=opacity,
        stroke_opacity=opacity,
        line_width=4,
        blend_mode=blend,
        alpha_is_shape=alpha_is_shape,
    )
    item = display.items[0]
    assert isinstance(item, PathPaintItem)
    return replace(item, graphics_soft_mask=mask)


@pytest.mark.parametrize("backdrop_alpha", [0, 128, 255])
@pytest.mark.parametrize("blend", [None, "Multiply", "Screen"])
def test_masked_element_preserves_backdrop_and_uses_blend_once(
    backdrop_alpha: int, blend: str | None
) -> None:
    backdrop = (*internal_BACKDROP[:3], backdrop_alpha)
    target = internal_target(backdrop)
    mask = internal_mask()
    target.paint_item(internal_paint(mask, opacity=0.5, blend=blend))
    control = internal_target(backdrop)
    # Original ca=0.5 is quantized to 128; the mask group contributes 128/255.
    control.paint_item(internal_paint(None, opacity=64 / 255, blend=blend))
    numpy.testing.assert_allclose(
        target.pixel_view(target.pixels)[3, 3],
        control.pixel_view(control.pixels)[3, 3],
        atol=1,
        rtol=0,
    )
    numpy.testing.assert_array_equal(target.pixel_view(target.pixels)[3, 9], backdrop)


@pytest.mark.parametrize("alpha_is_shape", [False, True])
@pytest.mark.parametrize("isolated", [False, True])
def test_masked_knockout_element_distinguishes_opacity_from_shape(
    alpha_is_shape: bool, isolated: bool
) -> None:
    target = internal_target()
    target.push_group(bytearray(len(target.pixels)), None, None, isolated=isolated, knockout=True)
    target.paint_item(internal_paint(None, color=(0, 1, 0)))
    target.paint_item(internal_paint(internal_mask(), alpha_is_shape=alpha_is_shape))
    target.composite_group(target.pop_group())
    actual = target.pixel_view(target.pixels)
    if alpha_is_shape:
        numpy.testing.assert_array_equal(actual[3, 9], [0, 255, 0, 255])
        numpy.testing.assert_allclose(actual[3, 3], [128, 127, 0, 255], atol=1, rtol=0)
    else:
        numpy.testing.assert_array_equal(actual[3, 9], internal_BACKDROP)
        expected = numpy.rint(numpy.array(internal_BACKDROP[:3]) * (127 / 255) + [128, 0, 0])
        numpy.testing.assert_allclose(actual[3, 3, :3], expected, atol=1, rtol=0)


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("blend", [None, "Multiply"])
def test_explicit_group_mask_is_applied_once_after_its_children(
    isolated: bool, blend: str | None
) -> None:
    target = internal_target()
    mask_alpha = internal_resolve_soft_mask(target, internal_mask())
    assert mask_alpha is not None
    target.push_group(
        bytearray(len(target.pixels)),
        0.5,
        blend,
        isolated=isolated,
        mask_alpha=mask_alpha,
    )
    target.paint_item(internal_paint(None, color=(0, 1, 0)))
    target.paint_item(internal_paint(None))
    target.composite_group(target.pop_group())
    control = internal_target()
    control.paint_item(internal_paint(None, opacity=64 / 255, blend=blend))
    numpy.testing.assert_allclose(
        target.pixel_view(target.pixels)[3, 3],
        control.pixel_view(control.pixels)[3, 3],
        atol=1,
        rtol=0,
    )
    numpy.testing.assert_array_equal(target.pixel_view(target.pixels)[3, 9], internal_BACKDROP)


def test_fillstroke_uses_mask_shape_on_each_internal_knockout_element() -> None:
    target = internal_target()
    item = internal_paint(internal_mask(width=12), alpha_is_shape=True, kind="fillstroke")
    target.paint_item(replace(item, stroke_color=(0, 0, 1)))
    mask_alpha = 128 / 255
    backdrop = numpy.array(internal_BACKDROP[:3], dtype=float)
    fill = backdrop * (1 - mask_alpha) + numpy.array([255, 0, 0]) * mask_alpha
    stroke = backdrop * (1 - mask_alpha) + numpy.array([0, 0, 255]) * mask_alpha
    expected = stroke + (1 - mask_alpha) * (fill - backdrop)
    numpy.testing.assert_allclose(
        target.pixel_view(target.pixels)[3, 1, :3], expected, atol=1, rtol=0
    )


def test_transfer_applies_to_zero_outside_group_bounds_and_follows_translation() -> None:
    mask = internal_mask(opacity=1, width=3, transfer=lambda value: (1 - value,), offset=(4, 0))
    target = internal_target()
    plane = internal_resolve_soft_mask(target, mask)
    assert plane is not None
    numpy.testing.assert_array_equal(plane[3], [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1])
    assert internal_resolve_soft_mask(target, mask) is plane
    assert not plane.flags.writeable
    target.paint_item(internal_paint(mask))
    numpy.testing.assert_array_equal(target.pixel_view(target.pixels)[3, 5], internal_BACKDROP)
    numpy.testing.assert_array_equal(target.pixel_view(target.pixels)[3, 9], [255, 0, 0, 255])


def test_alpha_mask_ignores_colors_and_destination_backdrop() -> None:
    red = internal_resolve_soft_mask(internal_target(), internal_mask(color=(1, 0, 0)))
    black = internal_resolve_soft_mask(
        internal_target((255, 255, 255, 0)), internal_mask(color=(0, 0, 0))
    )
    numpy.testing.assert_array_equal(red, black)


def test_enclosing_nonisolated_group_accumulates_masked_source_alpha_only() -> None:
    target = internal_target()
    target.push_group(bytearray(len(target.pixels)), 0.5, None, isolated=False)
    target.paint_item(internal_paint(internal_mask(), opacity=0.5))
    assert target.group_source_alpha is not None
    assert target.group_source_alpha[3, 3] == pytest.approx(64 / 255)
    assert target.group_source_alpha[3, 9] == 0
    target.composite_group(target.pop_group())
    control = internal_target()
    control.paint_item(internal_paint(None, opacity=32 / 255))
    numpy.testing.assert_allclose(
        target.pixel_view(target.pixels)[3, 3],
        control.pixel_view(control.pixels)[3, 3],
        atol=1,
        rtol=0,
    )
    numpy.testing.assert_array_equal(target.pixel_view(target.pixels)[3, 9], internal_BACKDROP)


def internal_image(mask: CapturedSoftMask, mask_kind: str) -> ImagePaintItem:
    source = ImageSource(
        b"\xff\x00\x00",
        {"Width": 1, "Height": 1, "BitsPerComponent": 8, "ColorSpace": "DeviceRGB"},
    )
    if mask_kind == "color-key":
        source.dictionary["Mask"] = [0, 0, 0, 0, 0, 0]
    elif mask_kind == "explicit":
        source.dictionary["Mask"] = PdfStream(raw_data=b"\x00")
    elif mask_kind == "soft":
        source.soft_mask = SoftMask(
            b"\xff", {"Width": 1, "Height": 1, "BitsPerComponent": 8, "ColorSpace": "DeviceGray"}
        )
    elif mask_kind.startswith("jpx"):
        source.dictionary["Filter"] = "JPXDecode"
        source.dictionary["SMaskInData"] = int(mask_kind[-1])
    elif mask_kind == "stencil":
        source.dictionary["ImageMask"] = True
        source.dictionary["BitsPerComponent"] = 1
        source.raw = b"\x00"
    display = DisplayList(internal_WIDTH, internal_HEIGHT)
    display.append("image", 0, image_source=source, bbox=(0, 0, 12, 8), fill=(1, 0, 0))
    item = display.items[0]
    assert isinstance(item, ImagePaintItem)
    return replace(item, graphics_soft_mask=mask)


@pytest.mark.parametrize("mask_kind", ["color-key", "explicit", "soft", "jpx1", "jpx2"])
def test_image_native_mask_overrides_graphics_mask_without_evaluating_it(mask_kind: str) -> None:
    mask = internal_mask()
    assert internal_graphics_soft_mask(internal_image(mask, mask_kind)) is None


@pytest.mark.parametrize("mask_kind", ["plain", "stencil", "jpx0"])
def test_plain_image_and_stencil_keep_graphics_soft_mask(mask_kind: str) -> None:
    mask = internal_mask()
    assert internal_graphics_soft_mask(internal_image(mask, mask_kind)) is mask


@pytest.mark.parametrize("mask_kind", ["plain", "stencil", "color-key", "soft"])
def test_image_raster_applies_either_graphics_mask_or_native_mask(mask_kind: str) -> None:
    target = internal_target()
    target.paint_item(internal_image(internal_mask(opacity=0, width=12), mask_kind))
    expected = [255, 0, 0, 255] if mask_kind in {"color-key", "soft"} else internal_BACKDROP
    numpy.testing.assert_array_equal(target.pixel_view(target.pixels)[3, 3], expected)

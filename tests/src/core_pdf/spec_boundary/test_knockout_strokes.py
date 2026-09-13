# SPDX-License-Identifier: AGPL-3.0-only
"""A stroke remains one knockout element when its raster pieces overlap."""

from typing import Any

import numpy
import pytest

from core_pdf.impl._impl.capture.records import CapturedPath, CapturedSubpath
from core_pdf.impl._impl.render.clipping import internal_ClipState
from core_pdf.impl._impl.render.display import DisplayList
from core_pdf.impl._impl.render.stroke_paint import internal_paint_stroke_once
from core_pdf.impl._impl.render.target import internal_RasterTarget
from core_pdf.impl._impl.runtime.array_views import uint8_image_view

internal_WIDTH = 24
internal_HEIGHT = 16


def internal_target(*, isolated: bool = False) -> internal_RasterTarget:
    pixels = bytearray((40, 120, 200, 255)) * (internal_WIDTH * internal_HEIGHT)
    target = internal_RasterTarget(
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
    target.push_group(bytearray(len(pixels)), None, None, isolated=isolated, knockout=True)
    return target


def internal_stroke(
    target: internal_RasterTarget,
    geometry: str,
    *,
    opacity: float,
    mode: str | None = None,
    alpha_is_shape: bool = False,
) -> None:
    if geometry == "caps":
        path = CapturedPath([CapturedSubpath([(5, 8), (19, 8)])])
    elif geometry == "crossing":
        path = CapturedPath(
            [CapturedSubpath([(3, 3), (21, 13)]), CapturedSubpath([(3, 13), (21, 3)])]
        )
    elif geometry == "join":
        path = CapturedPath([CapturedSubpath([(3, 8), (12, 8), (12, 14)])])
    else:
        raise AssertionError(geometry)
    display = DisplayList(internal_WIDTH, internal_HEIGHT)
    display.append(
        "stroke",
        0,
        path=path,
        stroke_color=(0.8, 0.4, 0.2),
        stroke_opacity=opacity,
        line_width=6,
        line_cap=1,
        line_join=1,
        blend_mode=mode,
        alpha_is_shape=alpha_is_shape,
    )
    target.paint_items(display.items)


@pytest.mark.parametrize(("geometry", "column"), [("caps", 6), ("crossing", 12), ("join", 12)])
@pytest.mark.parametrize("mode", [None, "Multiply"])
@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("alpha_is_shape", [False, True])
def test_stroke_overlap_uses_one_alpha_and_blend(
    geometry: str, column: int, mode: str | None, isolated: bool, alpha_is_shape: bool
) -> None:
    target = internal_target(isolated=isolated)
    internal_stroke(target, geometry, opacity=0.5, mode=mode, alpha_is_shape=alpha_is_shape)
    alpha = 128.0 / 255.0
    assert target.group_source_alpha is not None
    assert target.group_source_shape is not None
    assert target.group_source_alpha[8, column] == pytest.approx(alpha)
    assert target.group_source_shape[8, column] == pytest.approx(alpha if alpha_is_shape else 1.0)
    source = numpy.asarray([204.0, 102.0, 51.0])
    backdrop = numpy.asarray([40.0, 120.0, 200.0])
    if isolated:
        expected = source
    else:
        blended = source * backdrop / 255.0 if mode == "Multiply" else source
        expected = numpy.rint(alpha * blended + (1.0 - alpha) * backdrop)
    actual = target.pixel_view(target.pixels)[8, column]
    numpy.testing.assert_allclose(actual[:3], expected, rtol=0, atol=1)
    assert actual[3] == (128 if isolated else 255)


@pytest.mark.parametrize("geometry", ["caps", "crossing", "join"])
@pytest.mark.parametrize("alpha_is_shape", [False, True])
def test_zero_opacity_stroke_knocks_out_only_when_its_shape_remains(
    geometry: str, alpha_is_shape: bool
) -> None:
    target = internal_target()
    # First mark is fully opaque, so a later transparent stroke must restore
    # the initial backdrop when AIS=false, while AIS=true has zero shape.
    display = DisplayList(internal_WIDTH, internal_HEIGHT)
    path = CapturedPath()
    path.rect(0, 0, internal_WIDTH, internal_HEIGHT)
    display.append("fill", 0, path=path, fill_color=(1.0, 0.0, 0.0), fill_opacity=1.0)
    target.paint_items(display.items)
    internal_stroke(target, geometry, opacity=0.0, mode="Multiply", alpha_is_shape=alpha_is_shape)
    expected = [255, 0, 0, 255] if alpha_is_shape else [40, 120, 200, 255]
    numpy.testing.assert_array_equal(target.pixel_view(target.pixels)[8, 12], expected)
    assert target.group_source_alpha is not None
    assert target.group_source_alpha[8, 12] == (1.0 if alpha_is_shape else 0.0)


def test_stroke_opacity_does_not_exceed_constant_at_any_raster_overlap() -> None:
    target = internal_target(isolated=True)
    internal_stroke(target, "crossing", opacity=0.25)
    assert target.group_source_alpha is not None
    assert numpy.max(target.group_source_alpha) == pytest.approx(64.0 / 255.0)
    assert numpy.max(target.pixel_view(target.pixels)[..., 3]) == 64


def test_stroke_geometry_failure_restores_buffer_and_recording_planes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = internal_target()
    pixels, alpha, shape = target.pixels, target.group_source_alpha, target.group_source_shape
    before = bytes(pixels)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("geometry failed")

    monkeypatch.setattr(internal_RasterTarget, "stroke_path", fail)
    with pytest.raises(RuntimeError, match="geometry failed"):
        internal_paint_stroke_once(target, CapturedPath(), 6, (255, 0, 0, 128), None, None, 1, 1)
    assert target.pixels is pixels
    assert target.group_source_alpha is alpha
    assert target.group_source_shape is shape
    assert bytes(pixels) == before

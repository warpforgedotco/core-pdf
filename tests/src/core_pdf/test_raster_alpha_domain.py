import numpy as np
import pytest

from core_pdf.impl import render_target as raster
from core_pdf.impl.render_clipping import ClipState


def rounded_ratio(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient + (
        2 * remainder > denominator or (2 * remainder == denominator and quotient % 2 == 1)
    )


@pytest.mark.parametrize("route", ["generic", "span", "numpy-span"])
def test_every_byte_alpha_pair_matches_integer_source_over(monkeypatch, route):
    monkeypatch.setattr(
        raster, "RASTER_NUMPY_SPAN_MIN_PIXELS", 1 if route == "numpy-span" else 1000
    )
    pixels = bytearray(256 * 4)
    view = np.frombuffer(pixels, dtype=np.uint8).reshape(1, 256, 4)
    clip = ClipState(crop_x0=0, crop_y1=1, scale=1, width=256, height=1)
    target = raster.RasterTarget(
        pixels,
        None,
        clip=clip,
        width=256,
        height=1,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=1,
        page_view=view,
    )
    source = (231, 19, 127)
    destination = (13, 211, 67)
    for source_alpha in range(256):
        view[0, :, :3] = destination
        view[0, :, 3] = np.arange(256, dtype=np.uint8)
        rgba = (*source, source_alpha)
        if route in {"span", "numpy-span"}:
            target.blend_normal_solid_span(0, 0, 256, rgba)
        else:
            for destination_alpha in range(256):
                target.blend_px(destination_alpha * 4, rgba, None)
        expected = []
        for destination_alpha in range(256):
            if source_alpha == 0:
                expected.append((*destination, destination_alpha))
                continue
            denominator = source_alpha * 255 + destination_alpha * (255 - source_alpha)
            assert denominator > 0
            colors = tuple(
                rounded_ratio(
                    src * source_alpha * 255 + dst * destination_alpha * (255 - source_alpha),
                    denominator,
                )
                for src, dst in zip(source, destination, strict=True)
            )
            expected.append((*colors, rounded_ratio(denominator, 255)))
        expected_array = np.asarray(expected)
        assert np.max(np.abs(view[0, :, :3].astype(int) - expected_array[:, :3])) <= 1
        np.testing.assert_array_equal(view[0, :, 3], expected_array[:, 3])
        if source_alpha == 0:
            np.testing.assert_array_equal(view[0], expected_array)


@pytest.mark.parametrize("mode", ["multiply", "screen", "colordodge", "colorburn"])
@pytest.mark.parametrize("source_alpha", [0, 64, 255])
@pytest.mark.parametrize("destination_alpha", [0, 128, 255])
def test_non_normal_pixel_blending_matches_rational_composition(
    mode, source_alpha, destination_alpha
):
    from fractions import Fraction

    source = (200, 100, 50)
    destination = (90, 160, 240)
    pixels = bytearray((*destination, destination_alpha))
    view = np.frombuffer(pixels, dtype=np.uint8).reshape(1, 1, 4)
    clip = ClipState(crop_x0=0, crop_y1=1, scale=1, width=1, height=1)
    target = raster.RasterTarget(
        pixels,
        None,
        clip=clip,
        width=1,
        height=1,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=1,
        page_view=view,
    )
    target.blend_px(0, (*source, source_alpha), mode)
    if source_alpha == 0:
        assert tuple(pixels) == (*destination, destination_alpha)
        return
    sa, da = Fraction(source_alpha, 255), Fraction(destination_alpha, 255)
    alpha = sa + da * (1 - sa)
    expected = []
    for src, dst in zip(source, destination, strict=True):
        s, d = Fraction(src, 255), Fraction(dst, 255)
        blended = {
            "multiply": s * d,
            "screen": s + d - s * d,
            "colordodge": min(1, d / (1 - s)),
            "colorburn": 1 - min(1, (1 - d) / s),
        }[mode]
        premultiplied = (1 - sa) * da * d + (1 - da) * sa * s + sa * da * blended
        expected.append(round(premultiplied / alpha * 255))
    assert tuple(pixels[:3]) == pytest.approx(expected, abs=1)
    assert pixels[3] == round(alpha * 255)


@pytest.mark.parametrize("alpha", [0, 128, 255])
def test_pixel_blending_tracks_shape_independently_from_paint_alpha(alpha):
    pixels = bytearray(4)
    view = np.frombuffer(pixels, dtype=np.uint8).reshape(1, 1, 4)
    clip = ClipState(crop_x0=0, crop_y1=1, scale=1, width=1, height=1)
    target = raster.RasterTarget(
        pixels,
        None,
        clip=clip,
        width=1,
        height=1,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=1,
        page_view=view,
    )
    target.push_group(bytearray(4), None, None, isolated=False, track_shape=True)
    for _ in range(2):
        target.blend_px(0, (200, 50, 10, alpha), None, shape=128)
    group = target.pop_group()
    assert group.source_alpha is not None
    assert group.source_shape is not None
    assert group.source_alpha[0, 0] == pytest.approx(1 - (1 - alpha / 255) ** 2)
    assert group.source_shape[0, 0] == pytest.approx(1 - (1 - 128 / 255) ** 2)
    assert target.pixels is pixels
    assert pixels == bytearray(4)

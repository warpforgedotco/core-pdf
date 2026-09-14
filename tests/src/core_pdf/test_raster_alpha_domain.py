"""All byte-alpha pairs preserve source-over behavior across raster routes."""

import numpy as np
import pytest

from core_pdf.impl._impl.render import target as raster
from core_pdf.impl._impl.render.clipping import internal_ClipState


def rounded_ratio(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient + (
        2 * remainder > denominator or (2 * remainder == denominator and quotient % 2 == 1)
    )


@pytest.mark.parametrize("route", ["generic", "pixel", "span", "numpy-span"])
def test_every_byte_alpha_pair_matches_integer_source_over(monkeypatch, route):
    monkeypatch.setattr(
        raster, "RASTER_NUMPY_SPAN_MIN_PIXELS", 1 if route == "numpy-span" else 1000
    )
    pixels = bytearray(256 * 4)
    view = np.frombuffer(pixels, dtype=np.uint8).reshape(1, 256, 4)
    clip = internal_ClipState(crop_x0=0, crop_y1=1, scale=1, width=256, height=1)
    target = raster.internal_RasterTarget(
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
                index = destination_alpha * 4
                if route == "generic":
                    target.blend_px(index, rgba, None)
                else:
                    target.blend_normal_pixel(index, *rgba)
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
        # Floating implementations can differ from exact rational rounding by
        # one channel unit at ties; alpha and transparent no-op remain exact.
        assert np.max(np.abs(view[0, :, :3].astype(int) - expected_array[:, :3])) <= 1
        np.testing.assert_array_equal(view[0, :, 3], expected_array[:, 3])
        if source_alpha == 0:
            np.testing.assert_array_equal(view[0], expected_array)

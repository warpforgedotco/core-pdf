# SPDX-License-Identifier: AGPL-3.0-only

"""A colour converted once per profile, rendering and value.

transform caches conversions of up to SMALL_SAMPLE_ROWS samples, because lcms
rebuilds its transform on every call and a page repeats the same few colours
hundreds of times. These pin that the cache is invisible: the same bytes as an
uncached conversion, arrays a caller can write into without reaching the cache,
failures raised every time, and images never cached.
"""

import imagecodecs
import numpy as np
import pytest

from core_pdf.impl.graphics import icc_profiles
from core_pdf.impl.graphics.icc_profiles import (
    SMALL_SAMPLE_ROWS,
    IccProfileError,
    IccTransform,
    cached_cms_transform,
    cms_options,
    cms_transform,
    parse_icc_transform,
)
from core_pdf_spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
    ColorRendering,
    RenderingIntent,
)

SRGB = bytes(imagecodecs.cms_profile("srgb"))


@pytest.fixture
def rgb() -> IccTransform:
    cached_cms_transform.cache_clear()
    return parse_icc_transform(SRGB)


def direct(transform: IccTransform, samples: np.ndarray, rendering: ColorRendering) -> np.ndarray:
    intent, flags = cms_options(rendering)
    return cms_transform(transform.profile, transform.color_space, intent, flags, samples)


@pytest.mark.parametrize("intent", ["Perceptual", "RelativeColorimetric", "AbsoluteColorimetric"])
def test_cached_colours_are_the_uncached_bytes(rgb: IccTransform, intent: RenderingIntent) -> None:
    rendering = ColorRendering(intent=intent)
    rng = np.random.default_rng(7)
    for rows in (1, 3, SMALL_SAMPLE_ROWS):
        samples = rng.integers(0, 65536, size=(rows, 3), dtype=np.uint16)
        first = rgb.apply_uint16(samples, rendering=rendering)
        again = rgb.apply_uint16(samples.copy(), rendering=rendering)
        expected = direct(rgb, samples, rendering)
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(again, expected)
        assert first.dtype == np.uint8
        assert first.shape == (rows, 3)
    assert cached_cms_transform.cache_info().hits == 3


def test_a_result_is_the_callers_to_write_into(rgb: IccTransform) -> None:
    samples = np.array([[1000, 20000, 60000]], dtype=np.uint16)
    first = rgb.apply_uint16(samples)
    expected = first.copy()
    first[:] = 0
    np.testing.assert_array_equal(rgb.apply_uint16(samples), expected)


def test_strided_samples_convert_as_their_values(rgb: IccTransform) -> None:
    wide = np.array([[1000, 0, 20000, 0, 60000, 0]], dtype=np.uint16)
    strided = wide[:, ::2]
    assert not strided.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(
        rgb.apply_uint16(strided),
        direct(rgb, np.ascontiguousarray(strided), DEFAULT_COLOR_RENDERING),
    )


def test_images_bypass_the_cache(rgb: IccTransform) -> None:
    samples = np.zeros((SMALL_SAMPLE_ROWS + 1, 3), dtype=np.uint16)
    rgb.apply_uint16(samples)
    rgb.apply_uint16(samples)
    assert cached_cms_transform.cache_info().currsize == 0


def test_failures_are_raised_every_time(monkeypatch: pytest.MonkeyPatch, rgb: IccTransform) -> None:
    calls = 0

    def refuse(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise imagecodecs.CmsError("no")

    monkeypatch.setattr(icc_profiles.imagecodecs, "cms_transform", refuse)
    samples = np.array([[1, 2, 3]], dtype=np.uint16)
    for _ in range(2):
        with pytest.raises(IccProfileError):
            rgb.apply_uint16(samples)
    assert calls == 2

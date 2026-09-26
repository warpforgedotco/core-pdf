# SPDX-License-Identifier: AGPL-3.0-only

"""A colour converted once per profile, rendering and value.

transform keeps every sample row it converts in a call of up to MEMO_ROWS
rows, and converts only the rows it has not kept, because lcms rebuilds its
transform on every call and a page repeats the same few colours hundreds of
times. These pin that the memo is invisible: the same bytes as a direct
conversion, arrays a caller can write into without reaching it, failures
raised every time, and large images never kept.
"""

from typing import Any

import imagecodecs
import numpy as np
import pytest

from core_pdf.impl.graphics import icc_profiles
from core_pdf.impl.graphics.icc_profiles import (
    MEMO_ROWS,
    IccProfileError,
    IccTransform,
    cms_options,
    cms_transform,
    parse_icc_transform,
    row_memos,
)
from core_pdf_spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
    ColorRendering,
    RenderingIntent,
)

SRGB = bytes(imagecodecs.cms_profile("srgb"))


@pytest.fixture
def rgb() -> IccTransform:
    row_memos.clear()
    return parse_icc_transform(SRGB)


@pytest.fixture
def lcms_rows(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The row count of every call that reaches lcms."""
    counted: list[int] = []
    original = icc_profiles.imagecodecs.cms_transform

    def counting(data: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        counted.append(data.shape[0])
        return original(data, *args, **kwargs)

    monkeypatch.setattr(icc_profiles.imagecodecs, "cms_transform", counting)
    return counted


def direct(transform: IccTransform, samples: np.ndarray, rendering: ColorRendering) -> np.ndarray:
    intent, flags = cms_options(rendering)
    return cms_transform(transform.profile, transform.color_space, intent, flags, samples)


@pytest.mark.parametrize("intent", ["Perceptual", "RelativeColorimetric", "AbsoluteColorimetric"])
def test_kept_colours_are_the_direct_bytes(
    rgb: IccTransform, intent: RenderingIntent, lcms_rows: list[int]
) -> None:
    rendering = ColorRendering(intent=intent)
    rng = np.random.default_rng(7)
    for rows in (1, 3, 16, MEMO_ROWS):
        samples = rng.integers(0, 65536, size=(rows, 3), dtype=np.uint16)
        first = rgb.apply_uint16(samples, rendering=rendering)
        calls = len(lcms_rows)
        again = rgb.apply_uint16(samples.copy(), rendering=rendering)
        assert len(lcms_rows) == calls
        expected = direct(rgb, samples, rendering)
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(again, expected)
        assert first.dtype == np.uint8
        assert first.shape == (rows, 3)


def test_only_new_rows_reach_lcms(rgb: IccTransform, lcms_rows: list[int]) -> None:
    rng = np.random.default_rng(3)
    known = rng.integers(0, 65536, size=(40, 3), dtype=np.uint16)
    fresh = rng.integers(0, 65536, size=(5, 3), dtype=np.uint16)
    rgb.apply_uint16(known)
    mixed = np.concatenate([known[:20], fresh, known[20:], fresh[:2]])
    lcms_rows.clear()
    np.testing.assert_array_equal(
        rgb.apply_uint16(mixed), direct(rgb, mixed, DEFAULT_COLOR_RENDERING)
    )
    # The five new rows, and the repeat of two of them within the call,
    # go in one call; the forty kept ones do not.
    assert lcms_rows[0] == 7


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


def test_large_images_bypass_the_memo(rgb: IccTransform) -> None:
    samples = np.zeros((MEMO_ROWS + 1, 3), dtype=np.uint16)
    rgb.apply_uint16(samples)
    rgb.apply_uint16(samples)
    assert not any(row_memos.values())


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

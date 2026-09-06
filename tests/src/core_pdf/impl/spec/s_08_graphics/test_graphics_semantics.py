# SPDX-License-Identifier: AGPL-3.0-only
"""Device-independent rules and legacy graphics inputs have separate owners."""

import numpy
import pytest

from core_pdf.impl._impl.graphics.color_spec import color_spec_from_value as compatible_color_spec
from core_pdf.impl._impl.graphics.functions import (
    internal_compile_pdf_function as compatible_function,
)
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics.color import calgray_to_xyz
from core_pdf.impl.spec.s_08_graphics.color_kernels import unpack_subbyte_image_samples
from core_pdf.impl.spec.s_08_graphics.color_spec import color_spec_from_value
from core_pdf.impl.spec.s_08_graphics.pdf_function import internal_compile_pdf_function


def test_calgray_uses_its_gamma_and_white_point_before_device_conversion() -> None:
    assert calgray_to_xyz(0.5, 2.0, (0.8, 1.0, 1.2)) == pytest.approx((0.2, 0.25, 0.3))


def test_image_unpacking_keeps_each_row_byte_aligned() -> None:
    samples = unpack_subbyte_image_samples(b"\x6f\x93", 2, 3, 2, 1)
    numpy.testing.assert_array_equal(samples, (1, 2, 3, 2, 1, 0))


def test_missing_function_domain_is_a_compatibility_default() -> None:
    function = {"FunctionType": 2, "N": 2}
    with pytest.raises(ValueError):
        internal_compile_pdf_function(function)
    assert compatible_function(function)(0.5) == (0.25,)


def test_python_callable_is_an_application_function_input() -> None:
    def function(value: float) -> tuple[float]:
        return (value * 2,)

    with pytest.raises(ValueError):
        internal_compile_pdf_function(function)
    assert compatible_function(function)(0.25) == (0.5,)


def test_missing_icc_channel_count_is_only_filled_by_application_policy() -> None:
    value = ["ICCBased", {}]
    with pytest.raises(ValueError):
        color_spec_from_value(value)
    assert compatible_color_spec(value).channels == 3


def test_oversized_indexed_palette_is_an_application_compatibility_input() -> None:
    value = ["Indexed", "DeviceGray", 256, bytes(257)]
    with pytest.raises(ValueError):
        color_spec_from_value(value)
    assert compatible_color_spec(value).hival == 256


def test_malformed_sampled_encode_is_only_recovered_by_application_policy() -> None:
    function = PdfStream(
        {
            "FunctionType": 0,
            "BitsPerSample": 8,
            "Size": [2],
            "Domain": [0, 1],
            "Range": [0, 1],
            "Encode": ["bad", 1],
        },
        decoded_data=b"\x00\xff",
    )
    with pytest.raises(ValueError):
        internal_compile_pdf_function(function)
    assert compatible_function(function)(0.5) == (0.5,)


def test_application_function_normalization_preserves_decoded_stream_bytes() -> None:
    function = PdfStream(
        {
            "Filter": "FlateDecode",
            "FunctionType": 0,
            "BitsPerSample": 8,
            "Size": [2],
            "Domain": [0, 1],
            "Range": [0, 1],
        },
        decoded_data=b"\x00\xff",
    )
    assert compatible_function(function)(0.25) == (0.25,)

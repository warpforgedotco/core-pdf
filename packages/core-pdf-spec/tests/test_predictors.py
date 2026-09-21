"""Predictor wrappers enforce DecodeParms framing and translate kernel errors."""

import pytest

from core_pdf_spec.s_07_filters.decode_spec import FilterParams
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_filters.predictors import (
    apply_png_predictor,
    apply_predictor,
    apply_tiff_predictor,
)
from core_predictors.errors import PredictorError, UnsupportedPngFilterError
from core_predictors.png import png_predict
from core_predictors.tiff import tiff_predict


def test_png_rejects_an_incomplete_final_row_before_decoding_any_filter() -> None:
    data = b"\x05\x11\x00"
    with pytest.raises(PredictorError, match="^truncated PNG predictor row$"):
        png_predict(data, columns=1, colors=1, bits_per_component=8)
    with pytest.raises(FilterParseError, match="^truncated PNG predictor row$"):
        apply_png_predictor(data, FilterParams())


def test_png_unsupported_filter_error_is_translated_at_the_wrapper() -> None:
    with pytest.raises(UnsupportedPngFilterError, match="Unsupported PNG predictor filter 5"):
        png_predict(b"\x05\x11", columns=1, colors=1, bits_per_component=8)
    with pytest.raises(FilterUnsupportedError, match="Unsupported PNG predictor filter 5"):
        apply_png_predictor(b"\x05\x11", FilterParams())


def test_png_invalid_depth_error_is_translated_at_the_wrapper() -> None:
    with pytest.raises(PredictorError, match="^invalid PNG predictor bits 3$"):
        png_predict(b"", columns=1, colors=1, bits_per_component=3)
    with pytest.raises(FilterParseError, match="^invalid PNG predictor bits 3$"):
        apply_png_predictor(b"", FilterParams(bits_per_component=3))


def pack_samples(rows, bits):
    packed = bytearray()
    for row in rows:
        binary = "".join(f"{sample:0{bits}b}" for sample in row)
        binary += "0" * (-len(binary) % 8)
        packed.extend(int(binary, 2).to_bytes(len(binary) // 8, "big"))
    return bytes(packed)


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("colors", [1, 3])
@pytest.mark.parametrize("columns", [1, 3, 8, 129])
def test_tiff_differences_wrap_per_component_and_restart_each_row(bits, colors, columns):

    modulus = 1 << bits
    rows = [
        [(modulus - 1 - index * 3 + row * 7) % modulus for index in range(columns * colors)]
        for row in range(3)
    ]
    differences = [
        [
            value if index < colors else (value - row[index - colors]) % modulus
            for index, value in enumerate(row)
        ]
        for row in rows
    ]
    encoded, expected = pack_samples(differences, bits), pack_samples(rows, bits)
    assert (
        tiff_predict(memoryview(encoded), columns=columns, colors=colors, bits_per_component=bits)
        == expected
    )
    assert (
        apply_tiff_predictor(
            encoded, FilterParams(columns=columns, colors=colors, bits_per_component=bits)
        )
        == expected
    )


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
def test_tiff_wrapper_rejects_partial_rows_but_kernel_keeps_complete_rows(bits):

    params = FilterParams(columns=9, colors=3, bits_per_component=bits)
    row_size = (27 * bits + 7) // 8
    assert apply_tiff_predictor(b"", params) == b""
    assert tiff_predict(b"", columns=9, colors=3, bits_per_component=bits) == b""
    data = bytes(row_size + 1)
    with pytest.raises(FilterParseError, match="truncated TIFF"):
        apply_tiff_predictor(data, params)
    assert tiff_predict(data, columns=9, colors=3, bits_per_component=bits) == bytes(row_size)


def test_predictor_dispatch_preserves_passthrough_and_error_families():

    for params in (None, {}, FilterParams(), {"Predictor": 1}):
        assert apply_predictor(memoryview(b"payload"), params) == b"payload"
    assert apply_predictor(b"\x01\x02", {"Predictor": 2, "Columns": 2}) == b"\x01\x03"
    assert apply_predictor(b"\0\x01", {"Predictor": 15}) == b"\x01"
    with pytest.raises(FilterParseError, match="invalid stream predictor"):
        apply_predictor(b"", FilterParams(predictor=3))
    with pytest.raises(FilterParseError, match="invalid TIFF predictor bits"):
        apply_tiff_predictor(b"", FilterParams(bits_per_component=3))
    assert apply_png_predictor(b"", FilterParams()) == b""

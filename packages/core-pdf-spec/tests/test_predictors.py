"""PNG row decoding preserves strict framing and byte-oriented filter equations."""

import pytest

from core_pdf_spec.s_07_filters.decode_spec import FilterParams
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_filters.predictors import (
    PredictorError,
    UnsupportedPngFilterError,
    apply_png_predictor,
    png_predict,
)


@pytest.mark.parametrize("buffer_type", [bytes, memoryview])
def test_png_mixed_filters_use_the_previous_decoded_row(buffer_type: type) -> None:
    encoded = bytes(
        [
            0,
            250,
            10,
            20,
            30,
            1,
            5,
            255,
            2,
            250,
            2,
            250,
            252,
            251,
            5,
            3,
            129,
            129,
            130,
            129,
            4,
            1,
            2,
            3,
            4,
        ]
    )
    expected = bytes(
        [250, 10, 20, 30, 5, 4, 6, 0, 255, 0, 1, 5, 0, 129, 195, 229, 1, 131, 198, 233]
    )
    assert png_predict(buffer_type(encoded), columns=4, colors=1, bits_per_component=8) == expected


@pytest.mark.parametrize(
    ("filter_type", "expected"),
    [
        (0, [10, 20, 30, 40]),
        (1, [10, 30, 60, 100]),
        (2, [10, 20, 30, 40]),
        (3, [10, 25, 42, 61]),
        (4, [10, 30, 60, 100]),
    ],
)
def test_png_first_row_uses_zero_previous_bytes(filter_type: int, expected: list[int]) -> None:
    encoded = bytes([filter_type, 10, 20, 30, 40])
    assert png_predict(encoded, columns=4, colors=1, bits_per_component=8) == bytes(expected)


@pytest.mark.parametrize(
    ("previous", "encoded", "expected"),
    [
        (b"\x02\x03", b"\xfe\x07", b"\x00\x07"),
        (b"\x02\x00", b"\x01\x07", b"\x03\x07"),
        (b"\x0f\x14", b"\xfb\x07", b"\x0a\x16"),
    ],
)
def test_png_paeth_resolves_ties_in_prescribed_order(
    previous: bytes, encoded: bytes, expected: bytes
) -> None:
    # PNG Paeth chooses left, then above, then upper-left when distances tie.
    data = b"\x00" + previous + b"\x04" + encoded
    assert png_predict(data, columns=2, colors=1, bits_per_component=8) == previous + expected


@pytest.mark.parametrize(
    ("columns", "colors", "bits", "encoded", "expected"),
    [
        (2, 3, 8, b"\x01\x01\x02\x03\x04\x05\x06", b"\x01\x02\x03\x05\x07\x09"),
        (2, 1, 16, b"\x01\x01\x02\x03\x04", b"\x01\x02\x04\x06"),
        (9, 1, 1, b"\x01\x80\x40", b"\x80\xc0"),
    ],
)
def test_png_sub_filter_uses_byte_distance_for_the_pixel_layout(
    columns: int, colors: int, bits: int, encoded: bytes, expected: bytes
) -> None:
    assert png_predict(encoded, columns=columns, colors=colors, bits_per_component=bits) == expected


def test_png_empty_stream_decodes_to_empty_bytes() -> None:
    assert png_predict(b"", columns=3, colors=1, bits_per_component=2) == b""


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

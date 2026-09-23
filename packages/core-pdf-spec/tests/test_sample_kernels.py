import numpy
import pytest

from core_pdf_spec.s_07_filters.decode_spec import FilterParams
from core_pdf_spec.s_07_filters.errors import FilterParseError
from core_pdf_spec.s_07_filters.predictors import apply_tiff_predictor, tiff_predict_bits
from core_pdf_spec.s_08_graphics.color_kernels import (
    color_key_alpha,
    unpack_image_samples,
    unpack_subbyte_image_samples,
)


def test_color_key_requires_all_original_components_in_inclusive_ranges() -> None:
    values = numpy.asarray([[32768, 1], [32769, 1], [32768, 2]], dtype=numpy.uint16)
    assert color_key_alpha(values, (32768, 32768, 0, 1), 65535).tolist() == [0, 255, 255]


@pytest.mark.parametrize("mask", [(0, 65536), (2, 1), (0, 1 << 100), (False, 1), (0,)])
def test_color_key_rejects_invalid_source_ranges(mask: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="color key"):
        color_key_alpha(numpy.asarray([[1]], dtype=numpy.uint16), mask, 65535)


@pytest.mark.parametrize("buffer_kind", ["bytes", "memoryview", "ndarray"])
def test_sixteen_bit_words_keep_byte_order_low_bits_and_interleaved_rows(buffer_kind: str) -> None:
    packed = b"\x00\x00\x00\xff\x01\x00\x7f\xff\x80\x00\xff\xff"
    data = (
        memoryview(packed)
        if buffer_kind == "memoryview"
        else numpy.frombuffer(packed, dtype=numpy.uint8)
        if buffer_kind == "ndarray"
        else packed
    )
    values = unpack_image_samples(data, 16, 1, 2, 3)
    assert values.dtype == numpy.uint16
    assert values.tolist() == [0, 255, 256, 32767, 32768, 65535]


def test_word_unpacker_rejects_partial_last_word_and_ignores_trailing_bytes() -> None:
    with pytest.raises(ValueError, match="invalid image sample data"):
        unpack_image_samples(b"\xff\xff\x00", 16, 2, 1, 1)
    assert unpack_image_samples(b"\x12\x34\xff", 16, 1, 1, 1).tolist() == [0x1234]
    assert unpack_image_samples(b"\x12\xff", 8, 1, 1, 1).tolist() == [0x12]


@pytest.mark.parametrize("buffer_kind", ["bytes", "memoryview", "ndarray"])
@pytest.mark.parametrize(
    ("bits", "width", "components", "packed", "expected"),
    [
        (1, 3, 3, b"\xaa\xff\x55\x7f", [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        (1, 8, 1, b"\xa5\x3c", [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0]),
        (2, 3, 1, b"\x1b\xe7", [0, 1, 2, 3, 2, 1]),
        (4, 1, 3, b"\x12\x3f\xab\xcf", [1, 2, 3, 10, 11, 12]),
    ],
)
def test_image_samples_ignore_padding_at_each_row_boundary(
    bits: int,
    width: int,
    components: int,
    packed: bytes,
    expected: list[int],
    buffer_kind: str,
) -> None:
    data = (
        memoryview(packed)
        if buffer_kind == "memoryview"
        else numpy.frombuffer(packed, dtype=numpy.uint8)
        if buffer_kind == "ndarray"
        else packed
    )
    result = unpack_subbyte_image_samples(data, bits, width, 2, components)
    assert result.dtype == numpy.uint8
    assert result.shape == (len(expected),)
    assert result.tolist() == expected


def test_image_samples_ignore_bytes_after_the_declared_rows() -> None:
    result = unpack_subbyte_image_samples(b"\x1b\xe7\xff", 2, 3, 2, 1)
    assert result.tolist() == [0, 1, 2, 3, 2, 1]


def test_image_samples_reject_an_incomplete_declared_row() -> None:
    with pytest.raises(ValueError, match="^invalid image sample data$"):
        unpack_subbyte_image_samples(b"\x1b", 2, 3, 2, 1)


@pytest.mark.parametrize(
    ("bits", "width", "height", "components"),
    [(8, 1, 1, 1), (2, 0, 1, 1), (2, 1, 0, 1), (2, 1, 1, 0)],
)
def test_image_sample_layout_validation_stays_at_the_public_boundary(
    bits: int, width: int, height: int, components: int
) -> None:
    with pytest.raises(ValueError, match="^invalid image sample layout$"):
        unpack_subbyte_image_samples(b"", bits, width, height, components)


@pytest.mark.parametrize(
    ("bits", "columns", "colors", "encoded", "decoded"),
    [
        (1, 3, 3, b"\xae\xff\x55\x7f", b"\xb9\x80\x5e\x80"),
        (2, 3, 1, b"\xe7\x7f", b"\xd8\x4c"),
        (4, 3, 1, b"\xf2\x3f\x12\xff", b"\xf1\x40\x13\x20"),
    ],
)
def test_tiff_differences_wrap_per_component_and_reset_each_row(
    bits: int, columns: int, colors: int, encoded: bytes, decoded: bytes
) -> None:
    assert tiff_predict_bits(memoryview(encoded), columns, colors, bits) == decoded
    params = FilterParams(columns=columns, colors=colors, bits_per_component=bits)
    assert apply_tiff_predictor(encoded, params) == decoded


def test_raw_tiff_kernel_retains_complete_rows_but_strict_wrapper_rejects_truncation() -> None:
    encoded = b"\xf2\x3f\x12"
    assert tiff_predict_bits(encoded, 3, 1, 4) == b"\xf1\x40"
    assert tiff_predict_bits(b"\xf2", 3, 1, 4) == b""
    with pytest.raises(FilterParseError, match="^truncated TIFF predictor row$"):
        apply_tiff_predictor(encoded, FilterParams(columns=3, bits_per_component=4))

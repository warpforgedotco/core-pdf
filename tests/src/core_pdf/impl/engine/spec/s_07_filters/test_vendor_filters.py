from __future__ import annotations

import zlib

import imagecodecs
import numpy
import pytest

from core_pdf.impl.engine.spec.s_07_filters.codecs import (
    apply_ascii85,
    apply_ascii_hex,
    apply_lzw,
)
from core_pdf.impl.engine.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.engine.spec.s_07_filters.decoders import (
    decode_jpeg_image,
    decode_jpx_image,
    internal_jpx_thread_count,
)
from core_pdf.impl.engine.spec.s_07_filters.errors import FilterParseError
from core_pdf.impl.engine.spec.s_07_filters.flate import apply_flate
from core_pdf.impl.engine.spec.s_07_filters.pipeline import decode_stream_image_data
from core_pdf.impl.engine.spec.s_07_filters.predictor_impl import (
    png_predict,
    tiff_predict_16,
    tiff_predict_bits,
)
from core_pdf.impl.engine.spec.s_07_filters.predictors import (
    apply_png_predictor,
    apply_tiff_predictor,
)


def test_ascii_decoders() -> None:
    assert apply_ascii85(b"87cURD]j7BEbo80", {}) == b"Hello world!"
    assert apply_ascii_hex(b"61 62 63>", {}) == b"abc"


def test_lzw_decodes_default_early_change() -> None:
    # CLEAR, A, B, EOI encoded as PDF's MSB-first 9-bit codes.
    bits = "".join(f"{code:09b}" for code in (256, 65, 66, 257))
    bits += "0" * (-len(bits) % 8)
    encoded = bytes(int(bits[index : index + 8], 2) for index in range(0, len(bits), 8))

    assert apply_lzw(encoded, FilterParams()) == b"AB"


def test_image_filter_can_return_array_backed_jpeg_samples() -> None:
    source = numpy.array([[[10, 20, 30], [40, 50, 60]]], dtype=numpy.uint8)
    encoded = bytes(imagecodecs.jpeg_encode(source, level=95))

    decoded = decode_stream_image_data(
        encoded,
        {"Filter": "DCTDecode", "Width": 2, "Height": 1, "ColorSpace": "DeviceRGB"},
    )

    assert decoded is not None
    assert decoded.source == "jpeg"
    assert decoded.array.shape == (1, 2, 3)
    assert decoded.array.dtype == numpy.uint8


def test_imagecodecs_decoders_reuse_preallocated_output() -> None:
    source = numpy.zeros((2, 3), dtype=numpy.uint8)
    jpeg = bytes(imagecodecs.jpeg_encode(source))
    jpeg_output = numpy.empty_like(source)
    jpeg_decoded = decode_jpeg_image(memoryview(jpeg), out=jpeg_output)
    assert jpeg_decoded is jpeg_output

    jpx = bytes(imagecodecs.jpeg2k_encode(source))
    jpx_output = numpy.empty_like(source)
    jpx_decoded = decode_jpx_image(memoryview(jpx), out=jpx_output)
    assert jpx_decoded is jpx_output


def test_jpx_thread_count_is_bounded_and_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_PDF_JPX_THREADS", "2")
    assert internal_jpx_thread_count() == 2

    monkeypatch.setenv("CORE_PDF_JPX_THREADS", "99")
    assert internal_jpx_thread_count() == 4

    monkeypatch.setenv("CORE_PDF_JPX_THREADS", "invalid")
    assert 1 <= internal_jpx_thread_count() <= 4


def test_image_filter_rejects_non_identity_decode_array() -> None:
    source = numpy.zeros((1, 1, 3), dtype=numpy.uint8)
    encoded = bytes(imagecodecs.jpeg_encode(source))

    decoded = decode_stream_image_data(
        encoded,
        {
            "Filter": "DCTDecode",
            "Width": 1,
            "Height": 1,
            "ColorSpace": "DeviceRGB",
            "Decode": [1, 0, 1, 0, 1, 0],
        },
    )

    assert decoded is None


def test_image_filter_can_return_array_backed_cmyk_jpeg_samples() -> None:
    source = numpy.array([[[10, 20, 30, 40]]], dtype=numpy.uint8)
    encoded = bytes(imagecodecs.jpeg_encode(source))

    decoded = decode_stream_image_data(
        encoded,
        {"Filter": "DCTDecode", "Width": 1, "Height": 1, "ColorSpace": "DeviceCMYK"},
    )

    assert decoded is not None
    assert decoded.source == "jpeg"
    assert decoded.array.shape == (1, 1, 4)


def test_image_filter_can_return_array_backed_jpx_samples() -> None:
    source = numpy.zeros((1, 2, 3), dtype=numpy.uint8)
    encoded = bytes(imagecodecs.jpeg2k_encode(source))

    decoded = decode_stream_image_data(
        encoded,
        {"Filter": "JPXDecode", "Width": 2, "Height": 1},
    )

    assert decoded is not None
    assert decoded.source == "jpx"
    assert decoded.array.shape == (1, 2, 3)


def test_image_filter_can_return_array_backed_ccitt_samples() -> None:
    decoded = decode_stream_image_data(
        b"\xb6",
        {
            "Filter": "CCITTFaxDecode",
            "Width": 8,
            "Height": 1,
            "DecodeParms": {"Columns": 8, "Rows": 1, "K": 0},
        },
    )

    assert decoded is not None
    assert decoded.source == "ccitt"
    assert decoded.array.tolist() == [[255, 255, 255, 255, 0, 0, 0, 0]]


def test_flate_image_returns_native_array_samples() -> None:
    samples = bytes(range(12))
    decoded = decode_stream_image_data(
        zlib.compress(samples),
        {
            "Filter": "FlateDecode",
            "Width": 4,
            "Height": 3,
            "ColorSpace": "DeviceGray",
            "BitsPerComponent": 8,
        },
    )

    assert decoded is not None
    assert decoded.source == "flate"
    assert decoded.array.shape == (3, 4)
    assert decoded.array.tobytes() == samples


def test_flate_decodes_raw_stream() -> None:
    compressor = zlib.compressobj(wbits=-15)
    encoded = compressor.compress(b"hello") + compressor.flush()
    assert apply_flate(encoded, {}) == b"hello"


def test_flate_decodes_zlib_stream() -> None:
    assert apply_flate(zlib.compress(b"hello"), {}) == b"hello"


def test_flate_accepts_truncated_empty_raw_stream() -> None:
    assert apply_flate(bytes.fromhex("48890300"), {}) == b""


def test_flate_rejects_garbage() -> None:
    with pytest.raises(FilterParseError):
        apply_flate(b"not compressed data", {})


def test_predictors_reject_truncated_rows() -> None:
    params = FilterParams(columns=4, colors=1, bits_per_component=8)
    with pytest.raises(FilterParseError, match="truncated TIFF predictor row"):
        apply_tiff_predictor(b"abc", params)
    with pytest.raises(FilterParseError, match="truncated PNG predictor row"):
        apply_png_predictor(b"\x00abc", params)


def test_tiff_predict_16_matches_scalar_reference() -> None:
    columns = 17
    colors = 3
    rng = numpy.random.default_rng(123)
    source = rng.integers(0, 65536, 4 * columns * colors, dtype=numpy.uint16)
    encoded = source.astype(">u2").tobytes()
    expected = bytearray(encoded)
    samples_per_row = columns * colors
    row_bytes = samples_per_row * 2
    for row_start in range(0, len(expected), row_bytes):
        for sample in range(colors, samples_per_row):
            offset = row_start + sample * 2
            previous = row_start + (sample - colors) * 2
            value = int.from_bytes(expected[offset : offset + 2], "big")
            prior = int.from_bytes(expected[previous : previous + 2], "big")
            expected[offset : offset + 2] = ((value + prior) & 0xFFFF).to_bytes(2, "big")
    assert tiff_predict_16(encoded, columns, colors) == bytes(expected)


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_tiff_predict_bits_vector_path_matches_scalar_reference(bits: int) -> None:
    columns = 257
    colors = 3
    row_bytes = (columns * colors * bits + 7) // 8
    rng = numpy.random.default_rng(bits)
    encoded = rng.integers(0, 256, 7 * row_bytes, dtype=numpy.uint8).tobytes()
    expected = tiff_predict_bits(encoded[:512], columns, colors, bits)
    actual = tiff_predict_bits(encoded, columns, colors, bits)

    # The short input exercises the scalar branch; the full input exercises the
    # lookup-table/vectorized branch. Both must produce the same per-row result.
    assert actual[: len(expected)] == expected


def test_png_sub_predict_matches_scalar_reference() -> None:
    columns = 31
    colors = 4
    rng = numpy.random.default_rng(456)
    row_length = columns * colors
    encoded_rows = rng.integers(0, 256, 5 * row_length, dtype=numpy.uint8).tobytes()
    encoded = b"".join(
        b"\x01" + encoded_rows[index : index + row_length]
        for index in range(0, len(encoded_rows), row_length)
    )
    expected = bytearray()
    for row_start in range(0, len(encoded_rows), row_length):
        row = bytearray(encoded_rows[row_start : row_start + row_length])
        for index in range(colors, row_length):
            row[index] = (row[index] + row[index - colors]) & 0xFF
        expected.extend(row)
    assert png_predict(encoded, columns=columns, colors=colors, bits_per_component=8) == bytes(
        expected
    )


@pytest.mark.parametrize("filter_type", [3, 4])
def test_png_recursive_predictors_match_scalar_reference(filter_type: int) -> None:
    columns = 31
    colors = 4
    row_length = columns * colors
    rng = numpy.random.default_rng(filter_type)
    encoded_rows = rng.integers(0, 256, 5 * row_length, dtype=numpy.uint8).tobytes()
    encoded = b"".join(
        bytes([filter_type]) + encoded_rows[index : index + row_length]
        for index in range(0, len(encoded_rows), row_length)
    )
    expected = bytearray()
    previous = bytearray(row_length)
    for row_start in range(0, len(encoded_rows), row_length):
        encoded_row = encoded_rows[row_start : row_start + row_length]
        row = bytearray(encoded_row)
        for index in range(row_length):
            left = row[index - colors] if index >= colors else 0
            up = previous[index]
            if filter_type == 3:
                predictor = (left + up) >> 1
            else:
                up_left = previous[index - colors] if index >= colors else 0
                estimate = left + up - up_left
                distances = (abs(estimate - left), abs(estimate - up), abs(estimate - up_left))
                predictor = (left, up, up_left)[min(range(3), key=distances.__getitem__)]
            row[index] = (row[index] + predictor) & 0xFF
        expected.extend(row)
        previous = row
    assert png_predict(encoded, columns=columns, colors=colors, bits_per_component=8) == bytes(
        expected
    )

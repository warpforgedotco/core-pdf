# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import base64
import binascii
import zlib
from itertools import product

import numpy
import pytest

from core_pdf.impl.primitives import PdfName
from core_pdf.impl.spec.s_07_filters import predictors
from core_pdf.impl.spec.s_07_filters.codecs import (
    apply_ascii85,
    apply_ascii_hex,
    apply_flate,
    looks_like_pdf_content_stream,
)
from core_pdf.impl.spec.s_07_filters.decode_spec import (
    FilterParams,
    normalize_stream_decode_spec,
)
from core_pdf.impl.spec.s_07_filters.errors import FilterParseError
from core_pdf.impl.spec.s_07_filters.pipeline import decode_stream_data
from core_pdf.impl.spec.s_07_filters.predictors import (
    apply_png_predictor,
    apply_tiff_predictor,
)
from core_pdf.impl.spec.s_07_filters.registry import PREDICTOR_FILTERS
from core_pdf.impl.spec.s_07_syntax_primitives.content_operators import (
    PDF_CONTENT_OPERATOR_BYTES,
)
from core_pdf.impl.spec.s_08_graphics.color_kernels import (
    unpack_subbyte_image_samples,
)


def gzip_compress(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=zlib.MAX_WBITS | 16)
    return compressor.compress(data) + compressor.flush()


def raw_deflate_compress(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(data) + compressor.flush()


def test_filter_registry_pins_predictor_policy() -> None:
    assert frozenset({"FlateDecode", "Fl", "LZWDecode", "LZW"}) == PREDICTOR_FILTERS


@pytest.mark.parametrize("alias", ["FlateDecode", "FLATEDECODE", "PlateDecode"])
def test_filter_registry_preserves_flate_compatibility_aliases(alias: str) -> None:
    spec = normalize_stream_decode_spec({"Filter": PdfName.of(alias)})

    assert spec.filters == ("FlateDecode",)


def test_apply_flate_decodes_gzip_wrapped_stream() -> None:
    assert apply_flate(gzip_compress(b"hello"), {}) == b"hello"


def test_apply_flate_decodes_raw_deflate_stream() -> None:
    assert apply_flate(raw_deflate_compress(b"hello"), {}) == b"hello"


def test_apply_flate_decodes_complete_short_raw_deflate_stream() -> None:
    compressed = raw_deflate_compress(b"A")

    assert len(compressed) < 8
    assert apply_flate(compressed, {}) == b"A"


def test_apply_flate_recovers_bad_zlib_checksum() -> None:
    compressed = bytearray(zlib.compress(b"hello"))
    compressed[-1] ^= 0xFF

    assert apply_flate(bytes(compressed), {}) == b"hello"


def test_apply_flate_tolerates_mislabeled_content_stream() -> None:
    content = b"BT /F1 12 Tf 72 720 Td (Hello) Tj ET"

    assert apply_flate(content, {}) == content


def test_apply_flate_rejects_non_pdf_garbage() -> None:
    with pytest.raises(FilterParseError):
        apply_flate(b"not compressed data", {})


@pytest.mark.parametrize(
    "data",
    [
        b"(Tj Do q)",
        b"<546a 446f 71>",
        b"/Tj /Do /q",
        b"[Tj Do q]",
        b"<< /Operator Tj /Other Do >>",
        b"% Tj Do q\nnot content",
    ],
)
def test_content_stream_detection_ignores_operator_like_operands(data: bytes) -> None:
    assert not looks_like_pdf_content_stream(data)


@pytest.mark.parametrize("data", [b"(Tj)", b"/Do", b"[q]"])
def test_apply_flate_rejects_operator_like_uncompressed_operands(data: bytes) -> None:
    with pytest.raises(FilterParseError):
        apply_flate(data, {})


def test_content_stream_detection_finds_operator_after_complex_operands() -> None:
    data = b"[(Tj) /Do << /Value q >>] TJ"

    assert looks_like_pdf_content_stream(data)


@pytest.mark.parametrize("operator", sorted(PDF_CONTENT_OPERATOR_BYTES))
def test_content_stream_detection_uses_every_canonical_operator(operator: bytes) -> None:
    assert looks_like_pdf_content_stream(b"0 " + operator)


def test_content_stream_detection_supports_sliced_memoryview() -> None:
    content = b"(embedded Tj) not-content"
    data = memoryview(b"prefix" + content + b"suffix")[len(b"prefix") : -len(b"suffix")]

    assert not looks_like_pdf_content_stream(data)


def test_content_stream_detection_supports_reversed_memoryview() -> None:
    content = b"[(Tj)] TJ"

    assert looks_like_pdf_content_stream(memoryview(content[::-1])[::-1])


def reference_ascii_hex(data: bytes) -> bytes:
    filtered = bytearray()
    for byte in data:
        if byte in b"\x00\t\n\x0c\r ":
            continue
        if byte == 62:
            break
        if byte in b"0123456789ABCDEFabcdef":
            filtered.append(byte)
    if len(filtered) & 1:
        filtered.append(48)
    return binascii.unhexlify(filtered)


def test_apply_ascii_hex_matches_reference_exhaustively() -> None:
    alphabet = b"0Af Z\n>"
    for length in range(6):
        for value in product(alphabet, repeat=length):
            data = bytes(value)

            assert apply_ascii_hex(data, {}) == reference_ascii_hex(data)


def test_apply_ascii85_rejects_invalid_data() -> None:
    with pytest.raises(FilterParseError):
        apply_ascii85(b"!!!!~bad", {})


@pytest.mark.parametrize("data", [b"uuuuu~>", b"uuuu~>", b"!~>", b"z!~>"])
def test_apply_ascii85_rejects_overflow_and_incomplete_tuples(data: bytes) -> None:
    with pytest.raises(FilterParseError, match="invalid ASCII85Decode stream"):
        apply_ascii85(data, {})


def test_apply_ascii85_accepts_maximum_tuple() -> None:
    assert apply_ascii85(b"s8W-!~>", {}) == b"\xff\xff\xff\xff"


def test_apply_ascii85_matches_stdlib_across_tuple_lengths() -> None:
    for length in range(261):
        payload = bytes(index & 0xFF for index in range(length))
        encoded = base64.a85encode(payload, adobe=True)

        assert apply_ascii85(encoded, {}) == payload
        assert apply_ascii85(memoryview(encoded), {}) == payload

    zero_payload = b"\0" * 64
    assert apply_ascii85(base64.a85encode(zero_payload, adobe=True), {}) == zero_payload


def test_apply_ascii85_decodes_large_vectorizable_stream() -> None:
    payload = bytes((index * 17) & 0xFF for index in range(50_000))
    encoded = base64.a85encode(payload, adobe=True)

    assert apply_ascii85(encoded, {}) == payload


def test_apply_ascii85_rejects_large_overflowing_stream() -> None:
    with pytest.raises(FilterParseError, match="invalid ASCII85Decode stream"):
        apply_ascii85(b"u" * 5_000, {})


def test_tiff_predictor_rejects_truncated_row() -> None:
    params = FilterParams(columns=4, colors=1, bits_per_component=8)

    with pytest.raises(FilterParseError, match="truncated TIFF predictor row"):
        apply_tiff_predictor(b"abc", params)


def test_png_predictor_rejects_truncated_row() -> None:
    params = FilterParams(columns=4, colors=1, bits_per_component=8)

    with pytest.raises(FilterParseError, match="truncated PNG predictor row"):
        apply_png_predictor(b"\x00abc", params)


def test_png_predictor_can_recover_complete_rows_before_truncation() -> None:
    params = FilterParams(
        columns=4,
        colors=1,
        bits_per_component=8,
        damaged_rows_before_error=True,
    )

    assert apply_png_predictor(b"\x00full\x00bad", params) == b"full"


@pytest.mark.parametrize("bits_per_component", [1, 2, 4])
def test_subbyte_image_samples_match_expected_values(bits_per_component: int) -> None:
    width = 13
    height = 3
    components = 2
    values = [
        (index * 3 + 1) & ((1 << bits_per_component) - 1)
        for index in range(width * height * components)
    ]
    row_sample_count = width * components
    row_bits = row_sample_count * bits_per_component
    row_bytes = (row_bits + 7) // 8
    encoded = bytearray()
    for row_start in range(0, len(values), row_sample_count):
        bit_string = "".join(
            f"{value:0{bits_per_component}b}"
            for value in values[row_start : row_start + row_sample_count]
        )
        bit_string = bit_string.ljust(row_bytes * 8, "0")
        encoded.extend(
            int(bit_string[index : index + 8], 2) for index in range(0, len(bit_string), 8)
        )
    numpy.testing.assert_array_equal(
        unpack_subbyte_image_samples(
            bytes(encoded),
            bits_per_component,
            width,
            height,
            components,
        ),
        numpy.asarray(values, dtype=numpy.uint8),
    )


@pytest.mark.parametrize(
    ("columns", "colors", "bits_per_component"),
    [
        (64, 1, 8),
        (64, 3, 8),
        (64, 4, 8),
        (64, 1, 16),
        (32, 3, 16),
        (64, 1, 1),
        (64, 1, 2),
        (64, 1, 4),
    ],
)
def test_png_predictor_codec_path_matches_scalar_path(
    monkeypatch: pytest.MonkeyPatch, columns: int, colors: int, bits_per_component: int
) -> None:
    from core_pdf.impl.spec.s_07_filters import predictors as predictor_impl

    row_length = max(1, (columns * colors * bits_per_component + 7) // 8)
    rng = numpy.random.default_rng(columns * 31 + colors * 7 + bits_per_component)
    row_count = max(24, predictor_impl.PNG_CODEC_THRESHOLD // (row_length + 1) + 1)
    rows = rng.integers(0, 256, size=(row_count, row_length), dtype=numpy.uint8)
    encoded = bytearray()
    for index, row in enumerate(rows):
        encoded.append(index % 5)
        encoded.extend(row.tobytes())
    data = bytes(encoded)
    assert len(data) >= predictor_impl.PNG_CODEC_THRESHOLD

    codec = predictor_impl.png_predict(
        data, columns=columns, colors=colors, bits_per_component=bits_per_component
    )
    monkeypatch.setattr(predictor_impl, "PNG_CODEC_THRESHOLD", len(data) + 1)
    scalar = predictor_impl.png_predict(
        data, columns=columns, colors=colors, bits_per_component=bits_per_component
    )
    assert codec == scalar


def test_png_predictor_codec_path_falls_back_on_damaged_filter_byte() -> None:
    from core_pdf.impl.spec.s_07_filters import predictors as predictor_impl

    columns, colors = 512, 1
    good_row = b"\x02" + bytes(columns)
    bad_row = b"\x09" + bytes(columns)
    data = good_row * 2 + bad_row + good_row
    assert len(data) >= predictor_impl.PNG_CODEC_THRESHOLD

    recovered = predictor_impl.png_predict(
        data, columns=columns, colors=colors, bits_per_component=8, damaged_rows_before_error=True
    )
    assert recovered == bytes(columns * 2)
    with pytest.raises(predictor_impl.UnsupportedPngFilterError):
        predictor_impl.png_predict(data, columns=columns, colors=colors, bits_per_component=8)


def test_empty_png_predictor_avoids_row_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    params = FilterParams(columns=10**9, colors=4, bits_per_component=16)

    def unexpected_decode(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("empty predictor should not allocate or decode rows")

    monkeypatch.setattr(predictors, "png_predict", unexpected_decode)

    assert apply_png_predictor(b"", params) == b""


def test_decode_stream_data_preserves_reversed_memoryview_order() -> None:
    source = b"abcdef"

    assert decode_stream_data(memoryview(source)[::-1], None) == b"fedcba"


def test_crypt_filter_without_params_is_identity_after_security_stage() -> None:
    assert decode_stream_data(b"plain", {"Filter": PdfName.of("Crypt")}) == b"plain"

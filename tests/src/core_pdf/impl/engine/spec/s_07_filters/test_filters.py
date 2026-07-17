# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import binascii
import zlib
from itertools import product
from typing import cast

import pytest

from core_pdf.impl.engine.spec.s_07_filters import decoders
from core_pdf.impl.engine.spec.s_07_filters.codecs import apply_ascii85, apply_ascii_hex
from core_pdf.impl.engine.spec.s_07_filters.flate import (
    apply_flate,
    looks_like_pdf_content_stream,
)
from core_pdf.impl.engine.spec.s_07_filters.pipeline import decode_stream_data
from core_pdf.impl.engine.spec.s_07_security.standard_v4 import PdfStandardSecurityHandlerV4
from core_pdf.impl.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.primitives import PdfName
from core_pdf.impl.types import PdfDict


def gzip_compress(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=zlib.MAX_WBITS | 16)
    return compressor.compress(data) + compressor.flush()


def raw_deflate_compress(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(data) + compressor.flush()


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
    with pytest.raises(PdfParseError):
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
    with pytest.raises(PdfParseError):
        apply_flate(data, {})


def test_content_stream_detection_finds_operator_after_complex_operands() -> None:
    data = b"[(Tj) /Do << /Value q >>] TJ"

    assert looks_like_pdf_content_stream(data)


def test_content_stream_detection_supports_sliced_memoryview() -> None:
    content = b"(embedded Tj) not-content"
    data = memoryview(b"prefix" + content + b"suffix")[len(b"prefix") : -len(b"suffix")]

    assert not looks_like_pdf_content_stream(data)


def test_apply_ascii85_decodes_unterminated_pdf_stream() -> None:
    assert apply_ascii85(b"87cURD]j7BEbo80", {}) == b"Hello world!"


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
    with pytest.raises(PdfParseError):
        apply_ascii85(b"!!!!~bad", {})


@pytest.mark.parametrize(
    ("raw_data", "filters"),
    [
        (b"raw", PdfName.of("JPXDecode")),
        (b"726177>", [PdfName.of("ASCIIHexDecode"), PdfName.of("JPXDecode")]),
    ],
)
def test_jpx_receives_parent_colorspace_for_any_filter_position(
    monkeypatch: pytest.MonkeyPatch,
    raw_data: bytes,
    filters: object,
) -> None:
    embedded_color_flags: list[bool] = []

    def fake_decode_jpx(data: bytes, *, apply_embedded_color: bool) -> bytes:
        embedded_color_flags.append(apply_embedded_color)
        return data

    monkeypatch.setattr(decoders, "decode_jpx_impl", fake_decode_jpx)
    dictionary = {
        "Filter": filters,
        "ColorSpace": PdfName.of("DeviceRGB"),
    }
    stream = PdfStream(dictionary, raw_data, dictionary)

    assert stream.data == b"raw"
    assert embedded_color_flags == [False]


def test_crypt_filter_without_params_is_identity_after_security_stage() -> None:
    assert decode_stream_data(b"plain", {"Filter": PdfName.of("Crypt")}) == b"plain"


def make_v4_handler() -> PdfStandardSecurityHandlerV4:
    handler = object.__new__(PdfStandardSecurityHandlerV4)
    handler.encrypt_metadata = True
    handler.stmf = "Default"
    handler.strf = "Default"
    handler.cfm = {
        "Default": lambda _objid, _genno, data: b"default:" + data,
        "Special": lambda _objid, _genno, data: b"special:" + data,
    }
    return handler


def test_security_handler_uses_explicit_named_crypt_filter() -> None:
    handler = make_v4_handler()
    attrs = {
        "Filter": [PdfName.of("Crypt"), PdfName.of("FlateDecode")],
        "DecodeParms": [{"Name": PdfName.of("Special")}, None],
    }

    assert handler.decrypt(1, 0, b"ciphertext", cast(PdfDict, attrs)) == b"special:ciphertext"


def test_security_handler_defaults_explicit_crypt_to_identity() -> None:
    handler = make_v4_handler()

    assert handler.decrypt(1, 0, b"plain", {"Filter": PdfName.of("Crypt")}) == b"plain"


def test_security_handler_rejects_late_crypt_filter() -> None:
    handler = make_v4_handler()
    attrs = {"Filter": [PdfName.of("FlateDecode"), PdfName.of("Crypt")]}

    with pytest.raises(PdfParseError, match="first"):
        handler.decrypt(1, 0, b"ciphertext", cast(PdfDict, attrs))


def test_security_handler_rejects_unknown_named_crypt_filter() -> None:
    handler = make_v4_handler()
    attrs = {
        "Filter": PdfName.of("Crypt"),
        "DecodeParms": {"Name": PdfName.of("Unknown")},
    }

    with pytest.raises(PdfUnsupportedError, match="Undefined crypt filter"):
        handler.decrypt(1, 0, b"ciphertext", cast(PdfDict, attrs))

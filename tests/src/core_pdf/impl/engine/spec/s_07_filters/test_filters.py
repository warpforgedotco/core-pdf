# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import zlib

import pytest

from core_pdf.impl.engine.spec.s_07_filters.codecs import apply_ascii85
from core_pdf.impl.engine.spec.s_07_filters.flate import apply_flate
from core_pdf.impl.exceptions import PdfParseError


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


def test_apply_ascii85_decodes_unterminated_pdf_stream() -> None:
    assert apply_ascii85(b"87cURD]j7BEbo80", {}) == b"Hello world!"


def test_apply_ascii85_rejects_invalid_data() -> None:
    with pytest.raises(PdfParseError):
        apply_ascii85(b"!!!!~bad", {})

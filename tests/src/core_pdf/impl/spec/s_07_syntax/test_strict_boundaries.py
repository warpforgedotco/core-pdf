# SPDX-License-Identifier: AGPL-3.0-only
"""Strict PDF algorithms and explicitly composed reader recovery stay distinct."""

from __future__ import annotations

import pytest

from core_pdf.impl._impl.document.recovery.lexer import PdfLexer as ReaderLexer
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver as ReaderResolver
from core_pdf.impl._impl.document.recovery.xref import XRefScanner as ReaderXRefScanner
from core_pdf.impl._impl.document.recovery.xref import parse_xref_entry_line as read_xref_entry
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_filters.decode_spec import StreamDecodeSpec
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.resolver import ObjectResolver
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.xref import PdfXRefEntry, XRefScanner, parse_xref_entry_line
from core_pdf.impl.types import PdfReference, PdfString


@pytest.mark.parametrize("payload", [b"<4g86>", b"<< /Key >>", b"1 0 obj 42 endobx"])
def test_malformed_objects_require_the_reader_adapter(payload: bytes) -> None:
    with pytest.raises(PdfParseError):
        lexer = PdfLexer(payload)
        lexer.parse_indirect_object() if payload.startswith(b"1 0") else lexer.parse_object()
    lexer = ReaderLexer(payload)
    recovered = (
        lexer.parse_indirect_object() if payload.startswith(b"1 0") else lexer.parse_object()
    )
    assert recovered in (PdfString(b"H\x60"), {}, 42)


@pytest.mark.parametrize("length", [b"", b"/Length 2", b"/Length 99"])
def test_stream_delimiter_scanning_is_reader_recovery(length: bytes) -> None:
    source = b"<< " + length + b" >> stream\nabc\nendstream"
    with pytest.raises(PdfParseError):
        PdfLexer(source).parse_object()
    recovered = ReaderLexer(source).parse_object()
    assert isinstance(recovered, PdfStream)
    assert recovered.data == b"abc\n"


def test_missing_xref_marker_is_rejected_by_the_spec_reader() -> None:
    with pytest.raises(PdfParseError, match="xref table entry"):
        parse_xref_entry_line(b"0000000017 00000\n")
    assert read_xref_entry(b"0000000017 00000\n") == (17, 0, True)


def test_truncated_xref_rows_are_explicit_reader_recovery() -> None:
    stream = PdfStream({"Type": "XRef", "Size": 2, "W": [1, 1, 1]}, decoded_data=b"\x01\x10\x00")
    with pytest.raises(PdfParseError, match="length mismatch"):
        XRefScanner.parse_stream(stream)
    entries, _ = ReaderXRefScanner.parse_stream(stream)
    assert entries[0].offset == 16
    assert len(entries) == 1


def test_generation_zero_fallback_does_not_change_spec_resolution() -> None:
    data = b"1 0 obj 42 endobj"
    xref = {1 << 16: PdfXRefEntry(0)}
    strict = ObjectResolver(data, xref, {})
    reader = ReaderResolver(data, xref, {})
    try:
        assert strict.resolve(PdfReference(1, 1)) is None
        assert reader.resolve(PdfReference(1, 1)) == 42
    finally:
        strict.close()
        reader.close()


@pytest.mark.parametrize("resolver_class", [ObjectResolver, ReaderResolver])
def test_stream_resolution_and_replacement_preserve_an_explicit_decoder(
    resolver_class: type,
) -> None:
    calls: list[tuple[object, object]] = []

    def decoder(
        data: bytes | memoryview,
        dictionary: object | StreamDecodeSpec | None,
        *,
        parent_dictionary: object | None = None,
    ) -> bytes:
        calls.append((dictionary, parent_dictionary))
        return b"decoded"

    original = PdfStream(
        {"Filter": PdfReference(1)}, b"payload", {"Filter": PdfReference(1)}, decoder=decoder
    )
    resolver = resolver_class(b"1 0 obj /Example endobj", {1 << 16: PdfXRefEntry(0)}, {})
    try:
        resolved = resolver.resolve_stream(original).replace(raw_data=b"replacement")
        assert resolved.decoder is decoder
        assert resolved.data == b"decoded"
        assert calls == [(resolved.spec, resolved.dictionary)]
        assert original.dictionary == {"Filter": PdfReference(1)}
    finally:
        resolver.close()

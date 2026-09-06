# SPDX-License-Identifier: AGPL-3.0-only
"""Malformed generations must never alias another object's identity."""

from contextlib import closing

import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.document.recovery.lexer import PdfLexer as ReaderLexer
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl._impl.document.recovery.xref import XRefScanner as ReaderScanner
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.xref import key_for
from core_pdf.impl.types import PdfName, PdfReference, PdfString


@pytest.mark.parametrize("generation", [0, 65535])
def test_reference_generation_boundaries_keep_distinct_identities(generation: int) -> None:
    reference = PdfReference(2, generation)
    assert key_for(2, generation) != key_for(3)
    for lexer_type in (PdfLexer, ReaderLexer):
        with closing(lexer_type(f"2 {generation} R".encode())) as lexer:
            assert lexer.parse_object() == reference


@pytest.mark.parametrize("generation", [-1, 65536, 99999, 1 << 32])
def test_reference_and_packed_key_reject_invalid_generations(generation: int) -> None:
    with pytest.raises(ValueError):
        PdfReference(2, generation)
    with pytest.raises(ValueError):
        key_for(2, generation)


@pytest.mark.parametrize("lexer_type", [PdfLexer, ReaderLexer])
@pytest.mark.parametrize("generation", [65536, 99999])
def test_lexer_rejects_invalid_identity_before_parsing_payload(
    lexer_type: type[PdfLexer], generation: int
) -> None:
    with closing(lexer_type(f"2 {generation} R".encode())) as lexer:
        with pytest.raises(PdfParseError, match="reference"):
            lexer.parse_object()
    with closing(lexer_type(f"2 {generation} obj (forged) endobj".encode())) as lexer:
        with pytest.raises(PdfParseError, match="generation"):
            lexer.parse_indirect_object()


def test_recovery_scan_cannot_overwrite_a_valid_object_with_an_invalid_header() -> None:
    data = b"3 0 obj (legitimate) endobj\n2 65536 obj (forged) endobj\n"
    entries = ReaderScanner.brute_force_scan(data)
    assert set(entries) == {key_for(3)}
    with closing(ObjectResolver(data, entries, {})) as resolver:
        assert resolver.resolve(PdfReference(3)) == PdfString(b"legitimate")


@pytest.mark.parametrize("entry_type", [0, 1])
def test_recovered_stream_skips_invalid_generation_without_overwriting_valid_row(
    entry_type: int,
) -> None:
    # The second row used to overwrite object 3 because (2 << 16) | 65536 == 3 << 16.
    rows = bytes((1, 17, 0, 0, 0, entry_type, 29, 1, 0, 0))
    stream = PdfStream(
        {"Type": PdfName.of("XRef"), "Size": 4, "Index": [3, 1, 2, 1], "W": [1, 1, 3]},
        decoded_data=rows,
    )
    entries, _ = ReaderScanner.parse_stream(stream)
    assert set(entries) == {key_for(3)}
    assert entries[key_for(3)].offset == 17
    assert entries[key_for(3)].in_use


def test_invalid_classic_generation_still_allows_document_reconstruction() -> None:
    data = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(
        (
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Count 0 /Kids [] >>",
            b"<< /Title (Recovered) >>",
        ),
        1,
    ):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj ".encode() + body + b" endobj\n")
    xref = len(data)
    data.extend(b"xref\n1 3\n")
    for index, offset in enumerate(offsets):
        generation = 65536 if index == 1 else 0
        data.extend(f"{offset:010d} {generation:05d} n \n".encode())
    data.extend(
        b"trailer << /Size 4 /Root 1 0 R /Info 3 0 R >>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )
    with PdfDocument.open(bytes(data)) as document:
        assert document.xref_was_recovered
        assert document.get_metadata()["info"]["Title"] == "Recovered"
        assert len(document.pages) == 0

# SPDX-License-Identifier: AGPL-3.0-only
"""Core retains reader tolerance while selecting legacy cross-reference names."""

from __future__ import annotations

import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.document.recovery.xref import (
    XRefScanner,
    find_eof_marker,
    iter_indirect_object_headers,
)
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.standards import PdfVersion, SemanticContext


@pytest.mark.parametrize("version", [None, PdfVersion(1, 1), PdfVersion(1, 2), PdfVersion(9, 0)])
def test_reader_scans_nul_whitespace_with_legacy_or_unknown_context(
    version: PdfVersion | None,
) -> None:
    context = SemanticContext(version)
    assert XRefScanner.find_startxref(b"\x00startxref\n9\n%%EOF\x00", semantic_context=context) == 9
    assert find_eof_marker(b"\n%%EOF\x00", semantic_context=context) == 1
    data = b"1\x000\x00obj\x00"
    assert list(iter_indirect_object_headers(data, 0, len(data), semantic_context=context)) == [
        (0, 1, 0)
    ]
    table = b"xref\x000\x001\n0000000000 65535 f \ntrailer\n<<\x00/Size 1 /Old#20Key true >>"
    _, trailer, _, _ = XRefScanner.parse_table_section(table, 0, semantic_context=context)
    key = "Old#20Key" if version == PdfVersion(1, 1) else "Old Key"
    assert trailer[key] is True


def internal_document(version: str, *, with_xref: bool) -> bytes:
    data = f"%PDF-{version}\n".encode()
    offsets = []
    for number, body in enumerate(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Count 0 /Kids [] >>",
            b"<< /Author (Example) >>",
        ],
        1,
    ):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(data)
    if with_xref:
        data += b"xref\n0 4\n0000000000 65535 f \n"
        data += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets)
    data += b"trailer\n<< /Size 4 /Root 1 0 R /In#66o 3 0 R >>\n"
    if with_xref:
        data += f"startxref\n{xref}\n".encode()
    return data + b"%%EOF\n"


@pytest.mark.parametrize("with_xref", [True, False], ids=["xref", "recovered-trailer"])
@pytest.mark.parametrize(("version", "has_info"), [("1.1", False), ("1.2", True)])
def test_document_effective_context_reaches_later_trailer_recovery(
    version: str, has_info: bool, with_xref: bool
) -> None:
    with PdfDocument(internal_document(version, with_xref=with_xref)) as document:
        assert document.page_count() == 0
        assert ("Info" in document.trailer_dict) is has_info
        trailers = list(document.iter_literal_trailer_dictionaries())
        assert len(trailers) == 1
        assert ("Info" in trailers[0]) is has_info
        assert ("In#66o" in trailers[0]) is not has_info
        assert document.standards.effective_version == PdfVersion.parse(version)


def test_reader_previous_sections_keep_legacy_name_identity() -> None:
    data = b"%PDF-1.1\n"
    previous = len(data)
    data += b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Pr#65v 99999 >>\n"
    latest = len(data)
    data += (
        b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Prev "
        + str(previous).encode()
        + b" >>\n"
    )
    seen: set[int] = set()
    XRefScanner.load_section_chain(
        data, latest, seen, semantic_context=SemanticContext(PdfVersion(1, 1))
    )
    assert seen == {latest, previous}


@pytest.mark.parametrize("version", [PdfVersion(1, 1), PdfVersion(1, 5), PdfVersion(9, 0)])
def test_reader_hybrid_and_salvage_paths_keep_selected_name_rules(version: PdfVersion) -> None:
    # The legacy context is deliberately under-declared: the point is that
    # recovery cannot silently switch lexical versions in an auxiliary stream.
    data = b"%PDF-1.5\n"
    stream = len(data)
    data += (
        b"1 0 obj\n<< /Type /XRef /S#69ze 2 /W [1 1 1] /Index [1 1] /Length 3 >>\n"
        b"stream\n\x01\x09\x00\nendstream\nendobj\n"
    )
    table = len(data)
    data += (
        b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 2 /XRefStm "
        + str(stream).encode()
        + b" >>\n"
    )
    context = SemanticContext(version)
    salvaged = XRefScanner.parse_xref_stream_salvage(data, stream, semantic_context=context)
    assert salvaged is not None
    if version == PdfVersion(1, 1):
        assert salvaged.dictionary["S#69ze"] == 2
        assert "Size" not in salvaged.dictionary
        with pytest.raises(PdfParseError, match="invalid xref stream size"):
            XRefScanner.load_section_chain(data, table, set(), semantic_context=context)
    else:
        assert salvaged.dictionary["Size"] == 2
        entries, _ = XRefScanner.load_section_chain(data, table, set(), semantic_context=context)
        assert entries[1 << 16].offset == 9

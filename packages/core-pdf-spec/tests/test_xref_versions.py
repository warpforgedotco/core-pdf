# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from functools import partial

import pytest

from core_pdf_spec.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.xref import (
    XRefRevision,
    XRefScanner,
    find_eof_marker,
    iter_xref_revisions,
    parse_object_marker_prefix,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext


def context(version: str) -> SemanticContext:
    return SemanticContext(PdfVersion.parse(version))


def internal_revisions(data: bytes, start: int, context: SemanticContext) -> list[XRefRevision]:
    reader = partial(XRefScanner.parse_section_at, data, semantic_context=context)
    return list(iter_xref_revisions(start, reader))


def internal_table(trailer: bytes = b"", *, separator: bytes = b" ") -> bytes:
    return (
        b"xref\n0"
        + separator
        + b"1\n0000000000 65535 f \ntrailer\n<< /Size 1 "
        + trailer
        + b" >>\n"
    )


@pytest.mark.parametrize(("version", "key"), [("1.1", "Legacy#20Key"), ("1.2", "Legacy Key")])
def test_xref_trailer_names_follow_the_selected_version(version: str, key: str) -> None:
    data = internal_table(b"/Legacy#20Key true")
    _, trailer = XRefScanner.parse_table_section(data, 0, semantic_context=context(version))
    assert trailer[key] is True
    assert len(trailer) == 2


def test_explicit_xref_context_updates_a_supplied_trailer_lexer() -> None:
    data = internal_table(b"/Legacy#20Key true")
    lexer = PdfLexer(data)
    _, trailer = XRefScanner.parse_table_section(
        data, 0, lexer=lexer, semantic_context=context("1.1")
    )
    assert trailer["Legacy#20Key"] is True


def test_legacy_context_reaches_previous_classic_sections() -> None:
    data = b"%PDF-1.1\n"
    previous = len(data)
    data += internal_table(b"/Pr#65v 99999")
    latest = len(data)
    data += internal_table(f"/Prev {previous}".encode())
    revisions = internal_revisions(data, latest, context("1.1"))
    assert [revision.offset for revision in revisions] == [latest, previous]
    with pytest.raises(PdfParseError, match="invalid xref section"):
        internal_revisions(data, latest, context("1.2"))


def test_hybrid_stream_uses_context_for_escaped_type_names() -> None:
    data = b"%PDF-1.5\n"
    stream = len(data)
    data += (
        b"1 0 obj\n<< /Ty#70e /X#52ef /Size 2 /W [1 1 1] /Index [1 1] /Length 3 >>\n"
        b"stream\n\x01\x09\x00\nendstream\nendobj\n"
    )
    table = len(data)
    data += internal_table(f"/XRefStm {stream}".encode())
    revisions = internal_revisions(data, table, context("1.5"))
    assert revisions[0].entries[1 << 16].offset == 9
    with pytest.raises(PdfParseError, match="invalid xref stream type"):
        internal_revisions(data, table, context("1.1"))


def test_nul_whitespace_in_xref_subsections_follows_selected_rules() -> None:
    data = internal_table(separator=b"\x00")
    with pytest.raises(PdfParseError, match="invalid xref table subsection"):
        XRefScanner.parse_table_section(data, 0, semantic_context=context("1.2"))
    _, trailer = XRefScanner.parse_table_section(data, 0, semantic_context=context("1.3"))
    assert trailer["Size"] == 1


def test_startxref_eof_and_object_headers_share_whitespace_context() -> None:
    old = context("1.2")
    new = context("1.3")
    assert XRefScanner.find_startxref(b"\x00startxref\n9\n%%EOF\n", semantic_context=old) is None
    assert XRefScanner.find_startxref(b"\x00startxref\n9\n%%EOF\n", semantic_context=new) == 9
    assert find_eof_marker(b"\n%%EOF\x00", semantic_context=old) == -1
    assert find_eof_marker(b"\n%%EOF\x00", semantic_context=new) == 1
    data = b"1\x000\x00obj\x00"
    assert parse_object_marker_prefix(data, 4, semantic_context=old) is None
    assert parse_object_marker_prefix(data, 4, semantic_context=new) == (0, 1, 0)
    ignored = b"\x00% comment\n\x00xref"
    assert XRefScanner.skip_ignored(ignored, 0, semantic_context=old) == 0
    assert XRefScanner.skip_ignored(ignored, 0, semantic_context=new) == ignored.index(b"xref")
    assert XRefScanner.skip_ws(b"\x00xref", 0, semantic_context=old) == 0
    assert XRefScanner.skip_ws(b"\x00xref", 0, semantic_context=new) == 1
    assert XRefScanner.find_startxref(b"\nstartxref\n9\n%%EOF\n", semantic_context=old) == 9


@pytest.mark.parametrize("version", [None, PdfVersion(9, 0)])
def test_unknown_context_does_not_guess_xref_lexical_rules(version: PdfVersion | None) -> None:
    with pytest.raises(PdfUnsupportedError, match="recognized PDF version"):
        internal_revisions(internal_table(), 0, SemanticContext(version))

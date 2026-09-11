# SPDX-License-Identifier: AGPL-3.0-only
"""Version-selected lexical rules reach all cross-reference parsing paths."""

from __future__ import annotations

import pytest

from core_pdf_spec.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import (
    XRefScanner,
    XRefTable,
    find_eof_marker,
    parse_object_marker_prefix,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext


def internal_context(version: str) -> SemanticContext:
    return SemanticContext(PdfVersion.parse(version))


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
    # Adobe PDF Reference 1.2, 4.5: hexadecimal name escapes start with PDF 1.2.
    data = internal_table(b"/Legacy#20Key true")
    _, trailer, _, _ = XRefScanner.parse_table_section(
        data, 0, semantic_context=internal_context(version)
    )
    assert trailer[key] is True
    assert len(trailer) == 2


def test_explicit_xref_context_updates_a_supplied_trailer_lexer() -> None:
    data = internal_table(b"/Legacy#20Key true")
    lexer = PdfLexer(data)
    _, trailer, _, _ = XRefScanner.parse_table_section(
        data, 0, lexer=lexer, semantic_context=internal_context("1.1")
    )
    assert trailer["Legacy#20Key"] is True


def test_legacy_context_reaches_previous_classic_sections() -> None:
    data = b"%PDF-1.1\n"
    previous = len(data)
    data += internal_table(b"/Pr#65v 99999")
    latest = len(data)
    data += internal_table(f"/Prev {previous}".encode())
    seen: set[int] = set()
    XRefScanner.load_section_chain(data, latest, seen, semantic_context=internal_context("1.1"))
    assert seen == {latest, previous}
    # The same bytes name /Prev in 1.2; the out-of-bounds third section must fail.
    with pytest.raises(PdfParseError, match="invalid xref section"):
        XRefScanner.load_section_chain(
            data, latest, set(), semantic_context=internal_context("1.2")
        )


def test_hybrid_stream_uses_context_for_escaped_type_names() -> None:
    data = b"%PDF-1.5\n"
    stream = len(data)
    data += (
        b"1 0 obj\n<< /Ty#70e /X#52ef /Size 2 /W [1 1 1] /Index [1 1] /Length 3 >>\n"
        b"stream\n\x01\x09\x00\nendstream\nendobj\n"
    )
    table = len(data)
    data += internal_table(f"/XRefStm {stream}".encode())
    entries, _ = XRefScanner.load_section_chain(
        data, table, set(), semantic_context=internal_context("1.5")
    )
    assert entries[1 << 16].offset == 9
    # An intentionally under-declared context cannot silently use modern name
    # decoding inside the hybrid branch, even though the outer table is readable.
    with pytest.raises(PdfParseError, match="invalid xref stream type"):
        XRefScanner.load_section_chain(data, table, set(), semantic_context=internal_context("1.1"))


def test_nul_whitespace_in_xref_subsections_follows_selected_rules() -> None:
    # PDF 1.2, 4.4 omits NUL; PDF 1.3, Table 3.1 includes it.
    data = internal_table(separator=b"\x00")
    with pytest.raises(PdfParseError, match="invalid xref table subsection"):
        XRefScanner.parse_table_section(data, 0, semantic_context=internal_context("1.2"))
    _, trailer, _, _ = XRefScanner.parse_table_section(
        data, 0, semantic_context=internal_context("1.3")
    )
    assert trailer["Size"] == 1


def test_startxref_eof_and_object_headers_share_whitespace_context() -> None:
    old = internal_context("1.2")
    new = internal_context("1.3")
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
        XRefScanner.load_section_chain(
            internal_table(), 0, set(), semantic_context=SemanticContext(version)
        )


def test_no_context_retains_older_subclass_method_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyScanner(XRefScanner):
        pass

    original_section = LegacyScanner.parse_section_at
    original_table = LegacyScanner.parse_table_section
    calls: list[str] = []

    def legacy_parse_section(
        cls: type[XRefScanner], data: bytes, start: int
    ) -> tuple[XRefTable, PdfDict, int | None, int | None]:
        calls.append("section")
        return original_section(data, start)

    def legacy_parse_table(
        cls: type[XRefScanner], data: bytes, start: int
    ) -> tuple[XRefTable, PdfDict, int | None, int | None]:
        calls.append("table")
        return original_table(data, start)

    def legacy_skip_ws(data: bytes, start: int) -> int:
        calls.append("whitespace")
        return XRefScanner.skip_ws(data, start)

    monkeypatch.setattr(LegacyScanner, "parse_section_at", classmethod(legacy_parse_section))
    monkeypatch.setattr(LegacyScanner, "parse_table_section", classmethod(legacy_parse_table))
    monkeypatch.setattr(LegacyScanner, "skip_ws", staticmethod(legacy_skip_ws))
    _, trailer = LegacyScanner.load_section_chain(internal_table(), 0, set())
    assert trailer["Size"] == 1
    assert calls == ["section", "table", "whitespace", "whitespace", "whitespace"]

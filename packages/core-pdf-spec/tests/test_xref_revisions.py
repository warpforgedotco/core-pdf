"""Strict physical sections and shared revision traversal have distinct contracts."""

from functools import partial
from typing import cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import (
    ParsedXRefSection,
    PdfXRefEntry,
    XRefScanner,
    iter_xref_revisions,
    key_for,
    merge_xref_sections,
    overlay_xref_entries,
    parse_object_marker_prefix,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext


def internal_table(subsections: list[tuple[int, int, int]]) -> bytes:
    return (
        b"xref\n"
        + b"".join(
            f"{start} {count}\n".encode() + f"0000000009 {generation:05} n \n".encode() * count
            for start, count, generation in subsections
        )
        + b"trailer\n<< /Size 10 >>\n"
    )


@pytest.mark.parametrize(
    "subsections", [[(1, 1, 0), (1, 1, 0)], [(1, 1, 0), (1, 1, 1)], [(1, 3, 0), (3, 2, 0)]]
)
def test_classic_subsections_cannot_repeat_an_object_number(
    subsections: list[tuple[int, int, int]],
) -> None:
    # ISO 32000-1/2, 7.5.4 prohibits overlap, including across generations.
    with pytest.raises(PdfParseError, match="overlapping"):
        XRefScanner.parse_table_section(internal_table(subsections), 0)


def test_classic_subsections_may_be_in_descending_order() -> None:
    entries, trailer = XRefScanner.parse_table_section(internal_table([(5, 2, 0), (1, 2, 0)]), 0)
    assert {key >> 16 for key in entries} == {1, 2, 5, 6}
    assert trailer["Size"] == 10


@pytest.mark.parametrize("previous", [b"null", b"-1", b"99999", b"true", b"(bad)", b"[9]"])
def test_hybrid_stream_previous_pointer_has_no_meaning(previous: bytes) -> None:
    # ISO 32000-1/2, Table 17: only the primary trailer's /Prev is followed.
    data = b"%PDF-1.5\n"
    stream = len(data)
    data += (
        b"1 0 obj\n<< /Type /XRef /Size 2 /W [1 1 1] /Index [1 1] /Length 3 /Prev "
        + previous
        + b" >>\nstream\n\x01\x09\x00\nendstream\nendobj\n"
    )
    table = len(data)
    data += b"xref\ntrailer\n<< /Size 2 /XRefStm " + str(stream).encode() + b" >>\n"
    revisions = list(iter_xref_revisions(table, partial(XRefScanner.parse_section_at, data)))
    assert len(revisions) == 1
    assert revisions[0].offset == table
    assert revisions[0].entries[key_for(1)].offset == 9


def test_primary_table_then_supplemental_then_previous_precedence() -> None:
    old = {key_for(number): PdfXRefEntry(number) for number in (1, 2, 3)}
    supplemental = {key_for(number): PdfXRefEntry(100 + number) for number in (1, 2, 3)}
    current = {key_for(1, 1): PdfXRefEntry(201, 1), key_for(2, 1): PdfXRefEntry(0, 1, False)}
    sections = {
        10: ParsedXRefSection(10, "table", old, {}),
        20: ParsedXRefSection(20, "stream", supplemental, {"Prev": 20}),
        30: ParsedXRefSection(30, "table", current, {"Prev": 10, "XRefStm": 20}),
    }
    calls: list[tuple[int, bool]] = []

    def read(offset: int, *, stream_only: bool = False) -> ParsedXRefSection:
        calls.append((offset, stream_only))
        return sections[offset]

    revisions = list(iter_xref_revisions(30, read))
    merged = merge_xref_sections(revision.entries for revision in revisions)
    assert calls == [(30, False), (20, True), (10, False)]
    assert set(merged) == {key_for(1, 1), key_for(2, 1), key_for(3)}
    assert merged[key_for(1, 1)] is current[key_for(1, 1)]
    assert not merged[key_for(2, 1)].in_use
    assert merged[key_for(3)] is supplemental[key_for(3)]
    assert revisions[0].trailer is sections[30].trailer
    assert len(supplemental) == len(old) == 3


def test_supplemental_stream_can_be_reused_by_previous_revision() -> None:
    sections = {
        10: ParsedXRefSection(10, "table", {}, {"XRefStm": 20}),
        20: ParsedXRefSection(20, "stream", {}, {}),
        30: ParsedXRefSection(30, "table", {}, {"Prev": 10, "XRefStm": 20}),
    }

    def read(offset: int, *, stream_only: bool = False) -> ParsedXRefSection:
        return sections[offset]

    assert [revision.offset for revision in iter_xref_revisions(30, read)] == [30, 10]


@pytest.mark.parametrize("pointer", ["Prev", "XRefStm"])
@pytest.mark.parametrize("value", [-1, True, "9", 1.0, [9]])
def test_primary_chain_pointers_are_validated(pointer: str, value: object) -> None:
    def read(offset: int, *, stream_only: bool = False) -> ParsedXRefSection:
        return ParsedXRefSection(offset, "table", {}, cast(PdfDict, {pointer: value}))

    with pytest.raises(PdfParseError, match=pointer):
        list(iter_xref_revisions(0, read))


@pytest.mark.parametrize("actual_offset", [10, 20])
def test_primary_chain_detects_original_and_recovered_offset_loops(actual_offset: int) -> None:
    def read(offset: int, *, stream_only: bool = False) -> ParsedXRefSection:
        return ParsedXRefSection(actual_offset, "table", {}, {"Prev": actual_offset})

    with pytest.raises(PdfParseError, match="loop"):
        list(iter_xref_revisions(10, read))


def test_iterator_rejects_a_callback_returning_a_table_for_a_supplemental_stream() -> None:
    def read(offset: int, *, stream_only: bool = False) -> ParsedXRefSection:
        return ParsedXRefSection(offset, "table", {}, {"XRefStm": 20})

    with pytest.raises(PdfParseError, match="expected xref stream"):
        list(iter_xref_revisions(10, read))


def test_section_parser_enforces_stream_only_without_changing_table_parsing() -> None:
    data = internal_table([(1, 1, 0)])
    section = XRefScanner.parse_section_at(data, 0)
    assert section.offset == 0
    assert section.kind == "table"
    assert section.entries[key_for(1)].offset == 9
    with pytest.raises(PdfParseError, match="expected xref stream"):
        XRefScanner.parse_section_at(data, 0, stream_only=True)


def test_overlay_replaces_generations_and_preserves_unmentioned_objects() -> None:
    old = {key_for(1): PdfXRefEntry(10), key_for(2): PdfXRefEntry(20)}
    freed = {key_for(1, 1): PdfXRefEntry(0, 1, False)}
    replacement = {key_for(1, 1): PdfXRefEntry(30, 1)}
    overlay_xref_entries(old, freed)
    assert key_for(1) not in old
    assert not old[key_for(1, 1)].in_use
    overlay_xref_entries(old, replacement)
    assert old[key_for(1, 1)] is replacement[key_for(1, 1)]
    assert old[key_for(2)].offset == 20


@pytest.mark.parametrize("data", [b"0 0 obj ", b"01 0 obj ", b"1 00 obj ", b"1 0obj "])
def test_strict_object_markers_reject_noncanonical_or_joined_identifiers(data: bytes) -> None:
    assert parse_object_marker_prefix(data, data.index(b"obj")) is None


def test_object_marker_accepts_delimiters_and_version_selected_identifier_syntax() -> None:
    assert parse_object_marker_prefix(b"1 0 obj<<", 4) == (0, 1, 0)
    assert parse_object_marker_prefix(b"1 0 xxx ", 4) is None
    assert parse_object_marker_prefix(
        b"+01 00 obj ", 7, semantic_context=SemanticContext(PdfVersion(1, 7))
    ) == (0, 1, 0)

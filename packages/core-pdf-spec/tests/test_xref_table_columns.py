"""A canonical xref subsection read as columns reads as the per-row parse reads it."""

import random

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax import xref
from core_pdf_spec.s_07_syntax.xref import PdfXRefEntry, XRefScanner


def row(offset: int, generation: int, marker: bytes, eol: bytes = b"\r\n") -> bytes:
    return b"%010d %05d " % (offset, generation) + marker + eol


TABLES = [
    pytest.param(row(0, 65535, b"f") + row(17, 0, b"n") + row(81, 0, b"n"), 0, 3, id="crlf"),
    pytest.param(row(0, 65535, b"f", b" \n") + row(17, 3, b"n", b" \r"), 0, 2, id="space-eol"),
    pytest.param(row(9999999999, 99999, b"n"), 7, 1, id="big-generation"),
    pytest.param(row(9999999999, 65535, b"n") + row(0, 0, b"f"), 12, 2, id="widest-fields"),
    pytest.param(row(5, 0, b"n") + b"0000000009 00000 x\r\n", 0, 2, id="bad-marker"),
    pytest.param(row(5, 0, b"n"), 0, 2, id="short"),
    pytest.param(row(5, 0, b"n") * 3, (1 << 47) - 2, 3, id="huge-object-numbers"),
    pytest.param(row(5, 0, b"n") * 3, 1 << 60, 3, id="object-numbers-past-int64"),
]


def read(data: bytes, start: int, count: int) -> object:
    try:
        entries, pos, maximum = XRefScanner.read_subsection(data, 0, start, count)
    except PdfParseError as error:
        return f"raised {error}"
    return (
        [
            (key, tuple(entry), type(entry.offset), type(entry.in_use))
            for key, entry in entries.items()
        ],
        pos,
        maximum,
    )


@pytest.mark.parametrize(("data", "start", "count"), TABLES)
def test_columns_read_as_the_per_row_parse(
    data: bytes, start: int, count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    columns = read(data, start, count)
    monkeypatch.setattr(xref, "canonical_table_entries", lambda *_: None)
    assert columns == read(data, start, count)


def test_entries_are_the_named_tuple() -> None:
    table = xref.canonical_table_entries(row(17, 2, b"n") + row(0, 0, b"f"), 0, 5, 2)
    assert table == {
        (5 << 16) | 2: PdfXRefEntry(17, 2, True),
        6 << 16: PdfXRefEntry(0, 0, False),
    }
    assert all(type(entry) is PdfXRefEntry for entry in table.values())


def merged_row_by_row(sections: list[xref.XRefTable]) -> xref.XRefTable:
    merged: xref.XRefTable = {}
    claimed: set[int] = set()
    for section in sections:
        for key, entry in section.items():
            if key >> 16 not in claimed:
                merged[key] = entry  # noqa: PERF403 -- the reference is the loop itself
        claimed.update(key >> 16 for key in section)
    return merged


@pytest.mark.parametrize("seed", range(40))
def test_merged_sections_keep_what_the_row_loop_kept(seed: int) -> None:
    rng = random.Random(seed)
    sections = [
        {
            (rng.randrange(30) << 16) | rng.randrange(3): PdfXRefEntry(rng.randrange(1000))
            for _ in range(rng.randrange(12))
        }
        for _ in range(rng.randrange(6))
    ]
    merged = xref.merge_xref_sections(iter(sections))
    expected = merged_row_by_row(sections)
    assert list(merged.items()) == list(expected.items())
    assert all(merged[key] is expected[key] for key in merged)

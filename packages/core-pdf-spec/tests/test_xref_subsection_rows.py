"""An xref subsection read at once reads as its rows do one by one (ISO 32000-2 7.5.4)."""

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.xref import XRefScanner, key_for, parse_xref_entry_at


def row(offset: int, generation: int, marker: bytes, eol: bytes = b"\r\n") -> bytes:
    return b"%010d %05d " % (offset, generation) + marker + eol


def row_by_row(data: bytes, start: int, count: int) -> object:
    rows = []
    pos = 0
    try:
        for i in range(count):
            offset, generation, in_use, pos = parse_xref_entry_at(data, pos)
            rows.append((key_for(start + i, generation), offset, generation, in_use))
    except (PdfParseError, ValueError) as error:
        return f"raised {type(error).__name__}: {error}"
    return rows, pos, start + count - 1


def at_once(data: bytes, start: int, count: int) -> object:
    try:
        entries, pos, maximum = XRefScanner.read_subsection(data, 0, start, count)
    except (PdfParseError, ValueError) as error:
        return f"raised {type(error).__name__}: {error}"
    rows = [(key, e.offset, e.generation, e.in_use) for key, e in entries.items()]
    return rows, pos, maximum


@pytest.mark.parametrize(
    ("data", "start", "count"),
    [
        pytest.param(
            row(0, 65535, b"f") + row(17, 0, b"n", b" \n") + row(81, 2, b"n", b" \r"),
            0,
            3,
            id="canonical",
        ),
        pytest.param(row(0, 0, b"n") * 2 + b"trailer", 5, 2, id="trailer-after"),
        pytest.param(row(0, 0, b"n") + row(9, 70000, b"n"), 0, 2, id="big-generation"),
        pytest.param(row(0, 0, b"n") + b"000000009 000000 n\r\n", 0, 2, id="misaligned"),
        pytest.param(row(0, 0, b"n") + row(9, 0, b"n", b"\n\n"), 0, 2, id="bad-eol"),
        pytest.param(row(0, 0, b"n"), 0, 2, id="short"),
        pytest.param(row(0, 0, b"n"), -1, 1, id="negative-start"),
        pytest.param(b"", 3, 0, id="empty"),
    ],
)
def test_a_subsection_reads_as_its_rows(data: bytes, start: int, count: int) -> None:
    assert at_once(data, start, count) == row_by_row(data, start, count)

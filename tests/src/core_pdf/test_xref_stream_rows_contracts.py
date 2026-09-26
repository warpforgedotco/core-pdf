"""Xref stream rows decoded a column at a time read as decode_xref_row reads them."""

import random

import pytest

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.recovery_xref import decode_xref_stream_rows
from core_pdf_spec.s_07_syntax.xref import XRefTable, decode_xref_row


def row_by_row(
    data: bytes, w: list[int], available_index: list[int], effective_size: int
) -> XRefTable:
    entries: XRefTable = {}
    pos = 0
    for i in range(0, len(available_index), 2):
        start, count = available_index[i : i + 2]
        for object_number in range(start, start + count):
            row_pos = pos
            pos += sum(w)
            if object_number >= effective_size:
                continue
            try:
                key, entry, _ = decode_xref_row(data, row_pos, w, object_number)
            except PdfParseError:
                continue
            entries[key] = entry
    return entries


def flat(entries: XRefTable) -> list[tuple[object, ...]]:
    return [
        (key, tuple(e), tuple(type(field) for field in e), type(e)) for key, e in entries.items()
    ]


@pytest.mark.parametrize("seed", range(60))
def test_columns_read_as_rows(seed: int) -> None:
    rng = random.Random(seed)
    w = [rng.choice([0, 1, 2]), rng.randint(1, 8), rng.choice([0, 1, 2, 3, 8])]
    index: list[int] = []
    start = 0
    for _ in range(rng.randint(1, 4)):
        # Subsections may overlap, so a later row can replace an earlier key.
        start = max(0, start + rng.randint(-8, 5))
        count = rng.randint(0, 20)
        index += [start, count]
        start += count
    rows = sum(index[1::2])
    data = bytes(
        rng.choice([0, 1, 2, 3, 255]) if rng.random() < 0.3 else rng.randrange(256)
        for _ in range(rows * sum(w))
    )
    effective_size = rng.choice([rng.randint(0, start + 2), 1 << 70])
    assert flat(decode_xref_stream_rows(data, w, index, effective_size)) == flat(
        row_by_row(data, w, index, effective_size)
    )

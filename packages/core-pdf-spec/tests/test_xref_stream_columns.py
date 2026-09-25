"""Xref stream fields read a column at a time decode as row by row (ISO 32000-2 7.5.8.3)."""

import random

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.xref import decode_xref_row_at, decode_xref_rows


def row_by_row(data: bytes, w: list[int], index: list[int]) -> object:
    entries = {}
    pos = 0
    try:
        for i in range(0, len(index), 2):
            for object_number in range(index[i], index[i] + index[i + 1]):
                key, entry, pos = decode_xref_row_at(data, pos, w, object_number, sum(w))
                entries[key] = entry
    except PdfParseError as error:
        return f"raised {error}"
    return [
        (key, e.offset, e.generation, e.in_use, e.object_stream, e.index_in_stream)
        for key, e in entries.items()
    ]


def columns(data: bytes, w: list[int], index: list[int], size: int) -> object:
    try:
        entries = decode_xref_rows(data, w, index, size)
    except PdfParseError as error:
        return f"raised {error}"
    return [
        (key, e.offset, e.generation, e.in_use, e.object_stream, e.index_in_stream)
        for key, e in entries.items()
    ]


@pytest.mark.parametrize("seed", range(60))
def test_columns_decode_as_rows(seed: int) -> None:
    rng = random.Random(seed)
    w = [rng.choice([0, 1, 2]), rng.randint(1, 8), rng.choice([0, 1, 2, 3])]
    index: list[int] = []
    start = 0
    for _ in range(rng.randint(1, 3)):
        count = rng.randint(1, 20)
        index += [start, count]
        start += count + rng.randint(0, 4)
    rows = sum(index[1::2])
    data = bytes(
        rng.choice([0, 1, 2, 3]) if rng.random() < 0.4 else rng.randrange(256)
        for _ in range(rows * sum(w))
    )
    assert columns(data, w, index, start + 1) == row_by_row(data, w, index)

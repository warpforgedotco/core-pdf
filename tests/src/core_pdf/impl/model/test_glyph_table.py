# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import sys
import threading
from typing import Any

import pytest

from core_pdf.impl.exceptions import PdfContractError
from core_pdf.impl.model.glyph_table import GlyphTable
from core_pdf.impl.model.glyphs import GlyphObservation, GlyphSegment


def internal_observation(text: str, seqno: int = 1) -> GlyphObservation:
    return GlyphObservation(text, (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), seqno)


def internal_row_table(count: int) -> GlyphTable:
    """Build a table of capture-style rows, which materialize lazily."""
    segment = GlyphSegment(
        1,
        "Example",
        10.0,
        10.0,
        10.0,
        None,
        0,
        object(),
        0,
        None,
        None,
        None,
        1.0,
        0,
        0,
        None,
        None,
        None,
        0,
    )
    box = (0.0, 0.0, 1.0, 1.0)
    return GlyphTable(
        tuple(
            (
                segment,
                "A",
                box,
                box,
                None,
                b"A",
                65,
                1,
                1,
                True,
                1.0,
                "to_unicode",
                (),
                0,
                0,
                None,
                None,
                (),
                index,
            )
            for index in range(count)
        )
    )


def test_table_rejects_products_that_are_not_observations() -> None:
    # Deliberately invalid input, so the rows go in untyped.
    rows: Any = ("not a glyph",)
    with pytest.raises(PdfContractError, match="invalid glyph product"):
        GlyphTable.from_rows(rows)


def test_table_indexes_from_both_ends_and_refuses_slices() -> None:
    first = internal_observation("A")
    last = internal_observation("B")
    table = GlyphTable.from_rows((first, last))

    assert len(table) == 2
    assert bool(table)
    assert table[0] is first
    assert table[-1] is last
    span: Any = slice(0, 1)
    with pytest.raises(TypeError, match="slicing"):
        table[span]


def test_rows_materialize_once_per_table() -> None:
    table = internal_row_table(4)

    assert tuple(table)[0] is table[0]
    assert table[2] is table[2]


def test_rows_keep_one_identity_when_workers_materialize_together() -> None:
    # Page programs are cached per page and pages run on pool workers, so two
    # threads can reach an unmaterialized table at once. Every caller must
    # still observe the same row objects: the compat facades key maps by id()
    # over rows held from a single pass. A short switch interval makes the
    # unguarded interleaving reproducible rather than incidental.
    table = internal_row_table(4000)
    start = threading.Barrier(4)
    seen: list[tuple[GlyphObservation, ...]] = []
    collected = threading.Lock()

    def collect() -> None:
        start.wait()
        rows = tuple(table)
        with collected:
            seen.append(rows)

    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        workers = [threading.Thread(target=collect) for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
    finally:
        sys.setswitchinterval(previous_interval)

    assert len(seen) == 4
    for rows in seen:
        assert all(row is expected for row, expected in zip(rows, seen[0], strict=True))

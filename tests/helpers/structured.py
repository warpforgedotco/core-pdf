# SPDX-License-Identifier: AGPL-3.0-only
"""Builders for the structured output model: cells, stream tables, native blocks."""

from __future__ import annotations

from typing import Any

from core_pdf.impl.model.geometry import bbox_union
from core_pdf.impl.output import Block, BlockKind, Table, TableCell, TextLine
from core_pdf.impl.types import Rectangle


def cell(row: int, column: int, text: str, bbox: Rectangle | None = None) -> TableCell:
    return TableCell(row=row, column=column, text=text, bbox=bbox)


def stream_table(
    rows: tuple[tuple[TableCell, ...], ...],
    *,
    source: str = "stream",
    confidence: float | None = None,
    **metadata: Any,
) -> Table:
    """A table whose bbox is the union of its cells' boxes."""
    return Table(
        order=0,
        rows=rows,
        bbox=bbox_union(item.bbox for row in rows for item in row if item.bbox is not None),
        confidence=confidence,
        metadata={"source": source, **metadata},
    )


def native_block(text: str, bbox: Rectangle, *, kind: BlockKind = BlockKind.PARAGRAPH) -> Block:
    """One native block with a line per newline-separated piece of ``text``."""
    return Block(
        order=0,
        kind=kind,
        lines=tuple(TextLine(text=line, bbox=bbox, source="native") for line in text.split("\n")),
        bbox=bbox,
        provenance=("native",),
    )


__all__ = ("cell", "native_block", "stream_table")

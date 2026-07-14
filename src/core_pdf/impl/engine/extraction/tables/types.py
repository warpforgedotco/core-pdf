# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TypeAlias, TypedDict

Rect: TypeAlias = tuple[float, float, float, float]
TableRows: TypeAlias = list[list[str]]
TableSet: TypeAlias = list[TableRows]
TableBBoxes: TypeAlias = list[Rect | None]


class TableCellSpanRecord(TypedDict):
    text: str
    row_span: int
    col_span: int
    font_size: float


TableSpanRows: TypeAlias = list[list[TableCellSpanRecord]]
TableHeaderResult: TypeAlias = tuple[TableRows, TableSet]
TableSpanResult: TypeAlias = tuple[TableSet, TableSpanRows]
ExtractTablesResult: TypeAlias = TableSet | TableHeaderResult | TableSpanResult
TableSetWithBBoxes: TypeAlias = tuple[TableSet, TableBBoxes]
TableHeaderResultWithBBoxes: TypeAlias = tuple[TableHeaderResult, TableBBoxes]
TableSpanResultWithBBoxes: TypeAlias = tuple[TableSpanResult, TableBBoxes]
TableHeuristicResult: TypeAlias = (
    TableSet
    | TableHeaderResult
    | TableSpanResult
    | TableSetWithBBoxes
    | TableHeaderResultWithBBoxes
    | TableSpanResultWithBBoxes
)
TableCacheKey: TypeAlias = tuple[object, ...]


class TableExtractionResult(TypedDict):
    tables: TableSet
    spans: TableSpanRows
    bboxes: TableBBoxes
    header: TableRows


__all__ = (
    "Rect",
    "ExtractTablesResult",
    "TableBBoxes",
    "TableCacheKey",
    "TableCellSpanRecord",
    "TableExtractionResult",
    "TableHeaderResult",
    "TableHeaderResultWithBBoxes",
    "TableHeuristicResult",
    "TableRows",
    "TableSet",
    "TableSetWithBBoxes",
    "TableSpanResult",
    "TableSpanResultWithBBoxes",
    "TableSpanRows",
)

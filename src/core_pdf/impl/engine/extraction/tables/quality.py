# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Sequence


def bboxes_match(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    tolerance: float = 1e-3,
) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def bbox_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    if left_area <= 0.0 or right_area <= 0.0:
        return 0.0
    overlap_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    overlap_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    overlap_area = overlap_width * overlap_height
    return overlap_area / min(left_area, right_area)


def table_populated_cells(rows: Sequence[Sequence[str]]) -> int:
    return sum(1 for row in rows for cell in row if str(cell).strip())


def table_total_cells(rows: Sequence[Sequence[str]]) -> int:
    return sum(len(row) for row in rows)


def table_column_count(rows: Sequence[Sequence[str]]) -> int:
    return max((len(row) for row in rows), default=0)


def table_text_length(rows: Sequence[Sequence[str]]) -> int:
    return sum(len(str(cell).strip()) for row in rows for cell in row if str(cell).strip())


def table_quality_score(
    rows: list[list[str]],
    *,
    text_length: int | None = None,
) -> tuple[int, float, int, int, int]:
    row_count = len(rows)
    column_count = table_column_count(rows)
    populated_cells = table_populated_cells(rows)
    total_cells = table_total_cells(rows)
    density = populated_cells / total_cells if total_cells else 0.0
    if text_length is None:
        text_length = table_text_length(rows)
    substantial = int(row_count >= 2 and column_count >= 2 and populated_cells >= 4)
    quality = text_length * density
    return (substantial, quality, text_length, populated_cells, -total_cells)


__all__ = (
    "bbox_overlap_ratio",
    "bboxes_match",
    "table_column_count",
    "table_populated_cells",
    "table_quality_score",
    "table_text_length",
    "table_total_cells",
)

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf.impl._impl.model.geometry import overlap_ratio_of
from core_pdf.impl._impl.model.text import complete_text_covered, content_tokens
from core_pdf.impl._impl.output.model import Table


def internal_is_synthetic_chart(table: Table) -> bool:
    return table.metadata.get("source") == "chart-ocr" and bool(table.metadata.get("synthetic"))


def internal_remove_duplicate_tables(
    tables: tuple[Table, ...],
) -> tuple[Table, ...]:
    text_tokens = tuple(
        content_tokens(" ".join(cell.text for row in table.rows for cell in row))
        for table in tables
    )
    synthetic = tuple(internal_is_synthetic_chart(table) for table in tables)
    ranked = sorted(
        range(len(tables)),
        key=lambda index: (
            -len(text_tokens[index]),
            synthetic[index],
            tables[index].order,
            index,
        ),
    )
    retained: list[int] = []
    for index in ranked:
        table = tables[index]
        reference = next(
            (
                other_index
                for other_index in retained
                if table.bbox is not None
                and (
                    synthetic[index]
                    or (table.metadata.get("source") == "stream" and synthetic[other_index])
                )
                and (other_box := tables[other_index].bbox) is not None
                and overlap_ratio_of(table.bbox, other_box) >= 0.90
                and complete_text_covered(text_tokens[index], text_tokens[other_index])
            ),
            None,
        )
        if reference is None:
            retained.append(index)
    retained.sort()
    return tuple(tables[index] for index in retained)

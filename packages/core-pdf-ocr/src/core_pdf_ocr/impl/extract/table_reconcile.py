# SPDX-License-Identifier: AGPL-3.0-only
"""Reconcile recognized chart tables through shared spatial text projection."""

from __future__ import annotations

import re

from core_pdf.impl._impl.extract import table_reconcile as native_reconcile
from core_pdf.impl._impl.model.geometry import overlap_ratio_of
from core_pdf.impl._impl.output.model import Block, Table

internal_CONTENT_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def internal_is_synthetic_chart(table: Table) -> bool:
    return table.metadata.get("source") == "chart-ocr" and bool(table.metadata.get("synthetic"))


def internal_complete_text_covered(candidate: tuple[str, ...], reference: tuple[str, ...]) -> bool:
    if not candidate:
        return False
    size = len(candidate)
    for start in range(len(reference) - size + 1):
        if reference[start : start + size] != candidate:
            continue
        # Do not strip a surrounding operator from an expression: a bare 5
        # is not equivalent to > 5, even though its only word token matches.
        if start and not reference[start - 1].isalnum():
            continue
        if start + size < len(reference) and not reference[start + size].isalnum():
            continue
        return True
    return False


def internal_remove_duplicate_tables(
    tables: native_reconcile.internal_ProfiledTables,
) -> tuple[native_reconcile.internal_ProfiledTables, frozenset[int]]:
    """Select table copies and protect the references needed to preserve charts."""
    tables = native_reconcile.internal_remove_duplicate_tables(tables)
    text_tokens = tuple(
        tuple(
            internal_CONTENT_TOKEN_RE.findall(
                " ".join(cell.text for row in table.rows for cell in row)
            )
        )
        for table, _ in tables
    )
    synthetic = tuple(internal_is_synthetic_chart(table) for table, _ in tables)
    # Consider complete references first, with real tables preferred on ties.
    # Only retained tables may cover another candidate, so equal charts cannot
    # reject each other and every discarded copy has a surviving replacement.
    ranked = sorted(
        range(len(tables)),
        key=lambda index: (
            -len(text_tokens[index]),
            synthetic[index],
            tables[index][0].order,
            index,
        ),
    )
    retained: list[int] = []
    replacements: set[int] = set()
    for index in ranked:
        table, _ = tables[index]
        reference = next(
            (
                other_index
                for other_index in retained
                if table.bbox is not None
                and (
                    synthetic[index]
                    or (table.metadata.get("source") == "stream" and synthetic[other_index])
                )
                and (other_box := tables[other_index][0].bbox) is not None
                and overlap_ratio_of(table.bbox, other_box) >= 0.90
                and internal_complete_text_covered(text_tokens[index], text_tokens[other_index])
            ),
            None,
        )
        if reference is None:
            retained.append(index)
        else:
            replacements.add(reference)
    retained.sort()
    return tuple(tables[index] for index in retained), frozenset(
        output_index
        for output_index, index in enumerate(retained)
        if synthetic[index] or index in replacements
    )


def internal_project_text_and_tables(
    blocks: list[Block],
    parsed_tables: tuple[Table, ...],
) -> tuple[list[Block], tuple[Table, ...]]:
    """Apply recognition policies at the same stages as native projection."""
    tables, protected = internal_remove_duplicate_tables(
        native_reconcile.internal_profile_tables(parsed_tables)
    )
    tables = native_reconcile.internal_remove_block_duplicate_tables(
        blocks,
        tables,
        # Native filtering can discard a table with only partial block coverage.
        # Keep charts and the complete references that replaced them; overlapping
        # duplicate blocks can then be removed without losing chart content.
        protected_table_indexes=protected,
    )
    blocks = native_reconcile.internal_remove_table_duplicate_blocks(blocks, tables)
    projected_tables = native_reconcile.internal_remove_block_duplicate_table_rows(
        blocks, tuple(table for table, _ in tables)
    )
    return blocks, projected_tables

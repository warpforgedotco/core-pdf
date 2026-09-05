# SPDX-License-Identifier: AGPL-3.0-only
"""Reconcile recognized chart tables and short duplicate recognition artifacts."""

from __future__ import annotations

from collections import Counter

from core_pdf.impl._impl.extract import table_reconcile as native_reconcile
from core_pdf.impl._impl.output.model import Block, Table


def internal_covers_synthetic_chart_table(
    table: Table,
    profile: native_reconcile.internal_TableProfile,
    tables: native_reconcile.internal_ProfiledTables,
) -> bool:
    return any(
        other is not table
        and other.metadata.get("source") == "chart-ocr"
        and other.metadata.get("synthetic")
        and native_reconcile.internal_table_profile_token_coverage(
            other_profile,
            profile,
        )
        >= 0.95
        for other, other_profile in tables
    )


def internal_remove_block_duplicate_tables(
    blocks: list[Block],
    tables: native_reconcile.internal_ProfiledTables,
) -> native_reconcile.internal_ProfiledTables:
    if not blocks or not tables:
        return tables
    chart_coverage = frozenset(
        index
        for index, (table, profile) in enumerate(tables)
        if internal_covers_synthetic_chart_table(table, profile, tables)
    )
    rejected = frozenset(
        index
        for index in chart_coverage
        if tables[index][0].metadata.get("source") == "stream" and tables[index][0].bbox is not None
    )
    return native_reconcile.internal_remove_block_duplicate_tables(
        blocks,
        tables,
        protected_table_indexes=chart_coverage - rejected,
        rejected_table_indexes=rejected,
    )


def internal_remove_duplicate_tables(
    tables: native_reconcile.internal_ProfiledTables,
) -> native_reconcile.internal_ProfiledTables:
    rejected = frozenset(
        index
        for index, (table, profile) in enumerate(tables)
        if table.metadata.get("source") == "chart-ocr"
        and table.metadata.get("synthetic")
        and 0 < len(profile.tokens) <= 24
        and any(
            other_index != index
            and native_reconcile.internal_table_profile_token_coverage(profile, other_profile)
            >= 0.95
            for other_index, (_, other_profile) in enumerate(tables)
        )
    )
    return native_reconcile.internal_remove_duplicate_tables(
        tables, rejected_table_indexes=rejected
    )


def internal_remove_table_duplicate_blocks(
    blocks: list[Block],
    tables: native_reconcile.internal_ProfiledTables,
) -> list[Block]:
    if not blocks or not tables:
        return blocks
    token_counts = Counter(token for _, profile in tables for token in profile.tokens)
    blocks = [
        block
        for block in blocks
        if not (
            block.bbox is not None
            and block.provenance == ("ocr",)
            and 0 < len(tokens := native_reconcile.internal_emitted_text_tokens(block.text)) <= 3
            and all(token_counts[token] >= count for token, count in Counter(tokens).items())
        )
    ]
    return native_reconcile.internal_remove_table_duplicate_blocks(blocks, tables)


def internal_project_text_and_tables(
    blocks: list[Block],
    parsed_tables: tuple[Table, ...],
) -> tuple[list[Block], tuple[Table, ...]]:
    """Apply recognition policies at the same stages as native projection."""
    tables = internal_remove_duplicate_tables(
        internal_remove_block_duplicate_tables(
            blocks, native_reconcile.internal_profile_tables(parsed_tables)
        )
    )
    blocks = internal_remove_table_duplicate_blocks(blocks, tables)
    projected_tables = native_reconcile.internal_remove_block_duplicate_table_rows(
        blocks, tuple(table for table, _ in tables)
    )
    return blocks, projected_tables

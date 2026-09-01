# SPDX-License-Identifier: AGPL-3.0-only
"""Reconcile overlapping text and table projections."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from functools import lru_cache

from core_pdf.impl.extract.tables import (
    internal_character_spaced_cell,
    internal_structured_stream_table,
    internal_table_has_grid_shape,
)
from core_pdf.impl.model.geometry import overlap_ratio_min, overlap_ratio_of
from core_pdf.impl.output import Block, Table, TableCell

internal_EMITTED_TEXT_TOKEN_RE = re.compile(r"\w+")


@lru_cache(maxsize=256)
def internal_emitted_text_tokens(text: str) -> tuple[str, ...]:
    # The same table/block text is tokenized by many dedup heuristics per
    # page; the memo returns an immutable tuple every caller only reads.
    return tuple(
        match.group(0).casefold() for match in internal_EMITTED_TEXT_TOKEN_RE.finditer(text)
    )


def internal_wordlike_token(token: str) -> bool:
    return token.isalpha() and len(token) >= 3 and any(character in "aeiou" for character in token)


def internal_table_text(table: Table) -> str:
    return " ".join(cell.text for row in table.rows for cell in row if cell.text)


def internal_table_character_spaced_ratio(table: Table) -> float:
    filled_texts = [cell.text.strip() for row in table.rows for cell in row if cell.text.strip()]
    if not filled_texts:
        return 0.0
    return sum(internal_character_spaced_cell(text) for text in filled_texts) / len(filled_texts)


def internal_overlapping_block_token_coverage(table: Table, blocks: list[Block]) -> float:
    if table.bbox is None:
        return 0.0
    table_tokens = internal_emitted_text_tokens(internal_table_text(table))
    if len(table_tokens) < 16:
        return 0.0
    block_tokens: list[str] = []
    for block in blocks:
        if block.bbox is None or not block.text:
            continue
        if overlap_ratio_min(block.bbox, table.bbox) >= 0.45:
            block_tokens.extend(internal_emitted_text_tokens(block.text))
    if not block_tokens:
        return 0.0
    matched = (Counter(table_tokens) & Counter(block_tokens)).total()
    return matched / len(table_tokens)


def internal_stream_table_is_tabular(table: Table) -> bool:
    """Report a stream table whose shape is tabular rather than two-column prose.

    ``internal_table_has_grid_shape`` asks only that the rows divide, which a
    page of text set in two columns satisfies as readily as a table does. A
    third column and a third row are what prose does not produce by accident,
    and requiring both keeps a wrapped paragraph or a two-line caption from
    claiming to be a table it merely resembles.
    """
    if not internal_table_has_grid_shape(table):
        return False
    rows = [row for row in table.rows if row]
    columns = max((cell.column + cell.column_span for row in rows for cell in row), default=0)
    return columns >= 3 and len(rows) >= 3


def internal_stream_table_duplicated_by_blocks(table: Table, blocks: list[Block]) -> bool:
    table_tokens = internal_emitted_text_tokens(internal_table_text(table))
    token_count = len(table_tokens)
    if token_count < 16:
        return False
    # Keep structured numeric tables even when the layout pass also emits the
    # same glyphs as paragraph text.  ``internal_remove_table_duplicate_blocks``
    # removes those overlapping blocks after this decision; dropping the table
    # here loses the structure needed by downstream consumers.
    if internal_structured_stream_table(table):
        return False
    # The same reasoning applies to a table that is tabular in shape rather than
    # in content, and for the reason given on the ruled path below: the blocks
    # and the table hold the same glyphs, but only the table carries the rows and
    # columns, and the text survives either way. Judging this by numeric density
    # alone discarded comparison tables and schedules whose cells are sentences.
    if internal_stream_table_is_tabular(table):
        return False
    coverage = internal_overlapping_block_token_coverage(table, blocks)
    if coverage >= 0.80:
        return True
    return token_count >= 500 and coverage >= 0.35


def internal_table_duplicated_by_blocks(table: Table, blocks: list[Block]) -> bool:
    table_tokens = internal_emitted_text_tokens(internal_table_text(table))
    if len(table_tokens) < 24:
        return False
    if internal_structured_stream_table(table):
        return False
    if internal_table_has_grid_shape(table):
        # Blocks and a genuine table describing the same region both hold the
        # same glyphs, so one of them has to go. Dropping the table is the
        # wrong way round: the text survives either way, but only the table
        # carries the rows and columns. Keep it and let
        # internal_remove_table_duplicate_blocks take the blocks instead.
        return False
    return internal_overlapping_block_token_coverage(table, blocks) >= 0.90


def internal_small_table_duplicated_by_page_text(table: Table, blocks: list[Block]) -> bool:
    if table.metadata.get("source") == "stream":
        return False
    if internal_table_has_grid_shape(table):
        return False
    table_tokens = internal_emitted_text_tokens(internal_table_text(table))
    if not 4 <= len(table_tokens) < 24:
        return False
    block_counts: Counter[str] = Counter()
    for block in blocks:
        block_counts.update(internal_emitted_text_tokens(block.text))
    if not block_counts:
        return False
    matched = 0
    for token in table_tokens:
        if block_counts[token] > 0:
            matched += 1
            block_counts[token] -= 1
    return matched / len(table_tokens) >= 0.90


def internal_covers_synthetic_chart_table(table: Table, tables: tuple[Table, ...]) -> bool:
    return any(
        other is not table
        and other.metadata.get("source") == "chart-ocr"
        and other.metadata.get("synthetic")
        and internal_table_token_coverage(other, table) >= 0.95
        for other in tables
    )


def internal_fragmented_stream_table(table: Table) -> bool:
    if table.metadata.get("source") != "stream":
        return False
    tokens = internal_emitted_text_tokens(internal_table_text(table))
    if len(tokens) < 80:
        return False
    single_character = sum(len(token) == 1 for token in tokens)
    return single_character / len(tokens) >= 0.70


def internal_noisy_stream_table(table: Table) -> bool:
    if table.metadata.get("source") != "stream":
        return False
    tokens = internal_emitted_text_tokens(internal_table_text(table))
    if len(tokens) < 16:
        return False
    single_character = sum(len(token) == 1 for token in tokens)
    wordlike = sum(internal_wordlike_token(token) for token in tokens)
    return single_character / len(tokens) >= 0.55 and wordlike / len(tokens) < 0.30


def internal_remove_block_duplicate_tables(
    blocks: list[Block],
    tables: tuple[Table, ...],
) -> tuple[Table, ...]:
    if not blocks or not tables:
        return tables
    filtered: list[Table] = []
    block_boxes = tuple(block.bbox for block in blocks if block.bbox is not None and block.text)
    for table in tables:
        if (
            (
                (
                    internal_table_duplicated_by_blocks(table, blocks)
                    or internal_small_table_duplicated_by_page_text(table, blocks)
                )
                and not internal_covers_synthetic_chart_table(table, tables)
            )
            or table.metadata.get("source") == "stream"
            and table.bbox is not None
            and (
                internal_fragmented_stream_table(table)
                or internal_noisy_stream_table(table)
                or internal_covers_synthetic_chart_table(table, tables)
                or (
                    internal_table_character_spaced_ratio(table) >= 0.20
                    and any(
                        overlap_ratio_min(block_box, table.bbox) >= 0.85
                        for block_box in block_boxes
                    )
                )
                or internal_stream_table_duplicated_by_blocks(table, blocks)
            )
        ):
            continue
        filtered.append(table)
    return tuple(filtered)


def internal_remove_block_duplicate_table_rows(
    blocks: list[Block],
    tables: tuple[Table, ...],
) -> tuple[Table, ...]:
    """Drop table rows whose text is already emitted by an overlapping block.

    This mirrors internal_remove_table_duplicate_blocks in the opposite
    direction: block text is the primary reading order, so a row that
    repeats it (for example a schedule rendered both as body lines and as a
    stream table) is removed.  A row is dropped only when it and one block
    line share roughly the same token multiset, or when its tokens are
    scattered through the surrounding blocks without forming a fragment of a
    single line, so rows whose words merely echo a longer heading are kept.
    """
    if not blocks or not tables:
        return tables
    filtered: list[Table] = []
    for table in tables:
        if table.bbox is None or not table.rows:
            filtered.append(table)
            continue
        block_lines = [
            line
            for block in blocks
            if block.bbox is not None and overlap_ratio_min(block.bbox, table.bbox) >= 0.45
            for line in block.lines
        ]
        line_tokens = [
            tokens
            for line in block_lines
            for tokens in [internal_emitted_text_tokens(line.text)]
            if tokens
        ]
        if not line_tokens:
            filtered.append(table)
            continue
        block_counts = Counter(token for tokens in line_tokens for token in tokens)
        line_sets = [set(tokens) for tokens in line_tokens]
        kept_rows: list[tuple[TableCell, ...]] = []
        for row in table.rows:
            cells = [cell for cell in row if cell.text]
            row_tokens = internal_emitted_text_tokens(" ".join(cell.text for cell in cells))
            if not row_tokens:
                kept_rows.append(row)
                continue
            duplicated = False
            for line, line_set in zip(line_tokens, line_sets):
                matched = sum(1 for token in row_tokens if token in line_set)
                if matched / len(row_tokens) >= 0.9 and matched / len(line) >= 0.9:
                    duplicated = True
                    break
            if not duplicated:
                matched = sum(
                    min(count, block_counts[token]) for token, count in Counter(row_tokens).items()
                )
                fragment = any(
                    all(token in line_set for token in row_tokens) for line_set in line_sets
                )
                duplicated = matched / len(row_tokens) >= 0.9 and not fragment
            if not duplicated:
                kept_rows.append(row)
        if len(kept_rows) == len(table.rows):
            filtered.append(table)
            continue
        filtered.append(replace(table, rows=tuple(kept_rows)))
    return tuple(filtered)


def internal_table_token_coverage(candidate: Table, reference: Table) -> float:
    candidate_tokens = internal_emitted_text_tokens(internal_table_text(candidate))
    if not candidate_tokens:
        return 0.0
    reference_counts = Counter(internal_emitted_text_tokens(internal_table_text(reference)))
    matched = 0
    for token in candidate_tokens:
        if reference_counts[token] > 0:
            matched += 1
            reference_counts[token] -= 1
    return matched / len(candidate_tokens)


def internal_remove_duplicate_tables(tables: tuple[Table, ...]) -> tuple[Table, ...]:
    filtered: list[Table] = []
    for index, table in enumerate(tables):
        tokens = internal_emitted_text_tokens(internal_table_text(table))
        if not table.metadata and 0 < len(tokens) <= 8 and set(tokens) <= {"b", "i"}:
            continue
        if len(tables) < 2:
            filtered.append(table)
            continue
        if (
            table.metadata.get("source") == "chart-ocr"
            and table.metadata.get("synthetic")
            and 0 < len(tokens) <= 24
            and any(
                other_index != index and internal_table_token_coverage(table, other) >= 0.95
                for other_index, other in enumerate(tables)
            )
        ):
            continue
        filtered.append(table)
    return tuple(filtered)


def internal_remove_table_duplicate_blocks(
    blocks: list[Block], tables: tuple[Table, ...]
) -> list[Block]:
    if not blocks or not tables:
        return blocks
    table_boxes = [
        (table.bbox, set(internal_emitted_text_tokens(internal_table_text(table))))
        for table in tables
        if table.bbox is not None
    ]
    if not table_boxes:
        return blocks
    deduplicated: list[Block] = []
    for block in blocks:
        if block.bbox is None:
            deduplicated.append(block)
            continue
        block_tokens = internal_emitted_text_tokens(block.text)
        if not block_tokens:
            deduplicated.append(block)
            continue
        duplicate = False
        contained_line_boxes: list[tuple[float, float, float, float]] = []
        for table_bbox, tokens in table_boxes:
            overlap_ratio = overlap_ratio_min(block.bbox, table_bbox)
            if overlap_ratio >= 0.9:
                if sum(token in tokens for token in block_tokens) / len(block_tokens) >= 0.85:
                    duplicate = True
                    break
                contained_line_boxes.append(table_bbox)
        if duplicate:
            continue
        if contained_line_boxes:
            # A block that surrounds (or is surrounded by) a table but carries
            # additional non-table text: drop only the lines whose bbox lies
            # inside a table region so the table content is not emitted twice.
            filtered = tuple(
                line
                for line in block.lines
                if line.bbox is None
                or not any(overlap_ratio_of(line.bbox, box) >= 0.9 for box in contained_line_boxes)
            )
            if filtered and len(filtered) != len(block.lines):
                deduplicated.append(replace(block, lines=filtered))
                continue
        deduplicated.append(block)
    return deduplicated


def internal_remove_tiny_table_duplicate_blocks(
    blocks: list[Block],
    tables: tuple[Table, ...],
) -> list[Block]:
    if not blocks or not tables:
        return blocks
    table_token_counts: Counter[str] = Counter()
    for table in tables:
        table_token_counts.update(internal_emitted_text_tokens(internal_table_text(table)))
    if not table_token_counts:
        return blocks
    filtered: list[Block] = []
    for block in blocks:
        tokens = internal_emitted_text_tokens(block.text)
        if (
            block.provenance == ("ocr",)
            and 0 < len(tokens) <= 3
            and all(table_token_counts[token] >= count for token, count in Counter(tokens).items())
        ):
            continue
        filtered.append(block)
    return filtered


def internal_project_text_and_tables(
    blocks: list[Block],
    parsed_tables: tuple[Table, ...],
) -> tuple[list[Block], tuple[Table, ...]]:
    """Resolve overlap once, producing explicit text and table projections."""
    tables = internal_remove_duplicate_tables(
        internal_remove_block_duplicate_tables(blocks, parsed_tables)
    )
    text_blocks = internal_remove_table_duplicate_blocks(blocks, tables)
    text_blocks = internal_remove_tiny_table_duplicate_blocks(text_blocks, tables)
    tables = internal_remove_block_duplicate_table_rows(text_blocks, tables)
    return text_blocks, tables

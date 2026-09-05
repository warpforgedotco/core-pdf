# SPDX-License-Identifier: AGPL-3.0-only
"""Reconcile overlapping text and table projections."""

from __future__ import annotations

import re
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, replace

from core_pdf.impl._impl.extract.table_cleanup import (
    internal_structured_stream_table,
    internal_table_has_grid_shape,
    internal_table_with_bands,
)
from core_pdf.impl._impl.extract.table_facts import internal_TableFacts
from core_pdf.impl._impl.model.geometry import bbox_union, overlap_ratio_min, overlap_ratio_of
from core_pdf.impl._impl.model.spatial import SpatialFrame
from core_pdf.impl._impl.model.text import complete_text_covered, content_tokens
from core_pdf.impl._impl.output.model import Block, Table
from core_pdf.impl.types import Rectangle

internal_EMITTED_TEXT_TOKEN_RE = re.compile(r"\w+")
internal_BlockTokens = tuple[tuple[tuple[float, float, float, float], tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class internal_TableProfile:
    """Text and shape facts derived from one table."""

    tokens: tuple[str, ...]
    token_counts: Counter[str]
    token_set: frozenset[str]
    character_spaced_ratio: float
    has_grid_shape: bool
    structured_stream: bool
    stream_is_tabular: bool
    fragmented_stream: bool
    noisy_stream: bool


internal_ProfiledTables = tuple[tuple[Table, internal_TableProfile], ...]


def internal_profile_tables(tables: tuple[Table, ...]) -> internal_ProfiledTables:
    """Bind each immutable table snapshot to the facts used during one projection."""
    return tuple((table, internal_table_profile(table)) for table in tables)


def internal_emitted_text_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        match.group(0).casefold() for match in internal_EMITTED_TEXT_TOKEN_RE.finditer(text)
    )


def internal_wordlike_token(token: str) -> bool:
    return token.isalpha() and len(token) >= 3 and any(character in "aeiou" for character in token)


def internal_table_profile(table: Table) -> internal_TableProfile:
    text = " ".join(cell.text for row in table.rows for cell in row if cell.text)
    tokens = internal_emitted_text_tokens(text)
    token_counts = Counter(tokens)
    facts = internal_TableFacts.from_rows(table.rows)
    filled_texts = facts.filled_texts
    character_spaced_ratio = (
        facts.character_spaced_cells / len(filled_texts) if filled_texts else 0.0
    )
    has_grid_shape = internal_table_has_grid_shape(table, facts=facts)
    stream_is_tabular = has_grid_shape and facts.spanned_columns >= 3 and facts.nonempty_rows >= 3
    is_stream = table.metadata.get("source") == "stream"
    single_character = sum(len(token) == 1 for token in tokens)
    wordlike = sum(internal_wordlike_token(token) for token in tokens)
    return internal_TableProfile(
        tokens=tokens,
        token_counts=token_counts,
        token_set=frozenset(token_counts),
        character_spaced_ratio=character_spaced_ratio,
        has_grid_shape=has_grid_shape,
        structured_stream=internal_structured_stream_table(table, facts=facts),
        stream_is_tabular=stream_is_tabular,
        fragmented_stream=(
            is_stream and len(tokens) >= 80 and single_character / len(tokens) >= 0.70
        ),
        noisy_stream=(
            is_stream
            and len(tokens) >= 16
            and single_character / len(tokens) >= 0.55
            and wordlike / len(tokens) < 0.30
        ),
    )


def internal_overlapping_block_token_coverage(
    table: Table,
    blocks: list[Block],
    *,
    profile: internal_TableProfile | None = None,
    tokenized_blocks: internal_BlockTokens | None = None,
) -> float:
    if table.bbox is None:
        return 0.0
    table_profile = profile or internal_table_profile(table)
    table_tokens = table_profile.tokens
    if len(table_tokens) < 16:
        return 0.0
    block_entries = tokenized_blocks or tuple(
        (block.bbox, internal_emitted_text_tokens(block.text))
        for block in blocks
        if block.bbox is not None and block.text
    )
    overlapping_tokens = tuple(
        token
        for bbox, tokens in block_entries
        if overlap_ratio_min(bbox, table.bbox) >= 0.45
        for token in tokens
    )
    if not overlapping_tokens:
        return 0.0
    block_counts = Counter(overlapping_tokens)
    matched = sum(
        min(count, block_counts[token]) for token, count in table_profile.token_counts.items()
    )
    return matched / len(table_tokens)


def internal_stream_table_duplicated_by_blocks(
    table: Table,
    blocks: list[Block],
    *,
    profile: internal_TableProfile | None = None,
    tokenized_blocks: internal_BlockTokens | None = None,
) -> bool:
    table_profile = profile or internal_table_profile(table)
    table_tokens = table_profile.tokens
    token_count = len(table_tokens)
    if token_count < 16:
        return False
    # Keep structured numeric tables even when the layout pass also emits the
    # same glyphs as paragraph text.  ``internal_remove_table_duplicate_blocks``
    # removes those overlapping blocks after this decision; dropping the table
    # here loses the structure needed by downstream consumers.
    if table_profile.structured_stream:
        return False
    # The same reasoning applies to a table that is tabular in shape rather than
    # in content, and for the reason given on the ruled path below: the blocks
    # and the table hold the same glyphs, but only the table carries the rows and
    # columns, and the text survives either way. Judging this by numeric density
    # alone discarded comparison tables and schedules whose cells are sentences.
    if table_profile.stream_is_tabular:
        return False
    coverage = internal_overlapping_block_token_coverage(
        table, blocks, profile=table_profile, tokenized_blocks=tokenized_blocks
    )
    if coverage >= 0.80:
        return True
    return token_count >= 500 and coverage >= 0.35


def internal_table_duplicated_by_blocks(
    table: Table,
    blocks: list[Block],
    *,
    profile: internal_TableProfile | None = None,
    tokenized_blocks: internal_BlockTokens | None = None,
) -> bool:
    table_profile = profile or internal_table_profile(table)
    table_tokens = table_profile.tokens
    if len(table_tokens) < 24:
        return False
    if table_profile.structured_stream:
        return False
    if table_profile.has_grid_shape:
        # Blocks and a genuine table describing the same region both hold the
        # same glyphs, so one of them has to go. Dropping the table is the
        # wrong way round: the text survives either way, but only the table
        # carries the rows and columns. Keep it and let
        # internal_remove_table_duplicate_blocks take the blocks instead.
        return False
    return (
        internal_overlapping_block_token_coverage(
            table, blocks, profile=table_profile, tokenized_blocks=tokenized_blocks
        )
        >= 0.90
    )


def internal_remove_block_duplicate_tables(
    blocks: list[Block],
    tables: internal_ProfiledTables,
    *,
    protected_table_indexes: frozenset[int] = frozenset(),
) -> internal_ProfiledTables:
    if not blocks or not tables:
        return tables
    filtered: list[tuple[Table, internal_TableProfile]] = []
    needs_block_tokens = any(
        not profile.structured_stream and not profile.has_grid_shape and len(profile.tokens) >= 4
        for _, profile in tables
    )
    tokenized_blocks = (
        tuple(
            (block.bbox, internal_emitted_text_tokens(block.text))
            for block in blocks
            if block.bbox is not None and block.text
        )
        if needs_block_tokens
        else ()
    )
    block_boxes = tuple(block.bbox for block in blocks if block.bbox is not None and block.text)
    for index, (table, profile) in enumerate(tables):
        if index in protected_table_indexes:
            filtered.append((table, profile))
            continue
        if (
            internal_table_duplicated_by_blocks(
                table, blocks, profile=profile, tokenized_blocks=tokenized_blocks
            )
            or table.metadata.get("source") == "stream"
            and table.bbox is not None
            and (
                profile.fragmented_stream
                or profile.noisy_stream
                or (
                    profile.character_spaced_ratio >= 0.20
                    and any(
                        overlap_ratio_min(block_box, table.bbox) >= 0.85
                        for block_box in block_boxes
                    )
                )
                or internal_stream_table_duplicated_by_blocks(
                    table, blocks, profile=profile, tokenized_blocks=tokenized_blocks
                )
            )
        ):
            continue
        filtered.append((table, profile))
    return tuple(filtered)


def internal_remove_block_duplicate_table_rows(
    blocks: list[Block],
    tables: tuple[Table, ...],
) -> tuple[Table, ...]:
    """Drop rows reproduced completely by a surviving line in the same region."""
    if not blocks or not tables:
        return tables
    line_boxes_by_text: dict[str, list[Rectangle]] = {}
    for block in blocks:
        for line in block.lines:
            box = line.bbox or (block.bbox if len(block.lines) == 1 else None)
            text = " ".join(line.text.split())
            if box is not None and text:
                line_boxes_by_text.setdefault(text, []).append(box)
    filtered: list[Table] = []
    for table in tables:
        if not table.rows:
            filtered.append(table)
            continue
        kept_row_indexes: list[int] = []
        for row_index, row in enumerate(table.rows):
            cells = [cell for cell in row if cell.text]
            text = " ".join(" ".join(cell.text.split()) for cell in cells)
            cell_boxes = tuple(cell.bbox for cell in cells if cell.bbox is not None)
            row_box = bbox_union(cell_boxes)
            if not text or row_box is None or len(cell_boxes) != len(cells):
                kept_row_indexes.append(row_index)
                continue
            duplicated = any(
                overlap_ratio_of(line_box, row_box) >= 0.90
                and all(overlap_ratio_min(line_box, cell_box) > 0 for cell_box in cell_boxes)
                for line_box in line_boxes_by_text.get(text, ())
            )
            if not duplicated:
                kept_row_indexes.append(row_index)
        if len(kept_row_indexes) == len(table.rows):
            filtered.append(table)
            continue
        if not kept_row_indexes:
            # The blocks cover the rows, but need not include associated text.
            # Keep its table until that text can be projected independently.
            if table.title is not None or table.caption is not None:
                filtered.append(table)
            continue
        rows = tuple(
            tuple(
                replace(
                    cell,
                    row=index,
                    row_span=bisect_left(kept_row_indexes, old_index + cell.row_span) - index,
                )
                for cell in table.rows[old_index]
            )
            for index, old_index in enumerate(kept_row_indexes)
        )
        projected = internal_table_with_bands(replace(table, rows=rows))
        if table.row_bands:
            # A removed row must not shift another row's semantic role. Rebuild
            # column bounds, but retain the surviving row classifications.
            projected = replace(
                projected,
                row_bands=tuple(
                    replace(table.row_bands[old_index], index=index)
                    for index, old_index in enumerate(kept_row_indexes)
                ),
            )
        filtered.append(projected)
    return tuple(filtered)


def internal_line_duplicates_table(text: str, box: Rectangle, table: Table) -> bool:
    """Require complete text in the same cell region before dropping a line."""
    normalized = " ".join(text.split())
    if not normalized or table.bbox is None or overlap_ratio_of(box, table.bbox) < 0.90:
        return False
    tokens = content_tokens(normalized)
    participating_cells = []
    total_cells = 0
    for row in table.rows:
        row_cells = [cell for cell in row if cell.text]
        total_cells += len(row_cells)
        cells = [
            cell for cell in row_cells if cell.bbox is None or overlap_ratio_min(box, cell.bbox) > 0
        ]
        if not cells:
            continue
        participating_cells.extend(cells)
        cell_texts = [" ".join(cell.text.split()) for cell in cells]
        for cell, cell_text in zip(cells, cell_texts, strict=True):
            if (
                cell.bbox is not None
                and overlap_ratio_of(box, cell.bbox) >= 0.90
                and complete_text_covered(tokens, content_tokens(cell_text))
            ):
                return True
        for start in range(len(cells)):
            candidate = ""
            for end in range(start, len(cells)):
                candidate = f"{candidate} {cell_texts[end]}" if candidate else cell_texts[end]
                if not normalized.startswith(candidate):
                    break
                if normalized != candidate:
                    continue
                matched_cells = cells[start : end + 1]
                # Inferred column boundaries can graze the line's last glyph.
                # Test exact cell sequences instead of forcing neighboring
                # text into a match because of a small geometric intersection.
                if all(cell.bbox is not None for cell in matched_cells):
                    row_box = bbox_union(
                        cell.bbox for cell in matched_cells if cell.bbox is not None
                    )
                    if row_box is not None and overlap_ratio_of(box, row_box) >= 0.90:
                        return True
                elif start == 0 and end + 1 == len(cells) == len(row_cells):
                    # Without cell geometry, only the complete row can be
                    # compared using the containing table's known bounds.
                    return True
    # A single text line can represent an entire table in a competing layout.
    # Keep case, punctuation, order, and repeated values in this comparison.
    known_geometry = all(cell.bbox is not None for cell in participating_cells)
    if not known_geometry and len(participating_cells) != total_cells:
        return False
    covered_box = (
        bbox_union(cell.bbox for cell in participating_cells if cell.bbox is not None)
        if participating_cells and known_geometry
        else table.bbox
    )
    return (
        covered_box is not None
        and overlap_ratio_of(box, covered_box) >= 0.90
        and normalized == " ".join(" ".join(cell.text.split()) for cell in participating_cells)
    )


def internal_remove_table_duplicate_blocks(
    blocks: list[Block],
    tables: internal_ProfiledTables,
) -> list[Block]:
    if not blocks or not tables:
        return blocks
    located_tables = [table for table, _ in tables if table.bbox is not None]
    if not located_tables:
        return blocks
    table_frame = SpatialFrame.from_boxes(
        table.bbox for table in located_tables if table.bbox is not None
    )
    deduplicated: list[Block] = []
    for block in blocks:
        kept_lines = []
        for line in block.lines:
            # A tall block can include a heading well outside the table. Only
            # its individual lines establish which text actually repeats cells.
            box = line.bbox or (block.bbox if len(block.lines) == 1 else None)
            if box is None or not any(
                internal_line_duplicates_table(line.text, box, located_tables[int(index)])
                for index in table_frame.matching_overlap_min(box, 0.90)
            ):
                kept_lines.append(line)
        if len(kept_lines) == len(block.lines):
            deduplicated.append(block)
        elif kept_lines:
            box = (
                bbox_union(line.bbox for line in kept_lines if line.bbox is not None)
                if all(line.bbox is not None for line in kept_lines)
                else block.bbox
            )
            deduplicated.append(replace(block, lines=tuple(kept_lines), bbox=box))
    return deduplicated


def internal_project_text_and_tables(
    blocks: list[Block],
    parsed_tables: tuple[Table, ...],
) -> tuple[list[Block], tuple[Table, ...]]:
    """Resolve overlap once, producing explicit text and table projections."""
    tables = internal_remove_block_duplicate_tables(blocks, internal_profile_tables(parsed_tables))
    text_blocks = internal_remove_table_duplicate_blocks(blocks, tables)
    # Profiles describe the original row snapshots. Discard them before the
    # final row rewrite; another projection will derive facts from its new rows.
    projected_tables = internal_remove_block_duplicate_table_rows(
        text_blocks, tuple(table for table, _ in tables)
    )
    return text_blocks, projected_tables

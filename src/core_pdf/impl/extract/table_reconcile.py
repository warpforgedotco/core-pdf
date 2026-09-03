# SPDX-License-Identifier: AGPL-3.0-only
"""Reconcile overlapping text and table projections."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from types import MappingProxyType

from core_pdf.impl.extract.table_cleanup import (
    internal_character_spaced_cell,
    internal_structured_stream_table,
    internal_table_has_grid_shape,
)
from core_pdf.impl.model.geometry import overlap_ratio_min, overlap_ratio_of
from core_pdf.impl.output import Block, Table, TableCell

internal_EMITTED_TEXT_TOKEN_RE = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class internal_TableProfile:
    """Reusable text and shape facts for one reconciliation pass."""

    text: str
    tokens: tuple[str, ...]
    token_counts: Mapping[str, int]
    token_set: frozenset[str]
    character_spaced_ratio: float
    has_grid_shape: bool
    structured_stream: bool
    stream_is_tabular: bool
    fragmented_stream: bool
    noisy_stream: bool


internal_TableProfiles = Mapping[int, internal_TableProfile]


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


def internal_build_table_profile(table: Table) -> internal_TableProfile:
    text = internal_table_text(table)
    tokens = internal_emitted_text_tokens(text)
    token_counts = Counter(tokens)
    filled_texts = [cell.text.strip() for row in table.rows for cell in row if cell.text.strip()]
    character_spaced_ratio = (
        sum(internal_character_spaced_cell(text) for text in filled_texts) / len(filled_texts)
        if filled_texts
        else 0.0
    )
    has_grid_shape = internal_table_has_grid_shape(table)
    rows = [row for row in table.rows if row]
    columns = max((cell.column + cell.column_span for row in rows for cell in row), default=0)
    stream_is_tabular = has_grid_shape and columns >= 3 and len(rows) >= 3
    is_stream = table.metadata.get("source") == "stream"
    single_character = sum(len(token) == 1 for token in tokens)
    wordlike = sum(internal_wordlike_token(token) for token in tokens)
    return internal_TableProfile(
        text=text,
        tokens=tokens,
        token_counts=MappingProxyType(dict(token_counts)),
        token_set=frozenset(token_counts),
        character_spaced_ratio=character_spaced_ratio,
        has_grid_shape=has_grid_shape,
        structured_stream=internal_structured_stream_table(table),
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


def internal_table_profile(
    table: Table,
    profiles: internal_TableProfiles | None = None,
) -> internal_TableProfile:
    if profiles is not None:
        profile = profiles.get(id(table))
        if profile is not None:
            return profile
    return internal_build_table_profile(table)


def internal_table_character_spaced_ratio(
    table: Table,
    *,
    profile: internal_TableProfile | None = None,
) -> float:
    return (profile or internal_build_table_profile(table)).character_spaced_ratio


def internal_overlapping_block_token_coverage(
    table: Table,
    blocks: list[Block],
    *,
    profile: internal_TableProfile | None = None,
) -> float:
    if table.bbox is None:
        return 0.0
    table_profile = profile or internal_build_table_profile(table)
    table_tokens = table_profile.tokens
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
    block_counts = Counter(block_tokens)
    matched = sum(
        min(count, block_counts[token]) for token, count in table_profile.token_counts.items()
    )
    return matched / len(table_tokens)


def internal_stream_table_is_tabular(
    table: Table,
    *,
    profile: internal_TableProfile | None = None,
) -> bool:
    """Identify a stream table whose shape is tabular rather than two-column prose.

    ``internal_table_has_grid_shape`` asks only that the rows divide, which a
    page of text set in two columns satisfies as readily as a table does. A
    third column and a third row are what prose does not produce by accident,
    and requiring both keeps a wrapped paragraph or a two-line caption from
    claiming to be a table it merely resembles.
    """
    return (profile or internal_build_table_profile(table)).stream_is_tabular


def internal_stream_table_duplicated_by_blocks(
    table: Table,
    blocks: list[Block],
    *,
    profile: internal_TableProfile | None = None,
) -> bool:
    table_profile = profile or internal_build_table_profile(table)
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
    coverage = internal_overlapping_block_token_coverage(table, blocks, profile=table_profile)
    if coverage >= 0.80:
        return True
    return token_count >= 500 and coverage >= 0.35


def internal_table_duplicated_by_blocks(
    table: Table,
    blocks: list[Block],
    *,
    profile: internal_TableProfile | None = None,
) -> bool:
    table_profile = profile or internal_build_table_profile(table)
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
    return internal_overlapping_block_token_coverage(table, blocks, profile=table_profile) >= 0.90


def internal_small_table_duplicated_by_page_text(
    table: Table,
    blocks: list[Block],
    *,
    profile: internal_TableProfile | None = None,
    block_counts: Counter[str] | None = None,
) -> bool:
    table_profile = profile or internal_build_table_profile(table)
    if table.metadata.get("source") == "stream":
        return False
    if table_profile.has_grid_shape:
        return False
    table_tokens = table_profile.tokens
    if not 4 <= len(table_tokens) < 24:
        return False
    page_token_counts = block_counts
    if page_token_counts is None:
        page_token_counts = Counter()
        for block in blocks:
            page_token_counts.update(internal_emitted_text_tokens(block.text))
    if not page_token_counts:
        return False
    matched = sum(
        min(count, page_token_counts[token]) for token, count in table_profile.token_counts.items()
    )
    return matched / len(table_tokens) >= 0.90


def internal_covers_synthetic_chart_table(
    table: Table,
    tables: tuple[Table, ...],
    *,
    profiles: internal_TableProfiles | None = None,
) -> bool:
    table_profile = internal_table_profile(table, profiles)
    return any(
        other is not table
        and other.metadata.get("source") == "chart-ocr"
        and other.metadata.get("synthetic")
        and internal_table_token_coverage(
            other,
            table,
            candidate_profile=internal_table_profile(other, profiles),
            reference_profile=table_profile,
        )
        >= 0.95
        for other in tables
    )


def internal_fragmented_stream_table(
    table: Table,
    *,
    profile: internal_TableProfile | None = None,
) -> bool:
    return (profile or internal_build_table_profile(table)).fragmented_stream


def internal_noisy_stream_table(
    table: Table,
    *,
    profile: internal_TableProfile | None = None,
) -> bool:
    return (profile or internal_build_table_profile(table)).noisy_stream


def internal_remove_block_duplicate_tables(
    blocks: list[Block],
    tables: tuple[Table, ...],
    *,
    profiles: internal_TableProfiles | None = None,
) -> tuple[Table, ...]:
    if not blocks or not tables:
        return tables
    filtered: list[Table] = []
    block_boxes = tuple(block.bbox for block in blocks if block.bbox is not None and block.text)
    block_counts = Counter(
        token for block in blocks for token in internal_emitted_text_tokens(block.text)
    )
    for table in tables:
        profile = internal_table_profile(table, profiles)
        covers_synthetic_chart = internal_covers_synthetic_chart_table(
            table, tables, profiles=profiles
        )
        if (
            (
                (
                    internal_table_duplicated_by_blocks(table, blocks, profile=profile)
                    or internal_small_table_duplicated_by_page_text(
                        table,
                        blocks,
                        profile=profile,
                        block_counts=block_counts,
                    )
                )
                and not covers_synthetic_chart
            )
            or table.metadata.get("source") == "stream"
            and table.bbox is not None
            and (
                profile.fragmented_stream
                or profile.noisy_stream
                or covers_synthetic_chart
                or (
                    profile.character_spaced_ratio >= 0.20
                    and any(
                        overlap_ratio_min(block_box, table.bbox) >= 0.85
                        for block_box in block_boxes
                    )
                )
                or internal_stream_table_duplicated_by_blocks(table, blocks, profile=profile)
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


def internal_table_token_coverage(
    candidate: Table,
    reference: Table,
    *,
    candidate_profile: internal_TableProfile | None = None,
    reference_profile: internal_TableProfile | None = None,
) -> float:
    candidate_facts = candidate_profile or internal_build_table_profile(candidate)
    candidate_tokens = candidate_facts.tokens
    if not candidate_tokens:
        return 0.0
    reference_counts = (reference_profile or internal_build_table_profile(reference)).token_counts
    matched = sum(
        min(count, reference_counts.get(token, 0))
        for token, count in candidate_facts.token_counts.items()
    )
    return matched / len(candidate_tokens)


def internal_remove_duplicate_tables(
    tables: tuple[Table, ...],
    *,
    profiles: internal_TableProfiles | None = None,
) -> tuple[Table, ...]:
    filtered: list[Table] = []
    for index, table in enumerate(tables):
        profile = internal_table_profile(table, profiles)
        tokens = profile.tokens
        if not table.metadata and 0 < len(tokens) <= 8 and profile.token_set <= {"b", "i"}:
            continue
        if len(tables) < 2:
            filtered.append(table)
            continue
        if (
            table.metadata.get("source") == "chart-ocr"
            and table.metadata.get("synthetic")
            and 0 < len(tokens) <= 24
            and any(
                other_index != index
                and internal_table_token_coverage(
                    table,
                    other,
                    candidate_profile=profile,
                    reference_profile=internal_table_profile(other, profiles),
                )
                >= 0.95
                for other_index, other in enumerate(tables)
            )
        ):
            continue
        filtered.append(table)
    return tuple(filtered)


def internal_remove_table_duplicate_blocks(
    blocks: list[Block],
    tables: tuple[Table, ...],
    *,
    profiles: internal_TableProfiles | None = None,
) -> list[Block]:
    if not blocks or not tables:
        return blocks
    table_boxes = [
        (table.bbox, internal_table_profile(table, profiles).token_set)
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
    *,
    profiles: internal_TableProfiles | None = None,
) -> list[Block]:
    if not blocks or not tables:
        return blocks
    table_token_counts: Counter[str] = Counter()
    for table in tables:
        table_token_counts.update(internal_table_profile(table, profiles).token_counts)
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
    profiles = {id(table): internal_build_table_profile(table) for table in parsed_tables}
    tables = internal_remove_duplicate_tables(
        internal_remove_block_duplicate_tables(blocks, parsed_tables, profiles=profiles),
        profiles=profiles,
    )
    text_blocks = internal_remove_table_duplicate_blocks(blocks, tables, profiles=profiles)
    text_blocks = internal_remove_tiny_table_duplicate_blocks(
        text_blocks, tables, profiles=profiles
    )
    tables = internal_remove_block_duplicate_table_rows(text_blocks, tables)
    return text_blocks, tables

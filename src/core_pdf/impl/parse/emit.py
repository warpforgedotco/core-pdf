# SPDX-License-Identifier: AGPL-3.0-only
"""Normalize text and produce the final ParsedPage."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from statistics import fmean
from typing import Any

from core_pdf.impl.layout.spatial import (
    SpatialIndex,
)
from core_pdf.impl.model.geometry import (
    horizontal_overlap_ratio,
    interval_overlap,
    overlap_ratio_min,
    overlap_ratio_of,
    rect_tuple,
)
from core_pdf.impl.parse.layout import (
    internal_has_repeated_block_columns,
    layout_element_order,
)
from core_pdf.impl.parse.model import (
    ParsedLine,
    ParsedPage,
)
from core_pdf.impl.parse.tables import (
    internal_character_spaced_cell,
    internal_structured_stream_table,
    internal_table_has_grid_shape,
)
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing
from core_pdf.impl.structured.model import (
    Block,
    BlockKind,
    Diagnostic,
    Figure,
    Page,
    Table,
    TableCell,
    TextLine,
)
from core_pdf.impl.text import collapse_character_spaced


def internal_caption_for(
    caption_blocks: tuple[Block, ...],
    target_bbox: tuple[float, float, float, float] | None,
) -> Block | None:
    if target_bbox is None:
        return None
    candidates: list[tuple[float, Block]] = []
    for caption in caption_blocks:
        if caption.bbox is None or horizontal_overlap_ratio(caption.bbox, target_bbox) < 0.3:
            continue
        if caption.bbox[3] <= target_bbox[1]:
            gap = target_bbox[1] - caption.bbox[3]
        elif target_bbox[3] <= caption.bbox[1]:
            gap = caption.bbox[1] - target_bbox[3]
        else:
            continue
        caption_height = max(1.0, caption.bbox[3] - caption.bbox[1])
        if gap <= max(24.0, caption_height * 2.5):
            candidates.append((gap, caption))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def internal_attach_semantic_context(
    blocks: tuple[Block, ...],
    tables: list[Table],
    figures: list[Figure],
) -> tuple[list[Table], list[Figure]]:
    captions = tuple(block for block in blocks if block.kind is BlockKind.CAPTION)
    headings = tuple(block for block in blocks if block.kind is BlockKind.HEADING)

    def context(order: int, bbox: tuple[float, float, float, float] | None) -> dict[str, object]:
        metadata: dict[str, object] = {}
        caption = internal_caption_for(captions, bbox)
        if caption is not None:
            metadata["caption"] = caption.text
            metadata["caption_order"] = caption.order
        preceding = [
            heading
            for heading in headings
            if heading.order < order
            or (bbox is not None and heading.bbox is not None and heading.bbox[1] >= bbox[3])
        ]
        if preceding:
            heading = min(
                preceding,
                key=lambda item: (
                    abs((item.bbox or (0.0, 0.0, 0.0, 0.0))[1] - (bbox or (0.0, 0.0, 0.0, 0.0))[3]),
                    -item.order,
                ),
            )
            metadata["section"] = heading.text
            metadata["section_level"] = heading.level or 1
        return metadata

    tables = [
        replace(table, metadata={**table.metadata, **context(table.order, table.bbox)})
        for table in tables
    ]
    figures = [
        replace(figure, metadata={**figure.metadata, **context(figure.order, figure.bbox)})
        for figure in figures
    ]
    return tables, figures


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


internal_WELL_FORMED_NUMBER_RE = re.compile(r"^[+-]?\d+(?:[.,:/]\d+)*%?$")


def internal_symbol_characters(text: str) -> int:
    """Count punctuation that is not part of a well-formed number.

    The point in ``79.4`` is no more a symbol than the digits around it, and
    counting it made a table of decimals look like symbol soup -- the
    signature this module uses for a damaged text layer -- so numeric tables
    were deleted as corruption.

    The exemption is granted per token rather than per character: a damaged
    layer emits digits and punctuation interleaved (``1911*2.1,z,z``), where
    a separator happens to fall between two digits without the token being a
    number. Requiring the whole token to parse as one keeps that corruption
    visible.
    """
    symbols = 0
    for token in text.split():
        if internal_WELL_FORMED_NUMBER_RE.match(token):
            continue
        symbols += sum(1 for character in token if not character.isalnum())
    return symbols


def internal_corrupt_native_block(block: Block) -> bool:
    if "native" not in block.provenance:
        return False
    text = block.text
    tokens = internal_emitted_text_tokens(text)
    token_count = len(tokens)
    if token_count >= 24:
        wordlike = sum(internal_wordlike_token(token) for token in tokens)
        if wordlike / token_count >= 0.12:
            return False
    # One pass over the text collects every per-character count; separate
    # comprehensions per statistic walked the block text five times.
    is_ascii = text.isascii()
    nonspace_count = 0
    alphabetic = 0
    non_latin_alphabetic = 0
    alphanumeric = 0
    non_ascii = 0
    for character in text:
        if character.isspace():
            continue
        nonspace_count += 1
        if character.isalpha():
            alphabetic += 1
            if not is_ascii:
                if not ("a" <= character.casefold() <= "z"):
                    non_latin_alphabetic += 1
                if ord(character) > 127:
                    non_ascii += 1
            alphanumeric += 1
        else:
            if character.isalnum():
                alphanumeric += 1
            if not is_ascii and ord(character) > 127:
                non_ascii += 1
    if not nonspace_count:
        return False
    if non_latin_alphabetic and non_latin_alphabetic / alphabetic >= 0.50:
        return False
    if not alphanumeric:
        # A native block with no alphanumeric content is pure punctuation or
        # symbols.  Blocks made up solely of symbols/marks are semantic
        # emptiness -- isolated Braille glyphs, stray combining marks that
        # lost their base, ornaments, column rules -- and almost never appear
        # in the reference text; this mirrors `internal_corrupt_ocr_block`,
        # which drops short symbol-only OCR blocks.  Pure letter-like
        # punctuation (e.g. CJK fullwidth brackets "（）") and longer symbol
        # runs (which may be diagrams or math notation) are preserved.
        return nonspace_count <= 4 and any(
            unicodedata.category(character)[0] in ("S", "M")
            for character in text
            if not character.isspace()
        )
    symbol_ratio = internal_symbol_characters(text) / nonspace_count
    non_ascii_ratio = non_ascii / nonspace_count
    if token_count < 24:
        wordlike = sum(internal_wordlike_token(token) for token in tokens)
        # A compact row with at least one ordinary word per three tokens has
        # enough semantic evidence to preserve. PDF Reference 1.7 Table 3.20's
        # "1–2 Reserved; must be 0." meets that bar; longer mixed-case mojibake
        # fragments do not receive this narrow exemption.
        if token_count <= 12 and wordlike * 3 >= token_count:
            return False
    digit_bearing = sum(any(character.isdigit() for character in token) for token in tokens)
    if token_count < 24:
        return (
            wordlike == 0
            and (symbol_ratio > 0.30 or non_ascii_ratio > 0.10)
            or non_ascii_ratio > 0.02
            and symbol_ratio > 0.10
            and digit_bearing / max(1, token_count) >= 0.30
        )
    if digit_bearing / token_count < 0.35:
        return False
    return symbol_ratio > 0.25 or non_ascii_ratio > 0.02


def internal_corrupt_ocr_block(block: Block) -> bool:
    if block.provenance != ("ocr",):
        return False
    text = block.text.strip()
    if not text:
        return True
    nonspace = [character for character in text if not character.isspace()]
    return len(nonspace) <= 2 and not any(character.isalnum() for character in nonspace)


def internal_remove_corrupt_native_blocks(blocks: list[Block]) -> list[Block]:
    return [
        block
        for block in blocks
        if not internal_corrupt_native_block(block) and not internal_corrupt_ocr_block(block)
    ]


def internal_block_inside_page(block: Block, width: float, height: float) -> bool:
    if block.bbox is None:
        return True
    x0, y0, x1, y1 = block.bbox
    return min(width, x1) > max(0.0, x0) and min(height, y1) > max(0.0, y0)


def internal_remove_off_page_blocks(
    blocks: list[Block], width: float, height: float
) -> list[Block]:
    return [block for block in blocks if internal_block_inside_page(block, width, height)]


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


internal_ARABIC_INDIC_DIGITS = str.maketrans(
    {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)


internal_NUMERIC_PIPE_TOKEN = re.compile(r"^[($+-]?\d+(?:[.,]\d+)?[%)]?$")
internal_STANDALONE_ARTIFACT_TOKENS = frozenset({"]", "_", "□", "☐", "☒", "❖"})
internal_ARTIFACT_PROBE_TOKENS = (*internal_STANDALONE_ARTIFACT_TOKENS, ";", "�")
internal_BRACKET_JOIN_RE = re.compile(r"(?<=[0-9A-Za-z])\[(?=[0-9A-Za-z])")
internal_EXCLAMATION_NOISE_RE = re.compile(r"(?<=[%([])!(?=\s|$)")
internal_LINE_INITIAL_OCR_SUFFIX_FRAGMENTS = frozenset(
    {
        "able",
        "ating",
        "ducted",
        "ence",
        "ical",
        "ing",
        "lation",
        "ment",
        "ments",
        "tion",
        "tions",
        "ture",
    }
)


def internal_numeric_pipe_token(token: str) -> bool:
    return bool(internal_NUMERIC_PIPE_TOKEN.match(token.strip()))


def internal_wordlike_pipe_token(token: str) -> bool:
    letters = [character for character in token.casefold() if character.isalpha()]
    return len(letters) >= 3 and any(character in "aeiou" for character in letters)


def internal_ocr_artifact_token(token: str, line_tokens: list[str]) -> bool:
    if token in {"'", "[", "!"}:
        return True
    if (
        len(line_tokens) <= 2
        and len(token) == 2
        and token.startswith("0")
        and token[1].isdigit()
        and not any(internal_wordlike_pipe_token(line_token) for line_token in line_tokens)
    ):
        return True
    return token == "•" and not any(
        internal_wordlike_pipe_token(line_token) for line_token in line_tokens
    )


def internal_remove_line_initial_suffix_fragments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        tokens = line.split()
        if (
            len(tokens) >= 2
            and tokens[0].casefold() in internal_LINE_INITIAL_OCR_SUFFIX_FRAGMENTS
            and any(internal_wordlike_pipe_token(token) for token in tokens[1:3])
        ):
            tokens = tokens[1:]
            lines.append(" ".join(tokens))
            continue
        lines.append(line)
    return "\n".join(lines)


def internal_remove_sparse_ocr_artifacts(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        tokens = line.split()
        original_tokens = tokens
        if len(tokens) == 2 and tokens[0] == ">" and re.fullmatch(r"\d+(?:[.,]\d+)?", tokens[1]):
            lines.append(tokens[1])
            continue
        artifact_flags = [internal_ocr_artifact_token(token, tokens) for token in tokens]
        if any(artifact_flags):
            tokens = [
                token
                for token, is_artifact in zip(tokens, artifact_flags, strict=True)
                if not is_artifact
            ]
        if "|" not in tokens:
            lines.append(" ".join(tokens) if tokens != original_tokens else line)
            continue
        non_pipe_tokens = [token for token in tokens if token != "|"]
        if not non_pipe_tokens:
            continue
        if (
            all(internal_numeric_pipe_token(token) for token in non_pipe_tokens)
            or sum(internal_wordlike_pipe_token(token) for token in non_pipe_tokens) <= 1
        ):
            lines.append(" ".join(non_pipe_tokens))
            continue
        lines.append(line)
    return "\n".join(lines)


def internal_remove_standalone_artifact_tokens(text: str) -> str:
    return "\n".join(
        " ".join(token for token in line.split() if not internal_standalone_artifact_token(token))
        for line in text.splitlines()
    )


def internal_remove_nonword_bullet_lines(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        tokens = line.split()
        if "•" not in tokens or any(internal_wordlike_pipe_token(token) for token in tokens):
            lines.append(line)
            continue
        lines.append(" ".join(token for token in tokens if token != "•"))
    return "\n".join(lines)


def internal_standalone_artifact_token(token: str) -> bool:
    if token in internal_STANDALONE_ARTIFACT_TOKENS:
        return True
    if token == '"':
        return True
    if "�" in token:
        return True
    return ";" in token and not any(character.isalnum() for character in token)


def internal_normalize_latin_confusables(text: str) -> str:
    if not text:
        return text
    if any(token in text for token in internal_ARTIFACT_PROBE_TOKENS):
        text = internal_remove_standalone_artifact_tokens(text)
    if "•" in text:
        text = internal_remove_nonword_bullet_lines(text)
    if any(fragment in text for fragment in internal_LINE_INITIAL_OCR_SUFFIX_FRAGMENTS):
        text = internal_remove_line_initial_suffix_fragments(text)
    # The test only asks whether at least three Latin letters are present, so stop
    # there rather than folding every character in the line.  ASCII letters answer
    # themselves without folding, which is the overwhelmingly common case.
    latin_letters = 0
    for character in text:
        if "a" <= character <= "z" or "A" <= character <= "Z" or "a" <= character.casefold() <= "z":
            latin_letters += 1
            if latin_letters == 3:
                break
    if latin_letters < 3:
        return text
    normalized = text.translate(internal_ARABIC_INDIC_DIGITS).replace("؛", "")
    normalized = re.sub(
        r"(?<=[0-9A-Za-z])Η(?=[0-9A-Za-z])",
        "H",
        normalized,
    )
    return normalized


def internal_normalize_intrusive_punctuation(text: str) -> str:
    if not text or not any(character in text for character in "!["):
        return text
    normalized = internal_BRACKET_JOIN_RE.sub("", text) if text.count("[") == 1 else text
    normalized = internal_EXCLAMATION_NOISE_RE.sub("", normalized)
    return normalized


def internal_collapse_character_spaced_line(text: str) -> str:
    """Repair a native line whose glyph spacing was mistaken for word spacing."""
    return collapse_character_spaced(text, min_tokens=20, single_char_ratio=0.75)


def internal_normalize_emitted_text(text: str, source: str) -> str:
    if source == "native":
        text = internal_collapse_character_spaced_line(text)
    normalized = internal_normalize_latin_confusables(text)
    normalized = internal_normalize_intrusive_punctuation(normalized)
    if source == "native" and '"' in normalized:
        normalized = internal_remove_standalone_artifact_tokens(normalized)
    if source == "ocr":
        normalized = internal_remove_sparse_ocr_artifacts(normalized)
    return normalized


def internal_line_decoration_flags(
    line: ParsedLine,
    drawings: tuple[Any, ...],
    *,
    decoration_index: SpatialIndex[tuple[float, float, float, float]] | None = None,
) -> dict[str, bool]:
    """Infer simple text decorations from nearby, thin PDF paths."""
    if line.bbox is None:
        return {}
    x0, y0, x1, y1 = line.bbox
    line_height = max(1.0, y1 - y0)
    flags = {"underline": False, "strikeout": False}
    query = (x0, y0 - 3.0, x1, y0 + line_height * 0.75)
    if decoration_index is None:
        candidates = (
            bbox
            for drawing in drawings
            if (bbox := internal_line_decoration_bbox(drawing)) is not None
            and getattr(drawing, "kind", None) in {"fill", "fillstroke", "stroke"}
        )
    else:
        candidates = (hit.item for hit in decoration_index.candidate_hits(query))
    for bbox in candidates:
        dx0, dy0, dx1, dy1 = bbox
        width = dx1 - dx0
        height = dy1 - dy0
        if width < 2.0 or height > 2.5:
            continue
        overlap = interval_overlap(x0, x1, dx0, dx1) / width
        if overlap < 0.75:
            continue
        center_y = (dy0 + dy1) * 0.5
        if y0 - 3.0 <= center_y <= y0 + 1.5:
            flags["underline"] = True
        elif y0 + line_height * 0.25 <= center_y <= y0 + line_height * 0.75:
            flags["strikeout"] = True
        if flags["underline"] and flags["strikeout"]:
            break
    return flags


def internal_line_decoration_bbox(drawing: Any) -> tuple[float, float, float, float] | None:
    """Return a drawing bbox, materializing path geometry at most once."""
    bbox = getattr(drawing, "bbox", None)
    if bbox is None:
        rect = getattr(drawing, "rect", None)
        bbox = rect
        if bbox is None:
            path = getattr(drawing, "path", None)
            bbox_method = getattr(path, "bbox", None)
            bbox = bbox_method() if callable(bbox_method) else None
    return rect_tuple(bbox)


def internal_line_decoration_index(
    drawings: tuple[Any, ...],
) -> SpatialIndex[tuple[float, float, float, float]]:
    """Build the broad-phase index for thin path decoration candidates once."""
    entries = []
    for drawing in drawings:
        if getattr(drawing, "kind", None) not in {"fill", "fillstroke", "stroke"}:
            continue
        bbox = internal_line_decoration_bbox(drawing)
        if bbox is None:
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width >= 2.0 and height <= 2.5:
            entries.append((bbox, bbox))
    return SpatialIndex(entries, target_cell_count=max(64, len(entries) // 8 or 1))


def internal_remove_soft_line_end_hyphens(lines: list[str]) -> list[str]:
    if len(lines) < 2:
        return lines
    cleaned = list(lines)
    for index, text in enumerate(lines[:-1]):
        current = text.rstrip()
        next_text = lines[index + 1].lstrip()
        if (
            current.endswith("-")
            and len(current) >= 3
            and current[-2].islower()
            and next_text[:1].islower()
        ):
            cleaned[index] = f"{current[:-1]}{text[len(current) :]}"
    return cleaned


def internal_normalized_blocks(
    parsed: ParsedPage,
    drawings: tuple[CapturedDrawing, ...],
) -> list[Block]:
    """Build the normalized text candidate projection from parsed lines."""
    blocks: list[Block] = []
    decoration_index = internal_line_decoration_index(drawings)
    for index, parsed_block in enumerate(parsed.blocks):
        confidences = tuple(
            line.confidence
            for line in parsed_block.lines
            if line.confidence is not None and math.isfinite(line.confidence)
        )
        sources = tuple(dict.fromkeys(line.source for line in parsed_block.lines))
        normalized_line_texts = internal_remove_soft_line_end_hyphens(
            [internal_normalize_emitted_text(line.text, line.source) for line in parsed_block.lines]
        )
        decorated_lines: list[ParsedLine] = []
        for line in parsed_block.lines:
            flags = internal_line_decoration_flags(
                line,
                drawings,
                decoration_index=decoration_index,
            )
            decorated_lines.append(
                replace(
                    line,
                    underline=flags["underline"],
                    strikeout=flags["strikeout"],
                )
            )
        blocks.append(
            Block(
                order=index,
                kind=BlockKind(parsed_block.kind),
                lines=tuple(
                    TextLine(
                        text,
                        bbox=line.bbox,
                        source=line.source,
                        confidence=line.confidence,
                        contributing_sources=(line.source,),
                        bold=line.bold,
                        italic=line.italic,
                        underline=line.underline,
                        strikeout=line.strikeout,
                        mark=line.mark,
                        superscript=line.superscript,
                        subscript=line.subscript,
                        spans=line.spans,
                        words=line.words,
                    )
                    for line, text in zip(decorated_lines, normalized_line_texts, strict=True)
                ),
                bbox=parsed_block.bbox,
                column_index=parsed_block.column_index,
                rotation=(parsed_block.lines[0].rotation if parsed_block.lines else 0),
                confidence=(fmean(confidences) if confidences else None),
                level=parsed_block.level,
                provenance=sources,
            )
        )
    return blocks


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


def assemble_page(
    parsed: ParsedPage,
    drawings: tuple[CapturedDrawing, ...] = (),
) -> Page:
    blocks = internal_normalized_blocks(parsed, drawings)
    blocks = internal_remove_off_page_blocks(
        internal_remove_corrupt_native_blocks(blocks),
        parsed.width,
        parsed.height,
    )
    blocks, tables = internal_project_text_and_tables(blocks, parsed.tables)
    elements: list[tuple[str, object, tuple[float, float, float, float]]] = [
        ("block", block, block.bbox or (0.0, 0.0, 0.0, 0.0)) for block in blocks
    ]
    elements.extend(("table", table, table.bbox or (0.0, 0.0, 0.0, 0.0)) for table in tables)
    elements.extend(
        ("figure", figure, figure.bbox or (0.0, 0.0, 0.0, 0.0)) for figure in parsed.figures
    )
    ordered_blocks: list[Block] = []
    ordered_tables: list[Table] = []
    ordered_figures: list[Figure] = []
    element_boxes = tuple(item[2] for item in elements)
    if (
        parsed.full_page_image
        and len(element_boxes) > 1
        and internal_has_repeated_block_columns(parsed.blocks)
    ):
        element_order = tuple(
            sorted(
                range(len(element_boxes)),
                key=lambda index: (-element_boxes[index][3], element_boxes[index][0]),
            )
        )
    else:
        element_order = layout_element_order(
            element_boxes, parsed.rotation, parsed.width, parsed.height
        )
    for order, index in enumerate(element_order):
        kind, element, internal_bbox = elements[index]
        if kind == "block":
            assert isinstance(element, Block)
            ordered_blocks.append(replace(element, order=order))
        elif kind == "table":
            assert isinstance(element, Table)
            ordered_tables.append(replace(element, order=order))
        else:
            assert isinstance(element, Figure)
            ordered_figures.append(replace(element, order=order))
    ordered_tables, ordered_figures = internal_attach_semantic_context(
        tuple(ordered_blocks), ordered_tables, ordered_figures
    )
    header_parts = [
        block.text
        for block in ordered_blocks
        if block.bbox is not None
        and block.bbox[3] >= parsed.height * 0.88
        and block.bbox[3] - block.bbox[1] <= parsed.height * 0.08
        and len(block.text) <= 240
    ]
    footer_parts = [
        block.text
        for block in ordered_blocks
        if block.bbox is not None
        and block.bbox[1] <= parsed.height * 0.12
        and block.bbox[3] - block.bbox[1] <= parsed.height * 0.08
        and len(block.text) <= 240
    ]
    return Page(
        page_number=parsed.page_number,
        width=parsed.width,
        height=parsed.height,
        rotation=parsed.rotation,
        blocks=tuple(ordered_blocks),
        page_class=parsed.route.value,
        base_route=parsed.route.value,
        tables=tuple(ordered_tables),
        figures=tuple(ordered_figures),
        header="\n".join(header_parts),
        footer="\n".join(footer_parts),
        diagnostics=tuple(
            Diagnostic(
                code=message,
                message=(
                    "Reading order is ambiguous because differently rotated text shares "
                    "one layout block."
                    if message == "reading-order-ambiguous"
                    else message
                ),
                page_number=parsed.page_number,
            )
            for message in parsed.diagnostics
        ),
    )

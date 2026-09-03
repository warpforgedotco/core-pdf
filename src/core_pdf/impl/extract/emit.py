# SPDX-License-Identifier: AGPL-3.0-only
"""Normalize extracted products and produce the final page."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import replace
from statistics import fmean
from typing import Any

from core_pdf.impl.extract.block_layout import (
    internal_has_repeated_block_columns,
    layout_element_order,
)
from core_pdf.impl.extract.contracts import (
    PageRoute,
    ParsedBlock,
    ParsedLine,
)
from core_pdf.impl.extract.table_reconcile import (
    internal_emitted_text_tokens,
    internal_project_text_and_tables,
    internal_wordlike_token,
)
from core_pdf.impl.model.geometry import horizontal_overlap_ratio, interval_overlap, rect_tuple
from core_pdf.impl.output import (
    Block,
    BlockKind,
    Diagnostic,
    Figure,
    Page,
    Table,
    TextLine,
)
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing
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
    decoration_boxes: tuple[tuple[float, float, float, float], ...] | None = None,
) -> dict[str, bool]:
    """Infer simple text decorations from nearby, thin PDF paths."""
    if line.bbox is None:
        return {}
    x0, y0, x1, y1 = line.bbox
    line_height = max(1.0, y1 - y0)
    flags = {"underline": False, "strikeout": False}
    candidates = decoration_boxes
    if candidates is None:
        candidates = tuple(
            bbox
            for drawing in drawings
            if getattr(drawing, "kind", None) in {"fill", "fillstroke", "stroke"}
            and (bbox := internal_line_decoration_bbox(drawing)) is not None
        )
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
    parsed_blocks: tuple[ParsedBlock, ...],
    drawings: tuple[CapturedDrawing, ...],
) -> list[Block]:
    """Build the normalized text candidate projection from parsed lines."""
    decoration_boxes = tuple(
        bbox
        for drawing in drawings
        if getattr(drawing, "kind", None) in {"fill", "fillstroke", "stroke"}
        and (bbox := internal_line_decoration_bbox(drawing)) is not None
    )
    blocks: list[Block] = []
    for index, parsed_block in enumerate(parsed_blocks):
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
                decoration_boxes=decoration_boxes,
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


def assemble_page(
    blocks: tuple[ParsedBlock, ...],
    *,
    page_number: int,
    width: float,
    height: float,
    rotation: int,
    route: PageRoute,
    tables: tuple[Table, ...] = (),
    figures: tuple[Figure, ...] = (),
    diagnostics: tuple[str, ...] = (),
    full_page_image: bool = False,
    drawings: tuple[CapturedDrawing, ...] = (),
) -> Page:
    normalized_blocks = internal_normalized_blocks(blocks, drawings)
    normalized_blocks = internal_remove_off_page_blocks(
        internal_remove_corrupt_native_blocks(normalized_blocks),
        width,
        height,
    )
    normalized_blocks, projected_tables = internal_project_text_and_tables(
        normalized_blocks, tables
    )
    elements: list[tuple[str, object, tuple[float, float, float, float]]] = [
        ("block", block, block.bbox or (0.0, 0.0, 0.0, 0.0)) for block in normalized_blocks
    ]
    elements.extend(
        ("table", table, table.bbox or (0.0, 0.0, 0.0, 0.0)) for table in projected_tables
    )
    elements.extend(("figure", figure, figure.bbox or (0.0, 0.0, 0.0, 0.0)) for figure in figures)
    ordered_blocks: list[Block] = []
    ordered_tables: list[Table] = []
    ordered_figures: list[Figure] = []
    element_boxes = tuple(item[2] for item in elements)
    if full_page_image and len(element_boxes) > 1 and internal_has_repeated_block_columns(blocks):
        element_order = tuple(
            sorted(
                range(len(element_boxes)),
                key=lambda index: (-element_boxes[index][3], element_boxes[index][0]),
            )
        )
    else:
        element_order = layout_element_order(element_boxes, rotation, width, height)
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
        and block.bbox[3] >= height * 0.88
        and block.bbox[3] - block.bbox[1] <= height * 0.08
        and len(block.text) <= 240
    ]
    footer_parts = [
        block.text
        for block in ordered_blocks
        if block.bbox is not None
        and block.bbox[1] <= height * 0.12
        and block.bbox[3] - block.bbox[1] <= height * 0.08
        and len(block.text) <= 240
    ]
    return Page(
        page_number=page_number,
        width=width,
        height=height,
        rotation=rotation,
        blocks=tuple(ordered_blocks),
        page_class=route.value,
        base_route=route.value,
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
                page_number=page_number,
            )
            for message in diagnostics
        ),
    )

# SPDX-License-Identifier: AGPL-3.0-only
"""Normalize extracted products and produce the final page."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from statistics import fmean

from core_pdf.impl._impl.extract.block_layout import (
    internal_has_repeated_block_columns,
    layout_element_order,
)
from core_pdf.impl._impl.extract.contracts import (
    ParsedBlock,
    ParsedLine,
)
from core_pdf.impl._impl.extract.table_reconcile import (
    internal_project_text_and_tables,
)
from core_pdf.impl._impl.model.geometry import (
    horizontal_overlap_ratio,
    interval_overlap,
    rect_tuple,
)
from core_pdf.impl._impl.model.text import collapse_character_spaced
from core_pdf.impl._impl.output.model import (
    Block,
    BlockKind,
    Diagnostic,
    Figure,
    Page,
    Table,
    TextLine,
)
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing


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


def internal_block_inside_page(block: Block, width: float, height: float) -> bool:
    if block.bbox is None:
        return True
    x0, y0, x1, y1 = block.bbox
    return min(width, x1) > max(0.0, x0) and min(height, y1) > max(0.0, y0)


def internal_remove_off_page_blocks(
    blocks: list[Block], width: float, height: float
) -> list[Block]:
    return [block for block in blocks if internal_block_inside_page(block, width, height)]


def internal_collapse_character_spaced_line(text: str) -> str:
    """Repair a native line whose glyph spacing was mistaken for word spacing."""
    return collapse_character_spaced(text, min_tokens=20, single_char_ratio=0.75)


def internal_normalize_emitted_text(text: str, source: str) -> str:
    # Emission has no evidence that a decoded word, operator, or symbol is
    # spurious. Keep text intact apart from physical glyph-spacing repair.
    if source == "native":
        return internal_collapse_character_spaced_line(text)
    return text


def internal_line_decoration_flags(
    line: ParsedLine,
    drawings: tuple[CapturedDrawing, ...],
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
            if drawing.kind in {"fill", "fillstroke", "stroke"}
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


def internal_line_decoration_bbox(
    drawing: CapturedDrawing,
) -> tuple[float, float, float, float] | None:
    """Return a drawing bbox, materializing path geometry at most once."""
    bbox = drawing.bbox if drawing.bbox is not None else drawing.rect
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
    *,
    normalize_text: Callable[[str, str], str] = internal_normalize_emitted_text,
) -> list[Block]:
    """Build the normalized text candidate projection from parsed lines."""
    decoration_boxes = tuple(
        bbox
        for drawing in drawings
        if drawing.kind in {"fill", "fillstroke", "stroke"}
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
            [normalize_text(line.text, line.source) for line in parsed_block.lines]
        )
        lines: list[TextLine] = []
        for line, text in zip(parsed_block.lines, normalized_line_texts, strict=True):
            flags = internal_line_decoration_flags(
                line,
                drawings,
                decoration_boxes=decoration_boxes,
            )
            lines.append(
                TextLine(
                    text,
                    bbox=line.bbox,
                    source=line.source,
                    confidence=line.confidence,
                    contributing_sources=(line.source,),
                    bold=line.bold,
                    italic=line.italic,
                    underline=flags["underline"],
                    strikeout=flags["strikeout"],
                    mark=line.mark,
                    superscript=line.superscript,
                    subscript=line.subscript,
                    spans=line.spans,
                    words=line.words,
                )
            )
        blocks.append(
            Block(
                order=index,
                kind=BlockKind(parsed_block.kind),
                lines=tuple(lines),
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
    route: str,
    tables: tuple[Table, ...] = (),
    figures: tuple[Figure, ...] = (),
    diagnostics: tuple[str, ...] = (),
    full_page_image: bool = False,
    drawings: tuple[CapturedDrawing, ...] = (),
) -> Page:
    normalized_blocks = internal_normalized_blocks(blocks, drawings)
    normalized_blocks = internal_remove_off_page_blocks(
        normalized_blocks,
        width,
        height,
    )
    normalized_blocks, projected_tables = internal_project_text_and_tables(
        normalized_blocks, tables
    )
    return internal_compose_page(
        blocks,
        normalized_blocks,
        projected_tables,
        page_number=page_number,
        width=width,
        height=height,
        rotation=rotation,
        route=route,
        figures=figures,
        diagnostics=diagnostics,
        full_page_image=full_page_image,
    )


def internal_compose_page(
    blocks: tuple[ParsedBlock, ...],
    normalized_blocks: list[Block],
    projected_tables: tuple[Table, ...],
    *,
    page_number: int,
    width: float,
    height: float,
    rotation: int,
    route: str,
    figures: tuple[Figure, ...] = (),
    diagnostics: tuple[str, ...] = (),
    full_page_image: bool = False,
) -> Page:
    """Order projected products and attach semantic page context."""
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
        page_class=route,
        base_route=route,
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

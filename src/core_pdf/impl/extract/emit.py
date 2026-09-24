# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from copy import replace
from statistics import fmean
from typing import Any, ClassVar

import numpy

from core_pdf.impl.capture.records import CapturedDrawing
from core_pdf.impl.extract.block_layout import (
    has_repeated_block_columns,
    layout_element_order,
)
from core_pdf.impl.extract.contracts import ParsedBlock
from core_pdf.impl.extract.table_cleanup import table_with_bands
from core_pdf.impl.geometry import (
    bbox_area,
    bbox_intersects,
    bbox_union,
    horizontal_overlap_ratio,
    interval_overlap,
    overlap_ratio_min,
    overlap_ratio_of,
    rect_tuple,
)
from core_pdf.impl.output.model import (
    Block,
    BlockKind,
    Diagnostic,
    Figure,
    Page,
    Table,
    TableCell,
    TextLine,
)
from core_pdf.impl.text import collapse_ws, complete_text_covered, content_tokens
from core_pdf.impl.types import Record, Rectangle, frozen_setattr


def caption_for(
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


def attach_semantic_context(
    blocks: tuple[Block, ...],
    tables: list[Table],
    figures: list[Figure],
) -> tuple[list[Table], list[Figure]]:
    captions = tuple(block for block in blocks if block.kind is BlockKind.CAPTION)
    headings = tuple(block for block in blocks if block.kind is BlockKind.HEADING)

    def context(order: int, bbox: tuple[float, float, float, float] | None) -> dict[str, object]:
        metadata: dict[str, object] = {}
        caption = caption_for(captions, bbox)
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


def block_inside_page(block: Block, width: float, height: float) -> bool:
    if block.bbox is None:
        return True
    return bbox_intersects(block.bbox, (0.0, 0.0, width, height))


def remove_off_page_blocks(blocks: list[Block], width: float, height: float) -> list[Block]:
    return [block for block in blocks if block_inside_page(block, width, height)]


def line_decoration_flags(
    line: TextLine,
    decoration_boxes: tuple[tuple[float, float, float], ...],
    decoration_centers: tuple[float, ...],
) -> dict[str, bool]:
    if line.bbox is None:
        return {}
    x0, y0, x1, y1 = line.bbox
    line_height = max(1.0, y1 - y0)
    flags = {"underline": False, "strikeout": False}
    # Only a rule centred in the union of the underline and strikeout bands
    # below can set either flag, so bisect to that slice rather than scanning
    # every rule on the page. Drawing-heavy pages carry thousands of them, and
    # the scan was the dominant cost of assembling such a page.
    first = bisect_left(decoration_centers, y0 - 3.0)
    last = bisect_right(decoration_centers, y0 + max(1.5, line_height * 0.75))
    for dx0, dx1, center_y in decoration_boxes[first:last]:
        if interval_overlap(x0, x1, dx0, dx1) / (dx1 - dx0) < 0.75:
            continue
        if y0 - 3.0 <= center_y <= y0 + 1.5:
            flags["underline"] = True
        elif y0 + line_height * 0.25 <= center_y <= y0 + line_height * 0.75:
            flags["strikeout"] = True
        if flags["underline"] and flags["strikeout"]:
            break
    return flags


def line_decoration_bbox(
    drawing: CapturedDrawing,
) -> tuple[float, float, float, float] | None:
    bbox = drawing.bbox if drawing.bbox is not None else drawing.rect
    return rect_tuple(bbox)


def normalize_blocks(
    parsed_blocks: tuple[ParsedBlock, ...],
    drawings: tuple[CapturedDrawing, ...],
) -> list[Block]:
    # Sorted by centre so line_decoration_flags can bisect. The flags are
    # order-independent, so ordering the rules changes nothing it reports.
    decoration_boxes = tuple(
        sorted(
            (
                (bbox[0], bbox[2], (bbox[1] + bbox[3]) * 0.5)
                for drawing in drawings
                if drawing.kind in {"fill", "fillstroke", "stroke"}
                and (bbox := line_decoration_bbox(drawing)) is not None
                and bbox[2] - bbox[0] >= 2.0
                and bbox[3] - bbox[1] <= 2.5
            ),
            key=lambda box: box[2],
        )
    )
    decoration_centers = tuple(box[2] for box in decoration_boxes)
    blocks: list[Block] = []
    for index, parsed_block in enumerate(parsed_blocks):
        confidences = tuple(
            parsed.line.confidence
            for parsed in parsed_block.lines
            if parsed.line.confidence is not None and math.isfinite(parsed.line.confidence)
        )
        sources = tuple(dict.fromkeys(parsed.line.source for parsed in parsed_block.lines))
        lines: list[TextLine] = []
        for parsed in parsed_block.lines:
            flags = line_decoration_flags(parsed.line, decoration_boxes, decoration_centers)
            lines.append(
                replace(
                    parsed.line,
                    contributing_sources=(parsed.line.source,),
                    underline=flags["underline"],
                    strikeout=flags["strikeout"],
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
    normalized_blocks = normalize_blocks(blocks, drawings)
    normalized_blocks = remove_off_page_blocks(
        normalized_blocks,
        width,
        height,
    )
    normalized_blocks, projected_tables = project_text_and_tables(normalized_blocks, tables)
    return compose_page(
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


def compose_page(
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
    if full_page_image and len(element_boxes) > 1 and has_repeated_block_columns(blocks):
        element_order = tuple(
            sorted(
                range(len(element_boxes)),
                key=lambda index: (-element_boxes[index][3], element_boxes[index][0]),
            )
        )
    else:
        element_order = layout_element_order(element_boxes, rotation, width, height)
    for order, index in enumerate(element_order):
        kind, element, bbox = elements[index]
        if kind == "block":
            assert isinstance(element, Block)
            ordered_blocks.append(replace(element, order=order))
        elif kind == "table":
            assert isinstance(element, Table)
            ordered_tables.append(replace(element, order=order))
        else:
            assert isinstance(element, Figure)
            ordered_figures.append(replace(element, order=order))
    ordered_tables, ordered_figures = attach_semantic_context(
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


def remove_block_duplicate_table_rows(
    blocks: list[Block],
    tables: tuple[Table, ...],
) -> tuple[Table, ...]:
    if not blocks or not tables:
        return tables
    line_boxes_by_text: dict[str, list[Rectangle]] = {}
    for block in blocks:
        for line in block.lines:
            box = line.bbox or (block.bbox if len(block.lines) == 1 else None)
            text = collapse_ws(line.text)
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
            text = " ".join(collapse_ws(cell.text) for cell in cells)
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
        projected = table_with_bands(replace(table, rows=rows))
        if table.row_bands:
            projected = replace(
                projected,
                row_bands=tuple(
                    replace(table.row_bands[old_index], index=index)
                    for index, old_index in enumerate(kept_row_indexes)
                ),
            )
        filtered.append(projected)
    return tuple(filtered)


class IndexedRow(Record):
    __slots__ = ("cells", "texts", "tokens", "frame_indexes")

    cells: tuple[TableCell, ...]
    texts: tuple[str, ...]
    tokens: tuple[tuple[str, ...], ...]
    frame_indexes: tuple[int, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("cells", "texts", "tokens", "frame_indexes")
    __match_args__ = ("cells", "texts", "tokens", "frame_indexes")

    def __init__(
        self,
        cells: tuple[TableCell, ...],
        texts: tuple[str, ...],
        tokens: tuple[tuple[str, ...], ...],
        frame_indexes: tuple[int, ...],
    ) -> None:
        frozen_setattr(self, "cells", cells)
        frozen_setattr(self, "texts", texts)
        frozen_setattr(self, "tokens", tokens)
        frozen_setattr(self, "frame_indexes", frame_indexes)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.cells == other.cells
            and self.texts == other.texts
            and self.tokens == other.tokens
            and self.frame_indexes == other.frame_indexes
        )

    def __hash__(self) -> int:
        return hash((self.cells, self.texts, self.tokens, self.frame_indexes))


class TableIndex(Record):
    __slots__ = ("table", "rows", "frame")

    table: Table
    rows: tuple[IndexedRow, ...]
    frame: SpatialFrame | None

    __fields__: ClassVar[tuple[str, ...]] = ("table", "rows", "frame")
    __match_args__ = ("table", "rows", "frame")

    def __init__(
        self,
        table: Table,
        rows: tuple[IndexedRow, ...],
        frame: SpatialFrame | None,
    ) -> None:
        frozen_setattr(self, "table", table)
        frozen_setattr(self, "rows", rows)
        frozen_setattr(self, "frame", frame)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.table == other.table and self.rows == other.rows and self.frame == other.frame

    def __hash__(self) -> int:
        return hash((self.table, self.rows, self.frame))

    @classmethod
    def build(cls, table: Table) -> TableIndex:
        rows: list[IndexedRow] = []
        boxes: list[Rectangle] = []
        for row in table.rows:
            cells = tuple(cell for cell in row if cell.text)
            texts = tuple(collapse_ws(cell.text) for cell in cells)
            indexes: list[int] = []
            for cell in cells:
                if cell.bbox is None:
                    indexes.append(-1)
                else:
                    indexes.append(len(boxes))
                    boxes.append(cell.bbox)
            rows.append(
                IndexedRow(
                    cells,
                    texts,
                    tuple(content_tokens(text) for text in texts),
                    tuple(indexes),
                )
            )
        frame = SpatialFrame.from_boxes(boxes) if boxes else None
        return cls(table, tuple(rows), frame)


def line_duplicates_table(text: str, box: Rectangle, index: TableIndex) -> bool:
    table = index.table
    normalized = collapse_ws(text)
    if not normalized or table.bbox is None or overlap_ratio_of(box, table.bbox) < 0.90:
        return False
    tokens = content_tokens(normalized)
    if index.frame is None:
        touches = covered = numpy.zeros(0, dtype=bool)
    else:
        intersections = index.frame.intersection_areas(box)
        touches = intersections > 0.0
        box_area = bbox_area(box)
        covered = (
            intersections / box_area >= 0.90
            if box_area > 0.0
            else numpy.zeros(len(intersections), dtype=bool)
        )
    participating_cells: list[TableCell] = []
    total_cells = 0
    for row in index.rows:
        total_cells += len(row.cells)
        kept = [
            position
            for position, frame_index in enumerate(row.frame_indexes)
            if frame_index < 0 or touches[frame_index]
        ]
        if not kept:
            continue
        cells = [row.cells[position] for position in kept]
        participating_cells.extend(cells)
        cell_texts = [row.texts[position] for position in kept]
        for position in kept:
            frame_index = row.frame_indexes[position]
            if (
                frame_index >= 0
                and covered[frame_index]
                and complete_text_covered(tokens, row.tokens[position])
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
                if all(cell.bbox is not None for cell in matched_cells):
                    row_box = bbox_union(
                        cell.bbox for cell in matched_cells if cell.bbox is not None
                    )
                    if row_box is not None and overlap_ratio_of(box, row_box) >= 0.90:
                        return True
                elif start == 0 and end + 1 == len(cells) == len(row.cells):
                    return True
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
        and normalized == " ".join(collapse_ws(cell.text) for cell in participating_cells)
    )


def remove_table_duplicate_blocks(
    blocks: list[Block],
    tables: tuple[Table, ...],
) -> list[Block]:
    if not blocks or not tables:
        return blocks
    located_tables = [table for table in tables if table.bbox is not None]
    if not located_tables:
        return blocks
    table_frame = SpatialFrame.from_boxes(
        table.bbox for table in located_tables if table.bbox is not None
    )
    indexes = [TableIndex.build(table) for table in located_tables]
    deduplicated: list[Block] = []
    for block in blocks:
        kept_lines = []
        for line in block.lines:
            box = line.bbox or (block.bbox if len(block.lines) == 1 else None)
            if box is None or not any(
                line_duplicates_table(line.text, box, indexes[int(index)])
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


def project_text_and_tables(
    blocks: list[Block],
    parsed_tables: tuple[Table, ...],
) -> tuple[list[Block], tuple[Table, ...]]:
    text_blocks = remove_table_duplicate_blocks(blocks, parsed_tables)
    projected_tables = remove_block_duplicate_table_rows(text_blocks, parsed_tables)
    return text_blocks, projected_tables


class SpatialFrame(Record):
    __slots__ = ("boxes", "areas")

    boxes: numpy.ndarray[Any, Any]
    areas: numpy.ndarray[Any, Any]

    __fields__: ClassVar[tuple[str, ...]] = ("boxes", "areas")
    __match_args__ = ("boxes", "areas")

    def __init__(self, boxes: numpy.ndarray[Any, Any], areas: numpy.ndarray[Any, Any]) -> None:
        frozen_setattr(self, "boxes", boxes)
        frozen_setattr(self, "areas", areas)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.boxes == other.boxes and self.areas == other.areas

    def __hash__(self) -> int:
        return hash((self.boxes, self.areas))

    @classmethod
    def from_boxes(cls, boxes: Iterable[Rectangle]) -> SpatialFrame:
        packed = numpy.asarray(tuple(boxes), dtype=numpy.float64).reshape((-1, 4))
        widths = numpy.maximum(0.0, packed[:, 2] - packed[:, 0])
        heights = numpy.maximum(0.0, packed[:, 3] - packed[:, 1])
        packed.setflags(write=False)
        areas = widths * heights
        areas.setflags(write=False)
        return cls(packed, areas)

    def intersection_areas(self, box: Rectangle) -> numpy.ndarray[Any, Any]:
        widths = numpy.maximum(
            0.0,
            numpy.minimum(self.boxes[:, 2], box[2]) - numpy.maximum(self.boxes[:, 0], box[0]),
        )
        heights = numpy.maximum(
            0.0,
            numpy.minimum(self.boxes[:, 3], box[3]) - numpy.maximum(self.boxes[:, 1], box[1]),
        )
        return widths * heights

    def overlap_min(self, box: Rectangle) -> numpy.ndarray[Any, Any]:
        box_area = bbox_area(box)
        denominator = numpy.minimum(self.areas, box_area)
        return numpy.divide(
            self.intersection_areas(box),
            denominator,
            out=numpy.zeros_like(denominator),
            where=denominator > 0.0,
        )

    def matching_overlap_min(self, box: Rectangle, minimum: float) -> numpy.ndarray[Any, Any]:
        return numpy.flatnonzero(self.overlap_min(box) >= minimum)


__all__ = ("SpatialFrame",)

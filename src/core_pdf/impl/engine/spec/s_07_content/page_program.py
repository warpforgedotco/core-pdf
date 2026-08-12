# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable program emitted by one page content interpretation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from heapq import merge
from typing import Any

import numpy

from core_pdf.impl.engine.layout.glyphs import GlyphObservation
from core_pdf.impl.engine.layout.models import TextRun
from core_pdf.impl.engine.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLine,
)
from core_pdf.impl.exceptions import PdfContractError


class PageEventKind(IntEnum):
    TEXT = 1
    GLYPH = 2
    DRAWING = 3
    IMAGE = 4
    INLINE_IMAGE = 5


@dataclass(frozen=True, slots=True)
class LineTable:
    """Compact read-only storage for captured grid lines."""

    x0: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    y0: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    x1: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    y1: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    width: numpy.ndarray[Any, numpy.dtype[numpy.float64]]

    @classmethod
    def from_lines(cls, lines: Any) -> LineTable:
        values = tuple(lines)
        table = cls(
            numpy.asarray([line.x0 for line in values], dtype=numpy.float64),
            numpy.asarray([line.y0 for line in values], dtype=numpy.float64),
            numpy.asarray([line.x1 for line in values], dtype=numpy.float64),
            numpy.asarray([line.y1 for line in values], dtype=numpy.float64),
            numpy.asarray([line.line_width for line in values], dtype=numpy.float64),
        )
        for column in (table.x0, table.y0, table.x1, table.y1, table.width):
            internal_readonly(column)
        return table

    def __len__(self) -> int:
        return len(self.x0)

    def __iter__(self) -> Iterator[CapturedLine]:
        return (
            CapturedLine(x0, y0, x1, y1, width)
            for x0, y0, x1, y1, width in zip(self.x0, self.y0, self.x1, self.y1, self.width)
        )

    @property
    def nbytes(self) -> int:
        return sum(column.nbytes for column in (self.x0, self.y0, self.x1, self.y1, self.width))


def internal_readonly(array: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class PageEventStream:
    """A read-only event index over one interpreted page.

    Variable-size payloads remain in the existing typed capture products.  The
    numeric event index is deliberately small and gives all consumers one stable
    ordering and one visibility/geometry representation.
    """

    sequence: numpy.ndarray[Any, numpy.dtype[numpy.int64]]
    kind: numpy.ndarray[Any, numpy.dtype[numpy.uint8]]
    bbox: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    payload: numpy.ndarray[Any, numpy.dtype[numpy.int64]]
    visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
    non_text_indexes: numpy.ndarray[Any, numpy.dtype[numpy.int64]]
    runs: tuple[TextRun, ...]
    glyphs: tuple[GlyphObservation, ...]
    drawings: tuple[CapturedDrawing, ...]
    inline_images: tuple[CapturedInlineImage, ...]
    lines: LineTable

    def __post_init__(self) -> None:
        size = len(self.sequence)
        for column in (self.kind, self.payload, self.visible):
            if len(column) != size:
                raise ValueError("page event columns must have equal lengths")
        if self.bbox.shape != (size, 4):
            raise ValueError("page event bboxes must have shape (n, 4)")
        columns: tuple[numpy.ndarray[Any, Any], ...] = (
            self.sequence,
            self.kind,
            self.bbox,
            self.payload,
            self.visible,
        )
        for column in (*columns, self.non_text_indexes):
            internal_readonly(column)

    @classmethod
    def from_products(cls, products: PageProducts) -> PageEventStream:
        missing_box = (numpy.nan, numpy.nan, numpy.nan, numpy.nan)
        streams = (
            [
                (
                    run.seqno,
                    0,
                    PageEventKind.TEXT,
                    index,
                    (run.x0, run.y0, run.x1, run.y1),
                    run.visible,
                )
                for index, run in enumerate(products.runs)
            ],
            [
                (
                    glyph.seqno,
                    1,
                    PageEventKind.GLYPH,
                    index,
                    glyph.ink_bbox or missing_box,
                    glyph.visible,
                )
                for index, glyph in enumerate(products.glyphs)
                if glyph.has_paint
            ],
            [
                (
                    drawing.seqno,
                    2,
                    PageEventKind.IMAGE if drawing.kind == "image" else PageEventKind.DRAWING,
                    index,
                    (
                        missing_box
                        if drawing.rect is None
                        else (
                            drawing.rect.x0,
                            drawing.rect.y0,
                            drawing.rect.x1,
                            drawing.rect.y1,
                        )
                    ),
                    True,
                )
                for index, drawing in enumerate(products.drawings)
                if drawing.kind != "inline-image"
            ],
            [
                (
                    image.seqno,
                    3,
                    PageEventKind.INLINE_IMAGE,
                    index,
                    missing_box,
                    True,
                )
                for index, image in enumerate(products.inline_images)
            ],
        )
        for stream in streams:
            if any(left[0] > right[0] for left, right in zip(stream, stream[1:], strict=False)):
                stream.sort(key=lambda entry: entry[0])
        size = sum(len(stream) for stream in streams)
        sequence = numpy.empty(size, dtype=numpy.int64)
        kind = numpy.empty(size, dtype=numpy.uint8)
        bbox = numpy.empty((size, 4), dtype=numpy.float64)
        payload = numpy.empty(size, dtype=numpy.int64)
        visible = numpy.empty(size, dtype=numpy.bool_)
        entries = merge(*streams, key=lambda entry: (entry[0], entry[1]))
        for position, entry in enumerate(entries):
            sequence[position] = entry[0]
            kind[position] = int(entry[2])
            payload[position] = entry[3]
            bbox[position] = entry[4]
            visible[position] = entry[5]
        non_text_indexes = numpy.flatnonzero(
            (kind != int(PageEventKind.TEXT)) & (kind != int(PageEventKind.GLYPH))
        ).astype(numpy.int64, copy=False)
        return cls(
            sequence,
            kind,
            bbox,
            payload,
            visible,
            non_text_indexes,
            products.runs,
            products.glyphs,
            products.drawings,
            products.inline_images,
            products.lines,
        )


@dataclass(frozen=True, slots=True)
class PageProducts:
    """Typed payload columns owned by the canonical page program."""

    runs: tuple[TextRun, ...]
    glyphs: tuple[GlyphObservation, ...]
    drawings: tuple[CapturedDrawing, ...]
    inline_images: tuple[CapturedInlineImage, ...]
    lines: LineTable

    @classmethod
    def from_state(cls, state: Any) -> PageProducts:
        runs = tuple(state.runs)
        glyphs = tuple(state.glyphs)
        raw_drawings = tuple(state.drawings)
        if not all(isinstance(drawing, CapturedDrawing) for drawing in raw_drawings):
            raise PdfContractError("page state emitted an invalid drawing product")
        drawings = tuple(drawing for drawing in raw_drawings if drawing.kind != "inline-image")
        inline_images = tuple(state.inline_images)
        if not all(isinstance(run, TextRun) for run in runs):
            raise PdfContractError("page state emitted an invalid text-run product")
        if not all(isinstance(glyph, GlyphObservation) for glyph in glyphs):
            raise PdfContractError("page state emitted an invalid glyph product")
        if not all(isinstance(image, CapturedInlineImage) for image in inline_images):
            raise PdfContractError("page state emitted an invalid inline-image product")
        return cls(
            runs,
            glyphs,
            drawings,
            inline_images,
            LineTable.from_lines(state.lines),
        )


@dataclass(slots=True)
class PageProgram:
    """The canonical ordered program shared by extraction and rendering.

    A page is interpreted exactly once.  Consumers select products from this
    program instead of asking the content interpreter to run again with a
    different capture mode.
    """

    products: PageProducts
    internal_events: PageEventStream | None = field(default=None, init=False, repr=False)

    @property
    def events(self) -> PageEventStream:
        events = self.internal_events
        if events is None:
            events = PageEventStream.from_products(self.products)
            self.internal_events = events
        return events

    @classmethod
    def from_state(cls, state: Any) -> PageProgram:
        products = PageProducts.from_state(state)
        return cls(products)


__all__ = (
    "LineTable",
    "PageEventKind",
    "PageEventStream",
    "PageProducts",
    "PageProgram",
)

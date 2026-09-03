# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable program emitted by one page content interpretation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from heapq import merge
from typing import Any

import numpy

from core_pdf.impl.exceptions import PdfContractError
from core_pdf.impl.model.glyphs import GlyphObservation
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.runtime.array_views import readonly
from core_pdf.impl.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLine,
)


class PageEventKind(IntEnum):
    TEXT = 1
    GLYPH = 2
    DRAWING = 3
    IMAGE = 4
    INLINE_IMAGE = 5


LineColumns = tuple[
    "numpy.ndarray[Any, numpy.dtype[numpy.float64]]",
    "numpy.ndarray[Any, numpy.dtype[numpy.float64]]",
    "numpy.ndarray[Any, numpy.dtype[numpy.float64]]",
    "numpy.ndarray[Any, numpy.dtype[numpy.float64]]",
]


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
        x0_values: list[float] = []
        y0_values: list[float] = []
        x1_values: list[float] = []
        y1_values: list[float] = []
        width_values: list[float] = []
        for line in lines:
            x0_values.append(line.x0)
            y0_values.append(line.y0)
            x1_values.append(line.x1)
            y1_values.append(line.y1)
            width_values.append(line.line_width)
        table = cls(
            numpy.asarray(x0_values, dtype=numpy.float64),
            numpy.asarray(y0_values, dtype=numpy.float64),
            numpy.asarray(x1_values, dtype=numpy.float64),
            numpy.asarray(y1_values, dtype=numpy.float64),
            numpy.asarray(width_values, dtype=numpy.float64),
        )
        for column in (table.x0, table.y0, table.x1, table.y1, table.width):
            readonly(column)
        return table

    def __len__(self) -> int:
        return len(self.x0)

    def coordinate_columns(self) -> LineColumns:
        """Return the ``x0, y0, x1, y1`` columns without materializing rows.

        Callers that only need coordinates should prefer this over iteration:
        the columns already exist, so rebuilding one ``CapturedLine`` per row
        only to read the same four values back out is pure overhead.
        """
        return self.x0, self.y0, self.x1, self.y1

    def __iter__(self) -> Iterator[CapturedLine]:
        # Unbox each column once with ``tolist``; zipping the arrays directly
        # allocates a numpy scalar per field per row.
        return (
            CapturedLine(x0, y0, x1, y1, width)
            for x0, y0, x1, y1, width in zip(
                self.x0.tolist(),
                self.y0.tolist(),
                self.x1.tolist(),
                self.y1.tolist(),
                self.width.tolist(),
            )
        )

    @property
    def nbytes(self) -> int:
        return sum(column.nbytes for column in (self.x0, self.y0, self.x1, self.y1, self.width))


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
            readonly(column)

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
            if any(stream[index][0] > stream[index + 1][0] for index in range(len(stream) - 1)):
                stream.sort(key=lambda entry: entry[0])
        # Collect columns as Python lists and convert once; per-element numpy
        # stores (including a 4-wide bbox broadcast) cost a dispatch each.
        sequence_values: list[int] = []
        kind_values: list[int] = []
        bbox_values: list[Any] = []
        payload_values: list[int] = []
        visible_values: list[bool] = []
        for entry in merge(*streams, key=lambda entry: (entry[0], entry[1])):
            sequence_values.append(entry[0])
            kind_values.append(int(entry[2]))
            payload_values.append(entry[3])
            bbox_values.append(entry[4])
            visible_values.append(entry[5])
        sequence = numpy.asarray(sequence_values, dtype=numpy.int64)
        kind = numpy.asarray(kind_values, dtype=numpy.uint8)
        if bbox_values:
            bbox = numpy.asarray(bbox_values, dtype=numpy.float64)
        else:
            bbox = numpy.empty((0, 4), dtype=numpy.float64)
        payload = numpy.asarray(payload_values, dtype=numpy.int64)
        visible = numpy.asarray(visible_values, dtype=numpy.bool_)
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
        if not all(isinstance(glyph, GlyphObservation) for glyph in glyphs):
            raise PdfContractError("page state emitted an invalid glyph product")
        raw_drawings = tuple(state.drawings)
        if not all(isinstance(drawing, CapturedDrawing) for drawing in raw_drawings):
            raise PdfContractError("page state emitted an invalid drawing product")
        drawings = tuple(drawing for drawing in raw_drawings if drawing.kind != "inline-image")
        inline_images = tuple(state.inline_images)
        if not all(isinstance(run, TextRun) for run in runs):
            raise PdfContractError("page state emitted an invalid text-run product")
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


def line_coordinate_columns(lines: Any) -> LineColumns:
    """Return ``(x0, y0, x1, y1)`` float64 columns for any line collection.

    A ``LineTable`` already stores those columns, so this returns them
    directly.  Other sequences are folded into an array once.
    """
    if isinstance(lines, LineTable):
        return lines.coordinate_columns()
    values = tuple(lines)
    coordinates = numpy.fromiter(
        (value for line in values for value in (line.x0, line.y0, line.x1, line.y1)),
        dtype=numpy.float64,
        count=len(values) * 4,
    ).reshape((-1, 4))
    return coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]


__all__ = (
    "LineColumns",
    "LineTable",
    "PageEventKind",
    "PageEventStream",
    "PageProducts",
    "PageProgram",
    "line_coordinate_columns",
)

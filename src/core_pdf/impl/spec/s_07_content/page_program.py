# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable products emitted by one page content interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, TypeAlias

from core_pdf.impl.exceptions import PdfContractError
from core_pdf.impl.model.glyphs import GlyphObservation
from core_pdf.impl.model.runs import TextRun
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


PageEventPayload: TypeAlias = TextRun | GlyphObservation | CapturedDrawing | CapturedInlineImage


@dataclass(frozen=True, slots=True)
class PageEvent:
    """One directly typed entry in page-content order."""

    kind: PageEventKind
    payload: PageEventPayload

    @property
    def seqno(self) -> int:
        return self.payload.seqno


@dataclass(frozen=True, slots=True)
class PageProgram:
    """The canonical capture shared by extraction and rendering.

    A page is interpreted exactly once. Consumers read the ordinary immutable
    product tuples directly and request a merged event tuple only when ordering
    across product types matters.
    """

    runs: tuple[TextRun, ...] = ()
    glyphs: tuple[GlyphObservation, ...] = ()
    drawings: tuple[CapturedDrawing, ...] = ()
    inline_images: tuple[CapturedInlineImage, ...] = ()
    lines: tuple[CapturedLine, ...] = ()

    @classmethod
    def from_state(cls, state: Any) -> PageProgram:
        runs = tuple(state.runs)
        glyphs = tuple(state.glyphs)
        raw_drawings = tuple(state.drawings)
        inline_images = tuple(state.inline_images)
        lines = tuple(state.lines)
        if not all(isinstance(run, TextRun) for run in runs):
            raise PdfContractError("page state emitted an invalid text-run product")
        if not all(isinstance(glyph, GlyphObservation) for glyph in glyphs):
            raise PdfContractError("page state emitted an invalid glyph product")
        if not all(isinstance(drawing, CapturedDrawing) for drawing in raw_drawings):
            raise PdfContractError("page state emitted an invalid drawing product")
        if not all(isinstance(image, CapturedInlineImage) for image in inline_images):
            raise PdfContractError("page state emitted an invalid inline-image product")
        if not all(isinstance(line, CapturedLine) for line in lines):
            raise PdfContractError("page state emitted an invalid line product")
        return cls(
            runs,
            glyphs,
            tuple(drawing for drawing in raw_drawings if drawing.kind != "inline-image"),
            inline_images,
            lines,
        )

    @property
    def products(self) -> PageProgram:
        """Return this program for compatibility with callers outside ``impl``."""
        return self

    @property
    def events(self) -> tuple[PageEvent, ...]:
        """Return captured products merged into content-stream order."""
        events = [PageEvent(PageEventKind.TEXT, run) for run in self.runs]
        events.extend(
            PageEvent(PageEventKind.GLYPH, glyph) for glyph in self.glyphs if glyph.has_paint
        )
        events.extend(
            PageEvent(
                PageEventKind.IMAGE if drawing.kind == "image" else PageEventKind.DRAWING,
                drawing,
            )
            for drawing in self.drawings
            if drawing.kind != "inline-image"
        )
        events.extend(PageEvent(PageEventKind.INLINE_IMAGE, image) for image in self.inline_images)
        # Categories were appended in the tie-break order used by content
        # interpretation. Python's stable sort preserves it for equal seqnos.
        events.sort(key=lambda event: event.seqno)
        return tuple(events)


__all__ = (
    "PageEvent",
    "PageEventKind",
    "PageEventPayload",
    "PageProgram",
)

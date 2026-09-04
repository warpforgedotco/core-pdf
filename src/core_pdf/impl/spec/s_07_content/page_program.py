# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable products emitted by one page content interpretation."""

from __future__ import annotations

from dataclasses import dataclass, replace
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


class PageCommandKind(IntEnum):
    TEXT = 1
    GLYPH = 2
    DRAWING = 3
    IMAGE = 4
    INLINE_IMAGE = 5


PageCommandPayload: TypeAlias = TextRun | GlyphObservation | CapturedDrawing | CapturedInlineImage


@dataclass(frozen=True, slots=True)
class PageCommand:
    """One directly typed entry in page-content order."""

    kind: PageCommandKind
    payload: PageCommandPayload

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
    commands: tuple[PageCommand, ...] = ()

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
        drawings = tuple(drawing for drawing in raw_drawings if drawing.kind != "inline-image")
        commands = [PageCommand(PageCommandKind.TEXT, run) for run in runs]
        commands.extend(
            PageCommand(PageCommandKind.GLYPH, glyph) for glyph in glyphs if glyph.has_paint
        )
        commands.extend(
            PageCommand(
                PageCommandKind.IMAGE if drawing.kind == "image" else PageCommandKind.DRAWING,
                drawing,
            )
            for drawing in drawings
            if drawing.kind != "inline-image"
        )
        commands.extend(PageCommand(PageCommandKind.INLINE_IMAGE, image) for image in inline_images)
        commands.sort(key=lambda command: command.seqno)
        return cls(runs, glyphs, drawings, inline_images, lines, tuple(commands))

    def with_runs(self, runs: tuple[TextRun, ...]) -> PageProgram:
        """Return a program whose text projection and command stream agree."""
        commands = [
            command for command in self.commands if command.kind is not PageCommandKind.TEXT
        ]
        commands.extend(PageCommand(PageCommandKind.TEXT, run) for run in runs)
        commands.sort(key=lambda command: command.seqno)
        return replace(self, runs=runs, commands=tuple(commands))


__all__ = (
    "PageCommand",
    "PageCommandKind",
    "PageCommandPayload",
    "PageProgram",
)

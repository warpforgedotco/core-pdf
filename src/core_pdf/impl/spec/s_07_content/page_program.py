# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable products emitted by one page content interpretation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from core_pdf.impl.exceptions import PdfContractError
from core_pdf.impl.model.glyphs import GlyphObservation
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLine,
)

PageCommand: TypeAlias = TextRun | GlyphObservation | CapturedDrawing | CapturedInlineImage


@dataclass(frozen=True, slots=True)
class PageProgram:
    """The canonical capture shared by extraction and rendering.

    The typed projections are the source of truth. ``commands`` is derived from
    them once, in content-stream order, so every construction path produces the
    same program.
    """

    runs: tuple[TextRun, ...] = ()
    glyphs: tuple[GlyphObservation, ...] = ()
    drawings: tuple[CapturedDrawing, ...] = ()
    inline_images: tuple[CapturedInlineImage, ...] = ()
    lines: tuple[CapturedLine, ...] = ()
    commands: tuple[PageCommand, ...] = field(init=False)

    def __post_init__(self) -> None:
        runs = tuple(self.runs)
        glyphs = tuple(self.glyphs)
        drawings = tuple(self.drawings)
        inline_images = tuple(self.inline_images)
        lines = tuple(self.lines)
        validations: tuple[tuple[str, tuple[object, ...], type[object]], ...] = (
            ("text-run", runs, TextRun),
            ("glyph", glyphs, GlyphObservation),
            ("drawing", drawings, CapturedDrawing),
            ("inline-image", inline_images, CapturedInlineImage),
            ("line", lines, CapturedLine),
        )
        for name, products, product_type in validations:
            if not all(isinstance(product, product_type) for product in products):
                raise PdfContractError(f"page program contains an invalid {name} product")

        drawings = tuple(drawing for drawing in drawings if drawing.kind != "inline-image")
        commands: list[PageCommand] = [*runs]
        commands.extend(glyph for glyph in glyphs if glyph.has_paint)
        commands.extend(drawings)
        commands.extend(inline_images)
        commands.sort(key=lambda command: command.seqno)

        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "glyphs", glyphs)
        object.__setattr__(self, "drawings", drawings)
        object.__setattr__(self, "inline_images", inline_images)
        object.__setattr__(self, "lines", lines)
        object.__setattr__(self, "commands", tuple(commands))


__all__ = (
    "PageCommand",
    "PageProgram",
)

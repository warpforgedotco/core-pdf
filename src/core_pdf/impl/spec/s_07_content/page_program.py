# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable products emitted by one page content interpretation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from core_pdf.impl._impl.model.glyphs import GlyphObservation
from core_pdf.impl._impl.model.runs import TextRun
from core_pdf.impl.exceptions import PdfContractError
from core_pdf.impl.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLine,
)

PageCommand: TypeAlias = TextRun | GlyphObservation | CapturedDrawing | CapturedInlineImage


@dataclass(frozen=True, slots=True)
class CapturedProgram:
    """One capture scope, shared by page bodies, appearances, and pattern cells.

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


@dataclass(frozen=True, slots=True)
class AppearanceProgram:
    """An already-interpreted appearance and the source that owns its paint."""

    kind: Literal["widget", "annotation"]
    source: object
    clip_bbox: tuple[float, float, float, float]
    program: CapturedProgram


@dataclass(frozen=True, slots=True)
class PageProgram:
    """A body and ordered appearance scopes, with flattened extraction views.

    Renderers select whole appearance scopes rather than filtering sequence
    numbers, which may tie across paints and graphics-state boundaries.
    """

    body: CapturedProgram = field(default_factory=CapturedProgram)
    appearances: tuple[AppearanceProgram, ...] = ()

    runs: tuple[TextRun, ...] = field(init=False)
    glyphs: tuple[GlyphObservation, ...] = field(init=False)
    drawings: tuple[CapturedDrawing, ...] = field(init=False)
    inline_images: tuple[CapturedInlineImage, ...] = field(init=False)
    lines: tuple[CapturedLine, ...] = field(init=False)
    commands: tuple[PageCommand, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.body, CapturedProgram):
            raise PdfContractError("page program contains an invalid body")
        appearances = tuple(self.appearances)
        if not all(
            isinstance(appearance, AppearanceProgram)
            and appearance.kind in {"widget", "annotation"}
            and isinstance(appearance.program, CapturedProgram)
            for appearance in appearances
        ):
            raise PdfContractError("page program contains an invalid appearance")
        object.__setattr__(self, "appearances", appearances)
        programs = (self.body, *(appearance.program for appearance in appearances))
        for name in ("runs", "glyphs", "drawings", "inline_images", "lines", "commands"):
            object.__setattr__(
                self,
                name,
                tuple(item for program in programs for item in getattr(program, name))
                if appearances
                else getattr(self.body, name),
            )


__all__ = (
    "AppearanceProgram",
    "CapturedProgram",
    "PageCommand",
    "PageProgram",
)

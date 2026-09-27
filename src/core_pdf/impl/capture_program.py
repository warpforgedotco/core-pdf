# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass, field  # noqa: TID251
from functools import partial
from itertools import chain
from typing import Literal, TypeAlias

from core_pdf.impl.capture_records import (
    EMPTY_LINES,
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLines,
    CapturedTextBoundary,
)
from core_pdf.impl.glyphs import GlyphObservation
from core_pdf.impl.runs import TextRun
from core_pdf.impl.types import GeneratedRecord

PageCommand: TypeAlias = (
    TextRun | GlyphObservation | CapturedDrawing | CapturedInlineImage | CapturedTextBoundary
)


class CaptureOptions(GeneratedRecord):
    ink_bounds: bool = True
    text_runs: bool = True
    render_details: bool = True


DEFAULT_CAPTURE = CaptureOptions()
EXTRACTION_CAPTURE = CaptureOptions(render_details=False)


@dataclass(frozen=True, slots=True)
class CapturedProgram:
    runs: tuple[TextRun, ...] = ()
    glyphs: tuple[GlyphObservation, ...] = ()
    drawings: tuple[CapturedDrawing, ...] = ()
    inline_images: tuple[CapturedInlineImage, ...] = ()
    lines: CapturedLines = EMPTY_LINES
    text_boundaries: tuple[CapturedTextBoundary, ...] = field(default=(), kw_only=True)
    options: CaptureOptions = field(default=DEFAULT_CAPTURE, kw_only=True)
    _commands: tuple[PageCommand, ...] | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs", tuple(self.runs))
        object.__setattr__(self, "glyphs", tuple(self.glyphs))
        object.__setattr__(self, "drawings", tuple(self.drawings))
        object.__setattr__(self, "inline_images", tuple(self.inline_images))
        if type(self.lines) is not CapturedLines:
            object.__setattr__(self, "lines", CapturedLines(self.lines))
        object.__setattr__(self, "text_boundaries", tuple(self.text_boundaries))

    @property
    def commands(self) -> tuple[PageCommand, ...]:
        commands = self._commands
        if commands is None:
            if not self.options.render_details:
                raise ValueError(
                    "page program was captured without render details and cannot be drawn; "
                    "capture it with CaptureOptions(render_details=True)"
                )
            ordered: list[PageCommand] = [*self.text_boundaries, *self.runs]
            ordered.extend(glyph for glyph in self.glyphs if glyph.has_paint)
            ordered.extend(self.drawings)
            ordered.extend(self.inline_images)
            ordered.sort(key=lambda command: command.seqno)
            commands = tuple(ordered)
            object.__setattr__(self, "_commands", commands)
        return commands


class AppearanceProgram(GeneratedRecord):
    kind: Literal["widget", "annotation"]
    source: object
    clip_bbox: tuple[float, float, float, float]
    program: CapturedProgram


@dataclass(frozen=True, slots=True)
class PageProgram:
    body: CapturedProgram = field(default_factory=CapturedProgram)
    appearances: tuple[AppearanceProgram, ...] = ()
    runs: tuple[TextRun, ...] = field(init=False)
    glyphs: tuple[GlyphObservation, ...] = field(init=False)
    drawings: tuple[CapturedDrawing, ...] = field(init=False)
    inline_images: tuple[CapturedInlineImage, ...] = field(init=False)
    lines: CapturedLines = field(init=False)
    text_boundaries: tuple[CapturedTextBoundary, ...] = field(init=False)
    _commands: tuple[PageCommand, ...] | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        set_derived = partial(object.__setattr__, self)
        appearances = tuple(self.appearances)
        set_derived("appearances", appearances)
        body = self.body
        if not appearances:
            set_derived("runs", body.runs)
            set_derived("glyphs", body.glyphs)
            set_derived("drawings", body.drawings)
            set_derived("inline_images", body.inline_images)
            set_derived("lines", body.lines)
            set_derived("text_boundaries", body.text_boundaries)
            return
        programs = (body, *(appearance.program for appearance in appearances))
        merge = chain.from_iterable
        set_derived("runs", tuple(merge(p.runs for p in programs)))
        set_derived("glyphs", tuple(merge(p.glyphs for p in programs)))
        set_derived("drawings", tuple(merge(p.drawings for p in programs)))
        set_derived("inline_images", tuple(merge(p.inline_images for p in programs)))
        set_derived("lines", CapturedLines.concatenate(p.lines for p in programs))
        set_derived("text_boundaries", tuple(merge(p.text_boundaries for p in programs)))

    @property
    def options(self) -> CaptureOptions:
        return self.body.options

    @property
    def commands(self) -> tuple[PageCommand, ...]:
        commands = self._commands
        if commands is None:
            if not self.appearances:
                commands = self.body.commands
            else:
                programs = (self.body, *(a.program for a in self.appearances))
                commands = tuple(chain.from_iterable(p.commands for p in programs))
            object.__setattr__(self, "_commands", commands)
        return commands


__all__ = (
    "DEFAULT_CAPTURE",
    "EXTRACTION_CAPTURE",
    "AppearanceProgram",
    "CaptureOptions",
    "CapturedProgram",
    "PageCommand",
    "PageProgram",
)

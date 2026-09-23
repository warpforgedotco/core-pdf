# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Literal, TypeAlias

from core_pdf.impl.capture.records import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLine,
    CapturedTextBoundary,
)
from core_pdf.impl.model.glyphs import GlyphObservation
from core_pdf.impl.model.runs import TextRun

PageCommand: TypeAlias = (
    TextRun | GlyphObservation | CapturedDrawing | CapturedInlineImage | CapturedTextBoundary
)


@dataclass(frozen=True, slots=True)
class CapturedProgram:
    runs: tuple[TextRun, ...] = ()
    glyphs: tuple[GlyphObservation, ...] = ()
    drawings: tuple[CapturedDrawing, ...] = ()
    inline_images: tuple[CapturedInlineImage, ...] = ()
    lines: tuple[CapturedLine, ...] = ()
    text_boundaries: tuple[CapturedTextBoundary, ...] = field(default=(), kw_only=True)
    # Derived from the six above. init=False keeps it out of __init__,
    # __match_args__ and copy.replace, which is what the hand-written
    # __replace__ arranged by listing the other six explicitly.
    commands: tuple[PageCommand, ...] = field(init=False)

    def __post_init__(self) -> None:
        runs = tuple(self.runs)
        glyphs = tuple(self.glyphs)
        drawings = tuple(self.drawings)
        inline_images = tuple(self.inline_images)
        lines = tuple(self.lines)
        text_boundaries = tuple(self.text_boundaries)
        commands: list[PageCommand] = [*text_boundaries, *runs]
        commands.extend(glyph for glyph in glyphs if glyph.has_paint)
        commands.extend(drawings)
        commands.extend(inline_images)
        commands.sort(key=lambda command: command.seqno)

        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "glyphs", glyphs)
        object.__setattr__(self, "drawings", drawings)
        object.__setattr__(self, "inline_images", inline_images)
        object.__setattr__(self, "lines", lines)
        object.__setattr__(self, "text_boundaries", text_boundaries)
        object.__setattr__(self, "commands", tuple(commands))


@dataclass(frozen=True, slots=True)
class AppearanceProgram:
    kind: Literal["widget", "annotation"]
    source: object
    clip_bbox: tuple[float, float, float, float]
    program: CapturedProgram


@dataclass(frozen=True, slots=True)
class PageProgram:
    body: CapturedProgram = field(default_factory=CapturedProgram)
    appearances: tuple[AppearanceProgram, ...] = ()
    # Concatenations of body and appearances, rebuilt by __post_init__.
    runs: tuple[TextRun, ...] = field(init=False)
    glyphs: tuple[GlyphObservation, ...] = field(init=False)
    drawings: tuple[CapturedDrawing, ...] = field(init=False)
    inline_images: tuple[CapturedInlineImage, ...] = field(init=False)
    lines: tuple[CapturedLine, ...] = field(init=False)
    text_boundaries: tuple[CapturedTextBoundary, ...] = field(init=False)
    commands: tuple[PageCommand, ...] = field(init=False)

    def __post_init__(self) -> None:
        set_derived = partial(object.__setattr__, self)
        appearances = tuple(self.appearances)
        set_derived("appearances", appearances)
        body = self.body
        if not appearances:
            # Nothing to concatenate, so the body's own tuples stand as they are.
            set_derived("runs", body.runs)
            set_derived("glyphs", body.glyphs)
            set_derived("drawings", body.drawings)
            set_derived("inline_images", body.inline_images)
            set_derived("lines", body.lines)
            set_derived("text_boundaries", body.text_boundaries)
            set_derived("commands", body.commands)
            return
        # Body first, then each appearance in capture order. Nothing is
        # re-sorted: one TextState numbers the body and every appearance off a
        # single counter, so concatenating in that order is already by seqno.
        programs = (body, *(appearance.program for appearance in appearances))
        merge = chain.from_iterable
        set_derived("runs", tuple(merge(p.runs for p in programs)))
        set_derived("glyphs", tuple(merge(p.glyphs for p in programs)))
        set_derived("drawings", tuple(merge(p.drawings for p in programs)))
        set_derived("inline_images", tuple(merge(p.inline_images for p in programs)))
        set_derived("lines", tuple(merge(p.lines for p in programs)))
        set_derived("text_boundaries", tuple(merge(p.text_boundaries for p in programs)))
        set_derived("commands", tuple(merge(p.commands for p in programs)))


__all__ = (
    "AppearanceProgram",
    "CapturedProgram",
    "PageCommand",
    "PageProgram",
)

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
    # Derived from the six above, and built only when something asks for it.
    # init=False keeps it out of __init__, __match_args__ and copy.replace,
    # which is what the hand-written __replace__ arranged by listing the other
    # six explicitly. compare=False because it is a function of them.
    #
    # Lazy because the renderer is its only reader in the workspace: an
    # extract never touches it, and building it eagerly meant a has_paint call
    # on every glyph and a sort of every product on the page, for a tuple that
    # was then dropped.
    _commands: tuple[PageCommand, ...] | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs", tuple(self.runs))
        object.__setattr__(self, "glyphs", tuple(self.glyphs))
        object.__setattr__(self, "drawings", tuple(self.drawings))
        object.__setattr__(self, "inline_images", tuple(self.inline_images))
        object.__setattr__(self, "lines", tuple(self.lines))
        object.__setattr__(self, "text_boundaries", tuple(self.text_boundaries))

    @property
    def commands(self) -> tuple[PageCommand, ...]:
        commands = self._commands
        if commands is None:
            ordered: list[PageCommand] = [*self.text_boundaries, *self.runs]
            ordered.extend(glyph for glyph in self.glyphs if glyph.has_paint)
            ordered.extend(self.drawings)
            ordered.extend(self.inline_images)
            ordered.sort(key=lambda command: command.seqno)
            commands = tuple(ordered)
            object.__setattr__(self, "_commands", commands)
        return commands


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
    # False when the capture skipped the rasterizer's per-glyph payload -- the
    # glyph transforms and the bitmap requests. Such a program describes the
    # page's text correctly but cannot be drawn, and compose_page refuses it
    # rather than rendering a page with no glyphs on it.
    render_details: bool = field(default=True, kw_only=True)
    # Concatenations of body and appearances, rebuilt by __post_init__.
    runs: tuple[TextRun, ...] = field(init=False)
    glyphs: tuple[GlyphObservation, ...] = field(init=False)
    drawings: tuple[CapturedDrawing, ...] = field(init=False)
    inline_images: tuple[CapturedInlineImage, ...] = field(init=False)
    lines: tuple[CapturedLine, ...] = field(init=False)
    text_boundaries: tuple[CapturedTextBoundary, ...] = field(init=False)
    # Lazy for the same reason as CapturedProgram.commands, and it has to be:
    # reading body.commands here would force the body's.
    _commands: tuple[PageCommand, ...] | None = field(
        init=False, default=None, repr=False, compare=False
    )

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
    "AppearanceProgram",
    "CapturedProgram",
    "PageCommand",
    "PageProgram",
)

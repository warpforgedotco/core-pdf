# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Literal, TypeAlias

from core_pdf.impl.capture.records import (
    EMPTY_LINES,
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLines,
    CapturedTextBoundary,
)
from core_pdf.impl.glyphs import GlyphObservation
from core_pdf.impl.runs import TextRun

PageCommand: TypeAlias = (
    TextRun | GlyphObservation | CapturedDrawing | CapturedInlineImage | CapturedTextBoundary
)


@dataclass(frozen=True, slots=True)
class CaptureOptions:
    """What a capture records beyond the text itself.

    ink_bounds: each horizontal glyph's ink box, from the font's glyph bbox.
    text_runs: the layout runs, clusters and run geometry extraction builds on.
    render_details: the rasterizer's per-glyph payload -- glyph transforms and
    bitmap requests. A program captured without it describes the page's text
    correctly but cannot be drawn.
    """

    ink_bounds: bool = True
    text_runs: bool = True
    render_details: bool = True


DEFAULT_CAPTURE = CaptureOptions()
# Extraction composes blocks, not pixels, so it skips the render payload.
EXTRACTION_CAPTURE = CaptureOptions(render_details=False)


@dataclass(frozen=True, slots=True)
class CapturedProgram:
    runs: tuple[TextRun, ...] = ()
    glyphs: tuple[GlyphObservation, ...] = ()
    drawings: tuple[CapturedDrawing, ...] = ()
    inline_images: tuple[CapturedInlineImage, ...] = ()
    lines: CapturedLines = EMPTY_LINES
    text_boundaries: tuple[CapturedTextBoundary, ...] = field(default=(), kw_only=True)
    # What the capture recorded. commands refuses a program captured without
    # render details rather than draw a page with no glyphs on it.
    options: CaptureOptions = field(default=DEFAULT_CAPTURE, kw_only=True)
    # Derived from the six products above, and built only when something asks for it.
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
        # Accepted as any iterable of CapturedLine for construction by hand;
        # capture itself always hands over a CapturedLines.
        if type(self.lines) is not CapturedLines:
            object.__setattr__(self, "lines", CapturedLines(self.lines))
        object.__setattr__(self, "text_boundaries", tuple(self.text_boundaries))

    @property
    def commands(self) -> tuple[PageCommand, ...]:
        commands = self._commands
        if commands is None:
            if not self.options.render_details:
                # Without glyph transforms and bitmap requests every glyph
                # reports no paint, so the page would draw with no text on it.
                # Refusing is better than silently drawing that.
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
    lines: CapturedLines = field(init=False)
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
        set_derived("lines", CapturedLines.concatenate(p.lines for p in programs))
        set_derived("text_boundaries", tuple(merge(p.text_boundaries for p in programs)))

    @property
    def options(self) -> CaptureOptions:
        # One TextState captures the body and every appearance, so they share it.
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

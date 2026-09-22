# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar, Literal, Self, TypeAlias

from core_pdf.impl.capture.records import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLine,
    CapturedTextBoundary,
)
from core_pdf.impl.exceptions import PdfContractError
from core_pdf.impl.model.glyphs import GlyphObservation
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.records import Record, frozen_setattr

PageCommand: TypeAlias = (
    TextRun | GlyphObservation | CapturedDrawing | CapturedInlineImage | CapturedTextBoundary
)


class CapturedProgram(Record):
    __slots__ = (
        "runs",
        "glyphs",
        "drawings",
        "inline_images",
        "lines",
        "text_boundaries",
        "commands",
    )

    runs: tuple[TextRun, ...]
    glyphs: tuple[GlyphObservation, ...]
    drawings: tuple[CapturedDrawing, ...]
    inline_images: tuple[CapturedInlineImage, ...]
    lines: tuple[CapturedLine, ...]
    text_boundaries: tuple[CapturedTextBoundary, ...]
    commands: tuple[PageCommand, ...]

    __fields__: ClassVar[tuple[str, ...]] = (
        "runs",
        "glyphs",
        "drawings",
        "inline_images",
        "lines",
        "text_boundaries",
        "commands",
    )
    __match_args__ = ("runs", "glyphs", "drawings", "inline_images", "lines")

    def __init__(
        self,
        runs: tuple[TextRun, ...] = (),
        glyphs: tuple[GlyphObservation, ...] = (),
        drawings: tuple[CapturedDrawing, ...] = (),
        inline_images: tuple[CapturedInlineImage, ...] = (),
        lines: tuple[CapturedLine, ...] = (),
        *,
        text_boundaries: tuple[CapturedTextBoundary, ...] = (),
    ) -> None:
        frozen_setattr(self, "runs", runs)
        frozen_setattr(self, "glyphs", glyphs)
        frozen_setattr(self, "drawings", drawings)
        frozen_setattr(self, "inline_images", inline_images)
        frozen_setattr(self, "lines", lines)
        frozen_setattr(self, "text_boundaries", text_boundaries)
        self._post_init()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.runs == other.runs
            and self.glyphs == other.glyphs
            and self.drawings == other.drawings
            and self.inline_images == other.inline_images
            and self.lines == other.lines
            and self.text_boundaries == other.text_boundaries
            and self.commands == other.commands
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.runs,
                self.glyphs,
                self.drawings,
                self.inline_images,
                self.lines,
                self.text_boundaries,
                self.commands,
            )
        )

    def __replace__(self, /, **changes: Any) -> Self:
        runs = changes.pop("runs", self.runs)
        glyphs = changes.pop("glyphs", self.glyphs)
        drawings = changes.pop("drawings", self.drawings)
        inline_images = changes.pop("inline_images", self.inline_images)
        lines = changes.pop("lines", self.lines)
        text_boundaries = changes.pop("text_boundaries", self.text_boundaries)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            runs,
            glyphs,
            drawings,
            inline_images,
            lines,
            text_boundaries=text_boundaries,
        )

    def _post_init(self) -> None:
        runs = tuple(self.runs)
        glyphs = tuple(self.glyphs)
        drawings = tuple(self.drawings)
        inline_images = tuple(self.inline_images)
        lines = tuple(self.lines)
        text_boundaries = tuple(self.text_boundaries)
        validations: tuple[tuple[str, tuple[object, ...], type[object]], ...] = (
            ("text-run", runs, TextRun),
            ("glyph", glyphs, GlyphObservation),
            ("drawing", drawings, CapturedDrawing),
            ("inline-image", inline_images, CapturedInlineImage),
            ("line", lines, CapturedLine),
            ("text-boundary", text_boundaries, CapturedTextBoundary),
        )
        for name, products, product_type in validations:
            if not all(isinstance(product, product_type) for product in products):
                raise PdfContractError(f"page program contains an invalid {name} product")

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


class AppearanceProgram(Record):
    __slots__ = ("kind", "source", "clip_bbox", "program")

    kind: Literal["widget", "annotation"]
    source: object
    clip_bbox: tuple[float, float, float, float]
    program: CapturedProgram

    __fields__: ClassVar[tuple[str, ...]] = ("kind", "source", "clip_bbox", "program")
    __match_args__ = ("kind", "source", "clip_bbox", "program")

    def __init__(
        self,
        kind: Literal["widget", "annotation"],
        source: object,
        clip_bbox: tuple[float, float, float, float],
        program: CapturedProgram,
    ) -> None:
        frozen_setattr(self, "kind", kind)
        frozen_setattr(self, "source", source)
        frozen_setattr(self, "clip_bbox", clip_bbox)
        frozen_setattr(self, "program", program)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.kind == other.kind
            and self.source == other.source
            and self.clip_bbox == other.clip_bbox
            and self.program == other.program
        )

    def __hash__(self) -> int:
        return hash((self.kind, self.source, self.clip_bbox, self.program))


class PageProgram(Record):
    __slots__ = (
        "body",
        "appearances",
        "runs",
        "glyphs",
        "drawings",
        "inline_images",
        "lines",
        "text_boundaries",
        "commands",
    )

    body: CapturedProgram
    appearances: tuple[AppearanceProgram, ...]
    runs: tuple[TextRun, ...]
    glyphs: tuple[GlyphObservation, ...]
    drawings: tuple[CapturedDrawing, ...]
    inline_images: tuple[CapturedInlineImage, ...]
    lines: tuple[CapturedLine, ...]
    text_boundaries: tuple[CapturedTextBoundary, ...]
    commands: tuple[PageCommand, ...]

    __fields__: ClassVar[tuple[str, ...]] = (
        "body",
        "appearances",
        "runs",
        "glyphs",
        "drawings",
        "inline_images",
        "lines",
        "text_boundaries",
        "commands",
    )
    __match_args__ = ("body", "appearances")

    def __init__(
        self,
        body: CapturedProgram | None = None,
        appearances: tuple[AppearanceProgram, ...] = (),
    ) -> None:
        frozen_setattr(self, "body", CapturedProgram() if body is None else body)
        frozen_setattr(self, "appearances", appearances)
        self._post_init()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.body == other.body
            and self.appearances == other.appearances
            and self.runs == other.runs
            and self.glyphs == other.glyphs
            and self.drawings == other.drawings
            and self.inline_images == other.inline_images
            and self.lines == other.lines
            and self.text_boundaries == other.text_boundaries
            and self.commands == other.commands
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.body,
                self.appearances,
                self.runs,
                self.glyphs,
                self.drawings,
                self.inline_images,
                self.lines,
                self.text_boundaries,
                self.commands,
            )
        )

    def __replace__(self, /, **changes: Any) -> Self:
        body = changes.pop("body", self.body)
        appearances = changes.pop("appearances", self.appearances)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(body, appearances)

    def _post_init(self) -> None:
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
        for name in (
            "runs",
            "glyphs",
            "drawings",
            "inline_images",
            "lines",
            "text_boundaries",
            "commands",
        ):
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

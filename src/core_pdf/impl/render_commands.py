# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy

from core_pdf.impl.capture_program import CapturedProgram, PageCommand
from core_pdf.impl.capture_records import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedPath,
    CapturedSubpath,
    CapturedTextBoundary,
)
from core_pdf.impl.glyph_outlines import GlyphOutlineArrays
from core_pdf.impl.glyphs import GlyphObservation, Matrix6
from core_pdf.impl.render_display import DisplayList
from core_pdf.impl.render_model import PathPaintKind
from core_pdf.impl.runs import TextRun
from core_pdf.impl.types import RecordType, Rectangle, ReprFields
from core_pdf_cythonized import translated_outline_edges
from core_pdf_spec.s_07_content.model import NON_PAINTING_RENDER_MODES
from core_pdf_spec.s_08_graphics.geometry import unit_square_placement


def glyph_outline_path(
    glyph: GlyphObservation,
) -> tuple[CapturedPath, Rectangle | None, numpy.ndarray[Any, Any] | None] | None:
    if not glyph.paint_glyph:
        return None
    transform = glyph.glyph_transform
    decoder = glyph.font_decoder
    if transform is None or decoder is None:
        return None
    code = glyph.bitmap_code
    if code is None:
        code = glyph.cid if glyph.cid is not None else glyph.char_code
    if code is None:
        return None
    array_resolver = getattr(decoder, "glyph_outline_arrays", None)
    if not callable(array_resolver):
        return None
    arrays = array_resolver(code, glyph.gid, glyph.text)
    if arrays is None:
        return None
    return transformed_outline(arrays, transform)


def transformed_outline(
    arrays: GlyphOutlineArrays, transform: Matrix6
) -> tuple[CapturedPath, Rectangle | None, numpy.ndarray[Any, Any] | None] | None:
    a, b, c, d, e, f = transform
    linear_x, linear_y = arrays.linear_columns(a, b, c, d)
    column_x, column_y, edges, kept, dropped, bounds = translated_outline_edges(
        linear_x, linear_y, e, f, arrays.spans
    )
    if edges is None:
        return None
    path = CapturedPath.deferred_outline(column_x, column_y, kept)
    if dropped:
        return path, path.bbox(), edges
    return path, bounds, edges


def append_glyph_paint(
    display_list: DisplayList,
    glyph: GlyphObservation,
    clipping_subpaths: list[CapturedSubpath],
    *,
    include_paint: bool = True,
) -> bool:
    if glyph.visible is False and not glyph.clip_glyph:
        return True
    mode = int(glyph.text_render_mode)
    if mode == 3:
        return True
    if not include_paint and mode < 4:
        return True
    outline = glyph_outline_path(glyph)
    if outline is None:
        return False
    path, bbox, edge_array = outline
    if mode >= 4:
        clipping_subpaths.extend(path.subpaths)
    if not include_paint or mode in NON_PAINTING_RENDER_MODES or glyph.visible is False:
        return True
    paint_kind = (
        PathPaintKind.FILL
        if mode in {0, 4}
        else PathPaintKind.STROKE
        if mode in {1, 5}
        else PathPaintKind.FILL_STROKE
    )
    display_list.append_glyph_paint(paint_kind, glyph.seqno, bbox, path, edge_array, glyph.style)
    return True


class TextObjectState(ReprFields, metaclass=RecordType, frozen=False):
    active: bool
    group_open: bool
    group_pending: bool
    clipping_subpaths: list[CapturedSubpath]
    object_id: int | None

    def __init__(
        self,
        active: bool = False,
        group_open: bool = False,
        group_pending: bool = False,
        clipping_subpaths: list[CapturedSubpath] | None = None,
        object_id: int | None = None,
    ) -> None:
        self.active = active
        self.group_open = group_open
        self.group_pending = group_pending
        self.clipping_subpaths = [] if clipping_subpaths is None else clipping_subpaths
        self.object_id = object_id

    __hash__ = None  # type: ignore[assignment]


class ProgramLowering:
    __slots__ = (
        "display_list",
        "include_text",
        "explicit_text_boundaries",
        "text",
        "text_stack",
        "glyph_scope_depth",
    )

    def __init__(
        self, display_list: DisplayList, program: CapturedProgram, *, include_text: bool
    ) -> None:
        self.display_list = display_list
        self.include_text = include_text
        self.explicit_text_boundaries = bool(program.text_boundaries)
        self.text = TextObjectState()
        self.text_stack: list[TextObjectState] = []
        self.glyph_scope_depth = 0

    def lower(self, commands: tuple[PageCommand, ...]) -> None:
        for command in commands:
            command_type = type(command)
            if (
                command_type is not CapturedTextBoundary
                and not self.include_text
                and self.glyph_scope_depth
            ):
                continue
            lowering = COMMAND_LOWERINGS.get(command_type)
            if lowering is None:
                lowering = inherited_lowering(command_type)
            lowering(self, command)
        self.finish_text(len(commands))

    def flush_text_clip(self, seqno: int) -> None:
        subpaths = self.text.clipping_subpaths
        if not subpaths:
            return
        self.display_list.append(
            "clip",
            seqno,
            path=CapturedPath(list(subpaths)),
            fill_rule="nonzero",
        )
        subpaths.clear()

    def finish_text(self, seqno: int) -> None:
        text = self.text
        if text.group_open:
            self.display_list.append("group-end", seqno)
        self.flush_text_clip(seqno)
        text.active = False
        text.group_open = False
        text.group_pending = False
        text.object_id = None

    def open_pending_text_group(self, seqno: int) -> None:
        text = self.text
        if text.group_pending:
            text.group_pending = False
            text.group_open = True
            self.begin_text_group(seqno, knockout=True)

    def begin_text_group(self, seqno: int, *, knockout: bool) -> None:
        self.display_list.append(
            "group-begin",
            seqno,
            fill_opacity=1.0,
            blend_mode=None,
            group_isolated=False,
            group_knockout=knockout,
            group_track_shape=True,
        )

    def lower_boundary(self, command: CapturedTextBoundary) -> None:
        match command.kind:
            case "stream-begin":
                self.text_stack.append(self.text)
                self.text = TextObjectState()
                self.display_list.append("scope-begin", command.seqno)
            case "stream-end":
                self.finish_text(command.seqno)
                self.display_list.append("scope-end", command.seqno)
                if self.text_stack:
                    self.text = self.text_stack.pop()
            case "begin":
                self.finish_text(command.seqno)
                self.text.active = True
                self.text.group_pending = self.include_text and command.knockout
            case "end":
                self.finish_text(command.seqno)
            case "glyph-begin":
                self.glyph_scope_depth += 1
                if self.include_text:
                    self.open_pending_text_group(command.seqno)
                    self.begin_text_group(command.seqno, knockout=False)
            case "glyph-end":
                if self.glyph_scope_depth:
                    if self.include_text:
                        self.display_list.append("group-end", command.seqno)
                    self.glyph_scope_depth -= 1

    def lower_text_run(self, run: TextRun) -> None:
        if not self.include_text:
            return
        self.display_list.append(
            "text",
            run.seqno,
            text=run.text,
            bbox=(run.x0, run.y0, run.x1, run.y1),
            font_name=run.font_name,
            font_size=run.font_size,
            visible=run.visible,
            fill_color=run.fill_color,
            rotation_angle=run.rotation_angle,
        )

    def lower_glyph(self, glyph: GlyphObservation) -> None:
        include_text = self.include_text
        text = self.text
        glyph_text_object_id = glyph.text_object_id
        if (
            not self.explicit_text_boundaries
            and text.object_id is not None
            and glyph_text_object_id != text.object_id
        ):
            self.flush_text_clip(glyph.seqno)
        text.object_id = glyph_text_object_id
        glyph_paints = (
            include_text
            and glyph.text_render_mode not in NON_PAINTING_RENDER_MODES
            and glyph.visible is not False
        )
        if glyph_paints:
            self.open_pending_text_group(glyph.seqno)
        glyph_group_open = glyph_paints and self.explicit_text_boundaries and not text.group_open
        if glyph_group_open:
            self.begin_text_group(glyph.seqno, knockout=False)
        try:
            if append_glyph_paint(
                self.display_list,
                glyph,
                text.clipping_subpaths,
                include_paint=include_text,
            ):
                return
            if not include_text or glyph.text_render_mode in NON_PAINTING_RENDER_MODES:
                return
            bitmap = glyph.resolved_bitmap()
            if not bitmap:
                return
            self.display_list.append(
                "glyph",
                glyph.seqno,
                text=glyph.text,
                code=glyph.cid,
                gid=glyph.gid,
                font_name=glyph.font_name,
                unicode_source=glyph.unicode_source,
                alternates=glyph.alternates,
                bbox=glyph.ink_bbox,
                advance_bbox=glyph.advance_bbox,
                fill_color=glyph.fill,
                fill_opacity=glyph.fill_opacity,
                blend_mode=glyph.blend_mode,
                soft_mask_alpha=glyph.soft_mask_alpha,
                graphics_soft_mask=glyph.graphics_soft_mask,
                alpha_is_shape=glyph.alpha_is_shape,
                visible=glyph.visible,
                bitmap=bitmap,
                bitmap_width=glyph.bitmap_width,
                bitmap_height=glyph.bitmap_height,
            )
        finally:
            if glyph_group_open:
                self.display_list.append("group-end", glyph.seqno)

    def lower_drawing(self, drawing: CapturedDrawing) -> None:
        if not self.text.active:
            self.flush_text_clip(drawing.seqno)
        elif drawing.paints:
            self.open_pending_text_group(drawing.seqno)
        self.display_list.append_captured_drawing(drawing)

    def lower_inline_image(self, inline_image: CapturedInlineImage) -> None:
        assert isinstance(inline_image, CapturedInlineImage)
        bbox, quad = unit_square_placement(inline_image.ctm)
        text_active = self.text.active
        if not text_active:
            self.flush_text_clip(inline_image.seqno)
        if not inline_image.paints:
            return
        if text_active:
            self.open_pending_text_group(inline_image.seqno)
        self.display_list.append(
            "inline-image",
            inline_image.seqno,
            dictionary=dict(inline_image.dictionary),
            data=inline_image.data,
            image_source=inline_image.image_source,
            image_clip=inline_image.image_clip,
            ctm=inline_image.ctm,
            xobject_depth=inline_image.xobject_depth,
            blend_mode=inline_image.blend_mode,
            soft_mask_alpha=inline_image.soft_mask_alpha,
            graphics_soft_mask=inline_image.graphics_soft_mask,
            alpha_is_shape=inline_image.alpha_is_shape,
            fill=inline_image.fill,
            fill_opacity=inline_image.fill_opacity,
            bbox=bbox,
            quad=quad,
            raw_data=inline_image.data,
        )


COMMAND_LOWERINGS: dict[type, Callable[[ProgramLowering, Any], None]] = {
    CapturedTextBoundary: ProgramLowering.lower_boundary,
    TextRun: ProgramLowering.lower_text_run,
    GlyphObservation: ProgramLowering.lower_glyph,
    CapturedDrawing: ProgramLowering.lower_drawing,
    CapturedInlineImage: ProgramLowering.lower_inline_image,
}


def inherited_lowering(command_type: type) -> Callable[[ProgramLowering, Any], None]:
    for base, lowering in COMMAND_LOWERINGS.items():
        if issubclass(command_type, base):
            return lowering
    return ProgramLowering.lower_inline_image


def append_captured_program(
    display_list: DisplayList, page_program: CapturedProgram, *, include_text: bool
) -> None:
    ProgramLowering(display_list, page_program, include_text=include_text).lower(
        page_program.commands
    )

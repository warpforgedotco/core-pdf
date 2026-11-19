# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from core_pdf.impl.base.glyphs import GlyphObservation, GlyphStyle
from core_pdf.impl.base.runs import TextRun
from core_pdf.impl.base.types import RecordType, ReprFields, field
from core_pdf.impl.capture.program import CapturedProgram, PageCommand
from core_pdf.impl.capture.records import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedPath,
    CapturedSubpath,
    CapturedTextBoundary,
)
from core_pdf.impl.render.blend import declared_blend
from core_pdf.impl.render.display import PATH_PAINT_LAYOUT, DisplayList
from core_pdf.impl.render.model import DEFAULT_MITER_LIMIT, PathPaintKind
from core_pdf.impl.spec.s_07_content.model import NON_PAINTING_RENDER_MODES
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf.impl.spec.s_08_graphics.geometry import unit_square_placement
lazy from core_pdf_cythonized import glyph_outline_edges

PAINT_KINDS_BY_MODE = {
    0: PathPaintKind.FILL,
    4: PathPaintKind.FILL,
    1: PathPaintKind.STROKE,
    5: PathPaintKind.STROKE,
}
# The value lower_glyphs returns when the commands run out.
END_OF_COMMANDS: Any = object()


class GlyphStyleLowering:
    """What lowering a glyph takes from its style alone, worked out once for each style a
    run of glyphs shows; consecutive glyphs share one style object, and a style is never
    changed (setting a style field on a glyph replaces its style)."""

    __slots__ = (
        "style",
        "text_object_id",
        "mode",
        "clip_glyph",
        "paints",
        "outline_paints",
        "outline_arrays",
        "paint_kind",
        "plain_fill",
        "fields",
        "miter_limit",
        "stroke_matrix",
    )

    def __init__(self, style: GlyphStyle, display_list: DisplayList, include_text: bool) -> None:
        self.style = style
        self.text_object_id = style.text_object_id
        render_mode = style.text_render_mode
        self.mode = mode = int(render_mode)
        self.clip_glyph = style.clip_glyph
        # Whether a visible glyph of this style paints, tested on the mode as captured.
        self.paints = include_text and render_mode not in NON_PAINTING_RENDER_MODES
        self.outline_paints = include_text and mode not in NON_PAINTING_RENDER_MODES
        decoder = style.font_decoder
        resolver = None if decoder is None else getattr(decoder, "glyph_outline_arrays", None)
        self.outline_arrays = resolver if callable(resolver) else None
        self.paint_kind = paint_kind = PAINT_KINDS_BY_MODE.get(mode, PathPaintKind.FILL_STROKE)
        # Whether its items are plain fills (is_plain_fill), given a box.
        self.plain_fill = (
            paint_kind is PathPaintKind.FILL
            and style.fill_pattern is None
            and declared_blend(style.blend_mode) is None
        )
        self.fields = display_list.glyph_paint_fields_of(style)
        # A stroked glyph is stroked as a path is, with the miter limit and the stroke
        # matrix of its graphics state (ISO 32000-2 9.3.6); a filled one keeps the defaults.
        self.miter_limit = DEFAULT_MITER_LIMIT
        self.stroke_matrix = None
        if paint_kind is not PathPaintKind.FILL:
            limit = style.miter_limit
            self.miter_limit = float(limit) if is_pdf_number(limit) else DEFAULT_MITER_LIMIT
            self.stroke_matrix = style.stroke_matrix


class TextObjectState(ReprFields, metaclass=RecordType, frozen=False):
    active: bool = False
    group_open: bool = False
    group_pending: bool = False
    clipping_subpaths: list[CapturedSubpath] = field(default_factory=list)
    object_id: int | None = None

    __hash__ = None  # type: ignore[assignment]


class ProgramLowering:
    __slots__ = (
        "display_list",
        "include_text",
        "explicit_text_boundaries",
        "text",
        "text_stack",
        "glyph_scope_depth",
        "style_lowering",
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
        self.style_lowering: GlyphStyleLowering | None = None

    def lower(self, commands: tuple[PageCommand, ...]) -> None:
        iterator = iter(commands)
        command = next(iterator, END_OF_COMMANDS)
        while command is not END_OF_COMMANDS:
            if type(command) is GlyphObservation:
                command = self.lower_glyphs(command, iterator)
                continue
            command_type = type(command)
            if (
                command_type is not CapturedTextBoundary
                and not self.include_text
                and self.glyph_scope_depth
            ):
                command = next(iterator, END_OF_COMMANDS)
                continue
            lowering = COMMAND_LOWERINGS.get(command_type)
            if lowering is None:
                lowering = inherited_lowering(command_type)
            lowering(self, command)
            command = next(iterator, END_OF_COMMANDS)
        self.finish_text(len(commands))

    def flush_text_clip(self, seqno: int) -> None:
        subpaths = self.text.clipping_subpaths
        if not subpaths:
            return
        self.display_list.append(
            "clip", seqno, path=CapturedPath(list(subpaths)), fill_rule="nonzero"
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
        self.display_list.append_text(
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
        self.lower_glyphs(glyph, iter(()))

    def lower_glyphs(self, glyph: GlyphObservation, commands: Iterator[PageCommand]) -> Any:
        """Lower glyph and the glyphs and text runs that follow it in commands; return the
        first other command after them, or END_OF_COMMANDS.

        Each glyph whose outline the font gives becomes a path item; glyphs in a clipping
        render mode add their outline to the text clip; any other glyph that paints
        becomes a bitmap item.
        """
        if not self.include_text and self.glyph_scope_depth:
            for command in commands:
                if type(command) is not GlyphObservation:
                    return command
            return END_OF_COMMANDS
        include_text = self.include_text
        explicit_boundaries = self.explicit_text_boundaries
        display_list = self.display_list
        append_item = display_list.items.append
        append_plain_fill = display_list.append_plain_fill
        build_item = PATH_PAINT_LAYOUT.build
        text = self.text
        lowering = self.style_lowering
        while True:
            style = glyph.style
            if lowering is None or lowering.style is not style:
                lowering = self.style_lowering = GlyphStyleLowering(
                    style, display_list, include_text
                )
            seqno = glyph.seqno
            text_object_id = lowering.text_object_id
            if (
                not explicit_boundaries
                and text.object_id is not None
                and text_object_id != text.object_id
            ):
                self.flush_text_clip(seqno)
            text.object_id = text_object_id
            visible = glyph.visible
            glyph_paints = lowering.paints and visible is not False
            if glyph_paints and text.group_pending:
                self.open_pending_text_group(seqno)
            glyph_group_open = glyph_paints and explicit_boundaries and not text.group_open
            if glyph_group_open:
                self.begin_text_group(seqno, knockout=False)
            try:
                # The glyph's outline, when its font gives one.
                mode = lowering.mode
                if (
                    (visible is False and not lowering.clip_glyph)
                    or mode == 3
                    or (not include_text and mode < 4)
                ):
                    pass
                elif (
                    glyph.paint_glyph
                    and (transform := glyph.glyph_transform) is not None
                    and (outline_arrays := lowering.outline_arrays) is not None
                    and (
                        code := (
                            glyph.bitmap_code
                            if glyph.bitmap_code is not None
                            else glyph.cid
                            if glyph.cid is not None
                            else glyph.char_code
                        )
                    )
                    is not None
                    and (arrays := outline_arrays(code, glyph.gid, glyph.text)) is not None
                ):
                    a, b, c, d, e, f = transform
                    # Windings that never mix stay so wherever the outline is placed: its
                    # edges come as SimpleEdges, and the coverage kernels need not look for
                    # rows where they do.
                    column_x, column_y, edge_array, kept, dropped, bounds = glyph_outline_edges(
                        arrays.xs, arrays.ys, a, b, c, d, e, f, arrays.spans, arrays.simple
                    )
                    if edge_array is not None:
                        path = CapturedPath.deferred_outline(column_x, column_y, kept)
                        if mode >= 4:
                            text.clipping_subpaths.extend(path.subpaths)
                        if lowering.outline_paints and visible is not False:
                            # PathPaintItem's __init__ only assigns its fields, in this
                            # order.
                            item = build_item(
                                (
                                    lowering.paint_kind,
                                    seqno,
                                    path.bbox() if dropped else bounds,
                                    path,
                                    *lowering.fields,
                                    edge_array,
                                    lowering.miter_limit,
                                    lowering.stroke_matrix,
                                )
                            )
                            if lowering.plain_fill and not dropped and bounds is not None:
                                # The bounds are floats; x - x is 0.0 for each only when
                                # all four are finite.
                                x0, y0, x1, y1 = bounds
                                if x0 - x0 + (y0 - y0) + (x1 - x1) + (y1 - y1) == 0.0:
                                    append_plain_fill(item, x0, y0, x1, y1)
                                else:
                                    append_item(item)
                            else:
                                append_item(item)
                    elif include_text and glyph.text_render_mode not in NON_PAINTING_RENDER_MODES:
                        self.append_glyph_bitmap(glyph)
                elif include_text and glyph.text_render_mode not in NON_PAINTING_RENDER_MODES:
                    self.append_glyph_bitmap(glyph)
            finally:
                if glyph_group_open:
                    display_list.append("group-end", seqno)
            command = next(commands, END_OF_COMMANDS)
            # A text run's item comes between its glyphs' items.
            while type(command) is TextRun:
                self.lower_text_run(command)
                command = next(commands, END_OF_COMMANDS)
            if type(command) is not GlyphObservation:
                return command
            glyph = command

    def append_glyph_bitmap(self, glyph: GlyphObservation) -> None:
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

# SPDX-License-Identifier: AGPL-3.0-only
"""Pure capture-to-display conversion shared by pages and repeated pattern cells."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from core_pdf.impl._impl.capture.program import CapturedProgram
from core_pdf.impl._impl.capture.records import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedPath,
    CapturedSoftMask,
    CapturedSubpath,
    CapturedTextBoundary,
)
from core_pdf.impl._impl.model.glyphs import GlyphObservation
from core_pdf.impl._impl.model.runs import TextRun
from core_pdf.impl._impl.render.display import DisplayList
from core_pdf.impl._impl.render.model import (
    DisplayItem,
    DisplayListItem,
    ImagePaintItem,
    PathPaintItem,
)
from core_pdf.impl._impl.render.paths import internal_translate_rect
from core_pdf_spec.s_07_content.model import NON_PAINTING_RENDER_MODES
from core_pdf_spec.s_08_graphics.geometry import unit_square_placement


def internal_glyph_outline_path(glyph: GlyphObservation) -> CapturedPath | None:
    """Resolve and transform one captured embedded-font outline."""
    if not glyph.paint_glyph:
        return None
    transform = glyph.glyph_transform
    resolver = getattr(glyph.font_decoder, "glyph_outline", None)
    if transform is None or not callable(resolver):
        return None
    code = glyph.bitmap_code
    if code is None:
        code = glyph.cid if glyph.cid is not None else glyph.char_code
    if code is None:
        return None
    contours = resolver(code, glyph.gid, glyph.text)
    if not contours:
        return None
    subpaths: list[CapturedSubpath] = []
    for contour in contours:
        if len(contour) < 2:
            continue
        subpath = CapturedSubpath(list(contour), closed=True).transformed(transform)
        points = subpath.points
        if len(points) >= 2 and points[0] == points[-1]:
            points.pop()
        if len(points) >= 2:
            subpaths.append(subpath)
    return CapturedPath(subpaths) if subpaths else None


def internal_append_glyph_paint(
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
    path = internal_glyph_outline_path(glyph)
    if path is None:
        return False
    if mode >= 4:
        clipping_subpaths.extend(path.subpaths)
    if not include_paint or mode in NON_PAINTING_RENDER_MODES or glyph.visible is False:
        return True
    paint_kind = "fill" if mode in {0, 4} else "stroke" if mode in {1, 5} else "fillstroke"
    display_list.append(
        paint_kind,
        glyph.seqno,
        bbox=path.bbox(),
        path=path,
        fill=glyph.fill,
        fill_opacity=glyph.fill_opacity,
        stroke_color=glyph.stroke_color,
        stroke_opacity=glyph.stroke_opacity,
        line_width=glyph.line_width,
        line_cap=glyph.line_cap,
        line_join=glyph.line_join,
        dash_pattern=glyph.dash_pattern,
        fill_rule="nonzero",
        blend_mode=glyph.blend_mode,
        soft_mask_alpha=glyph.soft_mask_alpha,
        graphics_soft_mask=glyph.graphics_soft_mask
        if isinstance(glyph.graphics_soft_mask, CapturedSoftMask)
        else None,
        alpha_is_shape=glyph.alpha_is_shape,
    )
    return True


def append_captured_program(
    display_list: DisplayList, page_program: CapturedProgram, *, include_text: bool
) -> None:
    """Translate page and appearance captures through the same ordered paint path."""
    commands = page_program.commands
    text_clipping_subpaths: list[CapturedSubpath] = []
    current_text_object_id: int | None = None
    explicit_text_boundaries = bool(page_program.text_boundaries)
    text_active = False
    text_group_open = False
    glyph_scope_depth = 0
    text_stream_stack: list[tuple[bool, bool, list[CapturedSubpath], int | None]] = []

    def append_text_run(run: TextRun) -> None:
        display_list.append(
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

    def flush_text_clip(seqno: int) -> None:
        if not text_clipping_subpaths:
            return
        display_list.append(
            "clip",
            seqno,
            path=CapturedPath(list(text_clipping_subpaths)),
            fill_rule="nonzero",
        )
        text_clipping_subpaths.clear()

    def finish_text(seqno: int) -> None:
        nonlocal text_active, text_group_open, current_text_object_id
        if text_group_open:
            display_list.append("group-end", seqno)
        # ISO 32000-2 9.3.6/9.3.8: paint the text object before its
        # accumulated outlines modify the clipping path at ET.
        flush_text_clip(seqno)
        text_active = False
        text_group_open = False
        current_text_object_id = None

    def begin_text_group(seqno: int, *, knockout: bool) -> None:
        # Text and Type 3 glyph groups retain transparency on their children;
        # outer composition is always Normal, alpha=1 and no soft mask.
        display_list.append(
            "group-begin",
            seqno,
            fill_opacity=1.0,
            blend_mode=None,
            group_isolated=False,
            group_knockout=knockout,
            # Even TK=false glyphs are elementary objects: their stroke pieces
            # and combined fill/stroke need one geometric coverage result.
            group_track_shape=True,
        )

    for command in commands:
        if isinstance(command, CapturedTextBoundary):
            kind = command.kind
            if kind == "stream-begin":
                text_stream_stack.append(
                    (text_active, text_group_open, text_clipping_subpaths, current_text_object_id)
                )
                text_active = text_group_open = False
                text_clipping_subpaths = []
                current_text_object_id = None
                display_list.append("scope-begin", command.seqno)
            elif kind == "stream-end":
                finish_text(command.seqno)
                display_list.append("scope-end", command.seqno)
                if text_stream_stack:
                    text_active, text_group_open, text_clipping_subpaths, current_text_object_id = (
                        text_stream_stack.pop()
                    )
            elif kind == "begin":
                finish_text(command.seqno)
                text_active = True
                text_group_open = include_text and command.knockout
                if text_group_open:
                    begin_text_group(command.seqno, knockout=True)
            elif kind == "end":
                finish_text(command.seqno)
            elif kind == "glyph-begin":
                glyph_scope_depth += 1
                if include_text:
                    begin_text_group(command.seqno, knockout=False)
            elif kind == "glyph-end":
                if glyph_scope_depth:
                    if include_text:
                        display_list.append("group-end", command.seqno)
                    glyph_scope_depth -= 1
            continue
        if not include_text and glyph_scope_depth:
            # Type 3 paint is captured as paths/images, but remains text for
            # the renderer's include_text option. Scope records stay balanced.
            continue
        if not include_text and isinstance(command, TextRun):
            continue
        if isinstance(command, TextRun):
            append_text_run(command)
        elif isinstance(command, GlyphObservation):
            glyph = command
            glyph_text_object_id = glyph.text_object_id
            if (
                not explicit_text_boundaries
                and current_text_object_id is not None
                and glyph_text_object_id != current_text_object_id
            ):
                flush_text_clip(glyph.seqno)
            current_text_object_id = glyph_text_object_id
            glyph_group_open = include_text and explicit_text_boundaries and not text_group_open
            if glyph_group_open:
                begin_text_group(glyph.seqno, knockout=False)
            try:
                if internal_append_glyph_paint(
                    display_list,
                    glyph,
                    text_clipping_subpaths,
                    include_paint=include_text,
                ):
                    continue
                if not include_text or glyph.text_render_mode in NON_PAINTING_RENDER_MODES:
                    continue
                bitmap = glyph.resolved_bitmap()
                if not bitmap:
                    continue
                display_list.append(
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
                    graphics_soft_mask=glyph.graphics_soft_mask
                    if isinstance(glyph.graphics_soft_mask, CapturedSoftMask)
                    else None,
                    alpha_is_shape=glyph.alpha_is_shape,
                    visible=glyph.visible,
                    bitmap=bitmap,
                    bitmap_width=glyph.bitmap_width,
                    bitmap_height=glyph.bitmap_height,
                )
            finally:
                if glyph_group_open:
                    display_list.append("group-end", glyph.seqno)
        elif isinstance(command, CapturedDrawing):
            if not text_active:
                flush_text_clip(command.seqno)
            display_list.append_captured_drawing(command)
        else:
            assert isinstance(command, CapturedInlineImage)
            inline_image = command
            bbox, quad = unit_square_placement(inline_image.ctm)
            if not text_active:
                flush_text_clip(inline_image.seqno)
            if not inline_image.paints:
                continue
            display_list.append(
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
    finish_text(len(commands))


def internal_translated_soft_mask(
    mask: CapturedSoftMask | None, tx: float, ty: float
) -> CapturedSoftMask | None:
    """Move a mask with its repeated paint without recapturing its program."""
    if mask is None or (tx == 0 and ty == 0):
        return mask
    return replace(mask, offset=(mask.offset[0] + tx, mask.offset[1] + ty))


def translated_command(
    item: DisplayItem, tx: float, ty: float, parent_blend_mode: str | None = None
) -> DisplayItem:
    """Place one cell command without mutating the pattern's shared captures."""
    if isinstance(item, PathPaintItem):
        return replace(
            item,
            bbox=internal_translate_rect(item.bbox, tx, ty),
            path=item.path.translated(tx, ty) if isinstance(item.path, CapturedPath) else item.path,
            blend_mode=item.blend_mode or parent_blend_mode,
            graphics_soft_mask=internal_translated_soft_mask(item.graphics_soft_mask, tx, ty),
        )
    if isinstance(item, ImagePaintItem):
        return replace(
            item,
            bbox=internal_translate_rect(item.bbox, tx, ty),
            quad=tuple((x + tx, y + ty) for x, y in item.quad) if item.quad else None,
            image_clip=internal_translate_rect(item.image_clip, tx, ty),
            blend_mode=item.blend_mode or parent_blend_mode,
            graphics_soft_mask=internal_translated_soft_mask(item.graphics_soft_mask, tx, ty),
        )
    data: dict[str, Any] = dict(item.data)
    if isinstance(mask := data.get("graphics_soft_mask"), CapturedSoftMask):
        data["graphics_soft_mask"] = internal_translated_soft_mask(mask, tx, ty)
    for key in ("bbox", "rect"):
        if key in data:
            data[key] = internal_translate_rect(data[key], tx, ty)
    path = data.get("path")
    if isinstance(path, CapturedPath):
        data["path"] = path.translated(tx, ty)
    if item.kind == "shading" and isinstance(data.get("dictionary"), dict):
        dictionary = dict(data["dictionary"])
        coords = dictionary.get("Coords")
        if isinstance(coords, (list, tuple)):
            coords = list(coords)
            indexes = (0, 2) if dictionary.get("ShadingType") == 2 else (0, 3)
            for index in indexes:
                if len(coords) > index + 1:
                    coords[index] += tx
                    coords[index + 1] += ty
            dictionary["Coords"] = coords
        if "BBox" in dictionary:
            dictionary["BBox"] = internal_translate_rect(dictionary["BBox"], tx, ty)
        data["dictionary"] = dictionary
    data["blend_mode"] = data.get("blend_mode") or parent_blend_mode
    return DisplayListItem(item.kind, item.seqno, data)

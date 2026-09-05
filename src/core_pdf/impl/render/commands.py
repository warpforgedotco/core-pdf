# SPDX-License-Identifier: AGPL-3.0-only
"""Pure capture-to-display conversion shared by pages and repeated pattern cells."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from core_pdf.impl.model.glyphs import GlyphObservation
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.render.display import DisplayList
from core_pdf.impl.render.model import DisplayItem, DisplayListItem, ImagePaintItem, PathPaintItem
from core_pdf.impl.render.paths import internal_translate_rect
from core_pdf.impl.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedPath,
    CapturedSubpath,
)
from core_pdf.impl.spec.s_07_content.image_capture import unit_square_placement
from core_pdf.impl.spec.s_07_content.page_program import CapturedProgram
from core_pdf.impl.spec.s_07_content.state import internal_NON_PAINTING_RENDER_MODES


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
    if glyph.visible is False:
        return True
    mode = int(glyph.text_render_mode)
    if not include_paint and mode < 4:
        return True
    path = internal_glyph_outline_path(glyph)
    if path is None:
        return False
    if mode >= 4:
        clipping_subpaths.extend(path.subpaths)
    if not include_paint or mode in internal_NON_PAINTING_RENDER_MODES:
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
    )
    return True


def append_captured_program(
    display_list: DisplayList, page_program: CapturedProgram, *, include_text: bool
) -> None:
    """Translate page and appearance captures through the same ordered paint path."""
    commands = page_program.commands
    text_clipping_subpaths: list[CapturedSubpath] = []
    current_text_object_id: int | None = None

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

    for command in commands:
        if not include_text and isinstance(command, TextRun):
            continue
        if isinstance(command, TextRun):
            append_text_run(command)
        elif isinstance(command, GlyphObservation):
            glyph = command
            glyph_text_object_id = glyph.text_object_id
            if (
                current_text_object_id is not None
                and glyph_text_object_id != current_text_object_id
            ):
                flush_text_clip(glyph.seqno)
            current_text_object_id = glyph_text_object_id
            if internal_append_glyph_paint(
                display_list,
                glyph,
                text_clipping_subpaths,
                include_paint=include_text,
            ):
                continue
            if not include_text or glyph.text_render_mode in internal_NON_PAINTING_RENDER_MODES:
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
                visible=glyph.visible,
                bitmap=bitmap,
                bitmap_width=glyph.bitmap_width,
                bitmap_height=glyph.bitmap_height,
            )
        elif isinstance(command, CapturedDrawing):
            flush_text_clip(command.seqno)
            display_list.append_captured_drawing(command)
        else:
            assert isinstance(command, CapturedInlineImage)
            inline_image = command
            bbox, quad = unit_square_placement(inline_image.ctm)
            flush_text_clip(inline_image.seqno)
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
                fill=inline_image.fill,
                fill_opacity=inline_image.fill_opacity,
                bbox=bbox,
                quad=quad,
                raw_data=inline_image.data,
            )
    flush_text_clip(len(commands))


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
        )
    if isinstance(item, ImagePaintItem):
        return replace(
            item,
            bbox=internal_translate_rect(item.bbox, tx, ty),
            quad=tuple((x + tx, y + ty) for x, y in item.quad) if item.quad else None,
            image_clip=internal_translate_rect(item.image_clip, tx, ty),
            blend_mode=item.blend_mode or parent_blend_mode,
        )
    data: dict[str, Any] = dict(item.data)
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

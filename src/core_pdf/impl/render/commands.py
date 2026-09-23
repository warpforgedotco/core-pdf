# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from copy import replace
from typing import Any

import numpy

from core_pdf.impl.capture.program import CapturedProgram
from core_pdf.impl.capture.records import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedPath,
    CapturedSoftMask,
    CapturedSubpath,
    CapturedTextBoundary,
)
from core_pdf.impl.fonts.decoder import GlyphOutlineArrays
from core_pdf.impl.model.glyphs import GlyphObservation, Matrix6
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.render.display import DisplayList
from core_pdf.impl.render.model import (
    DisplayItem,
    DisplayListItem,
    ImagePaintItem,
    PathPaintItem,
)
from core_pdf.impl.render.paths import translate_rect
from core_pdf.impl.types import Rectangle
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
    if callable(array_resolver):
        arrays = array_resolver(code, glyph.gid, glyph.text)
        if arrays is None:
            return None
        return transformed_outline(arrays, transform)
    resolver = getattr(decoder, "glyph_outline", None)
    if not callable(resolver):
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
    if not subpaths:
        return None
    path = CapturedPath(subpaths)
    return path, path.bbox(), None


def transformed_outline(
    arrays: GlyphOutlineArrays, transform: Matrix6
) -> tuple[CapturedPath, Rectangle | None, numpy.ndarray[Any, Any] | None] | None:
    a, b, c, d, e, f = transform
    linear_x, linear_y = arrays.linear_columns(a, b, c, d)
    column_x = linear_x + e
    column_y = linear_y + f
    tx = column_x.tolist()
    ty = column_y.tolist()
    subpaths: list[CapturedSubpath] = []
    edge_blocks: list[numpy.ndarray[Any, Any]] = []
    dropped = False
    for start, end in arrays.spans:
        points = list(zip(tx[start:end], ty[start:end], strict=True))
        if points[0] == points[-1]:
            points.pop()
            end -= 1
        if len(points) >= 2:
            subpaths.append(CapturedSubpath(points, closed=True))
            xs = column_x[start:end]
            ys = column_y[start:end]
            edge_blocks.append(numpy.column_stack((xs[:-1], ys[:-1], xs[1:], ys[1:])))
            if points[0] != points[-1]:
                edge_blocks.append(
                    numpy.array([[xs[-1], ys[-1], xs[0], ys[0]]], dtype=numpy.float64)
                )
        else:
            dropped = True
    if not subpaths:
        return None
    path = CapturedPath(subpaths)
    edges = edge_blocks[0] if len(edge_blocks) == 1 else numpy.concatenate(edge_blocks)
    if dropped:
        return path, path.bbox(), edges
    # The columns are already numpy arrays here, so there is no conversion to
    # pay for: four reductions over a ~130-element ndarray beat four passes
    # over the Python lists by 3.2x, and float() keeps the result type.
    return (
        path,
        (
            float(column_x.min()),
            float(column_y.min()),
            float(column_x.max()),
            float(column_y.max()),
        ),
        edges,
    )


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
    paint_kind = "fill" if mode in {0, 4} else "stroke" if mode in {1, 5} else "fillstroke"
    display_list.append(
        paint_kind,
        glyph.seqno,
        bbox=bbox,
        path=path,
        edge_array=edge_array,
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
        graphics_soft_mask=glyph.graphics_soft_mask,
        alpha_is_shape=glyph.alpha_is_shape,
    )
    return True


def append_captured_program(
    display_list: DisplayList, page_program: CapturedProgram, *, include_text: bool
) -> None:
    commands = page_program.commands
    text_clipping_subpaths: list[CapturedSubpath] = []
    current_text_object_id: int | None = None
    explicit_text_boundaries = bool(page_program.text_boundaries)
    text_active = False
    text_group_open = False
    text_group_pending = False
    glyph_scope_depth = 0
    text_stream_stack: list[tuple[bool, bool, bool, list[CapturedSubpath], int | None]] = []

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
        nonlocal text_active, text_group_open, text_group_pending, current_text_object_id
        if text_group_open:
            display_list.append("group-end", seqno)
        flush_text_clip(seqno)
        text_active = False
        text_group_open = False
        text_group_pending = False
        current_text_object_id = None

    def open_pending_text_group(seqno: int) -> None:
        nonlocal text_group_open, text_group_pending
        if text_group_pending:
            text_group_pending = False
            text_group_open = True
            begin_text_group(seqno, knockout=True)

    def begin_text_group(seqno: int, *, knockout: bool) -> None:
        display_list.append(
            "group-begin",
            seqno,
            fill_opacity=1.0,
            blend_mode=None,
            group_isolated=False,
            group_knockout=knockout,
            group_track_shape=True,
        )

    for command in commands:
        if isinstance(command, CapturedTextBoundary):
            match command.kind:
                case "stream-begin":
                    text_stream_stack.append(
                        (
                            text_active,
                            text_group_open,
                            text_group_pending,
                            text_clipping_subpaths,
                            current_text_object_id,
                        )
                    )
                    text_active = text_group_open = text_group_pending = False
                    text_clipping_subpaths = []
                    current_text_object_id = None
                    display_list.append("scope-begin", command.seqno)
                case "stream-end":
                    finish_text(command.seqno)
                    display_list.append("scope-end", command.seqno)
                    if text_stream_stack:
                        (
                            text_active,
                            text_group_open,
                            text_group_pending,
                            text_clipping_subpaths,
                            current_text_object_id,
                        ) = text_stream_stack.pop()
                case "begin":
                    finish_text(command.seqno)
                    text_active = True
                    text_group_pending = include_text and command.knockout
                case "end":
                    finish_text(command.seqno)
                case "glyph-begin":
                    glyph_scope_depth += 1
                    if include_text:
                        open_pending_text_group(command.seqno)
                        begin_text_group(command.seqno, knockout=False)
                case "glyph-end":
                    if glyph_scope_depth:
                        if include_text:
                            display_list.append("group-end", command.seqno)
                        glyph_scope_depth -= 1
            continue
        if not include_text and glyph_scope_depth:
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
            glyph_paints = (
                include_text
                and glyph.text_render_mode not in NON_PAINTING_RENDER_MODES
                and glyph.visible is not False
            )
            if glyph_paints:
                open_pending_text_group(glyph.seqno)
            glyph_group_open = glyph_paints and explicit_text_boundaries and not text_group_open
            if glyph_group_open:
                begin_text_group(glyph.seqno, knockout=False)
            try:
                if append_glyph_paint(
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
                    graphics_soft_mask=glyph.graphics_soft_mask,
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
            elif command.paints:
                open_pending_text_group(command.seqno)
            display_list.append_captured_drawing(command)
        else:
            assert isinstance(command, CapturedInlineImage)
            inline_image = command
            bbox, quad = unit_square_placement(inline_image.ctm)
            if not text_active:
                flush_text_clip(inline_image.seqno)
            if not inline_image.paints:
                continue
            if text_active:
                open_pending_text_group(inline_image.seqno)
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


def translated_soft_mask(
    mask: CapturedSoftMask | None, tx: float, ty: float
) -> CapturedSoftMask | None:
    if mask is None or (tx == 0 and ty == 0):
        return mask
    return replace(mask, offset=(mask.offset[0] + tx, mask.offset[1] + ty))


def translated_command(
    item: DisplayItem, tx: float, ty: float, parent_blend_mode: str | None = None
) -> DisplayItem:
    if isinstance(item, PathPaintItem):
        return replace(
            item,
            bbox=translate_rect(item.bbox, tx, ty),
            path=item.path.translated(tx, ty) if isinstance(item.path, CapturedPath) else item.path,
            edge_array=(
                item.edge_array + numpy.array((tx, ty, tx, ty))
                if item.edge_array is not None
                else None
            ),
            blend_mode=item.blend_mode or parent_blend_mode,
            graphics_soft_mask=translated_soft_mask(item.graphics_soft_mask, tx, ty),
        )
    if isinstance(item, ImagePaintItem):
        return replace(
            item,
            bbox=translate_rect(item.bbox, tx, ty),
            quad=tuple((x + tx, y + ty) for x, y in item.quad) if item.quad else None,
            image_clip=translate_rect(item.image_clip, tx, ty),
            blend_mode=item.blend_mode or parent_blend_mode,
            graphics_soft_mask=translated_soft_mask(item.graphics_soft_mask, tx, ty),
        )
    data: dict[str, Any] = dict(item.data)
    if (mask := data.get("graphics_soft_mask")) is not None:
        data["graphics_soft_mask"] = translated_soft_mask(mask, tx, ty)
    for key in ("bbox", "rect"):
        if key in data:
            data[key] = translate_rect(data[key], tx, ty)
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
            dictionary["BBox"] = translate_rect(dictionary["BBox"], tx, ty)
        data["dictionary"] = dictionary
    data["blend_mode"] = data.get("blend_mode") or parent_blend_mode
    return DisplayListItem(item.kind, item.seqno, data)

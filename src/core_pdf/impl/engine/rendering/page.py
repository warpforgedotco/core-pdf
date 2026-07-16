# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_pdf.impl.engine.rendering.models import (
    DisplayList,
    RenderedPage,
    RenderOptions,
)
from core_pdf.impl.engine.spec.s_07_content import TextState
from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_08_graphics.matrix import (
    IDENTITY_MATRIX,
    Matrix,
)
from core_pdf.impl.objects import PdfStream


def compose_page(page: Any, options: RenderOptions | None = None) -> RenderedPage:
    options = options or RenderOptions()
    media_box = page.media_box or (0.0, 0.0, page.width, page.height)
    x0, y0, x1, y1 = media_box
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    display_list = DisplayList(width=width, height=height)

    if options.include_text:
        for run in page.chars:
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

    capture_state = None
    if options.include_text and hasattr(page, "capture_text_state"):
        try:
            capture_state = page.capture_text_state()
        except Exception:
            capture_state = None
    if capture_state is not None:
        for glyph in capture_state.glyphs:
            if not glyph.bitmap:
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
                bbox=(
                    glyph.ink_rect.x0,
                    glyph.ink_rect.y0,
                    glyph.ink_rect.x1,
                    glyph.ink_rect.y1,
                ),
                advance_bbox=(
                    glyph.advance_rect.x0,
                    glyph.advance_rect.y0,
                    glyph.advance_rect.x1,
                    glyph.advance_rect.y1,
                ),
                fill_color=glyph.fill,
                visible=glyph.visible,
                bitmap=glyph.bitmap,
                bitmap_width=glyph.bitmap_width,
                bitmap_height=glyph.bitmap_height,
            )

    for drawing in page.get_drawings():
        display_list.append(
            drawing["kind"],
            drawing["seqno"],
            bbox=drawing["rect"],
            fill=drawing["fill"],
            fill_pattern=drawing.get("fill_pattern"),
            fill_opacity=drawing["fill_opacity"],
            stroke_color=drawing["stroke_color"],
            stroke_pattern=drawing.get("stroke_pattern"),
            stroke_opacity=drawing["stroke_opacity"],
            line_width=drawing["line_width"],
            line_cap=drawing.get("line_cap"),
            line_join=drawing.get("line_join"),
            dash_pattern=drawing.get("dash_pattern"),
            fill_rule=drawing.get("fill_rule"),
            blend_mode=drawing.get("blend_mode"),
            soft_mask_alpha=drawing.get("soft_mask_alpha"),
            raw_data=drawing.get("raw_data"),
            dictionary=drawing.get("dictionary"),
            path=drawing.get("path"),
            items=drawing["items"],
        )

    graphics = page.get_graphics() if hasattr(page, "get_graphics") else None
    for inline_image in getattr(graphics, "inline_images", []):
        display_list.append(
            "inline-image",
            inline_image["seqno"],
            dictionary=dict(inline_image["dictionary"]),
            data=inline_image["data"],
            ctm=inline_image["ctm"],
            xobject_depth=inline_image["xobject_depth"],
            bbox=inline_image.get("bbox"),
            soft_mask_alpha=inline_image.get("soft_mask_alpha"),
            raw_data=inline_image.get("data"),
        )

    display_list.items.sort(key=lambda item: item.seqno)

    def append_capture(state: TextState) -> None:
        if options.include_text:
            for run in state.runs:
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
        for drawing in state.drawings:
            display_list.append(
                drawing.kind,
                drawing.seqno,
                bbox=drawing.rect,
                fill=drawing.fill,
                fill_pattern=drawing.fill_pattern,
                fill_opacity=drawing.fill_opacity,
                stroke_color=drawing.stroke_color,
                stroke_pattern=drawing.stroke_pattern,
                stroke_opacity=drawing.stroke_opacity,
                line_width=drawing.line_width,
                line_cap=drawing.line_cap,
                line_join=drawing.line_join,
                dash_pattern=drawing.dash_pattern,
                fill_rule=drawing.fill_rule,
                blend_mode=drawing.blend_mode,
                soft_mask_alpha=drawing.soft_mask_alpha,
                raw_data=drawing.raw_data,
                dictionary=drawing.dictionary,
                path=drawing.path,
                items=drawing.items,
            )

    def append_form_appearance(
        appearance: Any,
        rect: tuple[float, float, float, float],
        appearance_state: Any | None = None,
    ) -> bool:
        if not isinstance(appearance, dict):
            return False
        normal = lookup_dict_key(appearance, "N")
        if isinstance(normal, dict):
            selected = None
            if appearance_state is not None:
                state_name = normalize_pdf_name(appearance_state)
                if state_name is not None:
                    selected = lookup_dict_key(normal, state_name)
            if selected is None:
                off_appearance = lookup_dict_key(normal, "Off")
                yes_appearance = lookup_dict_key(normal, "Yes")
                selected = (
                    off_appearance
                    if off_appearance is not None
                    else yes_appearance
                    if yes_appearance is not None
                    else next(iter(normal.values()), None)
                )
            normal = selected
        if not isinstance(normal, PdfStream):
            normal = page.document.resolver.resolve(normal)
        if not isinstance(normal, PdfStream):
            return False
        form_dict = page.document.resolver.resolve_dict(normal.dictionary) or {}
        bbox = page.document.resolver.resolve_box(lookup_dict_key(form_dict, "BBox"))
        if bbox is None:
            bbox = rect
        try:
            matrix_operand = lookup_dict_key(form_dict, "Matrix")
            if isinstance(matrix_operand, (list, tuple)) and len(matrix_operand) > 6:
                matrix_operand = matrix_operand[:6]
            matrix = (
                Matrix.from_operand(matrix_operand)
                if matrix_operand is not None
                else IDENTITY_MATRIX
            )
        except ValueError:
            matrix = IDENTITY_MATRIX
        bx0, by0, bx1, by1 = bbox
        rx0, ry0, rx1, ry1 = rect
        bw = bx1 - bx0
        bh = by1 - by0
        if bw == 0 or bh == 0:
            return False
        if not hasattr(page.document.resolver, "resolve"):
            return False
        scale = Matrix(
            (rx1 - rx0) / bw,
            0.0,
            0.0,
            (ry1 - ry0) / bh,
            rx0 - bx0 * ((rx1 - rx0) / bw),
            ry0 - by0 * ((ry1 - ry0) / bh),
        )
        nested_ctm = matrix.multiply(scale)
        state = TextState(
            page.document,
            getattr(page, "page_dict", {}),
            capture_runs=True,
            capture_graphics=True,
            decoder_cache=getattr(page.document, "decoder_cache", {}),
        )
        resources = (
            page.document.resolver.resolve_dict(lookup_dict_key(form_dict, "Resources"))
            or page.cached_resources
        )
        state.consume_stream(normal, resources, nested_ctm, 0)
        append_capture(state)
        return True

    if options.include_layers:
        for field in page.get_fields():
            widget = field.widget or field.dict
            rect = field.rect
            appearance = None
            appearance_state = None
            if isinstance(widget, dict):
                appearance = lookup_dict_key(widget, "AP")
                appearance_state = lookup_dict_key(widget, "AS")
            if rect is None:
                continue
            display_list.append(
                "widget",
                -1,
                name=field.name,
                field_type=field.type,
                value=field.value_text,
                rect=rect,
                widget=dict(widget) if isinstance(widget, dict) else {},
                appearance=appearance,
                appearance_rendered=append_form_appearance(appearance, rect, appearance_state)
                if appearance is not None
                else False,
            )
    if options.include_annotations:
        for annot in page.get_annotations():
            appearance = lookup_dict_key(annot.dict, "AP") if isinstance(annot.dict, dict) else None
            appearance_state = (
                lookup_dict_key(annot.dict, "AS") if isinstance(annot.dict, dict) else None
            )
            rendered = False
            if appearance is not None:
                rendered = append_form_appearance(
                    appearance, annot.rect or (0.0, 0.0, 0.0, 0.0), appearance_state
                )
            display_list.append(
                "annotation",
                -1,
                subtype=annot.subtype,
                rect=annot.rect,
                contents=annot.contents,
                appearance=appearance,
                appearance_rendered=rendered,
            )
    return RenderedPage(
        page_number=getattr(page, "page_number", 0),
        width=width,
        height=height,
        rotate=(getattr(page, "rotation", 0) + options.rotate) % 360,
        display_list=display_list,
        metadata={
            "crop": options.crop,
            "group_alpha": (
                page.resolve_transparency_group_alpha()
                if hasattr(page, "resolve_transparency_group_alpha")
                else None
            ),
        },
    )

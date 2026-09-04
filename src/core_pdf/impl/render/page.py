# SPDX-License-Identifier: AGPL-3.0-only
"""Rendered-page rasterization and page-program composition."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

from core_pdf.impl.exceptions import PdfRasterTooLargeError
from core_pdf.impl.model.geometry import rect_tuple
from core_pdf.impl.model.glyphs import GlyphObservation
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.render.blend import (
    internal_color_rgba,
    internal_scale_rgba_alpha,
)
from core_pdf.impl.render.clipping import internal_ClipState
from core_pdf.impl.render.display import (
    RASTER_CONTROL_KINDS,
    DisplayList,
    internal_display_item_box,
)
from core_pdf.impl.render.model import (
    DisplayItem,
    DisplayListItem,
    ImagePaintItem,
    PathPaintItem,
    RasterImage,
    RenderOptions,
)
from core_pdf.impl.render.target import internal_RasterTarget
from core_pdf.impl.runtime.array_views import (
    uint8_image_view,
)
from core_pdf.impl.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedPath,
    CapturedSubpath,
)
from core_pdf.impl.spec.s_07_content.page_program import PageCommandKind, PageProgram
from core_pdf.impl.spec.s_07_content.state import (
    TextState,
    internal_NON_PAINTING_RENDER_MODES,
)
from core_pdf.impl.spec.s_07_document.annotation_appearance import (
    select_appearance_stream,
)
from core_pdf.impl.spec.s_08_graphics.image_metadata import (
    pdf_number,
)
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix


@dataclass(slots=True)
class RenderedPage:
    page_number: int
    width: float
    height: float
    rotate: int
    display_list: DisplayList
    metadata: dict[str, Any] = field(default_factory=dict)

    def internal_render_items(
        self,
        crop: tuple[float, float, float, float] | None,
    ) -> list[DisplayItem] | tuple[DisplayItem, ...]:
        if crop is None:
            return self.display_list.items
        selected: list[DisplayItem] = []
        for item in self.display_list.items:
            if type(item) is DisplayListItem and item.kind == "text":
                continue
            box = internal_display_item_box(item)
            always_render = box is None or (
                type(item) is DisplayListItem and item.kind in RASTER_CONTROL_KINDS
            )
            outside_crop = box is not None and (
                box[2] <= crop[0] or box[0] >= crop[2] or box[3] <= crop[1] or box[1] >= crop[3]
            )
            if always_render or not outside_crop:
                selected.append(item)
        return selected

    def internal_effective_crop(
        self,
        crop: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float, float, float] | None:
        value: object = crop if crop is not None else self.metadata.get("crop")
        parsed = rect_tuple(value)
        if parsed is not None:
            x0, y0, x1, y1 = parsed
            if x1 > x0 and y1 > y0:
                return x0, y0, x1, y1
        return None

    def unrotated_raster_size(
        self,
        scale: float = 1.0,
        *,
        crop: tuple[float, float, float, float] | None = None,
    ) -> tuple[int, int]:
        """Return the raster size before applying the page rotation."""
        scale = max(0.01, float(scale))
        effective_crop = self.internal_effective_crop(crop)
        if effective_crop is not None:
            width = max(1, int(round((effective_crop[2] - effective_crop[0]) * scale)))
            height = max(1, int(round((effective_crop[3] - effective_crop[1]) * scale)))
            return width, height
        return (
            max(1, int(round(self.width * scale))),
            max(1, int(round(self.height * scale))),
        )

    def raster_size(
        self,
        scale: float = 1.0,
        *,
        crop: tuple[float, float, float, float] | None = None,
    ) -> tuple[int, int]:
        """Return the width and height of the bytes produced by ``rasterize``."""
        width, height = self.unrotated_raster_size(scale, crop=crop)
        return (height, width) if self.rotate % 180 else (width, height)

    def validate_raster_size(
        self,
        scale: float = 1.0,
        max_pixels: int | None = None,
        *,
        crop: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Reject a raster request before allocating an oversized RGBA canvas."""
        if max_pixels is None or max_pixels <= 0:
            return
        width, height = self.unrotated_raster_size(scale, crop=crop)
        pixels = width * height
        if pixels > max_pixels:
            raise PdfRasterTooLargeError(
                "PDF page would render to too many pixels for safe processing: "
                f"page={self.page_number}, pixels={pixels}, maximum={max_pixels}. "
                "Try splitting the PDF, reducing the page dimensions, or using a lower render DPI."
            )

    def rasterize(
        self,
        *,
        background: tuple[int, int, int, int] = (255, 255, 255, 0),
        scale: float = 1.0,
        max_pixels: int | None = None,
        crop: tuple[float, float, float, float] | None = None,
    ) -> RasterImage:
        scale = max(0.01, float(scale))
        self.validate_raster_size(scale, max_pixels, crop=crop)
        crop = self.internal_effective_crop(crop)
        if crop is not None:
            crop_x0, crop_y0, internal_crop_x1, crop_y1 = crop
        else:
            crop_x0 = 0.0
            crop_y0 = 0.0
            crop_y1 = self.height
        width, height = self.unrotated_raster_size(scale, crop=crop)
        background_bytes = bytes(background)
        pixels = bytearray(background_bytes * (width * height))
        page_pixels = uint8_image_view(pixels, (height, width, 4))

        page_group_alpha = self.metadata.get("group_alpha")
        if not pdf_number(page_group_alpha):
            page_group_alpha = None
        clip_state = internal_ClipState(
            crop_x0=crop_x0,
            crop_y1=crop_y1,
            scale=scale,
            width=width,
            height=height,
        )
        raster_target = internal_RasterTarget(
            pixels,
            page_group_alpha,
            clip=clip_state,
            width=width,
            height=height,
            scale=scale,
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            crop_y1=crop_y1,
            page_view=page_pixels,
        )
        buffer_stack = raster_target.buffer_stack
        rotate = self.rotate % 360

        clip_state_stack: list[int] = []

        for item in self.internal_render_items(crop):
            if type(item) is PathPaintItem:
                raster_target.paint_typed_path(item)
                continue
            if type(item) is ImagePaintItem:
                raster_target.blit_image(item)
                continue
            generic_item = cast(DisplayListItem, item)
            data = generic_item.data
            blend_mode = data.get("blend_mode")
            if blend_mode == "Normal":
                blend_mode = None
            if generic_item.kind == "state-push":
                clip_state_stack.append(clip_state.depth)
                continue
            if generic_item.kind == "state-pop":
                if clip_state_stack:
                    clip_state.restore(clip_state_stack.pop())
                else:
                    clip_state.restore(0)
                continue
            if generic_item.kind == "clip":
                path = data.get("path")
                if type(path) is CapturedPath and path.has_segments():
                    clip_state.push(path, data.get("fill_rule") or "nonzero")
                continue
            if generic_item.kind == "group-begin":
                # A transparency group starts from a transparent backdrop, not
                # the page background: composite_group skips zero-alpha pixels,
                # so an opaque buffer would blend the whole page as though the
                # group had painted every pixel, flattening what it did paint.
                raster_target.push_group(
                    bytearray(width * height * 4),
                    data.get("fill_opacity"),
                    data.get("blend_mode"),
                )
                continue
            if generic_item.kind == "group-end":
                if len(buffer_stack) > 1:
                    child, group_alpha, group_blend_mode = raster_target.pop_group()
                    raster_target.composite_group(
                        child,
                        group_alpha if pdf_number(group_alpha) else data.get("fill_opacity"),
                        group_blend_mode
                        if type(group_blend_mode) is str
                        else data.get("blend_mode"),
                    )
                continue
            if generic_item.kind == "glyph":
                if data.get("visible") is False:
                    continue
                rgba = internal_color_rgba(data.get("fill_color"), None)
                raster_target.draw_glyph_bitmap(
                    data.get("bbox"),
                    data.get("bitmap"),
                    rgba,
                    blend_mode,
                    data.get("bitmap_width"),
                    data.get("bitmap_height"),
                )
            elif generic_item.kind == "shading":
                raster_target.paint_shading(data, blend_mode)
            elif generic_item.kind == "stroke":
                path = data.get("path")
                if type(path) is not CapturedPath:
                    continue
                stroke_rgba = internal_color_rgba(
                    data.get("stroke_color"), data.get("stroke_opacity")
                )
                soft_mask_alpha = data.get("soft_mask_alpha")
                if pdf_number(soft_mask_alpha):
                    stroke_rgba = internal_scale_rgba_alpha(stroke_rgba, soft_mask_alpha)
                raster_target.stroke_path(
                    path,
                    float(data.get("line_width") or 1.0),
                    stroke_rgba,
                    data.get("dash_pattern"),
                    blend_mode,
                    int(data.get("line_cap") or 0),
                    int(data.get("line_join") or 0),
                )
            elif generic_item.kind in {"fill", "fillstroke"}:
                rgba = internal_color_rgba(
                    data.get("fill") or data.get("fill_color"),
                    data.get("fill_opacity"),
                )
                soft_mask_alpha = data.get("soft_mask_alpha")
                if pdf_number(soft_mask_alpha):
                    rgba = internal_scale_rgba_alpha(rgba, soft_mask_alpha)
                path = data.get("path")
                if type(path) is not CapturedPath:
                    continue
                pattern_painted = generic_item.kind in {
                    "fill",
                    "fillstroke",
                } and raster_target.paint_fill_pattern(data, blend_mode)
                if generic_item.kind in {"fill", "fillstroke"} and not pattern_painted:
                    raster_target.fill_path(
                        path,
                        rgba,
                        blend_mode,
                        data.get("fill_rule") or "nonzero",
                    )
                if generic_item.kind == "fillstroke":
                    stroke_rgba = internal_color_rgba(
                        data.get("stroke_color"), data.get("stroke_opacity")
                    )
                    if pdf_number(soft_mask_alpha):
                        stroke_rgba = internal_scale_rgba_alpha(stroke_rgba, soft_mask_alpha)
                    raster_target.stroke_path(
                        path,
                        float(data.get("line_width") or 1.0),
                        stroke_rgba,
                        data.get("dash_pattern"),
                        blend_mode,
                        int(data.get("line_cap") or 0),
                        int(data.get("line_join") or 0),
                    )
            # An annotation whose appearance stream could not be rendered paints
            # nothing. This used to drop a translucent gold rectangle over every
            # such annotation and a blue one over every widget -- diagnostic
            # scaffolding, not page content. On a fillable form that tinted every
            # field: composited over white the pair produce (224, 226, 159),
            # which is exactly the wash that covered IRS-2023-Form-1095-A.
            # Readers draw nothing here, so neither do we; the items stay in the
            # display list because their /Rect still carries annotation geometry.
        if rotate in {90, 180, 270}:
            rotated = bytearray(background_bytes * (width * height))
            source_pixels = memoryview(raster_target.pixels).cast("I")
            rotated_pixels = memoryview(rotated).cast("I")
            if rotate == 90:
                for x in range(width):
                    start = x * height
                    rotated_pixels[start : start + height] = source_pixels[x::width][::-1]
            elif rotate == 270:
                for x in range(width):
                    start = (width - 1 - x) * height
                    rotated_pixels[start : start + height] = source_pixels[x::width]
            else:
                for y in range(height):
                    source = y * width
                    target = (height - 1 - y) * width
                    rotated_pixels[target : target + width] = source_pixels[
                        source : source + width
                    ][::-1]
            result = RasterImage(
                rotated,
                height if rotate in {90, 270} else width,
                width if rotate in {90, 270} else height,
                4,
            )
        else:
            # A rotation that is not a multiple of 90 rasterizes unrotated; the
            # reported dimensions match the buffer's unrotated layout.
            result = RasterImage(raster_target.pixels, width, height, 4)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "rotate": self.rotate,
            "display_list": [
                {
                    "kind": item.kind,
                    "seqno": item.seqno,
                    "data": (
                        item.to_data()
                        if isinstance(item, (PathPaintItem, ImagePaintItem))
                        else dict(item.data)
                    ),
                }
                for item in self.display_list.items
            ],
            "metadata": dict(self.metadata),
        }


# ===== page =====


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
    display_list: DisplayList, glyph: GlyphObservation, clipping_subpaths: list[CapturedSubpath]
) -> bool:
    if glyph.visible is False:
        return True
    path = internal_glyph_outline_path(glyph)
    if path is None:
        return False
    mode = int(glyph.text_render_mode)
    if mode >= 4:
        clipping_subpaths.extend(path.subpaths)
    if mode in internal_NON_PAINTING_RENDER_MODES:
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


def compose_page(
    page: Any,
    options: RenderOptions | None = None,
    *,
    page_program: PageProgram | None = None,
    fields: Iterable[Any] | None = None,
    annotations: Iterable[Any] | None = None,
) -> RenderedPage:
    options = options or RenderOptions()
    media_box = page.media_box or (0.0, 0.0, page.width, page.height)
    x0, y0, x1, y1 = media_box
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    display_list = DisplayList(width=width, height=height)

    if page_program is None and hasattr(page, "get_page_program"):
        page_program = page.get_page_program()
    if page_program is None:
        raise ValueError("compose_page requires the canonical page program")
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
        kind = command.kind
        if not options.include_text and kind in (PageCommandKind.TEXT, PageCommandKind.GLYPH):
            continue
        payload = command.payload
        if kind == PageCommandKind.TEXT:
            assert isinstance(payload, TextRun)
            append_text_run(payload)
        elif kind == PageCommandKind.GLYPH:
            assert isinstance(payload, GlyphObservation)
            glyph = payload
            glyph_text_object_id = glyph.text_object_id
            if (
                current_text_object_id is not None
                and glyph_text_object_id != current_text_object_id
            ):
                flush_text_clip(glyph.seqno)
            current_text_object_id = glyph_text_object_id
            if internal_append_glyph_paint(display_list, glyph, text_clipping_subpaths):
                continue
            if glyph.text_render_mode in internal_NON_PAINTING_RENDER_MODES:
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
        elif kind in (PageCommandKind.DRAWING, PageCommandKind.IMAGE):
            assert isinstance(payload, CapturedDrawing)
            flush_text_clip(payload.seqno)
            display_list.append_captured_drawing(payload)
        elif kind == PageCommandKind.INLINE_IMAGE:
            assert isinstance(payload, CapturedInlineImage)
            inline_image = payload
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
                bbox=None,
                raw_data=inline_image.data,
            )
    flush_text_clip(len(commands))

    def append_capture(state: TextState) -> None:
        if options.include_text:
            for run in state.runs:
                append_text_run(run)
        for drawing in state.drawings:
            display_list.append_captured_drawing(drawing)

    def append_form_appearance(
        appearance: Any,
        rect: tuple[float, float, float, float],
        appearance_state: Any | None = None,
    ) -> bool:
        # One implementation of 12.5.5's substate rule, shared with the capture
        # path. This used to have its own, which fell back to Off, then Yes,
        # then whatever /N happened to hold first when /AS named a state /N did
        # not contain -- so a checkbox with /AS /Off whose /N carries only the
        # checked stream rendered as checked. It must render as nothing.
        normal = select_appearance_stream(page.document.resolver, appearance, appearance_state)
        if normal is None:
            return False
        form_dict = page.document.resolver.resolve_dict(normal.dictionary) or {}
        bbox = page.document.resolver.resolve_box(form_dict.get("BBox"))
        if bbox is None:
            bbox = rect
        try:
            matrix_operand = form_dict.get("Matrix")
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
        state = TextState(page.document)
        resources = (
            page.document.resolver.resolve_dict(form_dict.get("Resources")) or page.resources
        )
        state.consume_stream(normal, resources, nested_ctm, 0)
        append_capture(state)
        return True

    if options.include_layers:
        field_records = fields
        if field_records is None:
            try:
                field_records = page.get_fields()
            except ValueError:
                # A malformed AcroForm must not prevent rendering the page's text and images.
                field_records = ()
        for field in field_records:
            widget = field.widget or field.dict
            rect = field.rect
            appearance = None
            appearance_state = None
            if isinstance(widget, dict):
                appearance = widget.get("AP")
                appearance_state = widget.get("AS")
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
        annotation_records = page.get_annotations() if annotations is None else annotations
        for annot in annotation_records:
            appearance = annot.dict.get("AP") if isinstance(annot.dict, dict) else None
            appearance_state = annot.dict.get("AS") if isinstance(annot.dict, dict) else None
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


__all__ = ("RenderedPage", "compose_page")

# SPDX-License-Identifier: AGPL-3.0-only
"""Rendered-page rasterization and page-program composition."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from core_pdf.impl.exceptions import PdfRasterTooLargeError
from core_pdf.impl.model.geometry import rect_tuple
from core_pdf.impl.render.clipping import internal_ClipState
from core_pdf.impl.render.commands import append_captured_program
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
    CapturedPath,
)
from core_pdf.impl.spec.s_07_content.page_program import PageProgram
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import is_pdf_number


class internal_RenderablePage(Protocol):
    """The stable page geometry required when composing a display list."""

    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...

    @property
    def media_box(self) -> tuple[float, float, float, float] | None: ...


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
        if not is_pdf_number(page_group_alpha):
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
        rotate = self.rotate % 360
        raster_target.paint_items(self.internal_render_items(crop))
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


def compose_page(
    page: internal_RenderablePage,
    options: RenderOptions | None = None,
    *,
    page_program: PageProgram | None = None,
    fields: Iterable[Any] | None = None,
    annotations: Iterable[Any] | None = None,
) -> RenderedPage:
    options = options or RenderOptions()
    fields = tuple(fields) if fields is not None else None
    annotations = tuple(annotations) if annotations is not None else None
    internal_page = cast(Any, page)
    media_box = getattr(page, "media_box", None) or (0.0, 0.0, page.width, page.height)
    x0, y0, x1, y1 = media_box
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    display_list = DisplayList(width=width, height=height)

    if page_program is None and hasattr(internal_page, "get_page_program"):
        capture_inputs: dict[str, Any] = {}
        if fields is not None:
            capture_inputs["fields"] = fields
        if annotations is not None:
            capture_inputs["annotations"] = annotations
        page_program = internal_page.get_page_program(**capture_inputs)
    if page_program is None:
        raise ValueError("compose_page requires the canonical page program")
    selected_appearances = tuple(
        appearance
        for appearance in page_program.appearances
        if (options.include_layers if appearance.kind == "widget" else options.include_annotations)
    )
    # A page's final clipping state does not belong to its annotations. Close
    # the body scope before replaying independent appearance streams.
    if selected_appearances:
        display_list.append("scope-begin", -1)
    append_captured_program(display_list, page_program.body, include_text=options.include_text)
    if selected_appearances:
        display_list.append("scope-end", -1)
    rendered_appearances: set[int] = set()
    for appearance_group in selected_appearances:
        clip_path = CapturedPath()
        ax0, ay0, ax1, ay1 = appearance_group.clip_bbox
        clip_path.rect(ax0, ay0, ax1 - ax0, ay1 - ay0)
        display_list.append("scope-begin", -1, path=clip_path)
        append_captured_program(
            display_list, appearance_group.program, include_text=options.include_text
        )
        display_list.append("scope-end", -1)
        rendered_appearances.add(id(appearance_group.source))

    if options.include_layers:
        field_records = fields
        if field_records is None:
            try:
                field_records = internal_page.get_fields()
            except ValueError:
                # A malformed AcroForm must not prevent rendering the page's text and images.
                field_records = ()
        for field in field_records:
            widget = field.widget or field.dict
            rect = field.rect
            appearance = None
            if isinstance(widget, dict):
                appearance = widget.get("AP")
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
                appearance_rendered=id(widget) in rendered_appearances,
            )
    if options.include_annotations:
        annotation_records = internal_page.get_annotations() if annotations is None else annotations
        for annot in annotation_records:
            appearance = annot.dict.get("AP") if isinstance(annot.dict, dict) else None
            display_list.append(
                "annotation",
                -1,
                subtype=annot.subtype,
                rect=annot.rect,
                contents=annot.contents,
                appearance=appearance,
                appearance_rendered=id(annot.dict) in rendered_appearances,
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
                internal_page.resolve_transparency_group_alpha()
                if hasattr(internal_page, "resolve_transparency_group_alpha")
                else None
            ),
        },
    )


__all__ = ("RenderedPage", "compose_page")

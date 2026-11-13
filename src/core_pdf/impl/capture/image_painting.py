# SPDX-License-Identifier: AGPL-3.0-only

"""TextState methods: painting images, inline images and shadings.

Each function takes the state as self; TextState lists them as its methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl.base.types import (
    PdfName,
)
from core_pdf.impl.base.values import recover_pdf_name
from core_pdf.impl.capture.records import (
    CapturedDrawing,
    CapturedInlineImage,
)
from core_pdf.impl.graphics.color_spec import raw_color_space_paints
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_08_graphics.color import color_space_paints
lazy from core_pdf.impl.graphics.images import image_overrides_graphics_soft_mask
lazy from core_pdf.impl.spec.s_08_graphics.geometry import unit_square_placement
lazy from core_pdf.impl.spec.s_08_graphics.image_spec import ImageSource

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_07_content.inline_images import InlineImage


from core_pdf.impl.capture.recording_support import (
    CachedImageSource,
    image_source_from_stream,
)

if TYPE_CHECKING:
    from core_pdf.impl.capture.recording import TextState


def captured_image_source(self: TextState, xobj: PdfStream) -> tuple[ImageSource, float | None]:
    rendering = self.graphics.color_rendering
    cache = self.caches.capture_image_sources
    key = (id(xobj), rendering)
    cached = cache.get_key(xobj, key)
    if cached is None:
        cached = cache.put_key(
            xobj,
            key,
            CachedImageSource(
                image_source_from_stream(xobj, self.resolver, color_rendering=rendering)
            ),
        )
    alpha = cached.soft_mask_alpha() if self.options.soft_mask_alpha else None
    return cached.source, alpha


def paint_image(self: TextState, state: object, xobj: PdfStream) -> None:
    xobj_dict = xobj.dictionary
    if self.is_graphics_visible():
        image_is_stencil = self.resolver.resolve(xobj_dict.get("ImageMask")) is True
        if image_is_stencil and self.initial_pattern(stroke=False):
            return
        width = self.resolver.resolve_int(xobj_dict.get("Width")) or 0
        height = self.resolver.resolve_int(xobj_dict.get("Height")) or 0
        bbox = None
        quad = None
        if width > 0 and height > 0:
            bbox, quad = unit_square_placement(self.graphics.ctm)
        source, smask_alpha = self.captured_image_source(xobj)
        paints = (
            color_space_paints(self.graphics.fill_space)
            if image_is_stencil
            else raw_color_space_paints(source.dictionary.get("ColorSpace"))
        )
        self.drawings.append(
            CapturedDrawing(
                seqno=self.sequence,
                fill=self.capture_color(stroke=False) if image_is_stencil else None,
                fill_opacity=self.graphics.fill_opacity,
                blend_mode=self.graphics.blend_mode,
                dash_pattern=self.transformed_dash_pattern(),
                soft_mask_alpha=smask_alpha,
                alpha_is_shape=self.graphics.alpha_is_shape,
                kind="image",
                graphics_soft_mask=None
                if image_overrides_graphics_soft_mask(source)
                else self.capture_graphics_soft_mask(),
                paints=paints,
                image_source=source,
                raw_data=xobj.raw_data,
                dictionary=dict(xobj_dict),
                image_clip=self.clip_bbox,
                items=[("quad", quad)] if quad is not None else [],
                bbox=bbox,
                stream_order=self.stream_order,
                xobject_depth=self.xobject_depth,
            )
        )
        self.sequence += 1


def paint_inline_image(self: TextState, state: object, image: InlineImage) -> None:
    if self.is_graphics_visible():
        dictionary = dict(image.dictionary)
        if dictionary.get("ImageMask") is True and self.initial_pattern(stroke=False):
            return
        data = getattr(image, "data", b"")
        color_name = recover_pdf_name(dictionary.get("ColorSpace"))
        if color_name is not None:
            color_resource = self.resolver.deep_resolve(
                self.lookup_page_resource("ColorSpace", color_name)
            )
            if color_resource is not None:
                dictionary[PdfName.of("ColorSpace")] = color_resource
        source = image_source_from_stream(
            PdfStream(raw_data=data, dictionary=dictionary),
            self.resolver,
            color_rendering=self.graphics.color_rendering,
        )
        paints = (
            color_space_paints(self.graphics.fill_space)
            if dictionary.get("ImageMask") is True
            else raw_color_space_paints(source.dictionary.get("ColorSpace"))
        )
        self.inline_images.append(
            CapturedInlineImage(
                seqno=self.sequence,
                dictionary=dictionary,
                data=data,
                image_source=source,
                image_clip=self.clip_bbox,
                ctm=self.graphics.ctm,
                graphics_soft_mask=None
                if image_overrides_graphics_soft_mask(source)
                else self.capture_graphics_soft_mask(),
                xobject_depth=self.xobject_depth,
                blend_mode=self.graphics.blend_mode,
                soft_mask_alpha=self.group_alpha,
                alpha_is_shape=self.graphics.alpha_is_shape,
                stream_order=self.stream_order,
                fill=self.capture_color(stroke=False)
                if dictionary.get("ImageMask") is True
                else None,
                fill_opacity=self.graphics.fill_opacity,
                paints=paints,
            )
        )
        self.sequence += 1


def paint_shading(self: TextState, state: object, shading: PdfDict) -> None:
    if not self.is_graphics_visible():
        return
    dictionary = self.capture_shading_dictionary(shading)
    self.drawings.append(
        CapturedDrawing(
            seqno=self.sequence,
            fill=self.capture_color(stroke=False),
            fill_opacity=self.graphics.fill_opacity,
            stroke_color=self.capture_color(stroke=True),
            stroke_opacity=self.graphics.stroke_opacity,
            line_width=self.graphics.line_width,
            line_cap=self.graphics.line_cap,
            line_join=self.graphics.line_join,
            dash_pattern=self.transformed_dash_pattern(),
            blend_mode=self.graphics.blend_mode,
            soft_mask_alpha=self.group_alpha,
            alpha_is_shape=self.graphics.alpha_is_shape,
            kind="shading",
            graphics_soft_mask=self.capture_graphics_soft_mask(),
            paints=raw_color_space_paints(dictionary.get("ColorSpace")),
            color_rendering=self.graphics.color_rendering,
            items=[],
            dictionary=dictionary,
            stream_order=self.stream_order,
            xobject_depth=self.xobject_depth,
        )
    )
    self.sequence += 1

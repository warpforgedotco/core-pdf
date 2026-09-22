# SPDX-License-Identifier: AGPL-3.0-only

import numpy

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfValueResolver
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.image_spec import ImageSource
from core_pdf_spec.s_08_graphics.image_spec import (
    image_source_from_stream as resolve_image_source,
)


def image_source_from_stream(
    stream: PdfStream,
    resolver: PdfValueResolver,
    *,
    color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> tuple[ImageSource, float | None]:
    source = resolve_image_source(
        stream,
        resolver,
        semantic_context=getattr(resolver, "semantic_context", None),
        color_rendering=color_rendering,
    )
    mask_alpha = None
    mask = source.soft_mask
    if mask is not None:
        width = resolver.resolve_int(mask.dictionary.get("Width")) or 0
        height = resolver.resolve_int(mask.dictionary.get("Height")) or 0
        data = mask.raw
        if width > 0 and height > 0 and data:
            total = min(len(data), width * height)
            mask_sum = numpy.frombuffer(data, numpy.uint8, count=total).sum(dtype=numpy.uint64)
            mask_alpha = int(mask_sum) / (255.0 * total)
    return source, mask_alpha

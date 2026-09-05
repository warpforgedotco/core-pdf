# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve embedded image inputs without depending on interpreter state."""

from __future__ import annotations

import numpy

from core_pdf.impl.primitives import PdfName
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfValueResolver
from core_pdf.impl.spec.s_08_graphics.image_decode import ImageSource, SoftMask


def image_source_from_stream(
    stream: PdfStream, resolver: PdfValueResolver
) -> tuple[ImageSource, float | None]:
    """Capture the image's samples, resolved colour space and optional soft mask.

    The scalar alpha is retained as capture metadata; rendering uses the full
    mask plane carried by the source.
    """
    soft_mask = None
    mask_alpha = None
    mask = stream.dictionary.get("SMask")
    mask_stream = resolver.resolve(mask) if mask is not None else None
    if isinstance(mask_stream, PdfStream):
        dictionary = resolver.resolve_dict(mask_stream.dictionary) or {}
        data = mask_stream.raw_data
        soft_mask = SoftMask(data, dict(dictionary))
        width = resolver.resolve_int(dictionary.get("Width")) or 0
        height = resolver.resolve_int(dictionary.get("Height")) or 0
        if width > 0 and height > 0 and data:
            total = min(len(data), width * height)
            mask_sum = numpy.frombuffer(data, numpy.uint8, count=total).sum(dtype=numpy.uint64)
            mask_alpha = int(mask_sum) / (255.0 * total)

    source_dictionary = dict(stream.dictionary)
    # Indexed palettes and base spaces may be indirect; the colour manager
    # reads this dictionary without going back through the PDF resolver.
    color_space = source_dictionary.get("ColorSpace")
    if color_space is not None:
        source_dictionary[PdfName.of("ColorSpace")] = resolver.deep_resolve(color_space)
    return ImageSource(stream.raw_data, source_dictionary, soft_mask=soft_mask), mask_alpha


__all__ = ("image_source_from_stream",)

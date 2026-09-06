# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve embedded image inputs without depending on interpreter state."""

from __future__ import annotations

from core_pdf.impl._impl.model.geometry import points_bbox
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfValueResolver
from core_pdf.impl.spec.s_08_graphics.image_spec import ImageSource, SoftMask
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix
from core_pdf.impl.types import PdfName, Rectangle


def unit_square_placement(matrix: Matrix) -> tuple[Rectangle, tuple[tuple[float, float], ...]]:
    """Image bounds and ordered affine sampling corners in page coordinates."""
    a, b, c, d, e, f = matrix
    quad = ((e, f), (a + e, b + f), (c + e, d + f), (a + c + e, b + d + f))
    bbox = points_bbox(quad)
    assert bbox is not None
    return bbox, quad


def image_source_from_stream(stream: PdfStream, resolver: PdfValueResolver) -> ImageSource:
    """Capture the image's samples, resolved colour space and optional soft mask.

    The returned descriptor contains inputs only; it performs no image preparation.
    """
    soft_mask = None
    mask = stream.dictionary.get("SMask")
    mask_stream = resolver.resolve(mask) if mask is not None else None
    if isinstance(mask_stream, PdfStream):
        dictionary = resolver.resolve_dict(mask_stream.dictionary) or {}
        data = mask_stream.raw_data
        soft_mask = SoftMask(data, dict(dictionary))

    source_dictionary = dict(stream.dictionary)
    # Image decoding has no resolver. Resolve its dimensions, sample layout,
    # Decode entries and colour space here without walking unrelated metadata
    # or changing the source stream dictionary.
    for key in ("Width", "Height", "BitsPerComponent", "Decode", "ColorSpace"):
        value = source_dictionary.get(key)
        if value is not None:
            source_dictionary[PdfName.of(key)] = resolver.deep_resolve(value)
    return ImageSource(stream.raw_data, source_dictionary, soft_mask=soft_mask)


__all__ = ("image_source_from_stream", "unit_square_placement")

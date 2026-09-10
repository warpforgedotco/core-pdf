# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve embedded image inputs without depending on interpreter state."""

from __future__ import annotations

from core_pdf_spec.s_07_syntax.resolver import STREAM_DECODE_KEYS
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfValueResolver
from core_pdf_spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf_spec.s_08_graphics.geometry import points_bbox
from core_pdf_spec.s_08_graphics.image_spec import ImageSource, SoftMask
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.types import Rectangle

internal_IMAGE_INPUT_KEYS = STREAM_DECODE_KEYS | {
    "Width",
    "Height",
    "BitsPerComponent",
    "Decode",
    "ColorSpace",
    "ImageMask",
}


def unit_square_placement(matrix: Matrix) -> tuple[Rectangle, tuple[tuple[float, float], ...]]:
    """Image bounds and ordered affine sampling corners in page coordinates."""
    a, b, c, d, e, f = matrix
    quad = ((e, f), (a + e, b + f), (c + e, d + f), (a + c + e, b + d + f))
    bbox = points_bbox(quad)
    assert bbox is not None
    return bbox, quad


def internal_resolve_image_dictionary(
    dictionary: dict[object, object], resolver: PdfValueResolver
) -> dict[object, object]:
    # Image and filter decoders have no resolver. Resolve their inputs for both
    # image kinds without walking unrelated metadata or changing the source.
    return {
        key: resolver.deep_resolve(value)
        if value is not None and normalize_pdf_name(key) in internal_IMAGE_INPUT_KEYS
        else value
        for key, value in dictionary.items()
    }


def image_source_from_stream(stream: PdfStream, resolver: PdfValueResolver) -> ImageSource:
    """Capture the image's samples, resolved colour space and optional soft mask.

    The returned descriptor contains inputs only; it performs no image preparation.
    """
    soft_mask = None
    mask = stream.dictionary.get("SMask")
    mask_stream = resolver.resolve(mask) if mask is not None else None
    if isinstance(mask_stream, PdfStream):
        dictionary = internal_resolve_image_dictionary(mask_stream.dictionary, resolver)
        data = mask_stream.raw_data
        soft_mask = SoftMask(data, dictionary)

    source_dictionary = internal_resolve_image_dictionary(stream.dictionary, resolver)
    return ImageSource(stream.raw_data, source_dictionary, soft_mask=soft_mask)


__all__ = ("image_source_from_stream", "unit_square_placement")

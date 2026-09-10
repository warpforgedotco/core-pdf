# SPDX-License-Identifier: AGPL-3.0-only
"""Resolved PDF image inputs and their soft masks, without image preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core_pdf_spec.s_07_syntax.resolver import STREAM_DECODE_KEYS
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfValueResolver
from core_pdf_spec.s_07_syntax_primitives.coercion import decoded_name, require_pdf_integer

internal_IMAGE_INPUT_KEYS = STREAM_DECODE_KEYS | {
    "Width",
    "Height",
    "BitsPerComponent",
    "Decode",
    "ColorSpace",
    "ImageMask",
}


@dataclass(frozen=True, slots=True)
class SoftMask:
    """The /SMask plane accompanying an image XObject.

    Carried as its own field rather than smuggled through the image's PDF
    dictionary: that dictionary is the real object dictionary and is exported
    verbatim to display consumers, so private keys in it leak downstream.
    """

    raw: bytes | memoryview
    dictionary: dict[Any, Any]


@dataclass(slots=True, eq=False)
class ImageSource:
    """Resolved PDF image inputs, without device preparation policy."""

    raw: bytes | memoryview
    dictionary: dict[Any, Any]
    soft_mask: SoftMask | None = field(default=None, kw_only=True)


def internal_resolve_image_dictionary(
    dictionary: dict[object, object], resolver: PdfValueResolver
) -> dict[object, object]:
    # Image and filter decoders have no resolver. Resolve their inputs for both
    # image kinds without walking unrelated metadata or changing the source.
    return {
        key: resolver.deep_resolve(value)
        if value is not None and decoded_name(key) in internal_IMAGE_INPUT_KEYS
        else value
        for key, value in dictionary.items()
    }


def image_source_from_stream(stream: PdfStream, resolver: PdfValueResolver) -> ImageSource:
    """Resolve the image's samples, colour space and optional soft mask.

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


def image_bits_per_component(dictionary: dict[object, object]) -> int | None:
    """Table 89: masks have one-bit samples; JPX defines its own sample depth."""
    filters = dictionary.get("Filter")
    filters = filters if isinstance(filters, (list, tuple)) else (filters,)
    if any(decoded_name(value) == "JPXDecode" for value in filters):
        return None
    value = dictionary.get("BitsPerComponent")
    if dictionary.get("ImageMask") is True:
        if value is None:
            return 1
        if require_pdf_integer(value, "invalid image bits-per-component") != 1:
            raise ValueError("invalid image mask bits-per-component")
        return 1
    if value is None:
        raise ValueError("missing image bits-per-component")
    bits = require_pdf_integer(value, "invalid image bits-per-component")
    if bits not in {1, 2, 4, 8, 16}:
        raise ValueError("invalid image bits-per-component")
    return bits


__all__ = (
    "image_bits_per_component",
    "SoftMask",
    "ImageSource",
    "image_source_from_stream",
)

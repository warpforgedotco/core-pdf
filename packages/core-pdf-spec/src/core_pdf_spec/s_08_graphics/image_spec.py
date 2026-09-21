# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_07_syntax.resolver import STREAM_DECODE_KEYS
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfValueResolver
from core_pdf_spec.s_07_syntax_primitives.coercion import decoded_name, require_pdf_integer
from core_pdf_spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
    ColorRendering,
    parse_rendering_intent,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext

internal_IMAGE_INPUT_KEYS = STREAM_DECODE_KEYS | {
    "Width",
    "Height",
    "BitsPerComponent",
    "Decode",
    "ColorSpace",
    "ImageMask",
    "Mask",
    "Matte",
    "SMaskInData",
    "Intent",
}


@dataclass(frozen=True, slots=True)
class SoftMask:
    raw: bytes | memoryview
    dictionary: dict[Any, Any]


@dataclass(slots=True, eq=False)
class ImageSource:
    raw: bytes | memoryview
    dictionary: dict[Any, Any]
    soft_mask: SoftMask | None = field(default=None, kw_only=True)
    semantic_context: SemanticContext | None = field(default=None, kw_only=True)
    color_rendering: ColorRendering = field(default=DEFAULT_COLOR_RENDERING, kw_only=True)


def internal_resolve_image_dictionary(
    dictionary: dict[object, object], resolver: PdfValueResolver
) -> dict[object, object]:
    return {
        key: resolver.deep_resolve(value)
        if value is not None and decoded_name(key) in internal_IMAGE_INPUT_KEYS
        else value
        for key, value in dictionary.items()
    }


def image_source_from_stream(
    stream: PdfStream,
    resolver: PdfValueResolver,
    *,
    semantic_context: SemanticContext | None = None,
    color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> ImageSource:
    soft_mask = None
    mask = stream.dictionary.get("SMask")
    mask_stream = resolver.resolve(mask) if mask is not None else None
    if isinstance(mask_stream, PdfStream):
        dictionary = internal_resolve_image_dictionary(mask_stream.dictionary, resolver)
        data = mask_stream.raw_data
        soft_mask = SoftMask(data, dictionary)

    source_dictionary = internal_resolve_image_dictionary(stream.dictionary, resolver)
    return ImageSource(
        stream.raw_data,
        source_dictionary,
        soft_mask=soft_mask,
        semantic_context=semantic_context,
        color_rendering=color_rendering,
    )


def image_color_rendering(
    dictionary: dict[object, object], rendering: ColorRendering = DEFAULT_COLOR_RENDERING
) -> ColorRendering:
    if dictionary.get("ImageMask") is True:
        return rendering
    intent = dictionary.get("Intent")
    return (
        rendering
        if intent is None
        else ColorRendering(parse_rendering_intent(intent), rendering.black_point_compensation)
    )


def image_decode_array_applies(
    dictionary: dict[object, object], *, context: SemanticContext | None = None
) -> bool:
    filters = dictionary.get("Filter")
    filters = filters if isinstance(filters, (list, tuple)) else (filters,)
    if not any(decoded_name(value) == "JPXDecode" for value in filters):
        return True
    if dictionary.get("ImageMask") is True:
        return True
    if context is None:
        return False
    if context.version is None or not context.version.recognized:
        raise PdfUnsupportedError("JPX Decode interpretation requires a recognized PDF version")
    return context.version >= PdfVersion(2, 0) and dictionary.get("ColorSpace") is not None


def image_smask_in_data(dictionary: dict[object, object]) -> int:
    filters = dictionary.get("Filter")
    filters = filters if isinstance(filters, (list, tuple)) else (filters,)
    if not any(decoded_name(value) == "JPXDecode" for value in filters):
        return 0
    selector = require_pdf_integer(dictionary.get("SMaskInData", 0), "invalid image SMaskInData")
    if selector not in {0, 1, 2}:
        raise ValueError("invalid image SMaskInData")
    return selector


def image_bits_per_component(dictionary: dict[object, object]) -> int | None:
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
    "image_color_rendering",
    "image_smask_in_data",
    "image_decode_array_applies",
    "image_bits_per_component",
    "SoftMask",
    "ImageSource",
    "image_source_from_stream",
)

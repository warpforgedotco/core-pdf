# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_syntax.resolver import STREAM_DECODE_KEYS
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import decoded_name, require_pdf_integer
from core_pdf.impl.spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
    ColorRendering,
    parse_rendering_intent,
)
from core_pdf.impl.spec.standards import PdfVersion, SemanticContext, require_recognized_version
from core_records import GeneratedRecord, RecordType, ReplaceFields, ReprFields, field

IMAGE_INPUT_KEYS = STREAM_DECODE_KEYS | {
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


class SoftMask(GeneratedRecord):
    raw: bytes | memoryview
    dictionary: dict[Any, Any]


class ImageSource(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False, eq=False):
    raw: bytes | memoryview
    dictionary: dict[Any, Any]
    soft_mask: SoftMask | None = field(default=None, kw_only=True)
    semantic_context: SemanticContext | None = field(default=None, kw_only=True)
    color_rendering: ColorRendering = field(default=DEFAULT_COLOR_RENDERING, kw_only=True)

    __match_args__ = ("raw", "dictionary")


def resolve_image_dictionary(dictionary: PdfDict, resolver: PdfValueResolver) -> PdfDict:
    return {
        key: resolver.deep_resolve(value)
        if value is not None and decoded_name(key) in IMAGE_INPUT_KEYS
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
        dictionary = resolve_image_dictionary(mask_stream.dictionary, resolver)
        data = mask_stream.raw_data
        soft_mask = SoftMask(data, dictionary)

    source_dictionary = resolve_image_dictionary(stream.dictionary, resolver)
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


def has_jpx_filter(dictionary: dict[object, object]) -> bool:
    filters = dictionary.get("Filter")
    filters = filters if isinstance(filters, (list, tuple)) else (filters,)
    return any(decoded_name(value) == "JPXDecode" for value in filters)


@cites(
    "ISO 32000-2",
    "7.4.9",
    since="2.0",
    rule="a JPX image's Decode array applies when it has a ColorSpace",
)
def image_decode_array_applies(
    dictionary: dict[object, object], *, context: SemanticContext | None = None
) -> bool:
    if not has_jpx_filter(dictionary):
        return True
    if dictionary.get("ImageMask") is True:
        return True
    version = require_recognized_version(
        context, "JPX Decode interpretation requires a recognized PDF version"
    )
    if version is None:
        return False
    return version >= PdfVersion(2, 0) and dictionary.get("ColorSpace") is not None


def image_smask_in_data(dictionary: dict[object, object]) -> int:
    if not has_jpx_filter(dictionary):
        return 0
    selector = require_pdf_integer(dictionary.get("SMaskInData", 0), "invalid image SMaskInData")
    if selector not in {0, 1, 2}:
        raise ValueError("invalid image SMaskInData")
    return selector


__all__ = (
    "image_color_rendering",
    "image_smask_in_data",
    "image_decode_array_applies",
    "SoftMask",
    "ImageSource",
    "image_source_from_stream",
)

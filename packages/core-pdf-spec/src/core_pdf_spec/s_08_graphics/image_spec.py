# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar, NoReturn, Self

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

frozen_setattr = object.__setattr__


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


class SoftMask:
    __slots__ = ("raw", "dictionary")

    raw: bytes | memoryview
    dictionary: dict[Any, Any]

    __fields__: ClassVar[tuple[str, ...]] = ("raw", "dictionary")
    __match_args__ = ("raw", "dictionary")

    def __init__(self, raw: bytes | memoryview, dictionary: dict[Any, Any]) -> None:
        frozen_setattr(self, "raw", raw)
        frozen_setattr(self, "dictionary", dictionary)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(raw={self.raw!r}, dictionary={self.dictionary!r})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.raw == other.raw and self.dictionary == other.dictionary

    def __hash__(self) -> int:
        return hash((self.raw, self.dictionary))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        raw = changes.pop("raw", self.raw)
        dictionary = changes.pop("dictionary", self.dictionary)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(raw, dictionary)


class ImageSource:
    __slots__ = ("raw", "dictionary", "soft_mask", "semantic_context", "color_rendering")

    raw: bytes | memoryview
    dictionary: dict[Any, Any]
    soft_mask: SoftMask | None
    semantic_context: SemanticContext | None
    color_rendering: ColorRendering

    __fields__: ClassVar[tuple[str, ...]] = (
        "raw",
        "dictionary",
        "soft_mask",
        "semantic_context",
        "color_rendering",
    )
    __match_args__ = ("raw", "dictionary")

    def __init__(
        self,
        raw: bytes | memoryview,
        dictionary: dict[Any, Any],
        *,
        soft_mask: SoftMask | None = None,
        semantic_context: SemanticContext | None = None,
        color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
    ) -> None:
        self.raw = raw
        self.dictionary = dictionary
        self.soft_mask = soft_mask
        self.semantic_context = semantic_context
        self.color_rendering = color_rendering

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"raw={self.raw!r}, "
            f"dictionary={self.dictionary!r}, "
            f"soft_mask={self.soft_mask!r}, "
            f"semantic_context={self.semantic_context!r}, "
            f"color_rendering={self.color_rendering!r}"
            ")"
        )

    def __replace__(self, /, **changes: Any) -> Self:
        raw = changes.pop("raw", self.raw)
        dictionary = changes.pop("dictionary", self.dictionary)
        soft_mask = changes.pop("soft_mask", self.soft_mask)
        semantic_context = changes.pop("semantic_context", self.semantic_context)
        color_rendering = changes.pop("color_rendering", self.color_rendering)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            raw,
            dictionary,
            soft_mask=soft_mask,
            semantic_context=semantic_context,
            color_rendering=color_rendering,
        )


def resolve_image_dictionary(
    dictionary: dict[object, object], resolver: PdfValueResolver
) -> dict[object, object]:
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

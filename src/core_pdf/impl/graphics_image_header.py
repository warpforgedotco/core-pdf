# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar

from core_pdf.impl.graphics_color_spec import ColorSpace, parse_color_space
from core_pdf.impl.pdf_names import lenient_int
from core_pdf.impl.types import Record, frozen_setattr


class ColorSpaceMemo:
    __slots__ = ("error", "parsed", "value")

    def __init__(self, value: object, parsed: ColorSpace | None = None) -> None:
        self.value = value
        self.parsed = parsed
        self.error: Exception | None = None

    def space(self) -> ColorSpace:
        parsed = self.parsed
        if parsed is not None:
            return parsed
        if self.error is not None:
            raise self.error
        try:
            parsed = self.parsed = parse_color_space(self.value)
        except Exception as error:
            self.error = error
            raise
        return parsed


class ImageHeader(Record):
    __slots__ = (
        "width",
        "height",
        "bits",
        "color_space",
        "decode",
        "mask",
        "smask",
        "filter",
        "parsed_space",
    )

    width: int
    height: int
    bits: int
    color_space: object
    decode: object
    mask: object
    smask: object
    filter: object
    parsed_space: ColorSpaceMemo

    __fields__: ClassVar[tuple[str, ...]] = (
        "width",
        "height",
        "bits",
        "color_space",
        "decode",
        "mask",
        "smask",
        "filter",
        "parsed_space",
    )
    __match_args__ = (
        "width",
        "height",
        "bits",
        "color_space",
        "decode",
        "mask",
        "smask",
        "filter",
        "parsed_space",
    )

    def __init__(
        self,
        width: int,
        height: int,
        bits: int,
        color_space: object,
        decode: object,
        mask: object,
        smask: object,
        filter: object,  # noqa: A002
        parsed_space: ColorSpaceMemo,
    ) -> None:
        frozen_setattr(self, "width", width)
        frozen_setattr(self, "height", height)
        frozen_setattr(self, "bits", bits)
        frozen_setattr(self, "color_space", color_space)
        frozen_setattr(self, "decode", decode)
        frozen_setattr(self, "mask", mask)
        frozen_setattr(self, "smask", smask)
        frozen_setattr(self, "filter", filter)
        frozen_setattr(self, "parsed_space", parsed_space)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.width == other.width
            and self.height == other.height
            and self.bits == other.bits
            and self.color_space == other.color_space
            and self.decode == other.decode
            and self.mask == other.mask
            and self.smask == other.smask
            and self.filter == other.filter
        )

    def __hash__(self) -> int:
        return hash((self.width, self.height, self.bits))

    def space(self) -> ColorSpace:
        return self.parsed_space.space()

    @property
    def has_color_key_mask(self) -> bool:
        return isinstance(self.mask, (list, tuple))


def read_image_header(
    dictionary: dict[Any, Any], *, space: ColorSpace | None = None
) -> ImageHeader:
    color_space = dictionary.get("ColorSpace")
    return ImageHeader(
        lenient_int(dictionary.get("Width"), 0),
        lenient_int(dictionary.get("Height"), 0),
        lenient_int(dictionary.get("BitsPerComponent"), 8),
        color_space,
        dictionary.get("Decode"),
        dictionary.get("Mask"),
        dictionary.get("SMask"),
        dictionary.get("Filter"),
        ColorSpaceMemo(color_space, space),
    )


__all__ = ("ColorSpaceMemo", "ImageHeader", "read_image_header")

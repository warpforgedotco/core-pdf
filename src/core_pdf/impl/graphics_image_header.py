# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

from core_pdf.impl.graphics_color_spec import ColorSpace, parse_color_space
from core_pdf.impl.pdf_names import lenient_int


class ImageHeader:
    __slots__ = (
        "width",
        "height",
        "declared_bits",
        "bits",
        "color_space",
        "decode",
        "mask",
        "filter",
        "parsed_space",
        "space_error",
    )

    def __init__(self, dictionary: dict[Any, Any]) -> None:
        self.width = lenient_int(dictionary.get("Width"), 0)
        self.height = lenient_int(dictionary.get("Height"), 0)
        declared_bits = lenient_int(dictionary.get("BitsPerComponent", 8), None)
        self.declared_bits = declared_bits
        self.bits = 8 if declared_bits is None else declared_bits
        self.color_space = dictionary.get("ColorSpace")
        self.decode = dictionary.get("Decode")
        self.mask = dictionary.get("Mask")
        self.filter = dictionary.get("Filter")
        self.parsed_space: ColorSpace | None = None
        self.space_error: Exception | None = None

    def checked_bits(self) -> int:
        bits = self.declared_bits
        if bits is None or bits <= 0:
            raise ValueError("invalid image bits-per-component")
        return bits

    def space(self) -> ColorSpace:
        parsed = self.parsed_space
        if parsed is not None:
            return parsed
        if self.space_error is not None:
            raise self.space_error
        try:
            parsed = self.parsed_space = parse_color_space(self.color_space)
        except Exception as error:
            self.space_error = error
            raise
        return parsed

    @property
    def has_color_key_mask(self) -> bool:
        return isinstance(self.mask, (list, tuple))


__all__ = ("ImageHeader",)

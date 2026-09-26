from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import decoded_name
from core_records import Record, frozen_setattr


def get_descendant(font: dict[Any, Any]) -> dict[Any, Any] | None:
    descendant_fonts = font.get("DescendantFonts")
    if descendant_fonts is None:
        return None
    if (
        not isinstance(descendant_fonts, (list, tuple))
        or len(descendant_fonts) != 1
        or not isinstance(descendant_fonts[0], dict)
    ):
        raise ValueError("invalid DescendantFonts array")
    return descendant_fonts[0]


class FontProgramInputs(Record):
    __slots__ = (
        "subtype",
        "original_subtype",
        "descendant",
        "font_file",
        "font_file2",
        "font_file3",
    )

    subtype: str | None
    original_subtype: str | None
    descendant: dict[str, Any] | None
    font_file: PdfStream | None
    font_file2: PdfStream | None
    font_file3: PdfStream | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "subtype",
        "original_subtype",
        "descendant",
        "font_file",
        "font_file2",
        "font_file3",
    )
    __match_args__ = (
        "subtype",
        "original_subtype",
        "descendant",
        "font_file",
        "font_file2",
        "font_file3",
    )

    def __init__(
        self,
        subtype: str | None,
        original_subtype: str | None,
        descendant: dict[str, Any] | None,
        font_file: PdfStream | None,
        font_file2: PdfStream | None,
        font_file3: PdfStream | None,
    ) -> None:
        frozen_setattr(self, "subtype", subtype)
        frozen_setattr(self, "original_subtype", original_subtype)
        frozen_setattr(self, "descendant", descendant)
        frozen_setattr(self, "font_file", font_file)
        frozen_setattr(self, "font_file2", font_file2)
        frozen_setattr(self, "font_file3", font_file3)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.subtype == other.subtype
            and self.original_subtype == other.original_subtype
            and self.descendant == other.descendant
            and self.font_file == other.font_file
            and self.font_file2 == other.font_file2
            and self.font_file3 == other.font_file3
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.subtype,
                self.original_subtype,
                self.descendant,
                self.font_file,
                self.font_file2,
                self.font_file3,
            )
        )


def font_descriptor(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("invalid font descriptor")
    return value


def descriptor_font_file(descriptor: dict[str, Any] | None, key: str) -> PdfStream | None:
    if descriptor is None:
        return None
    value = descriptor.get(key)
    if value is not None and not isinstance(value, PdfStream):
        raise ValueError(f"invalid font program stream: {key}")
    return value


def prepare_font_program_inputs(
    font: dict[str, Any],
    *,
    read_name: Callable[[object], str | None] = decoded_name,
    read_descendant: Callable[[dict[Any, Any]], dict[Any, Any] | None] = get_descendant,
    read_descriptor: Callable[[object], dict[str, Any] | None] = font_descriptor,
    read_font_file: Callable[[dict[str, Any] | None, str], PdfStream | None] = (
        descriptor_font_file
    ),
) -> FontProgramInputs:
    """The dictionaries and streams that select a font's embedded program.

    Each reader coerces one entry and, by default, raises ValueError on a
    malformed value; a caller that recovers passes readers that return None.
    """
    descendant = read_descendant(font)
    font_dict = descendant if descendant is not None else font
    original_descriptor = read_descriptor(font.get("FontDescriptor"))
    font_file = read_font_file(original_descriptor, "FontFile")
    descriptor = (
        read_descriptor(font_dict.get("FontDescriptor"))
        if font_dict is not font
        else original_descriptor
    )
    return FontProgramInputs(
        subtype=read_name(font_dict.get("Subtype")),
        original_subtype=read_name(font.get("Subtype")),
        descendant=descendant,
        font_file=font_file,
        font_file2=read_font_file(descriptor, "FontFile2"),
        font_file3=read_font_file(descriptor, "FontFile3"),
    )


__all__ = ["get_descendant", "FontProgramInputs", "prepare_font_program_inputs"]

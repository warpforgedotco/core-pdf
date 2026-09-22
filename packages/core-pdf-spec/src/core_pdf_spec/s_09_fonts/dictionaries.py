from __future__ import annotations

from typing import Any, ClassVar, NoReturn, Self, cast

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import decoded_name

frozen_setattr = object.__setattr__


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


class FontProgramInputs:
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"subtype={self.subtype!r}, "
            f"original_subtype={self.original_subtype!r}, "
            f"descendant={self.descendant!r}, "
            f"font_file={self.font_file!r}, "
            f"font_file2={self.font_file2!r}, "
            f"font_file3={self.font_file3!r}"
            ")"
        )

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
        subtype = changes.pop("subtype", self.subtype)
        original_subtype = changes.pop("original_subtype", self.original_subtype)
        descendant = changes.pop("descendant", self.descendant)
        font_file = changes.pop("font_file", self.font_file)
        font_file2 = changes.pop("font_file2", self.font_file2)
        font_file3 = changes.pop("font_file3", self.font_file3)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            subtype,
            original_subtype,
            descendant,
            font_file,
            font_file2,
            font_file3,
        )


def font_descriptor(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("invalid font descriptor")
    return cast(dict[str, Any], value)


def descriptor_font_file(descriptor: dict[str, Any] | None, key: str) -> PdfStream | None:
    if descriptor is None:
        return None
    value = descriptor.get(key)
    if value is not None and not isinstance(value, PdfStream):
        raise ValueError(f"invalid font program stream: {key}")
    return value


def prepare_font_program_inputs(font: dict[str, Any]) -> FontProgramInputs:
    descendant = get_descendant(font)
    font_dict = descendant if descendant is not None else font
    original_descriptor = font_descriptor(font.get("FontDescriptor"))
    font_file = descriptor_font_file(original_descriptor, "FontFile")
    descriptor = (
        font_descriptor(font_dict.get("FontDescriptor"))
        if font_dict is not font
        else original_descriptor
    )
    return FontProgramInputs(
        subtype=decoded_name(font_dict.get("Subtype")),
        original_subtype=decoded_name(font.get("Subtype")),
        descendant=descendant,
        font_file=font_file,
        font_file2=descriptor_font_file(descriptor, "FontFile2"),
        font_file3=descriptor_font_file(descriptor, "FontFile3"),
    )


__all__ = ["get_descendant", "FontProgramInputs", "prepare_font_program_inputs"]

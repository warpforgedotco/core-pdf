"""PDF font dictionary references, without backend selection or repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import decoded_name


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


@dataclass(frozen=True, slots=True)
class FontProgramInputs:
    """Structural font selection only; stream bytes stay lazy until attempted."""

    subtype: str | None
    original_subtype: str | None
    descendant: dict[str, Any] | None
    font_file: PdfStream | None
    font_file2: PdfStream | None
    font_file3: PdfStream | None


def internal_font_descriptor(value: object) -> dict[str, Any] | None:
    # ISO 32000-1, 7.3.9: a null dictionary entry is equivalent to omission.
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("invalid font descriptor")
    return cast(dict[str, Any], value)


def internal_font_file(descriptor: dict[str, Any] | None, key: str) -> PdfStream | None:
    if descriptor is None:
        return None
    value = descriptor.get(key)
    if value is not None and not isinstance(value, PdfStream):
        raise ValueError(f"invalid font program stream: {key}")
    return value


def prepare_font_program_inputs(font: dict[str, Any]) -> FontProgramInputs:
    descendant = get_descendant(font)
    font_dict = descendant if descendant is not None else font
    original_descriptor = internal_font_descriptor(font.get("FontDescriptor"))
    # Type 1 is selected from the original dictionary, not its descendant.
    font_file = internal_font_file(original_descriptor, "FontFile")
    descriptor = (
        internal_font_descriptor(font_dict.get("FontDescriptor"))
        if font_dict is not font
        else original_descriptor
    )
    return FontProgramInputs(
        subtype=decoded_name(font_dict.get("Subtype")),
        original_subtype=decoded_name(font.get("Subtype")),
        descendant=descendant,
        font_file=font_file,
        font_file2=internal_font_file(descriptor, "FontFile2"),
        font_file3=internal_font_file(descriptor, "FontFile3"),
    )


__all__ = ["get_descendant", "FontProgramInputs", "prepare_font_program_inputs"]

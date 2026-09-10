"""PDF font dictionary references, without backend selection or repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf_spec.s_09_fonts.widths import get_descendant


@dataclass(frozen=True, slots=True)
class FontProgramInputs:
    """Structural font selection only; stream bytes stay lazy until attempted."""

    subtype: str | None
    original_subtype: str | None
    descendant: dict[str, Any] | None
    font_file: PdfStream | None
    font_file2: PdfStream | None
    font_file3: PdfStream | None


def internal_font_file(descriptor: object, key: str) -> PdfStream | None:
    if descriptor is None:
        return None
    if not isinstance(descriptor, dict):
        raise ValueError("invalid font descriptor")
    value = descriptor.get(key)
    if value is not None and not isinstance(value, PdfStream):
        raise ValueError(f"invalid font program stream: {key}")
    return value


def prepare_font_program_inputs(font: dict[str, Any]) -> FontProgramInputs:
    descendant = get_descendant(font)
    font_dict = descendant if descendant is not None else font
    descriptor = font_dict.get("FontDescriptor")
    return FontProgramInputs(
        subtype=normalize_pdf_name(font_dict.get("Subtype")),
        original_subtype=normalize_pdf_name(font.get("Subtype")),
        descendant=descendant,
        # Type 1 is selected from the original dictionary, not its descendant.
        font_file=internal_font_file(font.get("FontDescriptor"), "FontFile"),
        font_file2=internal_font_file(descriptor, "FontFile2"),
        font_file3=internal_font_file(descriptor, "FontFile3"),
    )


__all__ = ["FontProgramInputs", "prepare_font_program_inputs"]

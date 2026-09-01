# SPDX-License-Identifier: AGPL-3.0-only
"""Ligature width detection across a font and its companion TrueType program."""

from __future__ import annotations

from typing import Any, Protocol, cast

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_float_strict,
    parse_int_strict,
)
from core_pdf.impl.spec.s_09_fonts.font_program_truetype import TrueTypeFontProgram
from core_pdf.impl.spec.s_09_fonts.helpers import strip_subset_tag


class FontResourceDocument(Protocol):
    def resolve(self, value: object) -> object: ...


def get_font_file(document: FontResourceDocument, font_obj: object) -> PdfStream | None:
    if not isinstance(font_obj, dict):
        return None
    descriptor = font_obj.get("FontDescriptor")
    if not isinstance(descriptor, dict):
        return None
    font_file = document.resolve(descriptor.get("FontFile2"))
    return font_file if isinstance(font_file, PdfStream) else None


def load_ligature_font_tables(tt_data: bytes) -> TrueTypeFontProgram | None:
    try:
        return TrueTypeFontProgram(tt_data)
    except ValueError:
        return None


def find_companion_font(
    document: FontResourceDocument,
    resources: object,
    base_name: str,
    ligature_starters: set[str],
) -> tuple[dict[int, float], dict[str, float], bytes | None]:
    if not isinstance(resources, dict):
        return {}, {}, None
    font_resources = resources.get("Font")
    if not isinstance(font_resources, dict):
        return {}, {}, None

    for fref in font_resources.values():
        fobj = document.resolve(fref)
        if not isinstance(fobj, dict):
            continue
        comp_base = strip_subset_tag(normalize_pdf_name(fobj.get("BaseFont")) or "")
        if comp_base != base_name:
            continue

        fc = fobj.get("FirstChar")
        lc = fobj.get("LastChar")
        try:
            fc_int = parse_int_strict(fc, "invalid font FirstChar")
            lc_int = parse_int_strict(lc, "invalid font LastChar")
        except ValueError:
            continue
        if lc_int < fc_int:
            continue

        widths_raw = fobj.get("Widths")
        if widths_raw is None:
            continue
        widths_raw = document.resolve(widths_raw)
        if not isinstance(widths_raw, (list, tuple)):
            raise ValueError("invalid font widths array")

        starter_widths: dict[int, float] = {}
        starter_chars: dict[str, float] = {}
        for i, width_value in enumerate(widths_raw):
            try:
                width = parse_float_strict(width_value, "invalid font widths array")
            except ValueError:
                continue
            if width <= 0:
                continue
            code = fc_int + i
            if code < 0 or code > 255:
                continue
            try:
                character = bytes([code]).decode("mac_roman")
            except UnicodeDecodeError:
                character = chr(code) if code < 128 else ""
            if character in ligature_starters:
                starter_widths[code] = width
                starter_chars[character] = width

        if starter_widths:
            font_file = get_font_file(document, fobj)
            if font_file is None:
                return starter_widths, starter_chars, None
            try:
                font_data = font_file.data
            except PdfParseError:
                font_data = None
            return starter_widths, starter_chars, font_data

    return {}, {}, None


def detect_ligature_overrides(
    document: FontResourceDocument,
    resources: object,
    font_obj: object,
) -> dict[int, str]:
    if not isinstance(font_obj, dict):
        return {}
    first_char = font_obj.get("FirstChar")
    last_char = font_obj.get("LastChar")
    try:
        first_char_int = parse_int_strict(first_char, "invalid font FirstChar")
        last_char_int = parse_int_strict(last_char, "invalid font LastChar")
    except ValueError:
        return {}

    base_name = strip_subset_tag(normalize_pdf_name(font_obj.get("BaseFont")) or "")
    if not base_name:
        return {}

    font_file = get_font_file(document, font_obj)
    if font_file is None:
        return {}
    try:
        tt_data = font_file.data
    except PdfParseError:
        return {}

    try:
        starter_widths, starter_chars, companion_data = find_companion_font(
            document, resources, base_name, set("ftscFTSC")
        )
    except ValueError:
        if load_ligature_font_tables(tt_data) is None:
            return {}
        raise

    if companion_data is None or not starter_widths:
        return {}

    parsed_primary = load_ligature_font_tables(tt_data)
    if parsed_primary is None:
        return {}

    try:
        TrueTypeFontProgram(companion_data)
    except ValueError:
        return {}

    lig_widths_raw = font_obj.get("Widths")
    if lig_widths_raw is not None and not isinstance(lig_widths_raw, (list, tuple)):
        raise ValueError("invalid font widths array")
    overrides: dict[int, str] = {}

    for pdf_code in range(first_char_int, last_char_int + 1):
        if pdf_code < 0 or pdf_code > 255:
            continue
        try:
            codepoint = ord(bytes([pdf_code]).decode("mac_roman"))
        except UnicodeDecodeError:
            codepoint = pdf_code

        if codepoint in parsed_primary.unicode_cmap:
            continue

        glyph_id = pdf_code - first_char_int
        body_bbox, is_composite = parsed_primary.composite_body_bbox(glyph_id)
        if not (is_composite and body_bbox):
            continue

        ft_width = starter_chars.get("f", 0.0) + starter_chars.get("t", 0.0)
        if ft_width <= 0:
            continue

        lig_width = 0.0
        if lig_widths_raw is not None:
            width_index = pdf_code - first_char_int
            if 0 <= width_index < len(lig_widths_raw):
                width_value = lig_widths_raw[width_index]
                if type(width_value) in (int, float):
                    lig_width = float(cast(Any, width_value))

        if lig_width and 0.85 <= lig_width / ft_width <= 0.98:
            overrides[pdf_code] = "ft"

    return overrides


__all__ = (
    "FontResourceDocument",
    "detect_ligature_overrides",
    "find_companion_font",
    "get_font_file",
    "load_ligature_font_tables",
)

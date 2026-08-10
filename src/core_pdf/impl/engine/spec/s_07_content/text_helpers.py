# SPDX-License-Identifier: AGPL-3.0-only
"""Native text spacing and normalization helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol, cast

from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts import TrueTypeFontProgram
from core_pdf.impl.engine.spec.s_09_fonts.widths import (
    require_font_float,
    require_font_int,
)
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.objects import PdfStream

NO_SPACE_BEFORE = frozenset(".,;:!?)]}%")
NO_SPACE_AFTER = frozenset("([{")


class FontResourceDocument(Protocol):
    def resolve(self, value: object) -> object: ...


@lru_cache(maxsize=256)
def cached_encode_latin1(s: str) -> bytes:
    return s.encode("latin-1", "replace")


def gap_separator(left: str, right: str, gap: float, run: Any) -> str:
    threshold = run.space_width * 0.12
    font_threshold = run.font_size * 0.10
    if font_threshold > threshold:
        threshold = font_threshold
    if threshold < 1.0:
        threshold = 1.0
    if gap <= threshold:
        return ""
    if not left or not right or left[-1].isspace() or right[0].isspace():
        return ""
    if right[0] in NO_SPACE_BEFORE or left[-1] in NO_SPACE_AFTER:
        return ""
    return " "


def can_merge_cross_font_word(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return (left[-1].isalnum() or left[-1] == "_") and (right[0].isalnum() or right[0] == "_")


@lru_cache(maxsize=4096)
def is_garbage_text(text: str) -> bool:
    if not text:
        return True
    for c in text:
        o = ord(c)
        if not (o < 32 or 0xE000 <= o <= 0xF8FF):
            return False
    return True


# Lone surrogates cannot survive encoding to UTF-8, so drop them here rather
# than letting them fail somewhere downstream.
NORMALIZE_EXTRACTED_TEXT_TABLE = dict.fromkeys(range(0xD800, 0xE000))


def normalize_extracted_text(text: str) -> str:
    if text.isascii():
        return text
    return text.translate(NORMALIZE_EXTRACTED_TEXT_TABLE)


def get_font_file(document: FontResourceDocument, font_obj: object) -> PdfStream | None:
    descriptor = lookup_dict_key(font_obj, "FontDescriptor")
    if not isinstance(descriptor, dict):
        return None
    font_file = document.resolve(lookup_dict_key(descriptor, "FontFile2"))
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
    font_resources = lookup_dict_key(resources, "Font")
    if not isinstance(font_resources, dict):
        return {}, {}, None

    for fref in font_resources.values():
        fobj = document.resolve(fref)
        if not isinstance(fobj, dict):
            continue
        comp_base = normalize_pdf_name(lookup_dict_key(fobj, "BaseFont")) or ""
        if "+" in comp_base:
            comp_base = comp_base.split("+", 1)[1]
        if comp_base != base_name:
            continue

        fc = lookup_dict_key(fobj, "FirstChar")
        lc = lookup_dict_key(fobj, "LastChar")
        try:
            fc_int = require_font_int(fc, "invalid font FirstChar")
            lc_int = require_font_int(lc, "invalid font LastChar")
        except ValueError:
            continue
        if lc_int < fc_int:
            continue

        widths_raw = lookup_dict_key(fobj, "Widths")
        if widths_raw is None:
            continue
        widths_raw = document.resolve(widths_raw)
        if not isinstance(widths_raw, (list, tuple)):
            raise ValueError("invalid font widths array")

        starter_widths: dict[int, float] = {}
        starter_chars: dict[str, float] = {}
        for i, width_value in enumerate(widths_raw):
            try:
                width = require_font_float(width_value, "invalid font widths array")
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
    first_char = lookup_dict_key(font_obj, "FirstChar")
    last_char = lookup_dict_key(font_obj, "LastChar")
    try:
        first_char_int = require_font_int(first_char, "invalid font FirstChar")
        last_char_int = require_font_int(last_char, "invalid font LastChar")
    except ValueError:
        return {}

    base_font_raw = normalize_pdf_name(lookup_dict_key(font_obj, "BaseFont")) or ""
    base_name = base_font_raw.split("+", 1)[1] if "+" in base_font_raw else base_font_raw
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

    lig_widths_raw = lookup_dict_key(font_obj, "Widths")
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
    "NO_SPACE_AFTER",
    "NO_SPACE_BEFORE",
    "cached_encode_latin1",
    "can_merge_cross_font_word",
    "detect_ligature_overrides",
    "find_companion_font",
    "gap_separator",
    "get_font_file",
    "is_garbage_text",
    "load_ligature_font_tables",
    "normalize_extracted_text",
)

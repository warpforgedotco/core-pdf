# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_09_fonts.encoding import PDFDOC_ENCODING_TABLE
from core_pdf.impl.engine.spec.s_09_fonts.glyphs import glyph_name_to_unicode

EncodingFallback = Callable[[int], str]
LIGATURE_TEXT_OVERRIDES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def fallback_with_pdfdoc(b: int) -> str:
    return normalize_ligature_text(PDFDOC_ENCODING_TABLE[b])


def normalize_ligature_text(text: str) -> str:
    return LIGATURE_TEXT_OVERRIDES.get(text, text)


@lru_cache(maxsize=256)
def cached_translate_table(table: tuple[str, ...]) -> dict[int, str]:
    return {i: text for i, text in enumerate(table) if text != chr(i)}


def decode_with_table(data: bytes, table: tuple[str, ...]) -> str:
    return data.decode("latin-1").translate(cached_translate_table(table))


def build_decode_table(
    fallback_fn: EncodingFallback,
    differences: dict[int, str] | None = None,
) -> tuple[str, ...]:
    gtn = glyph_name_to_unicode
    if not differences:
        return tuple(fallback_fn(b) for b in range(256))
    table = [fallback_fn(b) for b in range(256)]
    for code, glyph_name in differences.items():
        table[code] = gtn(glyph_name)
    return tuple(table)


@lru_cache(maxsize=256)
def cached_decode_table(
    key: str, differences_items: tuple[tuple[int, str], ...]
) -> tuple[str, ...]:
    differences = dict(differences_items)
    return build_decode_table(ENCODING_FALLBACKS.get(key, fallback_with_pdfdoc), differences)


def parse_differences(
    value: Any, resolve_name: Callable[[Any], str | None] | None = None
) -> dict[int, str]:
    differences: dict[int, str] = {}
    if value is None:
        return differences
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid encoding differences array")
    code = 0
    for item in value:
        if type(item) is int:
            if item < 0 or item > 255:
                continue
            code = item
            continue
        if resolve_name is not None:
            glyph_name = resolve_name(item)
        else:
            glyph_name = normalize_pdf_name(item)
        if glyph_name is None:
            continue
        if code < 0 or code > 255:
            continue
        differences[code] = glyph_name
        code += 1
    return differences


ENCODING_FALLBACKS: dict[str, EncodingFallback] = {
    "Type3": lambda b: "",
    "WinAnsiEncoding": lambda b: normalize_ligature_text(bytes([b]).decode("cp1252", "replace")),
    "MacRomanEncoding": lambda b: normalize_ligature_text(
        bytes([b]).decode("mac_roman", "replace")
    ),
}

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import unicodedata
from functools import lru_cache

from core_pdf.impl.engine.spec.s_09_fonts.data.core14 import (
    COMMON_GLYPHS,
    FONT_DATA,
    GLYPH_MAP,
    GLYPH_NAME_ALIASES,
    MODIFIER_NAMES,
    SYMBOL_GLYPH_NAME_ALIASES,
)

HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
TEX_GLYPH_ALIASES = {
    "lscript": "\u2113",
    "integraltext": "\u222b",
    "summationdisplay": "\u2211",
    "summationtext": "\u2211",
    "parenleftbig": "(",
    "parenleftBig": "(",
    "parenleftbigg": "(",
    "parenleftBigg": "(",
    "parenrightbig": ")",
    "parenrightBig": ")",
    "parenrightbigg": ")",
    "parenrightBigg": ")",
}


def ensure_glyph_map() -> dict[str, str]:
    if GLYPH_MAP:
        return GLYPH_MAP
    GLYPH_MAP.update(COMMON_GLYPHS)
    GLYPH_MAP.update(GLYPH_NAME_ALIASES)
    GLYPH_MAP.update(SYMBOL_GLYPH_NAME_ALIASES)
    GLYPH_MAP.update(FONT_DATA)
    return GLYPH_MAP


@lru_cache(maxsize=2048)
def glyph_name_to_unicode(name: str) -> str:
    if not name or name.startswith("."):
        return ""

    full = ensure_glyph_map()
    original_name = name
    name = name.split(".", 1)[0]
    if name.isdigit() or (name.startswith("i") and name[1:].isdigit()):
        return ""
    if len(name) == 1:
        return name
    if "_" in name:
        raw_parts = name.split("_")
        parts = [glyph_name_part_to_unicode(part, full) for part in raw_parts]
        if any(
            mapped == "" or "_" in mapped or (mapped == raw and len(raw) != 1)
            for raw, mapped in zip(raw_parts, parts, strict=True)
        ):
            return original_name
        return "".join(parts)

    return glyph_name_part_to_unicode(name, full, unknown_name=original_name)


def glyph_name_part_to_unicode(
    name: str, full: dict[str, str], *, unknown_name: str | None = None
) -> str:
    result = full.get(name)
    if result is not None:
        return result
    result = TEX_GLYPH_ALIASES.get(name)
    if result is not None:
        return result

    for base in (name.replace("_", "") if "_" in name else None,):
        if base and base != name and base in FONT_DATA:
            return FONT_DATA[base]

    for suffix in ("small", "superior", "inferior", "oldstyle", "fitted"):
        if name.endswith(suffix) and len(name) > len(suffix):
            base = name[: -len(suffix)]
            for candidate in (
                base,
                base.lower(),
                base[:1].upper() + base[1:].lower() if base else base,
            ):
                if len(candidate) > 1 and candidate in FONT_DATA:
                    return FONT_DATA[candidate]

    if is_uni_sequence(name):
        try:
            chars = []
            for i in range(3, len(name), 4):
                codepoint = int(name[i : i + 4], 16)
                if 0xD800 <= codepoint <= 0xDFFF:
                    return name
                chars.append(chr(codepoint))
            return "".join(chars)
        except ValueError:
            return name
    if is_u_codepoint(name):
        try:
            codepoint = int(name[1:], 16)
            if 0xD800 <= codepoint <= 0xDFFF:
                return name
            return chr(codepoint)
        except ValueError:
            return name

    modifier_parts = split_single_letter_modifier(name)
    if modifier_parts is not None:
        base, suffix = modifier_parts
        modifier = MODIFIER_NAMES.get(suffix)
        if modifier is not None:
            cat = "CAPITAL" if base.isupper() else "SMALL"
            try:
                return unicodedata.lookup(
                    f"LATIN {cat} LETTER {base.upper() if base.isupper() else base} WITH {modifier}"
                )
            except KeyError:
                pass

    try:
        return unicodedata.lookup(name)
    except KeyError:
        return unknown_name or name


def is_uni_sequence(name: str) -> bool:
    if len(name) < 7 or not name.startswith("uni") or (len(name) - 3) % 4 != 0:
        return False
    return all(ch in HEX_DIGITS for ch in name[3:])


def is_u_codepoint(name: str) -> bool:
    if len(name) < 5 or len(name) > 7 or not name.startswith("u"):
        return False
    return all(ch in HEX_DIGITS for ch in name[1:])


def split_single_letter_modifier(name: str) -> tuple[str, str] | None:
    if len(name) < 2:
        return None
    base = name[0]
    suffix = name[1:]
    if not base.isalpha() or not base.isascii():
        return None
    if not suffix.isalpha() or not suffix.islower() or not suffix.isascii():
        return None
    return base, suffix

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from core_pdf.impl.engine.spec.s_09_fonts.data.core14 import (
    COMMON_GLYPHS,
    GLYPH_FONT_DATA,
    GLYPH_MAP,
    GLYPH_NAME_ALIASES,
    MODIFIER_NAMES,
    SYMBOL_GLYPH_NAME_ALIASES,
)

GLYPH_MODIFIER_RE = re.compile(r"^([A-Za-z])([a-z]+)$")


def ensure_glyph_map() -> dict[str, str]:
    if GLYPH_MAP:
        return GLYPH_MAP
    GLYPH_MAP.update(COMMON_GLYPHS)
    GLYPH_MAP.update(GLYPH_NAME_ALIASES)
    GLYPH_MAP.update(SYMBOL_GLYPH_NAME_ALIASES)
    GLYPH_MAP.update(GLYPH_FONT_DATA)
    return GLYPH_MAP


@lru_cache(maxsize=2048)
def glyph_name_to_unicode(name: str) -> str:
    if not name or name.startswith("."):
        return ""
    if len(name) == 1:
        return name

    full = ensure_glyph_map()
    result = full.get(name)
    if result is not None:
        return result

    for base in (
        name.split(".", 1)[0] if "." in name else None,
        name.replace("_", "") if "_" in name else None,
    ):
        if base and base != name and base in GLYPH_FONT_DATA:
            return GLYPH_FONT_DATA[base]

    for suffix in ("small", "superior", "inferior", "oldstyle", "fitted"):
        if name.endswith(suffix) and len(name) > len(suffix):
            base = name[: -len(suffix)]
            for candidate in (
                base,
                base.lower(),
                base[:1].upper() + base[1:].lower() if base else base,
            ):
                if len(candidate) > 1 and candidate in GLYPH_FONT_DATA:
                    return GLYPH_FONT_DATA[candidate]

    if name.startswith("uni") and len(name) in {7, 11}:
        try:
            return chr(int(name[3:], 16))
        except ValueError:
            return name
    if name.startswith("u") and len(name) > 1:
        try:
            return chr(int(name[1:], 16))
        except ValueError:
            return name

    match = GLYPH_MODIFIER_RE.match(name)
    if match:
        modifier = MODIFIER_NAMES.get(match.group(2))
        if modifier is not None:
            base = match.group(1)
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
        return name

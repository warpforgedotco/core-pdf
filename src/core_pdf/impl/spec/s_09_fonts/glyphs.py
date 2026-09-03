# SPDX-License-Identifier: AGPL-3.0-only
"""Native glyph-name Unicode recovery helpers."""

from __future__ import annotations

import threading
import unicodedata

from core_pdf.impl.spec.s_09_fonts.data.core14 import (
    COMMON_GLYPHS,
    GLYPH_DATA,
    GLYPH_MAP,
    GLYPH_NAME_ALIASES,
    MODIFIER_NAMES,
    SYMBOL_GLYPH_NAME_ALIASES,
)
from core_pdf.impl.spec.s_09_fonts.data.zapf_dingbats import ZAPF_DINGBATS_GLYPHS

HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
# The Adobe glyph list places the Symbol font's bracket, brace and parenthesis
# pieces -- the segments a reader stacks to build a bracket taller than one
# line -- in Adobe's corporate private-use area. Unicode 3.2 gave every one of
# them a real codepoint, and a private-use character is never useful in
# extracted text: it renders as a blank box and carries no meaning to any
# consumer. Prefer the standard codepoints.
ADOBE_PUA_GLYPH_ALIASES = {
    "parenlefttp": "⎛",
    "parenleftex": "⎜",
    "parenleftbt": "⎝",
    "parenrighttp": "⎞",
    "parenrightex": "⎟",
    "parenrightbt": "⎠",
    "bracketlefttp": "⎡",
    "bracketleftex": "⎢",
    "bracketleftbt": "⎣",
    "bracketrighttp": "⎤",
    "bracketrightex": "⎥",
    "bracketrightbt": "⎦",
    "bracelefttp": "⎧",
    "braceleftmid": "⎨",
    "braceleftbt": "⎩",
    "braceex": "⎪",
    "bracerighttp": "⎫",
    "bracerightmid": "⎬",
    "bracerightbt": "⎭",
    "integralex": "⎮",
    "arrowhorizex": "⎯",
    "arrowvertex": "⏐",
    # Sans-serif variants of characters that already exist in Unicode; the
    # distinction is a typeface, not a different character.
    "registersans": "®",
    "copyrightsans": "©",
    "trademarksans": "™",
}

TEX_GLYPH_ALIASES = {
    "Ifractur": "ℑ",
    "Rfractur": "ℜ",
    # Adobe's legacy glyph list treats the common underscore ligature names
    # as the corresponding presentation-form characters. Preserve that
    # semantic identity before the generic component-name fallback runs.
    "f_f": "ﬀ",
    "f_f_i": "ﬃ",
    "f_f_l": "ﬄ",
    "epsilon1": "ϵ",
    "check": "✓",
    "circlecopyrt": "©",
    "radicalbt": "√",
    "lscript": "\u2113",
    "integraltext": "\u222b",
    "integraldisplay": "\u222b",
    "summationdisplay": "\u2211",
    "summationtext": "\u2211",
    "oint": "\u222e",
    "smallint": "\u222b",
    "coprod": "\u2210",
    "producttext": "\u220f",
    "uniontext": "\u22c3",
    "intersectiontext": "\u22c2",
    "coproducttext": "\u2210",
    "triangleleft": "\u25c1",
    "notexistential": "\u2204",
    "parenleftbig": "(",
    "parenleftBig": "(",
    "parenleftbigg": "(",
    "parenleftBigg": "(",
    "parenrightbig": ")",
    "parenrightBig": ")",
    "parenrightbigg": ")",
    "parenrightBigg": ")",
    "bracketleftbig": "[",
    "bracketleftBig": "[",
    "bracketleftbigg": "[",
    "bracketleftBigg": "[",
    "bracketrightbig": "]",
    "bracketrightBig": "]",
    "bracketrightbigg": "]",
    "bracketrightBigg": "]",
    "braceleftbig": "{",
    "braceleftBig": "{",
    "braceleftbigg": "{",
    "braceleftBigg": "{",
    "bracerightbig": "}",
    "bracerightBig": "}",
    "bracerightbigg": "}",
    "bracerightBigg": "}",
    "slashbig": "/",
    # Delimiter extension pieces: the segments TeX stacks to build a tall
    # vertical bar. Unicode has an extension codepoint for the single rule;
    # the double one is only ever read back as the character it draws.
    "vextendsingle": "\u23d0",
    "vextenddouble": "\u2016",
    "bardbl": "\u2016",
    # Wide accents, which cmex supplies in several widths for one character.
    "hatwide": "\u02c6",
    "hatwider": "\u02c6",
    "hatwidest": "\u02c6",
    "tildewide": "\u02dc",
    "tildewider": "\u02dc",
    "tildewidest": "\u02dc",
    # cmsy relations and delimiters with exact Unicode counterparts.
    "latticetop": "\u22a4",
    "star": "\u22c6",
    "mapsto": "\u21a6",
    "floorleft": "\u230a",
    "floorright": "\u230b",
    "ceilingleft": "\u2308",
    "ceilingright": "\u2309",
    "angbracketleft": "\u27e8",
    "angbracketright": "\u27e9",
    "lessmuch": "\u226a",
    "prime": "\u2032",
    "intercal": "\u22ba",
    # \not is drawn as an overlay, so the combining form composes with the
    # relation it negates instead of landing beside it.
    "negationslash": "\u0338",
    "radicalbig": "√",
    "radicalBig": "√",
    "radicalbigg": "√",
    "radicalBigg": "√",
    # Display-size variants of operators the table already carries in their
    # text size. cmex supplies one glyph per size; they are the same character.
    "uniondisplay": "\u22c3",
    "intersectiondisplay": "\u22c2",
    "productdisplay": "\u220f",
    # Big angle brackets, following the parenleftbigg/bracketleftbigg pattern
    # above: a delimiter grown for display maths is still the delimiter.
    "angbracketleftbigg": "\u27e8",
    "angbracketleftBigg": "\u27e8",
    "angbracketrightbigg": "\u27e9",
    "angbracketrightBigg": "\u27e9",
    # AMS symbol fonts (msam/msbm) reached through their builtin encodings.
    "measuredangle": "\u2221",
    "squaresolid": "\u25a0",
    "subsetnoteql": "\u228a",
    "owner": "\u220b",
    # Named for the hook it draws on the left of the stem, not for its
    # direction: cmsy's arrowhookleft is TeX's \hookrightarrow.
    "arrowhookleft": "\u21aa",
    # cmsy names a variant Greek letter with a trailing 1, as with epsilon1.
    "rho1": "\u03f1",
    # cmsy's "triangle" is read as the increment sign rather than the geometric
    # shape, following Adobe's reading of Delta. pypdf reports it the same way.
    "triangle": "\u2206",
}
internal_GLYPH_MAP_LOCK = threading.Lock()


def ensure_glyph_map() -> dict[str, str]:
    if GLYPH_MAP:
        return GLYPH_MAP
    with internal_GLYPH_MAP_LOCK:
        if GLYPH_MAP:
            return GLYPH_MAP
        GLYPH_MAP.update(COMMON_GLYPHS)
        GLYPH_MAP.update(GLYPH_NAME_ALIASES)
        GLYPH_MAP.update(SYMBOL_GLYPH_NAME_ALIASES)
        GLYPH_MAP.update(GLYPH_DATA)
    return GLYPH_MAP


def glyph_name_to_unicode(name: str) -> str:
    if not name or name.startswith("."):
        return ""

    full = ensure_glyph_map()
    original_name = name
    name = name.split(".", 1)[0]
    alias = TEX_GLYPH_ALIASES.get(name)
    if alias is not None:
        return alias
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
    result = ADOBE_PUA_GLYPH_ALIASES.get(name)
    if result is not None:
        return result
    result = ZAPF_DINGBATS_GLYPHS.get(name)
    if result is not None:
        return result
    result = full.get(name)
    if result is not None:
        return result
    result = TEX_GLYPH_ALIASES.get(name)
    if result is not None:
        return result

    for base in (name.replace("_", "") if "_" in name else None,):
        if base and base != name and base in GLYPH_DATA:
            return GLYPH_DATA[base]

    for suffix in ("small", "superior", "inferior", "oldstyle", "fitted"):
        if name.endswith(suffix) and len(name) > len(suffix):
            base = name[: -len(suffix)]
            for candidate in (
                base,
                base.lower(),
                base[:1].upper() + base[1:].lower() if base else base,
            ):
                if len(candidate) > 1 and candidate in GLYPH_DATA:
                    return GLYPH_DATA[candidate]

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

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any

from core_adobe_fonts.agl.glyph_list import GLYPH_DATA
from core_adobe_fonts.agl.mapping import glyph_component_to_unicode
from core_pdf.impl.pdf_values import recover_pdf_name
from core_pdf_spec.s_07_syntax_primitives.text_string import PDFDOC_ENCODING_TABLE
from core_pdf_spec.s_09_fonts.data.base_encodings import (
    MAC_ROMAN_ENCODING,
    STANDARD_ENCODING,
    WIN_ANSI_ENCODING,
)
from core_pdf_spec.s_09_fonts.helpers import (
    BASE_ENCODING_GLYPH_NAMES,
    get_base_encoding_glyph_names,
)
from core_pdf_spec.s_09_fonts.helpers import (
    build_simple_encoding_glyph_names as spec_simple_encoding_glyph_names,
)
from core_pdf_spec.standards import SemanticContext, recognized_version

HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


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
    "registersans": "®",
    "copyrightsans": "©",
    "trademarksans": "™",
}


TEX_GLYPH_ALIASES = {
    "Ifractur": "ℑ",
    "Rfractur": "ℜ",
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
    "vextendsingle": "\u23d0",
    "vextenddouble": "\u2016",
    "bardbl": "\u2016",
    "hatwide": "\u02c6",
    "hatwider": "\u02c6",
    "hatwidest": "\u02c6",
    "tildewide": "\u02dc",
    "tildewider": "\u02dc",
    "tildewidest": "\u02dc",
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
    "negationslash": "\u0338",
    "radicalbig": "√",
    "radicalBig": "√",
    "radicalbigg": "√",
    "radicalBigg": "√",
    "uniondisplay": "\u22c3",
    "intersectiondisplay": "\u22c2",
    "productdisplay": "\u220f",
    "angbracketleftbigg": "\u27e8",
    "angbracketleftBigg": "\u27e8",
    "angbracketrightbigg": "\u27e9",
    "angbracketrightBigg": "\u27e9",
    "measuredangle": "\u2221",
    "squaresolid": "\u25a0",
    "subsetnoteql": "\u228a",
    "owner": "\u220b",
    "arrowhookleft": "\u21aa",
    "rho1": "\u03f1",
    "triangle": "\u2206",
}


@lru_cache(maxsize=4096)
def glyph_name_to_unicode(name: str) -> str:
    if not name or name.startswith("."):
        return ""

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
        parts = [glyph_name_part_to_unicode(part) for part in raw_parts]
        if any(
            mapped == "" or "_" in mapped or (mapped == raw and len(raw) != 1)
            for raw, mapped in zip(raw_parts, parts, strict=True)
        ):
            return original_name
        return "".join(parts)

    return glyph_name_part_to_unicode(name, unknown_name=original_name)


def glyph_name_part_to_unicode(name: str, *, unknown_name: str | None = None) -> str:
    result = ADOBE_PUA_GLYPH_ALIASES.get(name)
    if result is not None:
        return result
    result = glyph_component_to_unicode(name, zapf_dingbats=True)
    if result:
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


MODIFIER_NAMES: dict[str, str] = {
    "acute": "ACUTE ACCENT",
    "grave": "GRAVE ACCENT",
    "circumflex": "CIRCUMFLEX ACCENT",
    "dieresis": "DIAERESIS",
    "tilde": "TILDE",
    "macron": "MACRON",
    "breve": "BREVE",
    "dotaccent": "DOT ABOVE",
    "ring": "RING ABOVE",
    "cedilla": "CEDILLA",
    "hungarumlaut": "DOUBLE ACUTE ACCENT",
    "ogonek": "OGONEK",
    "caron": "CARON",
}


def recover_strip_subset_tag(font_name: str) -> str:
    return font_name.split("+", 1)[-1]


LIGATURE_TEXT_OVERRIDES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


LEGITIMATE_MULTI_CHAR_GLYPHS = frozenset({"ff", "fi", "fl", "ffi", "ffl", "st"})


def normalize_ligature_text(text: str) -> str:
    return LIGATURE_TEXT_OVERRIDES.get(text, text)


def unicode_for_glyph_name(glyph_name: str) -> str | None:
    mapped = glyph_name_to_unicode(glyph_name)
    if not mapped or (mapped == glyph_name and len(glyph_name) != 1):
        return None
    return normalize_ligature_text(mapped)


def resolve_base_encoding(table: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        normalize_ligature_text(text) if text else (chr(code) if code < 32 else "")
        for code, text in enumerate(table)
    )


STANDARD_ENCODING_TABLE = resolve_base_encoding(STANDARD_ENCODING)


WIN_ANSI_ENCODING_TABLE = resolve_base_encoding(WIN_ANSI_ENCODING)


MAC_ROMAN_ENCODING_TABLE = resolve_base_encoding(MAC_ROMAN_ENCODING)


def build_decode_table(
    key: str,
    differences: dict[int, str] | tuple[tuple[int, str], ...] | None = None,
    *,
    context: SemanticContext | None = None,
) -> tuple[str, ...]:
    base = ENCODING_FALLBACKS.get(key, PDFDOC_FALLBACK_TABLE)
    if context is not None and key in BASE_ENCODING_GLYPH_NAMES:
        names = get_base_encoding_glyph_names(key, context=encoding_context(context))
        modern_names = BASE_ENCODING_GLYPH_NAMES[key]
        if names is not modern_names:
            base = tuple(
                (unicode_for_glyph_name(name) or "") if name != modern_names[code] else base[code]
                for code, name in enumerate(names)
            )
    if not differences:
        return base
    table = list(base)
    items = differences.items() if isinstance(differences, dict) else differences
    for code, glyph_name in items:
        mapped = unicode_for_glyph_name(glyph_name)
        if mapped is None:
            if glyph_name.isdecimal():
                continue
            table[code] = ""
            continue
        table[code] = mapped
    return tuple(table)


def recover_differences(
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
            glyph_name = recover_pdf_name(item)
        if glyph_name is None:
            continue
        if code < 0 or code > 255:
            continue
        differences[code] = glyph_name
        code += 1
    return differences


PDFDOC_FALLBACK_TABLE: tuple[str, ...] = tuple(
    normalize_ligature_text(text) for text in PDFDOC_ENCODING_TABLE
)


ENCODING_FALLBACKS: dict[str, tuple[str, ...]] = {
    "StandardEncoding": STANDARD_ENCODING_TABLE,
    "Type3": STANDARD_ENCODING_TABLE,
    "WinAnsiEncoding": WIN_ANSI_ENCODING_TABLE,
    "MacRomanEncoding": MAC_ROMAN_ENCODING_TABLE,
}


def build_simple_encoding_glyph_names(
    base_encoding: str | None,
    builtin_encoding: Mapping[int, str],
    differences: Mapping[int, str],
    *,
    authoritative_builtin: bool,
    context: SemanticContext | None = None,
) -> tuple[str, ...]:
    return spec_simple_encoding_glyph_names(
        base_encoding,
        {
            int(code): name or ".notdef"
            for code, name in builtin_encoding.items()
            if 0 <= code < 256
        },
        {int(code): name or ".notdef" for code, name in differences.items() if 0 <= code < 256},
        authoritative_builtin=authoritative_builtin,
        context=encoding_context(context),
    )


def encoding_context(context: SemanticContext | None) -> SemanticContext | None:
    return None if recognized_version(context) is None else context

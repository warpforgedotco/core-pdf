"""ISO 32000-2 9.8.2: a font descriptor's flags and style, and what they decide about a font.

The flags and the descriptor's FontWeight, ItalicAngle and FontFamily (9.8.1, Table 120) say
whether a font is symbolic, which decides a simple font's default base encoding (9.6.5.1,
Table 112) and how a TrueType font's codes select glyphs (9.6.5.4), and describe its style:
fixed pitch, serif, italic, bold. Many files carry wrong flags, so callers keep their own
evidence (the font program's cmap subtables, the font name) as the fallback.
"""

from __future__ import annotations

from typing import Any

from core_pdf.impl.adobe_fonts.encodings import MAC_ROMAN_ENCODING_GLYPH_NAMES
from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import decoded_name
from core_pdf.impl.spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf.impl.spec.types import PdfString
from core_records import GeneratedRecord

# Table 121: bit positions 1 to 32, low-order first.
FIXED_PITCH = 1 << 0
SERIF = 1 << 1
SYMBOLIC = 1 << 2
SCRIPT = 1 << 3
NONSYMBOLIC = 1 << 5
ITALIC = 1 << 6
ALL_CAP = 1 << 16
SMALL_CAP = 1 << 17
FORCE_BOLD = 1 << 18

# Table 120: FontWeight 400 is normal and 700 bold; 600 (semibold) and above read as bold.
BOLD_FONT_WEIGHT = 600.0


class FontDescriptorStyle(GeneratedRecord):
    # None where the descriptor is absent or the entry is missing or not a number.
    flags: int | None = None
    italic_angle: float | None = None
    font_weight: float | None = None
    font_family: str | None = None

    def has_flag(self, flag: int) -> bool:
        return self.flags is not None and bool(self.flags & flag)


NO_DESCRIPTOR_STYLE = FontDescriptorStyle()


def descriptor_number(value: object) -> float | None:
    if type(value) is int or type(value) is float:
        return float(value)
    return None


def descriptor_text(value: object) -> str | None:
    if isinstance(value, PdfString):
        try:
            return decode_pdf_text_string(value.data)
        except ValueError:
            return value.data.decode("latin-1")
    return decoded_name(value)


@cites(
    "ISO 32000-2", "9.8.2", rule="Flags is an unsigned 32-bit integer whose bits Table 121 defines"
)
def read_descriptor_style(descriptor: dict[str, Any] | None) -> FontDescriptorStyle:
    """The flags and style entries of a font descriptor; a negative Flags, which some writers
    produce from a signed 32-bit value, is read as its unsigned bit pattern."""
    if not isinstance(descriptor, dict):
        return NO_DESCRIPTOR_STYLE
    raw_flags = descriptor.get("Flags")
    flags: int | None = None
    if type(raw_flags) is int:
        flags = raw_flags & 0xFFFFFFFF
    elif type(raw_flags) is float and raw_flags.is_integer():
        flags = int(raw_flags) & 0xFFFFFFFF
    style = FontDescriptorStyle(
        flags,
        descriptor_number(descriptor.get("ItalicAngle")),
        descriptor_number(descriptor.get("FontWeight")),
        descriptor_text(descriptor.get("FontFamily")),
    )
    return NO_DESCRIPTOR_STYLE if style == NO_DESCRIPTOR_STYLE else style


@cites(
    "ISO 32000-2",
    "9.8.2",
    rule="a processor checks the Symbolic flag to tell a symbolic font from a nonsymbolic one",
)
def descriptor_symbolic(style: FontDescriptorStyle) -> bool | None:
    """Whether the font is symbolic, by its Symbolic flag alone (the Nonsymbolic flag is its
    redundant inverse), or None where the descriptor gives no Flags."""
    if style.flags is None:
        return None
    return bool(style.flags & SYMBOLIC)


@cites(
    "ISO 32000-2",
    "9.8.2",
    rule="the ForceBold flag and a FontWeight of 600 or more mark a bold font",
)
def descriptor_bold(style: FontDescriptorStyle) -> bool:
    """Bold by the descriptor: a FontWeight of 600 or more, or the ForceBold flag, which
    writers set only for bold faces. False means no evidence, not a light face."""
    weight = style.font_weight
    if weight is not None and weight >= BOLD_FONT_WEIGHT:
        return True
    return style.has_flag(FORCE_BOLD)


@cites(
    "ISO 32000-2",
    "9.8.2",
    rule="the Italic flag and a nonzero ItalicAngle mark an italic or oblique font",
)
def descriptor_italic(style: FontDescriptorStyle) -> bool:
    """Italic by the descriptor: an ItalicAngle of a degree or more from upright (Table 120
    gives it in degrees counterclockwise, negative for a rightward slant), or the Italic flag
    where ItalicAngle is absent. ItalicAngle is required and measures the glyphs, so an
    upright angle outweighs an Italic flag, which some symbol fonts set on upright glyphs.
    False means no evidence."""
    angle = style.italic_angle
    if angle is not None:
        return abs(angle) >= 1.0
    return style.has_flag(ITALIC)


# 9.6.5.1: the standard 14 fonts include two symbolic fonts, Symbol and ZapfDingbats; these
# are the base names that select them, without a subset tag or a ",Style" suffix.
STANDARD_SYMBOLIC_FONTS: dict[str, str] = {
    "Symbol": "Symbol",
    "SymbolMT": "Symbol",
    "ZapfDingbats": "ZapfDingbats",
    "ZapfDingbatsITC": "ZapfDingbats",
    "Dingbats": "ZapfDingbats",
}


@cites(
    "ISO 32000-2",
    "9.6.5.1",
    rule="Table 112: an unembedded symbolic font defaults to its built-in encoding",
)
def standard_symbolic_font(base_font_name: str | None) -> str | None:
    """Symbol or ZapfDingbats where the font names one of the two symbolic standard 14 fonts.

    Table 112 makes a symbolic font's built-in encoding the default base encoding of a font
    that is not embedded, StandardEncoding being that of a nonsymbolic one. Only these two
    fonts have a built-in encoding a processor knows without the program, and they are
    symbolic by definition (9.6.5.1), so the name decides even where the flags disagree."""
    if not base_font_name:
        return None
    name = base_font_name.split("+", 1)[-1].split(",", 1)[0]
    return STANDARD_SYMBOLIC_FONTS.get(name)


@cites(
    "ISO 32000-2",
    "9.6.5.4",
    rule="a symbolic TrueType font maps codes by its (3,0) or (1,0) cmap, others by glyph name",
)
def truetype_uses_symbolic_cmap(encoding: object, symbolic: bool | None) -> bool:
    """Whether a simple TrueType font's codes select glyphs directly through its (3,0) or (1,0)
    cmap subtable rather than through glyph names and its (3,1) or (1,0) subtable.

    The clause has a name path, for a font whose Encoding is MacRomanEncoding or
    WinAnsiEncoding or whose Nonsymbolic flag is set, and a direct path, for a font with no
    Encoding or whose Symbolic flag is set (its Encoding then ignored). A symbolic font with
    one of those two named encodings meets both; this takes the name path, which the clause
    states first and which keeps the encoding the writer gave. 9.8.2 says to judge the state by
    the Symbolic flag, so a font whose flags are present with Symbolic clear is nonsymbolic.
    Without flags, a font with no Encoding takes the direct path and one with an encoding
    dictionary the name path.
    """
    name = encoding if isinstance(encoding, str) else None
    if name in ("MacRomanEncoding", "WinAnsiEncoding"):
        return False
    if symbolic is not None:
        return symbolic
    return encoding is None


def mac_os_roman_codes() -> dict[str, int]:
    codes = {name: code for code, name in enumerate(MAC_ROMAN_ENCODING_GLYPH_NAMES) if name}
    codes.pop(".notdef", None)
    codes.pop("currency", None)
    codes.update(MAC_OS_ROMAN_ADDITIONS)
    return codes


# Table 113: the entries the Mac OS Roman encoding adds to MacRomanEncoding, Euro replacing
# currency at 219.
MAC_OS_ROMAN_ADDITIONS: dict[str, int] = {
    "notequal": 173,
    "infinity": 176,
    "lessequal": 178,
    "greaterequal": 179,
    "partialdiff": 182,
    "summation": 183,
    "product": 184,
    "pi": 185,
    "integral": 186,
    "Omega": 189,
    "radical": 195,
    "approxequal": 197,
    "Delta": 198,
    "lozenge": 215,
    "Euro": 219,
    "apple": 240,
}


@cites(
    "ISO 32000-2",
    "9.6.5.4",
    rule="without a (3,1) cmap, a glyph name maps to a (1,0) code by Mac OS Roman (Table 113)",
)
def mac_os_roman_code(glyph_name: str) -> int | None:
    """The Mac OS Roman code of a glyph name, for a lookup in a TrueType (1,0) subtable."""
    return MAC_OS_ROMAN_CODES.get(glyph_name)


MAC_OS_ROMAN_CODES: dict[str, int] = mac_os_roman_codes()

# 9.6.5.4: the (3,0) subtable's code ranges, each byte prefixed by the range's high byte.
SYMBOL_CMAP_PREFIXES = (0x0000, 0xF000, 0xF100, 0xF200)


__all__ = [
    "FontDescriptorStyle",
    "NO_DESCRIPTOR_STYLE",
    "SYMBOL_CMAP_PREFIXES",
    "descriptor_bold",
    "descriptor_italic",
    "descriptor_symbolic",
    "mac_os_roman_code",
    "read_descriptor_style",
    "standard_symbolic_font",
    "truetype_uses_symbolic_cmap",
]

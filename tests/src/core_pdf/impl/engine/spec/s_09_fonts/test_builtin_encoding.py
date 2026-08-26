# SPDX-License-Identifier: AGPL-3.0-only
"""A font program's built-in encoding is the implicit base encoding.

Table 114: when /BaseEncoding is absent, "for a font program that is embedded
in the PDF file, the implicit base encoding shall be the font program's
built-in encoding", and /Differences describes changes from that. core-pdf
could read it back out of a Type 1 program but not out of a CFF one, so
CFF-embedded fonts silently fell back to a standard table.
"""

from __future__ import annotations

import struct

from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.engine.spec.s_09_fonts.font_program import CFFFont, cff_font_for_data
from core_pdf.impl.objects import PdfStream


def index(items: list[bytes], off_size: int = 1) -> bytes:
    """Serialise a CFF INDEX (section 5)."""
    if not items:
        return struct.pack(">H", 0)
    out = struct.pack(">H", len(items)) + bytes([off_size])
    offset = 1
    for item in items:
        out += offset.to_bytes(off_size, "big")
        offset += len(item)
    out += offset.to_bytes(off_size, "big")
    return out + b"".join(items)


def dict_op(operands: list[int], op: int) -> bytes:
    """Serialise DICT operands followed by a one-byte operator."""
    out = b""
    for value in operands:
        # 5-byte integer form keeps offsets patchable at a fixed width.
        out += b"\x1d" + struct.pack(">i", value)
    return out + bytes([op])


def build_cff(encoding_bytes: bytes, glyph_names: list[str]) -> bytes:
    """Build a minimal CFF with a custom Encoding and a format 0 charset."""
    header = bytes([1, 0, 4, 1])
    name_index = index([b"TestFont"])
    strings = [name.encode() for name in glyph_names]
    string_index = index(strings)
    global_subrs = index([])
    # One empty charstring per glyph, plus .notdef.
    charstrings = index([b"\x0e"] * (len(glyph_names) + 1))
    # Custom SIDs start after the 391 standard strings.
    charset = bytes([0]) + b"".join(struct.pack(">H", 391 + i) for i in range(len(glyph_names)))

    # Lay the pieces out after a fixed-size top DICT, then patch the offsets.
    top_placeholder = dict_op([0], 15) + dict_op([0], 16) + dict_op([0], 17)
    top_index_size = len(index([top_placeholder]))
    base = len(header) + len(name_index) + top_index_size + len(string_index) + len(global_subrs)
    charset_off = base
    encoding_off = charset_off + len(charset)
    charstrings_off = encoding_off + len(encoding_bytes)
    top = dict_op([charset_off], 15) + dict_op([encoding_off], 16) + dict_op([charstrings_off], 17)
    assert len(top) == len(top_placeholder)
    return (
        header
        + name_index
        + index([top])
        + string_index
        + global_subrs
        + charset
        + encoding_bytes
        + charstrings
    )


def test_reads_a_format_0_custom_encoding() -> None:
    # Format 0: code array, one entry per glyph starting at glyph 1.
    encoding = bytes([0, 3]) + bytes([0x41, 0x42, 0x43])
    font = cff_font_for_data(build_cff(encoding, ["alpha", "beta", "gamma"]))
    assert font.builtin_encoding() == {0x41: "alpha", 0x42: "beta", 0x43: "gamma"}


def test_reads_a_format_1_range_encoding() -> None:
    # Format 1: one range of three sequential codes from 0x61.
    encoding = bytes([1, 1]) + bytes([0x61, 2])
    font = cff_font_for_data(build_cff(encoding, ["alpha", "beta", "gamma"]))
    assert font.builtin_encoding() == {0x61: "alpha", 0x62: "beta", 0x63: "gamma"}


def test_reads_supplements_appended_to_an_encoding() -> None:
    # High bit of the format byte means supplements follow; each gives a
    # second code for an already-encoded glyph, keyed by SID.
    encoding = (
        bytes([0x80, 2]) + bytes([0x41, 0x42]) + bytes([1]) + bytes([0x5A]) + struct.pack(">H", 391)
    )
    font = cff_font_for_data(build_cff(encoding, ["alpha", "beta"]))
    assert font.builtin_encoding() == {0x41: "alpha", 0x42: "beta", 0x5A: "alpha"}


def test_predefined_standard_encoding_reports_no_custom_encoding() -> None:
    # The decoder already applies StandardEncoding as the implicit base.
    font = CFFFont(build_cff(bytes([0, 1]) + bytes([0x41]), ["alpha"]))
    font.top_dict[16] = [0.0]

    assert font.builtin_encoding() == {}


def test_builtin_encoding_is_the_implicit_base_for_a_type1_font() -> None:
    font_program = b"""
    /Encoding 256 array
    0 1 255 {1 index exch /.notdef put} for
    dup 65 /ampersand put
    dup 66 /question put
    readonly def
    currentfile eexec
    """
    font = {
        "Subtype": "Type1",
        "BaseFont": "AAAAAA+Test",
        "FontDescriptor": {"FontFile": PdfStream(decoded_data=font_program)},
        "Encoding": {"Differences": [66, "percent"]},
    }
    decoder = FontDecoder(font)
    # The built-in encoding supplies 0101, and /Differences overrides 0102.
    assert decoder.decode(bytes([65])) == "&"
    assert decoder.decode(bytes([66])) == "%"


def test_an_explicit_base_encoding_still_wins() -> None:
    font_program = b"""
    /Encoding 256 array
    0 1 255 {1 index exch /.notdef put} for
    dup 65 /ampersand put
    readonly def
    currentfile eexec
    """
    font = {
        "Subtype": "Type1",
        "BaseFont": "AAAAAA+Test",
        "FontDescriptor": {"FontFile": PdfStream(decoded_data=font_program)},
        "Encoding": {"BaseEncoding": "WinAnsiEncoding"},
    }
    decoder = FontDecoder(font)
    # Naming a base encoding overrides the program's own table (Table 114).
    assert decoder.decode(bytes([65])) == "A"


def test_sparse_cff_encoding_drives_text_and_outline_selection() -> None:
    cff_data = build_cff(bytes([0, 1, 0x41]), ["alpha", "beta"])
    font = {
        "Subtype": "Type1",
        "BaseFont": "EmbeddedGreek",
        "FontDescriptor": {"FontFile3": PdfStream({"Subtype": "Type1C"}, decoded_data=cff_data)},
    }

    decoder = FontDecoder(font)
    glyphs = decoder.decode_glyphs(b"AB")

    assert isinstance(decoder.font_program, CFFFont)
    assert decoder.internal_simple_glyph_name(0x41) == "alpha"
    assert glyphs[0].unicode == "α"
    assert glyphs[0].gid == decoder.font_program.glyph_id_for_name("alpha") == 1
    # A custom CFF encoding is authoritative and sparse. Code 0x42 does not
    # inherit StandardEncoding's B merely because the font has a beta glyph.
    assert decoder.internal_simple_glyph_name(0x42) == ".notdef"
    assert glyphs[1].unicode == "�"
    assert glyphs[1].gid == 0


def test_pdf_differences_override_sparse_cff_encoding_for_text_and_outline() -> None:
    cff_data = build_cff(bytes([0, 1, 0x41]), ["alpha", "beta"])
    font = {
        "Subtype": "Type1",
        "BaseFont": "EmbeddedGreek",
        "Encoding": {"Differences": [0x42, "beta"]},
        "FontDescriptor": {"FontFile3": PdfStream({"Subtype": "Type1C"}, decoded_data=cff_data)},
    }

    decoder = FontDecoder(font)
    glyph = decoder.decode_glyphs(b"B")[0]

    assert isinstance(decoder.font_program, CFFFont)
    assert glyph.unicode == "β"
    assert glyph.gid == decoder.font_program.glyph_id_for_name("beta") == 2


def test_computer_modern_math_glyph_names_beat_a_useless_tounicode() -> None:
    """A Computer Modern math font's glyph names are authoritative.

    TeX's cmex, cmsy and cmmi carry ToUnicode maps that report the raw code,
    so a delimiter extension arrived as U+000C. glyph_decode already treats
    these fonts' glyph names as authoritative; the names simply had to resolve
    before that could take effect.
    """
    from core_pdf.impl.engine.spec.s_09_fonts.glyphs import glyph_name_to_unicode

    # cmex delimiter and accent pieces.
    assert glyph_name_to_unicode("vextendsingle") == "⏐"
    assert glyph_name_to_unicode("vextenddouble") == "‖"
    assert glyph_name_to_unicode("hatwide") == "ˆ"
    assert glyph_name_to_unicode("tildewide") == "˜"
    # cmsy relations and delimiters.
    assert glyph_name_to_unicode("latticetop") == "⊤"
    assert glyph_name_to_unicode("star") == "⋆"
    assert glyph_name_to_unicode("mapsto") == "↦"
    assert glyph_name_to_unicode("floorleft") == "⌊"
    assert glyph_name_to_unicode("floorright") == "⌋"
    assert glyph_name_to_unicode("angbracketleft") == "⟨"
    assert glyph_name_to_unicode("bardbl") == "‖"
    # \not overlays the relation it negates, so it composes rather than
    # standing beside it.
    assert glyph_name_to_unicode("negationslash") == "̸"

    for name in ("vextendsingle", "latticetop", "mapsto", "hatwide"):
        mapped = glyph_name_to_unicode(name)
        assert mapped
        assert mapped != name
        assert ord(mapped[0]) >= 32, f"{name} still decodes to a control character"

from core_pdf.impl._impl.fonts.decoder import FontDecoder
from core_pdf.impl._impl.fonts.font_program import CFFFont
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from tests.helpers.cff import build_cff


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
    from core_pdf.impl._impl.fonts.glyphs import glyph_name_to_unicode

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

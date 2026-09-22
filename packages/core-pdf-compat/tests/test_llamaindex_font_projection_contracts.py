from types import SimpleNamespace

import pytest

from core_pdf.impl.fonts.decoder import FontDecoder
from core_pdf_compat.llamaindex._operator_text import (
    Font,
    OperatorTextProjection,
    difference_text,
    internal_glyph_name_to_unicode,
)
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver


@pytest.fixture
def projection():
    resolver = ObjectResolver(b"", {})
    yield OperatorTextProjection(SimpleNamespace(document=SimpleNamespace(resolver=resolver)))
    resolver.close()


@pytest.mark.parametrize(
    ("widths", "expected"),
    [
        ([], {}),
        ([10, [100, 200, 300]], {10: 100.0, 11: 200.0, 12: 300.0}),
        ([10, 12, 250.5], {10: 250.5, 11: 250.5, 12: 250.5}),
        ([10, [100, 200], 11, 12, 350], {10: 100.0, 11: 350.0, 12: 350.0}),
        ([12, 10, 250], {}),
        (["bad", 10, [200]], {10: 200.0}),
        ([10], {}),
        ([10, "bad"], {}),
        ([10, "bad", None], {}),
        ([10, 12, None], {}),
        ([10, []], {}),
    ],
)
@pytest.mark.parametrize("default", [None, 750.25])
def test_cid_width_forms_expand_ranges_and_allow_later_overrides(
    projection, widths, expected, default
):
    font = {"DescendantFonts": [None, {"W": widths, "DW": default}]}
    actual, fallback = projection.internal_widths(font, FontDecoder({}))
    assert actual == expected
    assert fallback == (0.0 if default is None else default)


@pytest.mark.parametrize("missing", [None, 450.9, -12.9])
def test_simple_font_widths_truncate_individual_metrics_toward_zero(projection, missing):
    font = {
        "FirstChar": 65,
        "Widths": [500.9, -20.9, "600.8"],
        "FontDescriptor": {"MissingWidth": missing},
    }
    widths, fallback = projection.internal_widths(font, FontDecoder({}))
    assert widths == {65: 500.0, 66: -20.0, 67: 600.0}
    assert fallback == (0.0 if missing is None else float(int(missing)))


def test_simple_font_uses_only_positive_single_byte_decoder_widths_when_undeclared(projection):
    decoder = FontDecoder({})
    decoder.widths = {-1: 300, 0: 0, 65: 500, 66: -20, 255: 200, 256: 700}
    assert projection.internal_widths({}, decoder) == ({65: 500, 255: 200}, 0.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("f_f", "ﬀ"),
        ("f_f_i", "ﬃ"),
        ("f_f_l", "ﬄ"),
        ("negationslash", "⁄"),
        ("A_B", "A_B"),
        ("A", "A"),
    ],
)
def test_legacy_glyph_aliases_preserve_facade_specific_spelling(name, expected):
    assert internal_glyph_name_to_unicode(name) == expected


@pytest.mark.parametrize(
    ("name", "code", "expected"),
    [
        (".notdef", 65, "□"),
        ("a123", 65, "A"),
        ("123", 65, "/123"),
        ("uni0041", 65, "/uni0041"),
        ("uni00410042", 65, "/uni00410042"),
        ("A", 66, "A"),
        ("f_f", 65, "ﬀ"),
        ("unknown_glyph", 65, "/unknown_glyph"),
    ],
)
def test_encoding_differences_distinguish_names_from_unicode(name, code, expected):
    assert difference_text(name, code) == expected


@pytest.mark.parametrize(
    ("encoding", "data", "expected"),
    [
        ("latin-1", b"\xff", "ÿ"),
        ("utf-8", b"\xff", "ÿ"),
        ("missing-codec", b"A", "A"),
        ("utf-16-be", b"\x00A", "A"),
    ],
)
def test_font_encoding_fallbacks_preserve_bytes_when_the_declared_codec_fails(
    encoding, data, expected
):
    font = Font(FontDecoder({}), " ", 200, encoding, {}, {}, 500)
    assert font.encoded(data) == expected


def test_mapped_text_expansion_does_not_change_encoded_character_widths():
    encoding = tuple(chr(code) for code in range(256))
    font = Font(FontDecoder({}), "_", 250, encoding, {"A": "fi", "_": " "}, {65: 600}, 400)
    assert font.decode_parts(b"A_B") == (("fi", " ", "B"), 1250)


@pytest.mark.parametrize(
    "codespaces",
    [
        b"<00ff> <0100>",
        b"<0100> <00ff>",
        b"<00> <0100>",
        b"<zz> <0100>",
        b"<00ff> <0100> <02ff> <0300>",
    ],
)
def test_optional_cmap_preserves_explicit_mappings_despite_malformed_codespaces(
    projection, codespaces
):
    from core_pdf_spec.s_07_syntax.stream import PdfStream

    data = (
        b"1 begincodespacerange\n"
        + codespaces
        + b"\nendcodespacerange\n2 beginbfchar\n<00ff> <0041>\n<0100> <0042>\nendbfchar\n"
    )
    cmap = projection.internal_to_unicode({"ToUnicode": PdfStream({}, data)}, FontDecoder({}))
    assert cmap is not None
    assert cmap.mappings == {b"\x00\xff": "A", b"\x01\x00": "B"}


def test_unreadable_optional_cmap_preserves_font_usability(projection):
    from core_pdf_spec.s_07_syntax.stream import PdfStream

    stream = PdfStream({"Filter": "FlateDecode"}, b"\xff", {"Filter": "FlateDecode"})
    assert projection.internal_to_unicode({"ToUnicode": stream}, FontDecoder({})) is None


def test_valid_optional_cmap_and_visible_space_projection(projection):
    from core_pdf_spec.s_07_syntax.stream import PdfStream

    data = b"2 beginbfchar\n<20> <2423>\n<41> <00660069>\nendbfchar\n"
    decoder = FontDecoder({})
    cmap = projection.internal_to_unicode({"ToUnicode": PdfStream({}, data)}, decoder)
    assert cmap is not None
    assert projection.internal_character_map(decoder, cmap) == {" ": " ", "A": "fi"}


@pytest.mark.parametrize(
    ("font", "expected"),
    [
        ({"ToUnicode": object()}, True),
        ({}, True),
        ({"CharProcs": {"space": object()}}, True),
        ({"CharProcs": {"unknown_name": object()}}, False),
        ({"CharProcs": {None: object()}}, False),
    ],
)
def test_type3_interpretability_depends_on_mapping_or_recognized_glyph_names(
    projection, font, expected
):
    assert projection.type3_interpretable(font) is expected


@pytest.mark.parametrize(
    "codespaces",
    [
        b"<00ff> <0100>",
        b"<0100> <00ff>",
        b"<00> <0100>",
        b"<zz> <0100>",
        b"<00ff> <0100> <02ff> <0300>",
    ],
)
def test_invalid_optional_cmap_without_usable_mappings_is_discarded(projection, codespaces):
    from core_pdf_spec.s_07_syntax.stream import PdfStream

    data = b"1 begincodespacerange\n" + codespaces + b"\nendcodespacerange\n"
    assert (
        projection.internal_to_unicode({"ToUnicode": PdfStream({}, data)}, FontDecoder({})) is None
    )

import struct
from typing import Any

import pytest

from core_pdf.impl.fonts import decoder
from core_pdf.impl.types import PdfName
from core_pdf_spec.s_07_syntax.stream import PdfStream


def internal_sfnt(
    tag: bytes, payload: bytes, *, offset: int = 28, length: int | None = None
) -> bytes:
    return (
        struct.pack(">4sHHHH", b"OTTO", 1, 16, 0, 0)
        + struct.pack(">4sIII", tag, 0, offset, len(payload) if length is None else length)
        + payload
    )


@pytest.mark.parametrize("payload", [b"", b"cff payload", bytes(range(256))])
def test_cff_table_extraction_preserves_exact_table_bytes(payload):
    assert decoder.internal_extract_cff_table(internal_sfnt(b"CFF ", payload)) == payload


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"OTTO",
        b"not an sfnt file",
        internal_sfnt(b"glyf", b"other"),
        internal_sfnt(b"CFF ", b"short", length=100),
        internal_sfnt(b"CFF ", b"x", offset=1000),
    ],
)
def test_missing_or_truncated_cff_table_is_unavailable(data):
    assert decoder.internal_extract_cff_table(data) is None


@pytest.mark.parametrize(
    "subtype", ["Type1", "MMType1", "TrueType", "CIDFontType0", "CIDFontType2"]
)
@pytest.mark.parametrize("stream_key", ["FontFile", "FontFile2", "FontFile3"])
@pytest.mark.parametrize("stream_type", ["Type1C", "OpenType"])
def test_malformed_embedded_programs_leave_font_recovery_available(
    subtype, stream_key, stream_type
):
    descriptor = {stream_key: PdfStream({"Subtype": PdfName(stream_type.encode())}, b"damaged")}
    font: dict[str, Any] = {"Subtype": PdfName(subtype.encode()), "FontDescriptor": descriptor}
    if subtype.startswith("CID"):
        font = {"Subtype": PdfName(b"Type0"), "DescendantFonts": [font]}
    assert decoder.internal_font_program_for_pdf_font(font) is None


@pytest.mark.parametrize("descriptor", [None, 1, b"invalid", {"FontFile2": 1}, {"FontFile3": []}])
@pytest.mark.parametrize("descendant", [False, True])
def test_invalid_font_descriptors_are_contained_at_reader_boundary(descriptor, descendant):
    font: dict[str, Any] = {"Subtype": PdfName(b"TrueType"), "FontDescriptor": descriptor}
    if descendant:
        font = {"Subtype": PdfName(b"Type0"), "DescendantFonts": [font], "FontDescriptor": 1}
    assert decoder.internal_font_program_for_pdf_font(font) is None


@pytest.mark.parametrize("base", [None, "Base"])
@pytest.mark.parametrize("child_descriptor", [None, {}, {"FontName": PdfName(b"Child")}])
def test_base_font_name_precedes_descendant_then_parent_descriptor(base, child_descriptor):
    font: dict[str, Any] = {
        "Subtype": PdfName(b"Type0"),
        "FontDescriptor": {"FontName": PdfName(b"Parent")},
        "DescendantFonts": [{"FontDescriptor": child_descriptor}],
    }
    if base is not None:
        font["BaseFont"] = PdfName(base.encode())
    assert decoder.resolve_base_font_name(font, "Type0") == (
        base or ("Child" if child_descriptor else "Parent")
    )


@pytest.mark.parametrize(
    "encoding",
    [
        None,
        "WinAnsiEncoding",
        PdfName(b"WinAnsiEncoding"),
        {},
        {"Differences": ()},
        {"Differences": 1},
    ],
)
@pytest.mark.parametrize("subtype", ["Type1", "TrueType", "Type3"])
def test_simple_font_decoding_preserves_ascii_across_encoding_defaults(encoding, subtype):
    font = decoder.FontDecoder({"Subtype": PdfName(subtype.encode()), "Encoding": encoding})
    assert font.decode(b"AB") == "AB"
    assert font.decode(b"") == ""
    glyphs = font.decode_glyphs(memoryview(b"AB"))
    assert [(g.char_code, g.unicode) for g in glyphs] == [(65, "A"), (66, "B")]


@pytest.mark.parametrize("overrides", [None, {0xFB01: "fi"}, {ord("x"): "y"}])
def test_cid_unicode_mapping_and_ligature_overrides_preserve_source_code(overrides):
    font = decoder.FontDecoder(
        {
            "Subtype": PdfName(b"Type0"),
            "Encoding": PdfName(b"Identity-H"),
            "DescendantFonts": [{"Subtype": PdfName(b"CIDFontType2")}],
            "ToUnicode": PdfStream({}, b"1 beginbfchar\n<0001> <fb01>\nendbfchar\n"),
        },
        ligature_overrides=overrides,
    )
    (glyph,) = font.decode_glyphs(b"\0\1")
    assert glyph.code_bytes == b"\0\1"
    assert glyph.char_code == glyph.cid == 1
    assert glyph.unicode == ("fi" if overrides and 0xFB01 in overrides else "ﬁ")
    if overrides and 0xFB01 in overrides:
        assert "ﬁ" in glyph.alternates


@pytest.mark.parametrize("value", [b"Adobe", PdfName(b"Adobe"), "Adobe", 1, None])
@pytest.mark.parametrize("descendant", [False, True])
def test_cid_system_info_normalizes_names_and_strings(value, descendant):
    font: dict[str, Any] = {"CIDSystemInfo": {"Registry": value, "Ordering": b"Identity"}}
    if descendant:
        font = {"DescendantFonts": [font]}
    assert decoder.FontDecoder.internal_cid_system_info(font) == (
        "Adobe" if value not in (1, None) else None,
        "Identity",
    )

# SPDX-License-Identifier: AGPL-3.0-only
"""Preserve a replacement symbol explicitly mapped by both PDF and font."""

import zlib

import pytest

from core_pdf.impl._impl.fonts.decoder import FontDecoder
from core_pdf.impl._impl.fonts.font_program_truetype import TrueTypeFontProgram
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf, stream_obj

# Original two-glyph TrueType font generated with fontTools FontBuilder: an
# empty .notdef and a diamond/question-mark outline mapped only from U+FFFD.
# The font has 1000 units/em, a 600-unit advance, and a fixed 1970 timestamp.
internal_REPLACEMENT_FONT = (
    "789c7551318b134114fe66662ff10e54c48b085eb1721e2e729c9018b558b0f0d84e3c62936b4e97644d02bb9b25d9c0"
    "29228758588a95829558a610c442b4b4b0b051fc0982567a229649fc667691d5d337bc37dff7bd376fdeec420058c00e"
    "14eccb574e57d72717bf03c2a17aad15f909f66383fc3ef9994e78e37afa68fc9efc1df94637f0db9b2f9f3e261ed3cf"
    "7629ccdd140f883fd14f74a3745b368920ee3194c27ecb8763f81bcd237f3b410947c9f791dbb11f053fbeee7e06e412"
    "f987a43f4c5f3cbff48a7c977c0d7a56fab187df36af1e747f42a92fbadbad95673bc55d36e56b6e0a1299f18cb267e7"
    "7526cb9b4e4513469138c253a6047f9b924d7e8539587b32ff375b870b346036994db219948db77bebb2c97edfada758"
    "641466b74c278b4b984eb33c5a857738f9a9ecf50a8750018ecfcb955abd5629576af573f593cbe5c3b5fab2589d7e14"
    "ab8c775dd7733da7ed545d576c89ade9933b8b55b7edb96dc7998e5ddd6c01b7f33b84ee9763893259865541b70ab864"
    "305f60cd533985a51c4b1c80976355d0ad022e69dc0892d06f055110a78da0330886c35e3f261a85fe000d044810c247"
    "8b28a2c7488ddac18071c8d5439f6aa68d4cede09f7ff64f635ec8f228ee799eb70efc02ebfe726e"
)


def internal_explicit_replacement_pdf(ordering: str = "Identity") -> bytes:
    to_unicode = (
        b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def "
        b"/CMapName /ExplicitReplacement def /CMapType 2 def "
        b"1 begincodespacerange <0000> <ffff> endcodespacerange "
        b"1 beginbfchar <0039> <fffd> endbfchar "
        b"endcmap CMapName currentdict /CMap defineresource pop end end"
    )
    return one_page_pdf(
        b"BT /F1 18 Tf 36 740 Td (valid ) Tj /F2 18 Tf <0039> Tj /F1 18 Tf ( text) Tj ET",
        resources=b"<< /Font << /F1 5 0 R /F2 6 0 R >> >>",
        extra_objects=(
            b"<< /Type /Font /Subtype /Type0 /BaseFont /ReplacementRegression "
            b"/Encoding /Identity-H /DescendantFonts [7 0 R] /ToUnicode 10 0 R >>",
            b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /ReplacementRegression "
            + (
                f"/CIDSystemInfo << /Registry (Adobe) /Ordering ({ordering}) /Supplement 0 >> "
            ).encode()
            + b"/FontDescriptor 8 0 R /CIDToGIDMap 11 0 R /DW 600 >>",
            b"<< /Type /FontDescriptor /FontName /ReplacementRegression /Flags 32 "
            b"/FontBBox [0 0 600 700] /ItalicAngle 0 /Ascent 800 /Descent -200 "
            b"/CapHeight 700 /StemV 80 /FontFile2 9 0 R >>",
            stream_obj(zlib.decompress(bytes.fromhex(internal_REPLACEMENT_FONT))),
            stream_obj(to_unicode),
            stream_obj(bytes(0x39 * 2) + b"\x00\x01"),
        ),
    )


@pytest.mark.parametrize("ordering", ["Identity", "Japan1"])
def test_explicit_replacement_symbol_survives_native_document_extraction(ordering: str) -> None:
    # Before changing decoder expectations, qpdf 12.3.2 --check validated these
    # exact PDFs, and Poppler 26.07.0 pdftotext -layout returned "valid \ufffd text".
    # Its pdftoppm raster also paints our embedded diamond/question-mark glyph.
    # CID 0x0039 is arbitrary: treating it as Unicode invents the digit 9.
    # The explicit mapping and embedded glyph also override collection metadata.
    with open_pdf(internal_explicit_replacement_pdf(ordering)) as document:
        program = document.pages[0].get_page_program()
        result = document.extract()

    glyph = program.glyphs[6]
    assert isinstance(glyph.font_decoder, FontDecoder)
    assert isinstance(glyph.font_decoder.font_program, TrueTypeFontProgram)
    assert glyph.font_decoder.font_program.unicode_for_gid(1) == "\ufffd"
    assert glyph.text == "\ufffd"
    assert glyph.unicode_source == "to_unicode"
    assert "".join(glyph.text for glyph in program.glyphs) == "valid \ufffd text"
    assert result.text.strip() == "valid \ufffd text"

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_09_fonts.decoder import (
    FontDecoder,
    parse_type1_font_program_encoding,
)
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.primitives import PdfString


def test_parse_type1_font_program_encoding_reads_custom_array() -> None:
    font_program = b"""
    /Encoding 256 array
    0 1 255 {1 index exch /.notdef put} for
    dup 12 /fi put
    dup 65 /A put
    readonly def
    currentfile eexec
    dup 99 /Ignored put
    """

    assert parse_type1_font_program_encoding(font_program) == {12: "fi", 65: "A"}


def test_font_decoder_uses_embedded_type1_encoding_without_pdf_encoding() -> None:
    font_program = b"""
    /Encoding 256 array
    0 1 255 {1 index exch /.notdef put} for
    dup 12 /fi put
    dup 65 /A put
    readonly def
    currentfile eexec
    """
    font = {
        "Subtype": "Type1",
        "BaseFont": "WYXCRD+CMBX12",
        "FirstChar": 12,
        "LastChar": 65,
        "FontDescriptor": {
            "FontFile": PdfStream(decoded_data=font_program),
        },
    }

    decoder = FontDecoder(font)

    assert decoder.decode(b"A\x0c") == "A\ufb01"


def test_encoding_differences_default_to_standard_encoding() -> None:
    decoder = FontDecoder(
        {
            "Subtype": "Type1",
            "Encoding": {"Differences": [65, "A"]},
        }
    )

    # 128 is undefined in StandardEncoding; PDFDocEncoding would incorrectly
    # turn it into a bullet.
    assert decoder.decode(b"\x80A") == "A"


def cid_type0_font(encoding: str, *, ordering: str = "Japan1") -> dict[str, object]:
    return {
        "Subtype": "Type0",
        "BaseFont": "HeiseiMin-W3",
        "Encoding": encoding,
        "DescendantFonts": [
            {
                "Subtype": "CIDFontType0",
                "BaseFont": "HeiseiMin-W3",
                "CIDSystemInfo": {
                    "Registry": PdfString(b"Adobe"),
                    "Ordering": PdfString(ordering.encode("ascii")),
                    "Supplement": 7,
                },
            }
        ],
    }


def test_font_decoder_recovers_japanese_without_to_unicode() -> None:
    decoder = FontDecoder(cid_type0_font("90ms-RKSJ-H"))
    encoded = bytes.fromhex("93fa967b8cea82a982c8834a83698abf8e9a")

    glyphs = decoder.decode_glyphs(encoded)

    assert decoder.decode(encoded) == "日本語かなカナ漢字"
    assert [glyph.cid for glyph in glyphs] == [3284, 3722, 1952, 852, 883, 935, 966, 1533, 2248]
    assert {glyph.unicode_source for glyph in glyphs} == {"predefined_cmap"}


def test_vertical_cid_advance_uses_w2_and_dw2() -> None:
    font = cid_type0_font("Identity-V")
    descendant = font["DescendantFonts"][0]
    assert isinstance(descendant, dict)
    descendant.update({"DW": 200, "W": [1, [200]], "DW2": [880, 900], "W2": [1, [500, 0, -700]]})
    decoder = FontDecoder(font)

    # CID 1 uses W2's vertical displacement; CID 2 uses DW2. Horizontal W/DW
    # values must not affect vertical text positioning.
    advance = decoder.text_advance_vector(
        b"\x00\x01\x00\x02", font_size=10, char_space=0, word_space=0, horizontal_scale=1
    )
    assert advance == (0.0, -0.14)


def test_vertical_cid_w2_range_overrides_dw2() -> None:
    font = cid_type0_font("Identity-V")
    descendant = font["DescendantFonts"][0]
    assert isinstance(descendant, dict)
    descendant.update({"DW2": [880, -1000], "W2": [3, 4, -600, 250, 770]})
    decoder = FontDecoder(font)

    advance = decoder.text_advance_vector(
        b"\x00\x03\x00\x05", font_size=10, char_space=0, word_space=0, horizontal_scale=1
    )
    assert advance == (0.0, 0.16)
    assert decoder.vertical_metrics[3] == (-600.0, 250.0, 770.0)


def test_vertical_w2_position_vector_is_scaled_to_text_space() -> None:
    font = cid_type0_font("Identity-V")
    descendant = font["DescendantFonts"][0]
    assert isinstance(descendant, dict)
    descendant.update({"DW2": [880, -1000], "W2": [7, [600, 200, -450]]})

    decoder = FontDecoder(font)

    assert decoder.vertical_glyph_position(7, font_size=10) == pytest.approx((0.02, -0.045))


def test_to_unicode_is_authoritative_over_glyph_name_repairs() -> None:
    decoder = FontDecoder(
        {
            "Subtype": "Type1",
            "Encoding": {"Differences": [65, "A"]},
            "ToUnicode": PdfStream(
                decoded_data=b"""\n                /CIDInit /ProcSet findresource begin
                12 dict begin begincmap
                /CMapType 2 def
                1 begincodespacerange <00> <ff> endcodespacerange
                1 beginbfchar <41> <0058> endbfchar
                endcmap end
            """
            ),
        }
    )

    glyph = decoder.decode_glyphs(b"A")[0]
    assert glyph.unicode == "X"
    assert glyph.unicode_source == "to_unicode"


def test_font_decoder_recovers_japanese_identity_cids_without_to_unicode() -> None:
    decoder = FontDecoder(cid_type0_font("Identity-H"))
    encoded = bytes.fromhex("0cd40e8a07a0")

    assert decoder.decode(encoded) == "日本語"
    assert [glyph.cid for glyph in decoder.decode_glyphs(encoded)] == [3284, 3722, 1952]


def test_font_decoder_preserves_unicode_cmap_compatibility_character() -> None:
    decoder = FontDecoder(cid_type0_font("UniJIS-UTF32-H"))
    encoded = ord("⽇").to_bytes(4, "big")

    glyph = decoder.decode_glyphs(encoded)[0]

    assert decoder.decode(encoded) == "⽇"
    assert glyph.cid == 3284
    assert glyph.unicode_source == "predefined_cmap"


def test_font_decoder_recovers_vertical_japanese_punctuation() -> None:
    decoder = FontDecoder(cid_type0_font("90ms-RKSJ-V"))

    glyphs = decoder.decode_glyphs(bytes.fromhex("8169816a"))

    assert decoder.is_vertical
    assert decoder.decode(bytes.fromhex("8169816a")) == "（）"
    assert [glyph.cid for glyph in glyphs] == [7899, 7900]
    assert {glyph.unicode_source for glyph in glyphs} == {"predefined_cmap"}


def test_font_decoder_recovers_predefined_chinese_and_korean_encodings() -> None:
    cases = (
        ("GB1", "GBK-EUC-H", bytes.fromhex("d6d0cec4babad7d6"), "中文汉字"),
        ("CNS1", "ETen-B5-H", bytes.fromhex("a4a4a4e5ba7ea672"), "中文漢字"),
        ("Korea1", "KSCms-UHC-H", bytes.fromhex("c7d1b1b9beeec7d1b1db"), "한국어한글"),
    )

    for ordering, encoding, encoded, expected in cases:
        decoder = FontDecoder(cid_type0_font(encoding, ordering=ordering))
        glyphs = decoder.decode_glyphs(encoded)

        assert decoder.decode(encoded) == expected
        assert {glyph.unicode_source for glyph in glyphs} == {"predefined_cmap"}


def test_font_decoder_recovers_non_japanese_identity_cids_without_to_unicode() -> None:
    cases = (
        ("GB1", (4559, 3795, 1905, 4659), "中文汉字"),
        ("CNS1", (661, 726, 4111, 959), "中文漢字"),
        ("Korea1", (3296, 1204, 2479, 3296, 1238), "한국어한글"),
        ("KR", (2835, 353, 1887, 2835, 392), "한국어한글"),
    )

    for ordering, cids, expected in cases:
        encoded = b"".join(cid.to_bytes(2, "big") for cid in cids)
        decoder = FontDecoder(cid_type0_font("Identity-H", ordering=ordering))
        glyphs = decoder.decode_glyphs(encoded)

        assert decoder.decode(encoded) == expected
        assert {glyph.unicode_source for glyph in glyphs} == {"cid_collection"}


def test_font_decoder_recovers_adobe_kr_unicode_encoding() -> None:
    decoder = FontDecoder(cid_type0_font("UniAKR-UTF32-H", ordering="KR"))
    encoded = ord("한").to_bytes(4, "big")

    glyph = decoder.decode_glyphs(encoded)[0]

    assert glyph.unicode == "한"
    assert glyph.cid == 2835
    assert glyph.unicode_source == "predefined_cmap"

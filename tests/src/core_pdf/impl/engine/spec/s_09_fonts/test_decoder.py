# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from core_pdf.impl.engine.spec.s_09_fonts import decoder as decoder_module
from core_pdf.impl.engine.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.engine.spec.s_09_fonts.decoder import (
    FontDecoder,
    parse_type1_font_program_encoding,
)
from core_pdf.impl.engine.spec.s_09_fonts.encoding import decode_pdf_text_string
from core_pdf.impl.engine.spec.s_09_fonts.font_program_truetype import (
    internal_invert_unicode_cmap,
)
from core_pdf.impl.engine.spec.s_09_fonts.glyphs import glyph_name_to_unicode
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.primitives import PdfString

TESTS_DIR = Path(__file__).parents[6]


def test_glyph_name_to_unicode_handles_computer_modern_delimiter_aliases() -> None:
    assert glyph_name_to_unicode("bracketleftbig") == "["
    assert glyph_name_to_unicode("bracketrightBigg") == "]"
    assert glyph_name_to_unicode("braceleftbigg") == "{"
    assert glyph_name_to_unicode("bracerightBig") == "}"
    assert glyph_name_to_unicode("slashbig") == "/"
    assert glyph_name_to_unicode("radicalBigg") == "√"
    assert glyph_name_to_unicode("integraldisplay") == "∫"
    assert glyph_name_to_unicode("oint") == "∮"


@pytest.mark.parametrize("name", ["Helvetica", "Helveticasmall", "SYMBOL_ENCODING"])
def test_glyph_name_lookup_does_not_leak_non_glyph_font_data(name: str) -> None:
    assert glyph_name_to_unicode(name) == name


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


def test_decode_pdfdoc_encoding_accent_and_quote_bytes() -> None:
    assert decode_pdf_text_string(bytes(range(0x18, 0x20))) == "˘ˇˆ˙˝˛˚˜"
    assert decode_pdf_text_string(bytes(range(0x8D, 0x91))) == "“”‘’"


def test_decode_pdf_text_string_accepts_utf8_bom() -> None:
    assert decode_pdf_text_string(b"\xef\xbb\xbfPrice \xe2\x82\xac") == "Price €"


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

    # The built-in encoding puts the fi ligature at 014. Ligatures are expanded
    # wherever else they are decoded, so a glyph reached through a built-in
    # encoding reads the same as one reached through a base encoding table.
    assert decoder.decode(b"A\x0c") == "Afi"


def test_type3_font_defaults_to_standard_encoding() -> None:
    decoder = FontDecoder({"Subtype": "Type3"})

    assert decoder.decode(b"AZ ") == "AZ "


def test_type3_font_without_descriptor_uses_font_bbox_metrics() -> None:
    decoder = FontDecoder({"Subtype": "Type3", "FontBBox": [0, -200, 1000, 800]})

    assert decoder.descent == -200
    assert decoder.ascent == 800


def test_type3_font_bbox_with_positive_ymin_keeps_its_sign() -> None:
    decoder = FontDecoder({"Subtype": "Type3", "FontBBox": [0, 50, 1000, 700]})

    assert decoder.descent == 50
    assert decoder.ascent == 700


def test_text_advance_applies_character_spacing_to_each_glyph() -> None:
    decoder = FontDecoder({"Subtype": "Type1", "BaseFont": "Helvetica"})

    single_without_spacing = decoder.text_advance_vector(
        b"A", font_size=10, char_space=0, word_space=0, horizontal_scale=1
    )
    single_with_spacing = decoder.text_advance_vector(
        b"A", font_size=10, char_space=3, word_space=0, horizontal_scale=1
    )
    pair_without_spacing = decoder.text_advance_vector(
        b"AB", font_size=10, char_space=0, word_space=0, horizontal_scale=1
    )
    pair_with_spacing = decoder.text_advance_vector(
        b"AB", font_size=10, char_space=3, word_space=0, horizontal_scale=1
    )

    assert single_with_spacing[0] - single_without_spacing[0] == pytest.approx(0.03)
    assert pair_with_spacing[0] - pair_without_spacing[0] == pytest.approx(0.06)


def test_font_decoder_prefers_explicit_pdf_encoding_over_embedded_type1_encoding() -> None:
    font_program = b"""
    /Encoding 256 array
    0 1 255 {1 index exch /.notdef put} for
    dup 49 /m put
    readonly def
    currentfile eexec
    """
    font = {
        "Subtype": "Type1",
        "BaseFont": "Helvetica",
        "Encoding": "MacRomanEncoding",
        "FontDescriptor": {
            "FontFile": PdfStream(decoded_data=font_program),
        },
    }

    decoder = FontDecoder(font)

    assert decoder.decode(b"1") == "1"


def test_simple_font_decoder_reuses_glyphs_across_distinct_text_operands() -> None:
    decoder = FontDecoder({"Subtype": "Type1"})

    first = decoder.decode_glyphs(b"A" * 17)
    second = decoder.decode_glyphs(b"BA" * 9)

    assert first[0].unicode == "A"
    assert all(glyph is first[0] for glyph in first)
    assert all(glyph is first[0] for glyph in second if glyph.char_code == ord("A"))


def test_font_decoder_caches_cff_glyph_bboxes_including_missing_glyphs() -> None:
    class FakeCFFFont:
        def __init__(self) -> None:
            self.bbox_calls: list[int] = []

        def glyph_id_for_name(self, name: str) -> int:
            return {"A": 7, "B": 8}[name]

        def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
            self.bbox_calls.append(glyph_id)
            return (0.0, -2.0, 5.0, 8.0) if glyph_id == 7 else None

    decoder = FontDecoder({"Subtype": "Type1"})
    decoder.decode(b"A")
    cff_font = FakeCFFFont()
    decoder.cff_font = cast(Any, cff_font)

    assert decoder.glyph_bbox(65) == (0.0, -2.0, 5.0, 8.0)
    assert decoder.glyph_bbox(65) == (0.0, -2.0, 5.0, 8.0)
    assert decoder.glyph_bbox(66) is None
    assert decoder.glyph_bbox(66) is None
    assert cff_font.bbox_calls == [7, 8]


def test_font_decoder_does_not_emit_unknown_difference_names() -> None:
    font = {
        "Subtype": "Type1",
        "Encoding": {"Differences": [65, "/DefinitelyUnknownGlyphName"]},
    }

    decoder = FontDecoder(font)

    assert decoder.decode(b"A") == ""


def test_font_decoder_does_not_emit_unknown_dotted_difference_base_names() -> None:
    font = {
        "Subtype": "Type1",
        "Encoding": {"Differences": [65, "/DefinitelyUnknownGlyphName.alt"]},
    }

    decoder = FontDecoder(font)

    assert decoder.decode(b"A") == ""


def test_font_decoder_does_not_emit_unknown_underscore_difference_parts() -> None:
    font = {
        "Subtype": "Type1",
        "Encoding": {"Differences": [65, "/DefinitelyUnknown_OtherUnknown"]},
    }

    decoder = FontDecoder(font)

    assert decoder.decode(b"A") == ""


def test_type3_numeric_charproc_name_retains_base_encoding() -> None:
    decoder = FontDecoder(
        {
            "Subtype": "Type3",
            "Encoding": {"Differences": [65, "/65"]},
        }
    )

    glyph = decoder.decode_glyphs(b"A")[0]

    assert glyph.unicode == "A"
    assert glyph.unicode_source == "encoding"


def test_undefined_simple_font_code_retains_a_replacement_glyph() -> None:
    decoder = FontDecoder(
        {
            "Subtype": "Type3",
            "Encoding": {"Differences": [0, "/0"]},
        }
    )

    glyph = decoder.decode_glyphs(b"\x00")[0]

    assert glyph.unicode == "\ufffd"
    assert glyph.unicode_source == "undefined"


def test_font_decoder_keeps_single_character_difference_names() -> None:
    font = {
        "Subtype": "Type1",
        "Encoding": {"Differences": [12, "/A"]},
    }

    decoder = FontDecoder(font)

    assert decoder.decode(b"\x0c") == "A"


def test_font_decoder_keeps_known_underscore_difference_parts() -> None:
    font = {
        "Subtype": "Type1",
        "Encoding": {"Differences": [12, "/A_B"]},
    }

    decoder = FontDecoder(font)

    assert decoder.decode(b"\x0c") == "AB"


def test_font_decoder_keeps_known_dotted_difference_base_names() -> None:
    font = {
        "Subtype": "Type1",
        "Encoding": {"Differences": [12, "/A.alt"]},
    }

    decoder = FontDecoder(font)

    assert decoder.decode(b"\x0c") == "A"


def test_font_decoder_maps_tex_text_symbol_difference_names() -> None:
    font = {
        "Subtype": "Type1",
        "Encoding": {"Differences": [12, "/integraltext", "/summationtext"]},
    }

    decoder = FontDecoder(font)

    assert decoder.decode(b"\x0c\x0d") == "\u222b\u2211"


def test_glyph_u_codepoint_rejects_surrogates() -> None:
    assert glyph_name_to_unicode("uD800") == "uD800"


def test_font_decoder_identity_fallback_does_not_emit_surrogates() -> None:
    font = {
        "Subtype": "Type0",
        "Encoding": "Identity-H",
        "DescendantFonts": [{"Subtype": "CIDFontType2"}],
    }

    decoder = FontDecoder(font)

    assert decoder.decode(b"\xd8\x00") == "\ufffd"


def test_font_decoder_applies_learned_mapping_to_unknown_identity_code() -> None:
    decoder = FontDecoder(cid_type0_font("Identity-H", ordering="Unknown"))
    encoded = b"\x00A"

    assert decoder.decode_glyphs(encoded)[0].unicode_source == "identity"
    assert decoder.install_learned_unicode({encoded: "Z"}) == 1

    glyph = decoder.decode_glyphs(encoded)[0]
    assert glyph.unicode == "Z"
    assert glyph.unicode_source == "learned_ocr"


def test_cid_decoder_rejects_private_use_true_type_cmap_values() -> None:
    class FakeTrueTypeFont:
        def glyph_id_for_code(self, code: int) -> int:
            return code

        def has_glyph_id(self, gid: int) -> bool:
            return gid == 65

        def unicode_for_gid(self, gid: int) -> str:
            assert gid == 65
            return "\ue000"

    decoder = FontDecoder(cid_type0_font("Identity-H"))
    decoder.tt_font = cast(Any, FakeTrueTypeFont())

    glyph = decoder.decode_glyphs(b"\x00A")[0]

    assert glyph.unicode != "\ue000"
    assert glyph.unicode_source == "cid_collection"


def test_truetype_cmap_inversion_rejects_surrogates() -> None:
    assert internal_invert_unicode_cmap({0xD800: 1, 0x41: 2}) == {2: "A"}


def test_to_unicode_bfrange_does_not_emit_surrogates() -> None:
    cmap = ToUnicodeCMap(
        b"""
        /CIDInit /ProcSet findresource begin
        12 dict begin
        begincmap
        1 begincodespacerange
        <01> <02>
        endcodespacerange
        1 beginbfrange
        <01> <02> <D7FF>
        endbfrange
        endcmap
        CMapName currentdict /CMap defineresource pop
        end end
        """
    )

    assert cmap.decode(b"\x01\x02") == "\ud7ff\ufffd"


def test_to_unicode_destination_strings_stay_utf16be() -> None:
    cmap = ToUnicodeCMap(
        b"""
        /CIDInit /ProcSet findresource begin
        12 dict begin
        begincmap
        2 begincodespacerange
        <01> <02>
        endcodespacerange
        2 beginbfchar
        <01> <FEFF0041>
        <02> <FFFE4100>
        endbfchar
        endcmap
        CMapName currentdict /CMap defineresource pop
        end end
        """
    )

    assert cmap.decode(b"\x01") == "A"
    assert cmap.decode(b"\x02") != "A"


def test_to_unicode_fallback_does_not_emit_surrogates() -> None:
    cmap = ToUnicodeCMap(
        b"""
        /CIDInit /ProcSet findresource begin
        12 dict begin
        begincmap
        2 begincodespacerange
        <0000> <ffff>
        <000000> <ffffff>
        endcodespacerange
        endcmap
        CMapName currentdict /CMap defineresource pop
        end end
        """
    )

    assert cmap.decode(b"\xd8\x00") == "\ufffd"


def test_to_unicode_fixed_two_byte_fast_path_preserves_unmapped_identity() -> None:
    cmap = ToUnicodeCMap(
        b"""
        /CIDInit /ProcSet findresource begin
        12 dict begin
        begincmap
        1 begincodespacerange
        <0000> <ffff>
        endcodespacerange
        1 beginbfchar
        <0041> <0058>
        endbfchar
        endcmap
        CMapName currentdict /CMap defineresource pop
        end end
        """
    )

    assert cmap.decode(b"\x00\x41\x00\x42") == "XB"


def test_to_unicode_one_byte_fast_path_strips_nul_like_other_paths() -> None:
    cmap = ToUnicodeCMap(
        b"""
        /CIDInit /ProcSet findresource begin
        12 dict begin
        begincmap
        1 begincodespacerange
        <00> <ff>
        endcodespacerange
        1 beginbfchar
        <41> <0000>
        endbfchar
        endcmap
        CMapName currentdict /CMap defineresource pop
        end end
        """
    )

    assert cmap.decode(b"AB") == "B"


def test_to_unicode_rejects_invalid_codespace_ranges() -> None:
    for range_line in (b"<ff> <00>", b"<> <ff>", b"<00> <ffff>"):
        with pytest.raises(ValueError, match="^invalid ToUnicode CMap codespacerange$"):
            ToUnicodeCMap(
                b"""
                /CIDInit /ProcSet findresource begin
                12 dict begin
                begincmap
                1 begincodespacerange
                """
                + range_line
                + b"""
                endcodespacerange
                endcmap
                CMapName currentdict /CMap defineresource pop
                end end
                """
            )


def test_to_unicode_rejects_only_overlapping_codespace_ranges() -> None:
    with pytest.raises(ValueError, match="^invalid ToUnicode CMap codespacerange$"):
        ToUnicodeCMap(
            b"""
            /CIDInit /ProcSet findresource begin
            12 dict begin
            begincmap
            2 begincodespacerange
            <00> <7f>
            <40> <ff>
            endcodespacerange
            endcmap
            CMapName currentdict /CMap defineresource pop
            end end
            """
        )


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


def cid_to_unicode_stream() -> PdfStream:
    return PdfStream(
        decoded_data=b"""
        /CIDInit /ProcSet findresource begin
        12 dict begin begincmap
        /CMapType 2 def
        1 begincodespacerange <0000> <ffff> endcodespacerange
        1 beginbfchar <0041> <0058> endbfchar
        endcmap end
        """
    )


def test_cid_collection_map_stays_lazy_when_to_unicode_resolves_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, bool]] = []

    def resolve(registry: str, ordering: str, *, vertical: bool = False) -> dict[int, str]:
        calls.append((registry, ordering, vertical))
        return {66: "fallback"}

    monkeypatch.setattr(decoder_module, "resolve_cid_unicode_map", resolve)
    font = cid_type0_font("Identity-H")
    font["ToUnicode"] = cid_to_unicode_stream()
    decoder = FontDecoder(font)

    assert decoder.decode(b"\x00A") == "X"
    assert calls == []


def test_cff_unicode_repair_is_batched_on_first_suspicious_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRepairIndex:
        def __init__(self) -> None:
            self.calls: list[tuple[bytes, ...]] = []

        def repairs_for_codes(self, codes: object) -> dict[bytes, str]:
            requested = tuple(cast(Any, codes))
            self.calls.append(requested)
            return {b"\x00A": "X"}

    repair_index = FakeRepairIndex()
    monkeypatch.setattr(
        decoder_module,
        "build_cff_unicode_repair_index",
        lambda *internal_args: repair_index,
    )
    font = cid_type0_font("Identity-H")
    font["ToUnicode"] = PdfStream(
        decoded_data=b"""
        /CIDInit /ProcSet findresource begin 12 dict begin begincmap
        1 begincodespacerange <0000> <ffff> endcodespacerange
        1 beginbfchar <0041> <fffd> endbfchar
        endcmap end
        """
    )

    decoder = FontDecoder(font)

    assert repair_index.calls == []
    assert decoder.decode(b"\x00A") == "X"
    assert repair_index.calls == [(b"\x00A",)]


def test_cid_collection_map_resolves_once_on_first_unmapped_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, bool]] = []

    def resolve(registry: str, ordering: str, *, vertical: bool = False) -> dict[int, str]:
        calls.append((registry, ordering, vertical))
        return {66: "fallback"}

    monkeypatch.setattr(decoder_module, "resolve_cid_unicode_map", resolve)
    font = cid_type0_font("Identity-H")
    font["ToUnicode"] = cid_to_unicode_stream()
    decoder = FontDecoder(font)

    assert decoder.decode(b"\x00B") == "fallback"
    assert decoder.decode(b"\x00B") == "fallback"
    assert calls == [("Adobe", "Japan1", False)]


def test_font_decoder_recovers_japanese_without_to_unicode() -> None:
    decoder = FontDecoder(cid_type0_font("90ms-RKSJ-H"))
    encoded = bytes.fromhex("93fa967b8cea82a982c8834a83698abf8e9a")

    glyphs = decoder.decode_glyphs(encoded)

    assert decoder.decode(encoded) == "日本語かなカナ漢字"
    assert [glyph.cid for glyph in glyphs] == [3284, 3722, 1952, 852, 883, 935, 966, 1533, 2248]
    assert {glyph.unicode_source for glyph in glyphs} == {"predefined_cmap"}


def test_vertical_cid_advance_uses_w2_and_dw2() -> None:
    font = cid_type0_font("Identity-V")
    descendant = cast(dict[str, object], cast(list[object], font["DescendantFonts"])[0])
    assert isinstance(descendant, dict)
    descendant.update({"DW": 200, "W": [1, [200]], "DW2": [880, 900], "W2": [1, [500, 0, -700]]})
    decoder = FontDecoder(font)

    # CID 1 uses W2's vertical displacement; CID 2 uses DW2. Horizontal W/DW
    # values must not affect vertical text positioning.
    advance = decoder.text_advance_vector(
        b"\x00\x01\x00\x02", font_size=10, char_space=0, word_space=0, horizontal_scale=1
    )
    assert advance == (0.0, -14.0)


def test_vertical_cid_w2_range_overrides_dw2() -> None:
    font = cid_type0_font("Identity-V")
    descendant = cast(dict[str, object], cast(list[object], font["DescendantFonts"])[0])
    assert isinstance(descendant, dict)
    descendant.update({"DW2": [880, -1000], "W2": [3, 4, -600, 250, 770]})
    decoder = FontDecoder(font)

    advance = decoder.text_advance_vector(
        b"\x00\x03\x00\x05", font_size=10, char_space=0, word_space=0, horizontal_scale=1
    )
    assert advance == (0.0, 16.0)
    assert decoder.vertical_metrics[3] == (-600.0, 250.0, 770.0)


def test_vertical_w2_position_vector_is_scaled_to_text_space() -> None:
    font = cid_type0_font("Identity-V")
    descendant = cast(dict[str, object], cast(list[object], font["DescendantFonts"])[0])
    assert isinstance(descendant, dict)
    descendant.update({"DW2": [880, -1000], "W2": [7, [600, 200, -450]]})

    decoder = FontDecoder(font)

    assert decoder.vertical_glyph_position(7, font_size=10) == pytest.approx((-2.0, 4.5))


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


def test_font_decoder_uses_explicit_glyph_name_when_to_unicode_omits_code() -> None:
    decoder = FontDecoder(
        {
            "Subtype": "Type1",
            "BaseFont": "TeXGyrePagella-Regular",
            "Encoding": {"Differences": [65, "ff"]},
            "ToUnicode": PdfStream(
                decoded_data=b"""\n                /CIDInit /ProcSet findresource begin
                12 dict begin begincmap
                /CMapType 2 def
                1 begincodespacerange <00> <ff> endcodespacerange
                0 beginbfchar endbfchar
                endcmap end
            """
            ),
        }
    )

    glyph = decoder.decode_glyphs(b"A")[0]
    assert glyph.unicode == "ff"
    assert glyph.unicode_source == "glyph_name"


def test_font_decoder_falls_through_replacement_to_predefined_mapping() -> None:
    decoder = FontDecoder(
        {
            "Subtype": "Type1",
            "BaseFont": "Helvetica",
            "Encoding": "WinAnsiEncoding",
            "ToUnicode": PdfStream(
                decoded_data=b"""\n                /CIDInit /ProcSet findresource begin
                12 dict begin begincmap
                /CMapType 2 def
                1 begincodespacerange <00> <ff> endcodespacerange
                1 beginbfchar <41> <fffd> endbfchar
                endcmap end
            """
            ),
        }
    )

    glyph = decoder.decode_glyphs(b"A")[0]
    assert glyph.unicode == "A"
    assert glyph.unicode_source == "encoding"
    assert "�" in glyph.alternates


def test_font_decoder_uses_glyph_name_when_to_unicode_is_nul() -> None:
    decoder = FontDecoder(
        {
            "Subtype": "Type1",
            "BaseFont": "TeXGyrePagella-Regular",
            "Encoding": {"Differences": [65, "ff"]},
            "ToUnicode": PdfStream(
                decoded_data=b"""\n                /CIDInit /ProcSet findresource begin
                12 dict begin begincmap
                /CMapType 2 def
                1 begincodespacerange <00> <ff> endcodespacerange
                1 beginbfchar <41> <0000> endbfchar
                endcmap end
            """
            ),
        }
    )

    glyph = decoder.decode_glyphs(b"A")[0]
    assert glyph.unicode == "ff"
    assert glyph.unicode_source == "glyph_name"


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

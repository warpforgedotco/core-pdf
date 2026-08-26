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
from core_pdf.impl.engine.spec.s_09_fonts.font_program import CFFFont
from core_pdf.impl.engine.spec.s_09_fonts.font_program_truetype import (
    TrueTypeFontProgram,
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


def test_cid_text_advance_uses_decoded_one_byte_cid_width() -> None:
    font = cid_type0_font("Identity-H")
    font["Encoding"] = PdfStream(
        decoded_data=b"""
        1 begincodespacerange <00> <ff> endcodespacerange
        1 begincidchar <41> 100 endcidchar
        """
    )
    descendant = cast(dict[str, object], cast(list[object], font["DescendantFonts"])[0])
    descendant["W"] = [65, [100], 100, [900]]
    decoder = FontDecoder(font)

    implicit = decoder.text_advance_vector(
        b"A", font_size=10, char_space=0, word_space=0, horizontal_scale=100
    )
    glyphs = decoder.decode_glyphs(b"A")
    explicit = decoder.text_advance_vector(
        b"A",
        font_size=10,
        char_space=0,
        word_space=0,
        horizontal_scale=100,
        glyphs=glyphs,
    )

    assert glyphs[0].width_code == 100
    assert implicit == explicit == pytest.approx((9.0, 0.0))


def test_cid_text_advance_respects_mapping_free_codespace_decode() -> None:
    font = cid_type0_font("Identity-H")
    font["Encoding"] = PdfStream(
        decoded_data=b"1 begincodespacerange <0000> <ffff> endcodespacerange"
    )
    descendant = cast(dict[str, object], cast(list[object], font["DescendantFonts"])[0])
    descendant["W"] = [0, [300], 65, [900]]
    decoder = FontDecoder(font)

    glyphs = decoder.decode_glyphs(b"\x00A")
    advance = decoder.text_advance_vector(
        b"\x00A", font_size=10, char_space=0, word_space=0, horizontal_scale=100
    )

    assert glyphs[0].width_code == 0
    assert advance == pytest.approx((3.0, 0.0))


@pytest.mark.parametrize(
    ("source", "mapped_cid", "expected_advance"),
    [
        (b" ", 100, 7.0),
        (b"\x01", 32, 5.0),
        (b"\x00 ", 32, 5.0),
    ],
)
def test_word_spacing_depends_on_encoded_single_byte_code_32(
    source: bytes, mapped_cid: int, expected_advance: float
) -> None:
    code_hex = source.hex().encode("ascii")
    font = cid_type0_font("Identity-H")
    font["Encoding"] = PdfStream(
        decoded_data=(
            b"1 begincodespacerange <"
            + (b"0000> <ffff" if len(source) == 2 else b"00> <ff")
            + b"> endcodespacerange 1 begincidchar <"
            + code_hex
            + b"> "
            + str(mapped_cid).encode("ascii")
            + b" endcidchar"
        )
    )
    descendant = cast(dict[str, object], cast(list[object], font["DescendantFonts"])[0])
    descendant["W"] = [mapped_cid, [500]]
    decoder = FontDecoder(font)

    advance = decoder.text_advance_vector(
        source, font_size=10, char_space=0, word_space=2, horizontal_scale=100
    )

    assert advance == pytest.approx((expected_advance, 0.0))


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


def test_font_decoder_slots_follow_its_annotated_state() -> None:
    decoder = FontDecoder({"Subtype": "Type1"})

    assert FontDecoder.__slots__ == tuple(FontDecoder.__annotations__)
    assert not hasattr(decoder, "__dict__")


def test_cid_font_decoder_reuses_glyphs_across_distinct_long_operands() -> None:
    decoder = FontDecoder(cid_type0_font("Identity-H"))

    first = decoder.decode_glyphs(b"\x00A" * 9)
    second = decoder.decode_glyphs(b"\x00B\x00A" * 5)

    assert all(glyph is first[0] for glyph in first)
    assert all(glyph is first[0] for glyph in second if glyph.cid == ord("A"))


def test_cid_glyph_cache_stops_accepting_new_codes_at_its_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(decoder_module, "internal_CID_GLYPH_CACHE_LIMIT", 1)
    decoder = FontDecoder(cid_type0_font("Identity-H"))

    cached = decoder.internal_decode_cid_glyphs(b"\x00A")[0]
    uncached_first = decoder.internal_decode_cid_glyphs(b"\x00B")[0]
    uncached_second = decoder.internal_decode_cid_glyphs(b"\x00B")[0]

    assert decoder.internal_decode_cid_glyphs(b"\x00A")[0] is cached
    assert uncached_second is not uncached_first


def test_font_decoder_caches_cff_glyph_bboxes_including_missing_glyphs() -> None:
    class FakeCFFFont(CFFFont):
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
    decoder.font_program = cast(Any, cff_font)

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

    assert decoder.decode(encoded) == "A"
    assert decoder.decode_cache
    assert decoder.glyphs_cache
    assert decoder.cid_glyph_cache
    assert decoder.install_learned_unicode({encoded: "Z"}) == 1
    assert not decoder.decode_cache
    assert not decoder.glyphs_cache
    assert not decoder.cid_glyph_cache

    glyph = decoder.decode_glyphs(encoded)[0]
    assert glyph.unicode == "Z"
    assert glyph.unicode_source == "learned_ocr"


def test_cid_decoder_rejects_private_use_true_type_cmap_values() -> None:
    class FakeTrueTypeFont(TrueTypeFontProgram):
        def glyph_id_for_code(self, code: int) -> int:
            return code

        def has_glyph_id(self, gid: int) -> bool:
            return gid == 65

        def unicode_for_gid(self, gid: int) -> str:
            assert gid == 65
            return "\ue000"

    decoder = FontDecoder(cid_type0_font("Identity-H"))
    decoder.font_program = FakeTrueTypeFont.__new__(FakeTrueTypeFont)

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


def test_cff_unicode_repair_invalidates_cid_glyphs_only_when_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRepairIndex:
        repair = "X"

        def repairs_for_codes(self, codes: object) -> dict[bytes, str]:
            assert tuple(cast(Any, codes)) == (b"\x00A",)
            return {b"\x00A": self.repair}

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

    first = decoder.internal_decode_cid_glyphs(b"\x00A")[0]
    unchanged = decoder.internal_decode_cid_glyphs(b"\x00A")[0]
    decoder.decode(b"\x00A")
    assert decoder.decode_cache
    assert decoder.glyphs_cache
    repair_index.repair = "Y"
    changed = decoder.internal_decode_cid_glyphs(b"\x00A")[0]

    assert first.unicode == "X"
    assert unchanged is first
    assert changed.unicode == "Y"
    assert changed is not first
    assert not decoder.decode_cache
    assert not decoder.glyphs_cache


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
    assert advance == (0.0, 14.0)


def test_vertical_cid_omitted_dw2_matches_spec_default() -> None:
    omitted = FontDecoder(cid_type0_font("Identity-V"))
    explicit_font = cid_type0_font("Identity-V")
    descendant = cast(dict[str, object], cast(list[object], explicit_font["DescendantFonts"])[0])
    descendant["DW2"] = [880, -1000]
    explicit = FontDecoder(explicit_font)

    omitted_advance = omitted.text_advance_vector(
        b"\x00\x01", font_size=10, char_space=0, word_space=0, horizontal_scale=100
    )
    explicit_advance = explicit.text_advance_vector(
        b"\x00\x01", font_size=10, char_space=0, word_space=0, horizontal_scale=100
    )

    assert omitted.vertical_glyph_metric(1) == explicit.vertical_glyph_metric(1)
    assert omitted_advance == explicit_advance == pytest.approx((0.0, -10.0))


def test_vertical_cid_w2_range_overrides_dw2() -> None:
    font = cid_type0_font("Identity-V")
    descendant = cast(dict[str, object], cast(list[object], font["DescendantFonts"])[0])
    assert isinstance(descendant, dict)
    descendant.update({"DW2": [880, -1000], "W2": [3, 4, -600, 250, 770]})
    decoder = FontDecoder(font)

    advance = decoder.text_advance_vector(
        b"\x00\x03\x00\x05", font_size=10, char_space=0, word_space=0, horizontal_scale=1
    )
    assert advance == (0.0, -16.0)
    assert decoder.vertical_metrics[3] == (-600.0, 250.0, 770.0)


def test_vertical_w2_position_vector_is_scaled_to_text_space() -> None:
    font = cid_type0_font("Identity-V")
    descendant = cast(dict[str, object], cast(list[object], font["DescendantFonts"])[0])
    assert isinstance(descendant, dict)
    descendant.update({"DW2": [880, -1000], "W2": [7, [600, 200, -450]]})

    decoder = FontDecoder(font)

    assert decoder.vertical_glyph_metric(7) == (600.0, 200.0, -450.0)
    assert decoder.vertical_glyph_position(7, font_size=10) == pytest.approx((-2.0, 4.5))


def test_vertical_glyph_metric_uses_dw2_and_horizontal_width_fallback() -> None:
    font = cid_type0_font("Identity-V")
    descendant = cast(dict[str, object], cast(list[object], font["DescendantFonts"])[0])
    assert isinstance(descendant, dict)
    descendant.update({"DW": 400, "DW2": [750, -900]})
    decoder = FontDecoder(font)

    assert decoder.vertical_glyph_metric(7) == (-900.0, 200.0, 750.0)


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


TO_UNICODE_WA = b"""
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
1 begincodespacerange
<00> <ff>
endcodespacerange
1 beginbfrange
<20> <7e> <0020>
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end end
"""


@pytest.mark.parametrize(
    ("encoding", "code", "expected"),
    [
        ("WinAnsiEncoding", ord("W"), "W"),
        ("WinAnsiEncoding", ord("r"), "r"),
        ("WinAnsiEncoding", 0xA9, "copyright"),
        ("MacRomanEncoding", ord("W"), "W"),
        ("MacRomanEncoding", 0xA9, "copyright"),
    ],
)
def test_named_base_encoding_names_glyphs_even_with_a_tounicode_cmap(
    encoding: str, code: int, expected: str
) -> None:
    """A ToUnicode CMap must not cost the font its glyph names.

    ``byte_decode_table`` is only built when a simple font has no ToUnicode
    CMap, so glyph-name lookup cannot read the base encoding off it. Before
    this was handled, every code in a WinAnsi/MacRoman simple font resolved to
    ``.notdef``, and CFF/Type1 outlines rendered as the font's own box glyph.
    """
    font = {
        "Subtype": "Type1",
        "BaseFont": "FSElliotPro",
        "Encoding": encoding,
        "ToUnicode": PdfStream(decoded_data=TO_UNICODE_WA),
    }

    decoder = FontDecoder(cast(Any, font))

    assert decoder.byte_decode_table is None
    assert decoder.internal_simple_glyph_name(code) == expected


def test_differences_still_win_over_the_named_base_encoding() -> None:
    font = {
        "Subtype": "Type1",
        "BaseFont": "FSElliotPro",
        "Encoding": {"BaseEncoding": "WinAnsiEncoding", "Differences": [ord("W"), "alpha"]},
        "ToUnicode": PdfStream(decoded_data=TO_UNICODE_WA),
    }

    decoder = FontDecoder(cast(Any, font))

    assert decoder.internal_simple_glyph_name(ord("W")) == "alpha"
    assert decoder.internal_simple_glyph_name(ord("r")) == "r"


def test_simple_truetype_outline_selection_is_independent_of_tounicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = object.__new__(TrueTypeFontProgram)
    program.cid_to_gid = None
    program.cmap = {0x20AC: 7}
    monkeypatch.setattr(
        decoder_module,
        "internal_font_program_for_pdf_font",
        lambda ignored_font: program,
    )
    font = {
        "Subtype": "TrueType",
        "BaseFont": "EmbeddedSans",
        "Encoding": "WinAnsiEncoding",
        "ToUnicode": PdfStream(
            decoded_data=b"""
            1 begincodespacerange <00> <ff> endcodespacerange
            1 beginbfchar <80> <20ac> endbfchar
            """
        ),
    }

    decoder = FontDecoder(cast(Any, font))
    glyph = decoder.decode_glyphs(b"\x80")[0]

    assert decoder.byte_decode_table is None
    assert decoder.internal_simple_glyph_name(0x80) == "Euro"
    assert glyph.unicode == "€"
    assert glyph.gid == 7

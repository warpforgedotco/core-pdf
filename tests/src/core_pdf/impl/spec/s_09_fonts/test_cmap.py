import numpy

from core_pdf.impl.spec.s_09_fonts.cid_unicode import resolve_cid_unicode_map
from core_pdf.impl.spec.s_09_fonts.cmap_decoder import CMapDecoder
from core_pdf.impl.spec.s_09_fonts.cmap_ranges import iter_codespace_range
from core_pdf.impl.spec.s_09_fonts.cmap_resources import resolve_cmap_decoder
from core_pdf.impl.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap


def test_cidrange_rejects_unbounded_expansion() -> None:
    cmap = ToUnicodeCMap(b"1 begincidrange <00000000> <ffffffff> 32 endcidrange")

    assert cmap.mappings == {}


def test_iter_codespace_range_carries_across_ff() -> None:
    assert list(iter_codespace_range(b"\x00\xfe", b"\x01\xff")) == [
        b"\x00\xfe",
        b"\x00\xff",
        b"\x01\xfe",
        b"\x01\xff",
    ]


def test_cmap_uses_shortest_matching_code_space_first() -> None:
    cmap = CMapDecoder(
        b"""\n            2 begincodespacerange
            <00> <ff>
            <0000> <ffff>
            endcodespacerange
            1 begincidchar <01> 11 endcidchar
            1 begincidchar <0102> 22 endcidchar
        """
    )

    assert cmap.decode_entries(b"\x01\x02") == [(b"\x01", 11), (b"\x02", 0)]


def test_identity_vertical_cmaps_preserve_wmode() -> None:
    horizontal = resolve_cmap_decoder("Identity-H")
    vertical = resolve_cmap_decoder("Identity-V")
    one_byte_vertical = resolve_cmap_decoder("OneByteIdentityV")
    assert horizontal is not None
    assert horizontal.wmode == 0
    assert vertical is not None
    assert vertical.wmode == 1
    assert one_byte_vertical is not None
    assert one_byte_vertical.wmode == 1


def test_identity_cmap_numeric_decode_matches_scalar_decode() -> None:
    for cmap, data in (
        (CMapDecoder.identity(byte_width=1), b"\x00A\xff"),
        (CMapDecoder.identity(byte_width=2), b"\x00A\x12\xff\x80"),
    ):
        values = cmap.decode_cids_array(data)
        assert values is not None
        expected = numpy.asarray(
            [cid for internal_code, cid in cmap.decode_entries(data)], dtype=numpy.int64
        )
        numpy.testing.assert_array_equal(values, expected)


def test_numeric_decode_falls_back_for_explicit_cmap_mappings() -> None:
    cmap = CMapDecoder(
        b"1 begincodespacerange <00> <ff> endcodespacerange 1 begincidchar <41> 99 endcidchar"
    )
    assert cmap.decode_cids_array(b"A") is None


def test_cid_mapping_sections_apply_in_source_order() -> None:
    range_then_char = CMapDecoder(
        b"1 begincidrange <41> <41> 100 endcidrange 1 begincidchar <41> 200 endcidchar"
    )
    char_then_range = CMapDecoder(
        b"1 begincidchar <41> 200 endcidchar 1 begincidrange <41> <41> 100 endcidrange"
    )

    assert range_then_char.decode_entries(b"A") == [(b"A", 200)]
    assert char_then_range.decode_entries(b"A") == [(b"A", 100)]


def test_notdef_mapping_sections_apply_in_source_order() -> None:
    range_then_char = CMapDecoder(
        b"1 beginnotdefrange <41> <41> 100 endnotdefrange 1 beginnotdefchar <41> 200 endnotdefchar"
    )
    char_then_range = CMapDecoder(
        b"1 beginnotdefchar <41> 200 endnotdefchar 1 beginnotdefrange <41> <41> 100 endnotdefrange"
    )

    assert range_then_char.decode_entries(b"A") == [(b"A", 200)]
    assert char_then_range.decode_entries(b"A") == [(b"A", 100)]


def test_cid_decoder_rejects_empty_source_codes() -> None:
    cmap = CMapDecoder(b"1 begincidchar <> 7 endcidchar 1 beginnotdefchar <> 8 endnotdefchar")

    assert b"" not in cmap.cid_mappings
    assert b"" not in cmap.notdef_mappings
    assert all(length > 0 for length in cmap.decode_lengths)
    assert cmap.decode_entries(b"A") == [(b"A", 0)]


def test_cid_decoder_rejects_invalid_cid_destinations_and_range_endpoints() -> None:
    cmap = CMapDecoder(
        b"2 begincidchar <41> -1 <42> 65536 endcidchar "
        b"2 begincidchar <43> 123 <47> 65535 endcidchar "
        b"1 begincidrange <43> <44> 65535 endcidrange "
        b"1 begincidrange <48> <49> 65534 endcidrange "
        b"1 beginnotdefchar <45> -1 endnotdefchar "
        b"2 beginnotdefrange <46> <46> 65536 <4a> <4a> 65535 endnotdefrange"
    )

    assert cmap.decode_entries(b"ABCDEFGHIJ") == [
        (b"A", 0),
        (b"B", 0),
        (b"C", 123),
        (b"D", 0),
        (b"E", 0),
        (b"F", 0),
        (b"G", 65535),
        (b"H", 65534),
        (b"I", 65535),
        (b"J", 65535),
    ]


def test_cmap_ignores_procedures_and_operators_outside_begincmap_scope() -> None:
    cmap = CMapDecoder(
        b"1 begincidchar <41> 10 endcidchar "
        b"begincmap "
        b"{ /Parent usecmap /WMode 1 def 1 begincidchar <42> 20 endcidchar } bind def "
        b"/WMode 0 def 1 begincidchar <43> 30 endcidchar "
        b"endcmap "
        b"1 begincidchar <44> 40 endcidchar",
        usecmap_resolver=lambda name: (
            b"1 begincidchar <41> 99 endcidchar" if name == "Parent" else None
        ),
    )

    assert cmap.wmode == 0
    assert cmap.cid_mappings == {b"C": 30}
    assert cmap.decode_entries(b"ABCD") == [
        (b"A", 0),
        (b"B", 0),
        (b"C", 30),
        (b"D", 0),
    ]


def test_cmap_ignores_section_operators_in_comments_and_larger_identifiers() -> None:
    cmap = CMapDecoder(
        b"""
        % begincidchar
        <41> 99
        endcidchar
        1 notbegincidchar <42> 100 endcidchar
        """
    )

    assert cmap.cid_mappings == {}
    assert cmap.decode_entries(b"AB") == [(b"A", 0), (b"B", 0)]


def test_cmap_usecmap_inheritance_obeys_local_range_and_char_precedence() -> None:
    parent = b"""
        1 begincodespacerange <00> <ff> endcodespacerange
        1 begincidchar <41> 100 endcidchar
        1 begincidrange <42> <43> 200 endcidrange
    """
    child = b"""
        % /Ignored usecmap
        (/AlsoIgnored usecmap)
        /Parent usecmap
        1 begincidchar <43> 400 endcidchar
        1 begincidrange <41> <42> 300 endcidrange
    """

    cmap = CMapDecoder(child, usecmap_resolver=lambda name: parent if name == "Parent" else None)

    assert cmap.decode_entries(b"ABC") == [(b"A", 300), (b"B", 301), (b"C", 400)]


def test_tounicode_usecmap_inherits_and_allows_local_override() -> None:
    parent = b"""
        1 begincodespacerange <00> <ff> endcodespacerange
        2 beginbfchar
        <41> <0041>
        <43> <0043>
        endbfchar
    """
    child = b"""
        /Parent usecmap
        1 beginbfchar <41> <0058> endbfchar
        1 beginbfchar <42> <0042> endbfchar
    """

    cmap = ToUnicodeCMap(child, usecmap_resolver=lambda name: parent if name == "Parent" else None)

    assert cmap.decode(b"ABC") == "XBC"


def test_tounicode_mapping_sections_apply_in_source_order() -> None:
    range_then_char = ToUnicodeCMap(
        b"1 beginbfrange <41> <41> <0058> endbfrange 1 beginbfchar <41> <0059> endbfchar"
    )
    char_then_range = ToUnicodeCMap(
        b"1 beginbfchar <41> <0059> endbfchar 1 beginbfrange <41> <41> <0058> endbfrange"
    )

    assert range_then_char.decode(b"A") == "Y"
    assert char_then_range.decode(b"A") == "X"


def test_tounicode_numeric_cidrange_recovery_applies_in_source_order() -> None:
    range_then_char = ToUnicodeCMap(
        b"1 begincidrange <41> <41> 89 endcidrange 1 beginbfchar <41> <0058> endbfchar"
    )
    char_then_range = ToUnicodeCMap(
        b"1 beginbfchar <41> <0058> endbfchar 1 begincidrange <41> <41> 89 endcidrange"
    )

    assert range_then_char.decode(b"A") == "X"
    assert char_then_range.decode(b"A") == "Y"


def test_tounicode_rejects_empty_source_codes() -> None:
    cmap = ToUnicodeCMap(
        b"2 beginbfchar <> <0041> <42> <0042> endbfchar "
        b"2 beginbfrange <> <> [<0041>] <43> <43> <0043> endbfrange "
        b"2 begincidrange <> <> 65 <44> <44> 68 endcidrange"
    )

    assert b"" not in cmap.mappings
    assert all(length > 0 for length in cmap.decode_lengths)
    assert cmap.decode(b"BCD") == "BCD"


def test_tounicode_ignores_procedures_and_operators_outside_begincmap_scope() -> None:
    cmap = ToUnicodeCMap(
        b"1 beginbfchar <41> <0041> endbfchar "
        b"begincmap "
        b"{ 1 beginbfchar <42> <0042> endbfchar } bind def "
        b"1 beginbfchar <43> <0043> endbfchar "
        b"endcmap "
        b"1 beginbfchar <44> <0044> endbfchar"
    )

    assert cmap.mappings == {b"C": "C"}


def test_tounicode_without_begincmap_stops_at_endcmap() -> None:
    cmap = ToUnicodeCMap(
        b"1 beginbfchar <41> <0041> endbfchar endcmap 1 beginbfchar <42> <0042> endbfchar"
    )

    assert cmap.mappings == {b"A": "A"}


def test_tounicode_end_operator_inside_literal_string_does_not_end_block() -> None:
    cmap = ToUnicodeCMap(
        b"""
        2 beginbfchar
        <41> (endbfchar)
        <42> <0042>
        endbfchar
        """
    )

    assert b"A" in cmap.mappings
    assert cmap.mappings[b"B"] == "B"


def test_tounicode_retains_hex_prefix_before_corrupt_nested_delimiter() -> None:
    cmap = ToUnicodeCMap(
        b"""
        3 beginbfchar
        <01> <0044>
        <02> <000<>
        <03> <006d>
        endbfchar
        """
    )

    assert cmap.mappings == {b"\x01": "D", b"\x02": "\x00"}


def test_tounicode_can_distinguish_explicit_null_mappings() -> None:
    cmap = ToUnicodeCMap(
        b"""
        1 begincodespacerange <0000> <ffff> endcodespacerange
        2 beginbfchar
        <0001> <0041>
        <0002> <0000>
        endbfchar
        """
    )

    assert cmap.decode(b"\x00\x01\x00\x02") == "A"
    assert cmap.decode(b"\x00\x01\x00\x02", preserve_nulls=True) == "A\x00"


def test_tounicode_accepts_numeric_cid_ranges_as_unicode_scalars() -> None:
    cmap = ToUnicodeCMap(
        b"""
        1 begincodespacerange <0000> <ffff> endcodespacerange
        2 begincidrange
        <4e00> <4e02> 19968
        <00ff> <0100> 255
        endcidrange
        """
    )

    assert cmap.decode(b"\x4e\x00\x4e\x01\x4e\x02\x00\xff\x01\x00") == "一丁丂ÿĀ"


def test_tounicode_retains_explicit_mappings_with_malformed_codespace() -> None:
    cmap = ToUnicodeCMap(
        b"""
        1 begincodespacerange <0083> <020c> endcodespacerange
        3 beginbfrange
        <020b> <020b> <0028>
        <0083> <0083> <0061>
        <020c> <020c> <0029>
        endbfrange
        """
    )

    assert cmap.decode(b"\x02\x0b\x00\x83\x02\x0c") == "(a)"


def test_non_japanese_cjk_maps_cover_full_collections() -> None:
    cases = (
        ("GB1", 30300, ((115, "〈"), (4559, "中"), (1905, "汉"), (22047, "⺁"))),
        ("CNS1", 18780, ((148, "〈"), (661, "中"), (4111, "漢"), (14164, "𥴠"))),
        ("Korea1", 17190, ((104, "·"), (3296, "한"), (1204, "국"))),
        ("KR", 21140, ((2835, "한"), (353, "국"), (1887, "어"))),
    )

    for ordering, minimum_size, expected in cases:
        horizontal = resolve_cid_unicode_map("Adobe", ordering)
        vertical = resolve_cid_unicode_map("Adobe", ordering, vertical=True)

        assert horizontal is not None
        assert vertical is not None
        assert horizontal.get(minimum_size - 1) is not None
        assert vertical.get(minimum_size - 1) is not None
        for cid, text in expected:
            assert horizontal.get(cid) == text

from core_pdf.impl._impl.fonts.cid_unicode import resolve_cid_unicode_map
from core_pdf.impl._impl.fonts.cmap_tounicode import ToUnicodeCMap


def test_cidrange_rejects_unbounded_expansion() -> None:
    cmap = ToUnicodeCMap(b"1 begincidrange <00000000> <ffffffff> 32 endcidrange")

    assert cmap.mappings == {}


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

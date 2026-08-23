import numpy

from core_pdf.impl.engine.spec.s_09_fonts.cid_unicode import resolve_cid_unicode_map
from core_pdf.impl.engine.spec.s_09_fonts.cmap_decoder import CMapDecoder
from core_pdf.impl.engine.spec.s_09_fonts.cmap_ranges import iter_codespace_range
from core_pdf.impl.engine.spec.s_09_fonts.cmap_resources import resolve_cmap_decoder
from core_pdf.impl.engine.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap


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


def test_tounicode_usecmap_inherits_and_allows_local_override() -> None:
    parent = b"""
        1 begincodespacerange <00> <ff> endcodespacerange
        1 beginbfchar <41> <0041> endbfchar
    """
    child = b"""
        /Parent usecmap
        1 beginbfchar <41> <0058> endbfchar
        1 beginbfchar <42> <0042> endbfchar
    """

    cmap = ToUnicodeCMap(child, usecmap_resolver=lambda name: parent if name == "Parent" else None)

    assert cmap.decode(b"AB") == "XB"


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

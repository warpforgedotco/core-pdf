from core_pdf.impl.third_party.cid.cmap import CMapDecoder, ToUnicodeCMap, iter_codespace_range
from core_pdf.impl.third_party.cid.resource_loader import (
    resolve_cid_unicode_map,
    resolve_cmap_decoder,
)


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

    assert cmap.decode(b"\x01\x02") == [(b"\x01", 11), (b"\x02", 0)]


def test_identity_vertical_cmaps_preserve_wmode() -> None:
    assert resolve_cmap_decoder("Identity-H").wmode == 0
    assert resolve_cmap_decoder("Identity-V").wmode == 1
    assert resolve_cmap_decoder("OneByteIdentityV").wmode == 1


def test_tounicode_usecmap_inherits_and_allows_local_override() -> None:
    parent = b"""\n+        1 begincodespacerange <00> <ff> endcodespacerange
        1 beginbfchar <41> <0041> endbfchar
    """
    child = b"""\n+        /Parent usecmap
        1 beginbfchar <41> <0058> endbfchar
        1 beginbfchar <42> <0042> endbfchar
    """

    cmap = ToUnicodeCMap(child, usecmap_resolver=lambda name: parent if name == "Parent" else None)

    assert cmap.decode(b"AB") == "XB"


def test_japan1_unicode_map_disambiguates_and_covers_legacy_cids() -> None:
    horizontal = resolve_cid_unicode_map("Adobe", "Japan1")
    vertical = resolve_cid_unicode_map("Adobe", "Japan1", vertical=True)

    assert horizontal is not None
    assert vertical is not None
    assert horizontal[114] == "\u2012"
    assert horizontal[633] == "\u3000"
    assert horizontal[3284] == "日"
    assert horizontal[7639] == "欝"
    assert vertical[7899] == "（"


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
        assert len(horizontal) >= minimum_size
        assert len(vertical) >= minimum_size
        for cid, text in expected:
            assert horizontal[cid] == text

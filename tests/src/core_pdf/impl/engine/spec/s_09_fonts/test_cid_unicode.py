from core_pdf.impl.engine.spec.s_09_fonts.cid_unicode import (
    CompactCMap,
    internal_compact_cmap,
    resolve_cid_unicode_map,
)
from core_pdf.impl.engine.spec.s_09_fonts.cmap_decoder import CMapDecoder


def test_compact_cmap_returns_only_effective_codes() -> None:
    cmap = CompactCMap(
        effective_codes_by_cid={100: (b"A",), 102: (b"C",), 300: (b"B",)},
    )

    assert cmap.codes_for_cid(100) == (b"A",)
    assert cmap.codes_for_cid(101) == ()
    assert cmap.codes_for_cid(200) == ()
    assert cmap.codes_for_cid(300) == (b"B",)


def test_compact_cmap_uses_decoder_effective_usecmap_precedence() -> None:
    parent = b"""
        1 begincidchar <41> 100 endcidchar
        1 begincidrange <42> <43> 200 endcidrange
    """
    child = b"""
        /Parent usecmap
        1 begincidchar <43> 400 endcidchar
        1 begincidrange <41> <42> 300 endcidrange
    """
    decoder = CMapDecoder(child, usecmap_resolver=lambda name: parent if name == "Parent" else None)

    cmap = internal_compact_cmap(decoder)

    assert cmap.codes_for_cid(100) == ()
    assert cmap.codes_for_cid(200) == ()
    assert cmap.codes_for_cid(300) == (b"A",)
    assert cmap.codes_for_cid(301) == (b"B",)
    assert cmap.codes_for_cid(400) == (b"C",)


def test_compact_cmap_preserves_later_explicit_mapping_over_range() -> None:
    decoder = CMapDecoder(
        b"1 begincidrange <41> <41> 100 endcidrange 1 begincidchar <41> 200 endcidchar"
    )

    cmap = internal_compact_cmap(decoder)

    assert cmap.codes_for_cid(100) == ()
    assert cmap.codes_for_cid(200) == (b"A",)


def test_compact_cmap_excludes_mappings_outside_declared_codespace() -> None:
    decoder = CMapDecoder(
        b"1 begincodespacerange <00> <7f> endcodespacerange "
        b"1 begincidchar <ff> 99 endcidchar "
        b"1 begincidrange <7f> <80> 100 endcidrange"
    )

    cmap = internal_compact_cmap(decoder)

    assert decoder.decode_entries(b"\x7f\x80\xff") == [
        (b"\x7f", 100),
        (b"\x80", 0),
        (b"\xff", 0),
    ]
    assert cmap.codes_for_cid(99) == ()
    assert cmap.codes_for_cid(100) == (b"\x7f",)
    assert cmap.codes_for_cid(101) == ()


def test_compact_japan1_map_preserves_legacy_and_vertical_choices() -> None:
    horizontal = resolve_cid_unicode_map("Adobe", "Japan1")
    vertical = resolve_cid_unicode_map("Adobe", "Japan1", vertical=True)

    assert horizontal is not None
    assert vertical is not None
    assert horizontal.get(114) == "\u2012"
    assert horizontal.get(633) == "\u3000"
    assert horizontal.get(3284) == "日"
    assert horizontal.get(7639) == "欝"
    assert vertical.get(7899) == "（"


def test_compact_cid_map_caches_missing_cids() -> None:
    mapping = resolve_cid_unicode_map("Adobe", "Japan1")

    assert mapping is not None
    assert mapping.get(-1) is None
    assert mapping.get(-1, "missing") == "missing"
    assert -1 in mapping.cache

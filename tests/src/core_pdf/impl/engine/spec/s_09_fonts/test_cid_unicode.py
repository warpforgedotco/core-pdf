from core_pdf.impl.engine.spec.s_09_fonts.cid_unicode import (
    CompactCMap,
    code_for_cid,
    remove_codes_covered_by_ranges,
    resolve_cid_unicode_map,
)
from core_pdf.impl.third_party.cid.cmap import CIDRange


def test_code_for_cid_uses_cmap_mixed_radix_ranges() -> None:
    cid_range = CIDRange(b"\x81\x40", b"\x82\x42", 100)

    assert code_for_cid(cid_range, 100) == b"\x81\x40"
    assert code_for_cid(cid_range, 102) == b"\x81\x42"
    assert code_for_cid(cid_range, 103) == b"\x82\x40"
    assert code_for_cid(cid_range, 105) == b"\x82\x42"
    assert code_for_cid(cid_range, 106) is None


def test_compact_cmap_preserves_explicit_and_later_range_precedence() -> None:
    cmap = CompactCMap(
        mappings_by_cid={300: (b"B",)},
        mapped_codes=frozenset({b"B"}),
        ranges=(CIDRange(b"A", b"C", 100), CIDRange(b"B", b"B", 200)),
        effective_codes_by_cid={100: (b"A",), 102: (b"C",), 300: (b"B",)},
    )

    assert cmap.codes_for_cid(100) == (b"A",)
    assert cmap.codes_for_cid(101) == ()
    assert cmap.codes_for_cid(200) == ()
    assert cmap.codes_for_cid(300) == (b"B",)


def test_remove_codes_covered_by_ranges_handles_mixed_radix_range() -> None:
    mappings = {b"\x81\x40": 1, b"\x81\x41": 2, b"\x82\x42": 3, b"\x82\x43": 4}

    assert remove_codes_covered_by_ranges(mappings, [CIDRange(b"\x81\x40", b"\x82\x42", 100)]) == {
        b"\x82\x43": 4
    }


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

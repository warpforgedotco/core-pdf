"""CID Unicode recovery honors effective mappings and ordered voting tiers."""

import pytest

from core_pdf.impl._impl.fonts import cid_unicode as cid
from core_pdf.impl._impl.fonts.cmap_decoder import CMapDecoder
from core_pdf.impl._impl.fonts.cmap_resources import unicode_scalar_from_cmap_code


@pytest.mark.parametrize("codespace", [b"", b"1 begincodespacerange <41> <43> endcodespacerange"])
def test_compact_inversion_respects_later_ranges_explicit_overrides_and_codespaces(codespace):
    decoder = CMapDecoder(
        codespace
        + b"""
    1 begincidrange <40> <44> 10 endcidrange
    1 begincidrange <42> <43> 20 endcidrange
    2 begincidchar <43> 30 <45> 40 endcidchar
    """
    )
    result = cid.internal_compact_cmap(decoder)
    expected = {11: (b"A",), 20: (b"B",), 30: (b"C",)}
    if not codespace:
        expected.update({10: (b"@",), 14: (b"D",), 40: (b"E",)})
    assert result.effective_codes_by_cid == expected
    assert result.codes_for_cid(999) == ()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize(
    ("primary", "opposite", "fallback", "expected"),
    [
        ((("A", 1), ("B", 3)), (("C", 9),), (("D", 0),), "B"),
        ((("B", 2), ("A", 2)), (), (), "A"),
        ((("B", 2), ("A", 1), ("A", 2)), (), (), "A"),
        (((None, 3),), (("Z", 0), ("C", 2)), (("D", 0),), "C"),
        ((), (), (("D", 0), ("E", -2)), "D"),
        (((None, 3),), ((None, 2),), ((None, 0),), None),
        ((("\ue000", 2), ("A", 2)), (), (), "A"),
        ((("\uf900", 2), ("一", 2)), (), (), "一"),
        ((("\u0301", 2), ("é", 2)), (), (), "é"),
    ],
)
def test_unicode_votes_obey_orientation_weight_fallback_and_ties(
    monkeypatch, vertical, primary, opposite, fallback, expected
):
    maps = {}

    def sources(entries, prefix):
        result = []
        for index, (text, weight) in enumerate(entries):
            name = f"{prefix}{index}"
            maps[name] = cid.CompactCMap({7: (text.encode("utf-8"),)} if text else {})
            result.append((name, "utf-8", weight))
        return tuple(result)

    collection = {
        vertical: sources(primary, "p") + sources(fallback, "f"),
        not vertical: sources(opposite, "o"),
    }
    monkeypatch.setitem(cid.CID_COLLECTION_UNICODE_SOURCES, ("Test", "Votes"), collection)
    monkeypatch.setattr(cid, "compact_cmap", maps.get)
    mapping = cid.CIDUnicodeMap("Test", "Votes", vertical)
    assert mapping.get(7) == expected
    assert mapping.get(7, "missing") == (expected or "missing")
    # Both successful and unsuccessful votes are cached, not caller defaults.
    maps.clear()
    assert mapping.get(7, "another") == (expected or "another")


def test_collection_override_precedes_source_votes(monkeypatch):
    monkeypatch.setitem(cid.CID_COLLECTION_UNICODE_OVERRIDES, ("Test", "Override"), {7: "X"})
    mapping = cid.CIDUnicodeMap("Test", "Override", False)
    assert mapping.get(7) == "X"
    assert mapping.get(8, "missing") == "missing"


@pytest.mark.parametrize("codes", [(b"B", b"A"), (b"A", b"B"), (b"\xff", b"A"), (b"AB", b"A")])
def test_unicode_candidate_selection_is_order_independent_and_rejects_non_scalars(
    monkeypatch, codes
):
    monkeypatch.setattr(cid, "compact_cmap", lambda name: cid.CompactCMap({7: codes}))
    assert cid.preferred_unicode_for_cid("test", "utf-8", 7) == "A"
    assert cid.preferred_unicode_for_cid("test", "utf-8", 8) is None


@pytest.mark.parametrize(
    ("code", "codec", "expected"),
    [
        (b"A", "gb2312_7bit", "A"),
        (b"VP", "gb2312_7bit", "中"),
        (b"A", "euc_kr_7bit", "A"),
        (b"0!", "euc_kr_7bit", "가"),
        (b"\x80", "gb2312_7bit", None),
        (b"abc", "euc_kr_7bit", None),
        (b"A", "jis_x0208", "A"),
        (b"$\x22", "jis_x0208", "あ"),
        (b"abc", "jis_x0208", None),
        (b"\xff\xff", "jis_x0208", None),
        (b"", "utf-8", None),
        (b"AB", "utf-8", None),
        (b"\xd8\x00", "utf-16-be", None),
        (b"\xff", "utf-8", None),
    ],
)
def test_legacy_cmap_codes_decode_to_exactly_one_unicode_scalar(code, codec, expected):
    assert unicode_scalar_from_cmap_code(code, codec) == expected


def test_unknown_collections_and_resources_have_no_unicode_mapping():
    assert cid.resolve_cid_unicode_map("Unknown", "Unknown") is None
    assert cid.compact_cmap("No-Such-CMap") is None
    assert cid.preferred_unicode_for_cid("No-Such-CMap", "utf-8", 7) is None


@pytest.mark.parametrize("vertical", [False, True])
def test_packaged_collection_maps_are_cached_by_orientation(vertical):
    mapping = cid.resolve_cid_unicode_map("Adobe", "GB1", vertical=vertical)
    assert mapping is not None
    assert mapping.vertical is vertical
    assert cid.resolve_cid_unicode_map("Adobe", "GB1", vertical=vertical) is mapping
    # GB1 CID 34 denotes ASCII A in the packaged Unicode CMap.
    assert mapping.get(34) == "A"

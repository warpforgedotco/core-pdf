import pytest

from core_pdf.impl import fonts_unicode
from core_pdf.impl.fonts_cmap import CID_COLLECTION_UNICODE_SOURCES
from scripts.build_cid_unicode_tables import OUTPUT, build_table


@pytest.mark.parametrize(("registry", "ordering"), sorted(CID_COLLECTION_UNICODE_SOURCES))
def test_packaged_table_matches_the_live_votes(registry, ordering):
    path = OUTPUT / fonts_unicode.cid_unicode_table_name(registry, ordering)
    assert path.read_bytes() == build_table(registry, ordering), (
        "stale table: run scripts/build_cid_unicode_tables.py"
    )


def test_table_round_trips_vertical_changes_and_removals():
    horizontal = {1: "A", 2: "B", 3: "C"}
    vertical = {1: "A", 2: "︱", 4: "D"}
    data = fonts_unicode.encode_cid_unicode_table(horizontal, vertical)
    assert fonts_unicode.decode_cid_unicode_table(data, vertical=False) == horizontal
    assert fonts_unicode.decode_cid_unicode_table(data, vertical=True) == vertical


def test_packaged_collections_are_answered_without_voting(monkeypatch):
    def refuse(self, cid):
        raise AssertionError("voted live")

    monkeypatch.setattr(fonts_unicode.CIDUnicodeMap, "vote_from_sources", refuse)
    mapping = fonts_unicode.CIDUnicodeMap("Adobe", "Japan1", False)
    assert mapping.get(34) == "A"
    assert mapping.get(10**6, "missing") == "missing"

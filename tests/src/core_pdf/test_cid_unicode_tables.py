import pytest

from core_pdf.impl import fonts_unicode
from scripts.build_cid_unicode_tables import (
    CID_COLLECTION_UNICODE_SOURCES,
    OUTPUT,
    build_table,
    encode_cid_unicode_table,
)


@pytest.mark.parametrize(("registry", "ordering"), sorted(CID_COLLECTION_UNICODE_SOURCES))
def test_packaged_table_matches_the_live_votes(registry, ordering):
    path = OUTPUT / fonts_unicode.cid_unicode_table_name(registry, ordering)
    assert path.read_bytes() == build_table(registry, ordering), (
        "stale table: run scripts/build_cid_unicode_tables.py"
    )


def test_table_round_trips_vertical_changes_and_removals():
    horizontal = {1: "A", 2: "B", 3: "C"}
    vertical = {1: "A", 2: "︱", 4: "D"}
    data = encode_cid_unicode_table(horizontal, vertical)
    assert fonts_unicode.decode_cid_unicode_table(data, vertical=False) == horizontal
    assert fonts_unicode.decode_cid_unicode_table(data, vertical=True) == vertical


def test_packaged_collections_are_answered_from_their_tables():
    mapping = fonts_unicode.CIDUnicodeMap("Adobe", "Japan1", False)
    assert mapping.get(34) == "A"
    assert mapping.get(10**6, "missing") == "missing"


def test_a_collection_without_a_table_has_no_map():
    assert fonts_unicode.resolve_cid_unicode_map("Test", "Votes") is None


@pytest.mark.parametrize(
    ("registry", "ordering"),
    [("../cid_unicode/Adobe", "Japan1"), (str(OUTPUT / "Adobe"), "Japan1")],
)
def test_a_pdf_named_collection_cannot_steer_the_table_lookup(registry, ordering):
    assert fonts_unicode.resolve_cid_unicode_map(registry, ordering) is None


def test_every_collection_has_a_listed_table():
    assert fonts_unicode.packaged_cid_unicode_table_names() == {
        fonts_unicode.cid_unicode_table_name(registry, ordering)
        for registry, ordering in CID_COLLECTION_UNICODE_SOURCES
    }

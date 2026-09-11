"""ToUnicode compilation preserves reader recovery and source-order semantics."""

from itertools import permutations

import pytest

from core_pdf.impl._impl.fonts.cmap_tounicode import ToUnicodeCMap, parse_to_unicode_cmap
from core_pdf_spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap as StrictToUnicodeCMap

CODESPACE = b"1 begincodespacerange <00> <ff> endcodespacerange\n"
MAPPING_BLOCKS = {
    "bfchar": b"2 beginbfchar <01> <0041> <02> <0061> endbfchar",
    "bfrange": b"1 beginbfrange <01> <02> <0042> endbfrange",
    "cidrange": b"1 begincidrange <01> <02> 68 endcidrange",
}


@pytest.mark.parametrize("order", list(permutations(MAPPING_BLOCKS)))
def test_last_mapping_definition_wins_across_block_types(order: tuple[str, ...]) -> None:
    # cidrange in a ToUnicode map is reader recovery; it participates in the
    # same source ordering as the prescribed bfchar and bfrange definitions.
    data = CODESPACE + b"\n".join(MAPPING_BLOCKS[name] for name in order)
    expected = {"bfchar": "Aa", "bfrange": "BC", "cidrange": "DE"}[order[-1]]
    parsed = parse_to_unicode_cmap(data)
    assert parsed.mappings == {b"\x01": expected[0], b"\x02": expected[1]}
    assert ToUnicodeCMap(data).decode(b"\x01\x02") == expected


@pytest.mark.parametrize(
    "codespace",
    [
        b"1 begincodespacerange <0083> <020c> endcodespacerange",
        b"2 begincodespacerange <00> <ff> <00> <7f> endcodespacerange",
    ],
    ids=["malformed-rectangle", "overlap"],
)
def test_invalid_codespaces_keep_explicit_mapping_evidence(codespace: bytes) -> None:
    mapping = b"1 beginbfchar <0100> <0041> endbfchar"
    parsed = parse_to_unicode_cmap(codespace + b" " + mapping)
    assert parsed.code_space_ranges == ()
    assert parsed.mappings == {b"\x01\x00": "A"}
    cmap = ToUnicodeCMap(codespace + b" " + mapping)
    assert cmap.decode_lengths == (2,)
    assert cmap.decode(b"\x01\x00") == "A"
    # The malformed codespace alone supplies no usable recovery evidence.
    with pytest.raises(ValueError, match="codespacerange"):
        ToUnicodeCMap(codespace)


def test_codespace_recovery_retains_valid_disjoint_entries() -> None:
    data = (
        b"3 begincodespacerange <00> <3f> <ff> <80> <80> <ff> endcodespacerange "
        b"2 beginbfchar <01> <0041> <80> <0042> endbfchar"
    )
    parsed = parse_to_unicode_cmap(data)
    assert parsed.code_space_ranges == ((b"\x00", b"\x3f"), (b"\x80", b"\xff"))
    assert ToUnicodeCMap(data).decode(b"\x01\x80") == "AB"


@pytest.mark.parametrize(
    "invalid",
    [
        b"1 beginbfrange <03> <01> <0041> endbfrange",
        b"1 beginbfrange <01> <02> [] endbfrange",
        b"1 beginbfrange <01> <02> endbfrange",
    ],
    ids=["reversed", "empty-destinations", "incomplete"],
)
def test_invalid_only_ranges_reject_even_with_a_valid_character_mapping(invalid: bytes) -> None:
    data = CODESPACE + b"1 beginbfchar <09> <005A> endbfchar " + invalid
    with pytest.raises(ValueError, match="bfrange"):
        parse_to_unicode_cmap(data)
    with pytest.raises(ValueError, match="bfrange"):
        ToUnicodeCMap(data)


@pytest.mark.parametrize("invalid_first", [False, True])
def test_valid_range_allows_recovery_from_invalid_ranges_in_other_blocks(
    invalid_first: bool,
) -> None:
    valid = b"1 beginbfrange <01> <02> [<0041> <0042>] endbfrange"
    invalid = b"1 beginbfrange <03> <01> <0058> endbfrange"
    blocks = (invalid, valid) if invalid_first else (valid, invalid)
    cmap = ToUnicodeCMap(CODESPACE + b" ".join(blocks))
    assert cmap.mappings == {b"\x01": "A", b"\x02": "B"}
    assert cmap.decode(b"\x01\x02") == "AB"


@pytest.mark.parametrize(
    "parent",
    [
        None,
        b"1 beginbfrange <03> <01> <0041> endbfrange",
        b"1 begincodespacerange <ff> <00> endcodespacerange",
    ],
    ids=["missing", "bad-mappings", "bad-codespace"],
)
def test_parent_recovery_preserves_child_mapping(parent: bytes | None) -> None:
    child = CODESPACE + b"/Parent usecmap 1 beginbfchar <01> <0043> endbfchar"
    cmap = ToUnicodeCMap(child, usecmap_resolver=lambda name: parent)
    assert cmap.decode(b"\x01") == "C"
    assert cmap.mappings == {b"\x01": "C"}


def test_child_overrides_parent_without_discarding_other_parent_mappings() -> None:
    parent = CODESPACE + b"2 beginbfchar <01> <0041> <02> <0050> endbfchar"
    child = b"/Parent usecmap 1 beginbfchar <01> <0043> endbfchar"
    parsed = parse_to_unicode_cmap(child)
    assert parsed.usecmap_name == "Parent"
    assert parsed.code_space_ranges == ()
    assert parsed.mappings == {b"\x01": "C"}
    cmap = ToUnicodeCMap(child, usecmap_resolver=lambda name: parent)
    assert cmap.decode(b"\x01\x02") == "CP"


def test_parent_resolver_unexpected_errors_propagate() -> None:
    def resolve(name: str) -> bytes | None:
        raise RuntimeError("parent source failed")

    with pytest.raises(RuntimeError, match="parent source failed"):
        ToUnicodeCMap(CODESPACE + b"/Parent usecmap", usecmap_resolver=resolve)


def test_parent_depth_limit_keeps_sixteen_levels_and_skips_deeper_maps() -> None:
    visited: list[int] = []

    def resolve(name: str) -> bytes:
        level = int(name)
        visited.append(level)
        mapping = f"1 beginbfchar <{level:02x}> <{65 + level:04x}> endbfchar".encode()
        return f"/{level + 1} usecmap ".encode() + mapping

    child = CODESPACE + b"/1 usecmap 1 beginbfchar <00> <0041> endbfchar"
    cmap = ToUnicodeCMap(child, usecmap_resolver=resolve)
    assert cmap.decode(bytes(range(17))) == "ABCDEFGHIJKLMNOPQ"
    assert cmap.lookup(b"\x11") is None
    assert visited == list(range(1, 18))


def test_empty_source_has_reader_fallback_but_no_strict_codespace() -> None:
    parsed = parse_to_unicode_cmap(b"")
    assert parsed.code_space_ranges == ()
    assert parsed.mappings == {}
    assert parsed.usecmap_name is None
    cmap = ToUnicodeCMap(b"")
    assert cmap.decode_lengths == (1,)
    assert cmap.decode(b"\x00A") == "A"
    assert cmap.decode(b"\x00A", preserve_nulls=True) == "\x00A"
    with pytest.raises(ValueError, match="codespace"):
        StrictToUnicodeCMap(b"")
    assert StrictToUnicodeCMap(CODESPACE).decode_lengths == (1,)

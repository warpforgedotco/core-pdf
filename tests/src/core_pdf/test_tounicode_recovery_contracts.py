import pytest

from core_pdf.impl.fonts.cmap_ranges import expand_range
from core_pdf.impl.fonts.cmap_tounicode import ToUnicodeCMap, decode_utf16be_text


def block(kind, records):
    return b"1 begin" + kind + b"\n" + records + b"\nend" + kind + b"\n"


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (b"", ""),
        (b"A", "A"),
        (b"\0A", "A"),
        (b"\xfe\xff\0A", "A"),
        (b"\x41\0B", "AB"),
        (b"\xd8\0", "\ufffd"),
        (b"\xd8\x3d\xde\0", "😀"),
    ],
)
def test_destination_decoding_recovers_bom_odd_bytes_and_invalid_surrogates(encoded, expected):
    assert decode_utf16be_text(encoded) == expected


@pytest.mark.parametrize("buffer_type", [bytes, bytearray, memoryview])
@pytest.mark.parametrize("preserve_nulls", [False, True])
def test_mapping_lengths_nulls_and_unmapped_bytes_have_stable_precedence(
    buffer_type, preserve_nulls
):
    source = block(b"bfchar", b"<01> <0041> <0102> <0058> <03> <0000> <04> <00420043>")
    cmap = ToUnicodeCMap(buffer_type(source))
    assert cmap.decode(b"\1\2\3\4", preserve_nulls=preserve_nulls) == (
        "A\2\0BC" if preserve_nulls else "A\2BC"
    )
    assert cmap.decode(b"") == ""
    assert cmap.mappings[b"\1\2"] == "X"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [(b"\0A", "A"), (b"\0\0", "\ufffd"), (b"\xd8\0", "\ufffd"), (b"\0A\x42", "AB")],
)
def test_two_byte_codespace_uses_unicode_fallback_and_retains_trailing_byte(payload, expected):
    cmap = ToUnicodeCMap(block(b"codespacerange", b"<0000> <ffff>"))
    assert cmap.decode(payload) == expected


@pytest.mark.parametrize(
    "records", [b"<0083> <020c>", b"<00> <ff> <20> <30>", b"<00> <ffff>", b"<ff> <00>"]
)
def test_bad_codespace_keeps_explicit_mapping_but_rejects_empty_corruption(records):
    codespace = block(b"codespacerange", records)
    with pytest.raises(ValueError, match="codespacerange"):
        ToUnicodeCMap(codespace)
    cmap = ToUnicodeCMap(codespace + block(b"bfchar", b"<01> <0041>"))
    assert cmap.decode(b"\1") == "A"
    assert cmap.code_space_ranges == ()


def test_codespace_retains_valid_ranges_and_discards_trailing_operand():
    cmap = ToUnicodeCMap(block(b"codespacerange", b"<00> <7f> <ffff> <0000> <80>"))
    assert cmap.code_space_ranges == ((b"\0", b"\x7f"),)


@pytest.mark.parametrize(
    ("kind", "records"),
    [
        (b"bfchar", b"<01> <005a>"),
        (b"bfrange", b"<01> <01> <005a>"),
        (b"cidrange", b"<01> <01> 90"),
    ],
)
def test_later_mapping_blocks_override_earlier_definitions(kind, records):
    initial = block(b"bfchar", b"<01> <0041>")
    assert ToUnicodeCMap(initial + block(kind, records)).decode(b"\1") == "Z"
    assert ToUnicodeCMap(block(kind, records) + initial).decode(b"\1") == "A"


@pytest.mark.parametrize(
    ("destination", "expected"),
    [
        (b"<0041>", {b"\1": "A", b"\2": "B", b"\3": "C"}),
        (b"[<0041> <0042>]", {b"\1": "A", b"\2": "B"}),
        (b"[<0041> <0042> <0043> <0044>]", {b"\1": "A", b"\2": "B", b"\3": "C"}),
        (b"(A)", {b"\1": "A", b"\2": "B", b"\3": "C"}),
    ],
)
def test_range_destinations_preserve_available_entries_and_clip_excess(destination, expected):
    cmap = ToUnicodeCMap(block(b"bfrange", b"<01> <03> " + destination))
    assert cmap.mappings == expected


@pytest.mark.parametrize(
    "records",
    [
        b"<01>",
        b"(a) (c) <0041>",
        b"<03> <01> <0041>",
        b"<01> <0003> <0041>",
        b"<01> <03> []",
        b"<01> <03> [<zz>]",
        b"<01> <03> <zz>",
    ],
)
def test_invalid_range_block_raises_only_when_no_valid_range_survives(records):
    invalid = block(b"bfrange", records)
    with pytest.raises(ValueError, match="bfrange"):
        ToUnicodeCMap(invalid)
    assert ToUnicodeCMap(invalid + block(b"bfrange", b"<04> <04> <0044>")).mappings[b"\4"] == "D"


@pytest.mark.parametrize(
    "parent",
    [
        None,
        b"1 begincodespacerange <ff> <00> endcodespacerange",
        block(b"bfchar", b"<01> <0041> <02> <0042>"),
    ],
)
def test_usecmap_ignores_unusable_parent_and_child_mapping_wins(parent):
    requests = []

    def resolve(name):
        requests.append(name)
        return parent

    child = ToUnicodeCMap(
        b"/Parent usecmap\n" + block(b"bfchar", b"<01> <005a>"), usecmap_resolver=resolve
    )
    assert requests == ["Parent"]
    assert child.mappings[b"\1"] == "Z"
    assert (b"\2" in child.mappings) == (parent is not None and b"bfchar" in parent)


def test_usecmap_cycles_terminate_and_keep_local_mappings():
    source = b"/Loop usecmap\n" + block(b"bfchar", b"<01> <0041>")
    calls = []

    def resolve(name):
        calls.append(name)
        return source

    assert ToUnicodeCMap(source, usecmap_resolver=resolve).decode(b"\1") == "A"
    assert len(calls) == 17


@pytest.mark.parametrize("base", ["", "A", "prefixA", "\ud800A", "\U0010ffff"])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_range_expansion_preserves_prefix_and_replaces_invalid_final_scalars(base, width):
    result = expand_range(1, 3, width, base)

    def valid(character):
        value = ord(character)
        return character if value < 0x110000 and not 0xD800 <= value <= 0xDFFF else "\ufffd"

    for offset in range(3):
        if not base:
            expected = ""
        else:
            final = ord(base[-1]) + offset
            expected = "".join(valid(c) for c in base[:-1]) + (
                valid(chr(final)) if final <= 0x10FFFF else "\ufffd"
            )
        assert result[(offset + 1).to_bytes(width, "big")] == expected


@pytest.mark.parametrize(
    "records", [b"<> <0041>", b"<zz> <0041>", b"<01> <zz>", b"<zz> <0041<02> <0042>"]
)
def test_unusable_character_mapping_does_not_hide_later_valid_block(records):
    cmap = ToUnicodeCMap(block(b"bfchar", records) + block(b"bfchar", b"<03> <0043>"))
    assert cmap.mappings == {b"\3": "C"}


@pytest.mark.parametrize(("prefix", "expected"), [(b"0041", "A"), (b"041", "А")])
def test_nested_destination_delimiter_retains_prefix_and_abandons_misaligned_block(
    prefix, expected
):
    cmap = ToUnicodeCMap(
        block(b"bfchar", b"<01> <" + prefix + b"<02> <0042>") + block(b"bfchar", b"<03> <0043>")
    )
    assert cmap.mappings == {b"\1": expected, b"\3": "C"}


@pytest.mark.parametrize("records", [b"<03> <01> 65", b"<01> <02> bad", b"<000000> <010000> 65"])
def test_unusable_numeric_cid_ranges_are_skipped(records):
    cmap = ToUnicodeCMap(block(b"cidrange", records) + block(b"bfchar", b"<01> <0041>"))
    assert cmap.mappings == {b"\1": "A"}


@pytest.mark.parametrize(
    ("start", "end", "width"), [(1, 0, 1), (0, 256, 1), (0, 0, 0), (0, 65536, 3)]
)
def test_range_expansion_rejects_reversed_overflowing_and_excessive_ranges(start, end, width):
    with pytest.raises(ValueError, match="bfrange"):
        expand_range(start, end, width, "A")

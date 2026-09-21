from __future__ import annotations

import pytest

from core_adobe_fonts.cmap.decoder import CMapDecoder
from core_adobe_fonts.cmap.ranges import validate_codespace_range
from core_adobe_fonts.cmap.tokenizer import CMapProgram
from core_pdf_spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap, parse_to_unicode_cmap

CODESPACE = b"1 begincodespacerange <00> <7f> endcodespacerange\n"


@pytest.mark.parametrize("reader", [CMapDecoder, ToUnicodeCMap])
@pytest.mark.parametrize("width", [1, 4, 5])
def test_pdf_codespace_supports_one_to_four_bytes(
    reader: type[CMapDecoder | ToUnicodeCMap], width: int
) -> None:
    start, end = b"00" * width, b"ff" * width
    data = b"1 begincodespacerange <" + start + b"> <" + end + b"> endcodespacerange"
    if width == 5:
        with pytest.raises(ValueError, match="code length"):
            reader(data)
    else:
        cmap = reader(data)
        assert cmap.decode_lengths == (width,)
        assert tuple(cmap.code_space_ranges) == ((b"\x00" * width, b"\xff" * width),)


def test_generic_postscript_range_validation_has_no_pdf_byte_limit() -> None:
    validate_codespace_range(b"\x00" * 5, b"\xff" * 5)


@pytest.mark.parametrize("reader", [CMapDecoder, ToUnicodeCMap])
@pytest.mark.parametrize(
    "codespace",
    [
        b"",
        b"1 begincodespacerange <> <> endcodespacerange",
        b"1 begincodespacerange <00> <ffff> endcodespacerange",
        b"1 begincodespacerange <0083> <020c> endcodespacerange",
        b"2 begincodespacerange <00> <7f> <40> <ff> endcodespacerange",
    ],
    ids=["missing", "empty-code", "different-lengths", "nonrectangular", "overlap"],
)
def test_complete_codespace_rejects_malformed_ranges(
    reader: type[CMapDecoder | ToUnicodeCMap], codespace: bytes
) -> None:
    with pytest.raises(ValueError):
        reader(codespace)


@pytest.mark.parametrize("reader", [CMapDecoder, ToUnicodeCMap])
@pytest.mark.parametrize(
    "local_codespace", [CODESPACE, b"1 begincodespacerange <80> <ff> endcodespacerange"]
)
def test_inheriting_cmap_cannot_redeclare_its_codespace(
    reader: type[CMapDecoder | ToUnicodeCMap], local_codespace: bytes
) -> None:
    with pytest.raises(ValueError, match="usecmap.*redefine"):
        reader(b"/Parent usecmap " + local_codespace, usecmap_resolver=lambda name: CODESPACE)


@pytest.mark.parametrize("reader", [CMapDecoder, ToUnicodeCMap])
def test_inheriting_cmap_accepts_overrides_only_inside_parent_codespace(
    reader: type[CMapDecoder | ToUnicodeCMap],
) -> None:
    if reader is CMapDecoder:
        parent = CODESPACE + b"2 begincidchar <01> 7 <02> 8 endcidchar"
        child = b"/Parent usecmap 1 begincidchar <01> 9 endcidchar"
        invalid = b"/Parent usecmap 1 begincidchar <80> 9 endcidchar"
    else:
        parent = CODESPACE + b"2 beginbfchar <01> <0041> <02> <0042> endbfchar"
        child = b"/Parent usecmap 1 beginbfchar <01> <0043> endbfchar"
        invalid = b"/Parent usecmap 1 beginbfchar <80> <0043> endbfchar"
    cmap = reader(child, usecmap_resolver=lambda name: parent)
    assert tuple(cmap.code_space_ranges) == ((b"\x00", b"\x7f"),)
    if isinstance(cmap, CMapDecoder):
        assert cmap.decode_entries(b"\x01\x02") == [(b"\x01", 9), (b"\x02", 8)]
    else:
        assert (cmap.lookup(b"\x01"), cmap.lookup(b"\x02")) == ("C", "B")
    with pytest.raises(ValueError, match="outside codespace"):
        reader(invalid, usecmap_resolver=lambda name: parent)


def test_standalone_tounicode_parser_defers_parent_membership_until_resolution() -> None:
    child = b"/Parent usecmap 1 beginbfchar <80> <0041> endbfchar"
    parsed = parse_to_unicode_cmap(child)
    assert parsed.code_space_ranges == ()
    assert parsed.mappings == {b"\x80": "A"}
    assert parsed.usecmap_name == "Parent"
    with pytest.raises(ValueError, match="outside codespace"):
        ToUnicodeCMap(child, usecmap_resolver=lambda name: CODESPACE)
    with pytest.raises(ValueError, match="outside codespace"):
        parse_to_unicode_cmap(CODESPACE + b"1 beginbfchar <80> <0041> endbfchar")


@pytest.mark.parametrize("token", [b"(usecmap)", b"{usecmap}", b"[usecmap]"])
def test_operator_looking_group_contents_do_not_declare_inheritance(token: bytes) -> None:
    cmap = CMapDecoder(CODESPACE + token + b" pop")
    assert cmap.decode_lengths == (1,)
    CMapProgram.parse(b"/Parent usecmap (begincodespacerange) {begincodespacerange}")


def test_codespace_redeclaration_check_only_uses_the_scoped_program() -> None:
    cmap = CMapDecoder(b"usecmap begincmap " + CODESPACE + b" endcmap usecmap")
    assert cmap.decode_lengths == (1,)


@pytest.mark.parametrize(
    "mapping",
    [b"1 begincidrange <30> <90> 7 endcidrange", b"1 beginnotdefrange <30> <90> 0 endnotdefrange"],
)
def test_cid_mapping_range_must_fit_a_single_codespace(mapping: bytes) -> None:
    data = b"2 begincodespacerange <00> <3f> <80> <ff> endcodespacerange "
    with pytest.raises(ValueError, match="mapping range outside"):
        CMapDecoder(data + mapping)
    cmap = CMapDecoder(data + b"1 begincidrange <30> <3f> 7 endcidrange")
    assert cmap.decode_entries(b"\x30\x3f") == [(b"\x30", 7), (b"\x3f", 22)]


@pytest.mark.parametrize(
    "ranges",
    [[(b"\x00" * 5, b"\xff" * 5)], [(b"\x00", b"\x7f"), (b"\x40", b"\xff")]],
    ids=["five-byte", "overlap"],
)
def test_cid_parent_resource_cannot_bypass_effective_codespace_validation(
    ranges: list[tuple[bytes, bytes]],
) -> None:
    entries = b" ".join(
        b"<" + start.hex().encode() + b"> <" + end.hex().encode() + b">" for start, end in ranges
    )
    parent = str(len(ranges)).encode() + b" begincodespacerange " + entries + b" endcodespacerange"
    with pytest.raises(ValueError):
        CMapDecoder(b"/Parent usecmap", usecmap_resolver=lambda name: parent)

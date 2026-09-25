from collections.abc import Iterable

import pytest

import core_pdf.impl.fonts.decoder as decoder_module
from core_pdf.impl.fonts.cmap_tokenizer import CMapDecoder
from core_pdf.impl.fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.fonts.decoder import FontDecoder, single_code_mapping, split_code_bytes


def block(kind, records):
    return b"1 begin" + kind + b"\n" + records + b"\nend" + kind + b"\n"


@pytest.mark.parametrize("kind", [b"cidchar", b"notdefchar"])
@pytest.mark.parametrize(
    "record",
    [
        b"<01> 10",
        b"<01> 10 <02>",
        b"bad 10 <01> 10",
        b"<zz> 2 <01> 10",
        b"<> 2 <01> 10",
        b"<02> -1 <01> 10",
        b"<02> 65536 <01> 10",
        b"<02> bad <01> 10",
    ],
)
def test_character_blocks_keep_valid_records_and_drop_malformed_operands(kind, record):
    cmap = CMapDecoder(block(kind, record))
    lookup = cmap.mapped_cid if kind == b"cidchar" else cmap.mapped_notdef
    assert lookup(b"\1") == 10
    assert lookup(b"\2") is None


@pytest.mark.parametrize("kind", [b"cidrange", b"notdefrange"])
@pytest.mark.parametrize(
    "bad",
    [
        b"",
        b"<04>",
        b"bad bad 1",
        b"<zz> <03> 1",
        b"<03> <01> 1",
        b"<01> <03> bad",
        b"<01> <03> -1",
        b"<01> <03> 65536",
    ],
)
def test_range_blocks_retain_valid_triples_after_malformed_input(kind, bad):
    source = block(kind, bad) + block(kind, b"<01> <03> 10")
    cmap = CMapDecoder(source)
    lookup = cmap.mapped_cid if kind == b"cidrange" else cmap.mapped_notdef
    assert [lookup(bytes([i])) for i in (1, 2, 3)] == (
        [10, 11, 12] if kind == b"cidrange" else [10, 10, 10]
    )


def test_cid_range_overflow_is_rejected_but_notdef_range_is_constant():
    assert CMapDecoder(block(b"cidrange", b"<01> <03> 65535")).mapped_cid(b"\1") is None
    cmap = CMapDecoder(block(b"notdefrange", b"<01> <03> 65535"))
    assert cmap.mapped_notdef(b"\3") == 65535


@pytest.mark.parametrize(
    ("parent", "width", "mode"),
    [
        ("OneByteIdentityH", 1, 0),
        ("OneByteIdentityV", 1, 1),
        ("Identity-H", 2, 0),
        ("Identity-V", 2, 1),
    ],
)
def test_identity_parent_aliases_preserve_code_width_and_writing_mode(parent, width, mode):
    cmap = CMapDecoder(f"/{parent} usecmap".encode())
    encoded = (65).to_bytes(width, "big")
    assert cmap.decode_entries(encoded) == [(encoded, 65)]
    assert cmap.wmode == mode


@pytest.mark.parametrize("parent_data", [None, block(b"cidchar", b"<01> 10 <02> 20")])
def test_custom_parent_inheritance_preserves_child_override(parent_data):
    child = CMapDecoder(
        b"/Parent usecmap\n" + block(b"cidchar", b"<01> 30"),
        usecmap_resolver=lambda name: parent_data,
    )
    assert child.mapped_cid(b"\1") == 30
    assert child.mapped_cid(b"\2") == (None if parent_data is None else 20)


def test_named_parent_cycle_stops_without_discarding_local_mappings():
    data = b"/Loop usecmap\n" + block(b"cidchar", b"<01> 10")
    calls = []

    def resolve(name):
        calls.append(name)
        return data

    assert CMapDecoder(data, usecmap_resolver=resolve).mapped_cid(b"\1") == 10
    assert calls == ["Loop"]
    with pytest.raises(ValueError, match="nesting"):
        CMapDecoder(data, inheritance_depth=6)


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("font_size", [0, 12, -8])
@pytest.mark.parametrize("horizontal_scale", [0, 80, 120])
@pytest.mark.parametrize("spacing", [(0, 0), (1.5, 3)])
def test_text_advance_equals_ordered_sum_of_glyph_advances(
    vertical, font_size, horizontal_scale, spacing
):
    font = FontDecoder({"Subtype": "Type1", "BaseFont": "Helvetica"})
    font.is_vertical = vertical
    font.vertical_metrics = {65: (-700, 200, 800)}
    data = b"A B"
    glyphs = font.decode_glyphs(data)
    parameters = {
        "font_size": font_size,
        "char_space": spacing[0],
        "word_space": spacing[1],
        "horizontal_scale": horizontal_scale,
    }
    advances = [
        font.glyph_advance_vector(
            glyph.width_code, **parameters, encoded_space=glyph.code_bytes == b" "
        )
        for glyph in glyphs
    ]
    expected = tuple(sum(axis) for axis in zip(*advances, strict=True))
    assert font.text_advance_vector(data, **parameters) == pytest.approx(
        expected, rel=1e-14, abs=1e-14
    )
    assert font.text_advance_vector(memoryview(data), glyphs=glyphs, **parameters) == pytest.approx(
        expected, rel=1e-14, abs=1e-14
    )
    assert font.text_advance_vector(b"", **parameters) == (0, 0)


@pytest.mark.parametrize(
    "cmap",
    [
        None,
        CMapDecoder.identity(byte_width=1),
        CMapDecoder.identity(byte_width=2),
        ToUnicodeCMap(block(b"codespacerange", b"<8100> <81ff>")),
    ],
)
def test_code_splitting_preserves_all_bytes_including_outside_codespace_and_trailing_bytes(cmap):
    for data in (b"", b"A", b"\x81\x01B", b"\x81\x01\x81\x02"):
        chunks = split_code_bytes(data, cmap)
        assert b"".join(chunks) == data
        assert all(chunks)
    if cmap is None or cmap.decode_lengths == (1,):
        assert split_code_bytes(b"AB", cmap) == [b"A", b"B"]
    elif cmap.code_space_ranges == [(b"\0\0", b"\xff\xff")]:
        assert split_code_bytes(b"ABC", cmap) == [b"AB", b"C"]
    else:
        assert split_code_bytes(b"A\x81\1B", cmap) == [b"A", b"\x81\1", b"B"]


@pytest.mark.parametrize("limit", [None, 10, 100])
def test_single_code_mapping_uses_cid_mapping_and_excludes_long_codes(limit):
    text = ToUnicodeCMap(block(b"bfchar", b"<01> <0041> <0203> <0042> <040506> <0043>"))
    cmap = CMapDecoder(
        block(b"codespacerange", b"<01> <01> <0200> <02ff>")
        + block(b"cidchar", b"<01> 10 <0203> 20")
    )
    expected = {b"\1": (10, "A"), b"\2\3": (20, "B")}
    assert single_code_mapping(text, cmap, limit) == {
        code: value for code, value in expected.items() if limit is None or value[0] < limit
    }


@pytest.mark.parametrize("length", [float("inf"), float("-inf"), float("nan")])
def test_invalid_type1_length_metadata_does_not_prevent_font_recovery(length):
    from core_pdf_spec.s_07_syntax.stream import PdfStream

    font = FontDecoder(
        {
            "Subtype": "Type1",
            "BaseFont": "Helvetica",
            "FontDescriptor": {"FontFile": PdfStream({"Length1": length}, b"damaged program")},
        }
    )
    assert font.font_program is None
    assert font.decode_glyphs(b"A")[0].unicode == "A"


def test_a_short_string_is_decoded_once_and_shared() -> None:
    font = FontDecoder({"Subtype": "Type1", "BaseFont": "Helvetica"})
    first = font.decode_glyphs(b"AB")
    assert [glyph.unicode for glyph in first] == ["A", "B"]
    assert font.decode_glyphs(b"AB") is first
    assert font.decode_glyphs(memoryview(b"AB")) is first
    assert font.decode_glyphs(bytearray(b"AB")) is first


def test_long_strings_are_not_kept() -> None:
    font = FontDecoder({"Subtype": "Type1", "BaseFont": "Helvetica"})
    text = b"A" * (decoder_module.STRING_GLYPH_CACHE_MAX_BYTES + 1)
    assert "".join(glyph.unicode for glyph in font.decode_glyphs(text)) == text.decode()
    assert text not in font.string_glyph_cache


def test_the_string_cache_is_cleared_when_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(decoder_module, "STRING_GLYPH_CACHE_MAX_ENTRIES", 2)
    font = FontDecoder({"Subtype": "Type1", "BaseFont": "Helvetica"})
    for text in (b"A", b"B", b"C"):
        assert font.decode_glyphs(text)[0].unicode == text.decode()
    assert len(font.string_glyph_cache) <= 2


def test_a_cff_repair_that_changes_a_mapping_clears_cached_strings() -> None:
    from core_pdf_spec.s_07_syntax.stream import PdfStream

    to_unicode = (
        b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        b"1 begincodespacerange <0000> <FFFF> endcodespacerange\n"
        b"2 beginbfchar <0041> <FFFD> <0042> <0042> endbfchar\n"
        b"endcmap CMapName currentdict /CMap defineresource pop end end"
    )
    font = FontDecoder(
        {
            "Subtype": "Type0",
            "BaseFont": "X",
            "Encoding": "Identity-H",
            "DescendantFonts": [
                {
                    "Subtype": "CIDFontType0",
                    "BaseFont": "X",
                    "CIDSystemInfo": {"Registry": "Adobe", "Ordering": "Identity", "Supplement": 0},
                }
            ],
            "ToUnicode": PdfStream({}, to_unicode),
        }
    )

    class Repairs:
        answer: dict[bytes, str] = {}

        def repairs_for_codes(self, codes: Iterable[bytes]) -> dict[bytes, str]:
            list(codes)
            return dict(self.answer)

    repairs = Repairs()
    font.cff_unicode_repair_index = repairs  # ty: ignore[invalid-assignment]
    assert [glyph.unicode for glyph in font.decode_glyphs(b"\x00\x41")] == ["�"]
    assert [glyph.unicode for glyph in font.decode_glyphs(b"\x00\x41\x00\x42")] == ["�", "B"]
    # A later string repairs code 0041; the string cached before must not
    # keep answering with the old mapping.
    repairs.answer = {b"\x00\x41": "A"}
    assert [glyph.unicode for glyph in font.decode_glyphs(b"\x00\x42\x00\x41")] == ["B", "A"]
    assert [glyph.unicode for glyph in font.decode_glyphs(b"\x00\x41")] == ["A"]

"""Literal PDF font mappings and outline events."""

import pytest

from core_pdf.impl.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.spec.s_09_fonts.font_program import execute_type2_charstring
from core_pdf.impl.spec.s_09_fonts.glyphs import glyph_name_to_unicode
from core_pdf.impl.spec.s_09_fonts.widths import parse_font_widths


def test_mapping_records_retain_operand_order_and_incomplete_operands() -> None:
    from core_pdf.impl.spec.s_09_fonts.cmap_tokenizer import CMapProgram
    from core_pdf.impl.spec.s_09_fonts.cmap_tounicode import cmap_mapping_blocks

    program = CMapProgram.parse(
        b"2 beginbfchar <41> (endbfchar) <42> endbfchar "
        b"1 begincidrange <43> <44> 65 endcidrange "
        b"1 beginbfrange <45> <46> [<feff0041>] endbfrange"
    )
    blocks = list(cmap_mapping_blocks(program, include_cid_ranges=True))
    assert [block.operator for block in blocks] == [
        b"beginbfchar",
        b"begincidrange",
        b"beginbfrange",
    ]
    assert [block.trailing_operand_count for block in blocks] == [1, 0, 0]
    assert [
        (record.source, record.source_end, record.destination)
        for block in blocks
        for record in block.records()
    ] == [
        (b"<41>", None, b"(endbfchar)"),
        (b"<43>", b"<44>", b"65"),
        (b"<45>", b"<46>", b"[<feff0041>]"),
    ]
    assert [block.operator for block in cmap_mapping_blocks(program)] == [
        b"beginbfchar",
        b"beginbfrange",
    ]


def cmap(body: bytes, codespace: bytes = b"<00> <ff>") -> bytes:
    return b"1 begincodespacerange " + codespace + b" endcodespacerange " + body


def test_literal_tounicode_keeps_null_and_bom_characters() -> None:
    mapping = ToUnicodeCMap(cmap(b"2 beginbfchar <01> <0000> <02> <feff0041> endbfchar"))
    assert mapping.lookup(b"\x01") == "\x00"
    assert mapping.lookup(b"\x02") == "\ufeffA"
    assert mapping.lookup(b"\x03") is None


@pytest.mark.parametrize(
    ("name", "literal"),
    [
        ("A.alt", "A"),
        ("A_unknown", "A"),
        ("uni00410042", "AB"),
        ("uni004a", ""),
        ("u1F600", "\U0001f600"),
        ("uniD800", ""),
        ("unknown", ""),
    ],
)
def test_agl_name_rules(name: str, literal: str) -> None:
    assert glyph_name_to_unicode(name) == literal


def test_cid_metric_defaults_are_pdf_rules() -> None:
    metrics = parse_font_widths({"DescendantFonts": [{}]}, "Type0")
    assert metrics.default_width == 1000
    assert metrics.default_vertical_origin_y == 880
    assert metrics.default_vertical_displacement_y == -1000


def test_type2_execution_emits_curves_without_selecting_a_rasterizer() -> None:
    events: list[tuple[str, tuple[float, ...]]] = []
    active = False

    def move(x: float, y: float) -> None:
        nonlocal active
        active = True
        events.append(("move", (x, y)))

    def line(x: float, y: float) -> None:
        events.append(("line", (x, y)))

    def curve(a: float, b: float, c: float, d: float, e: float, f: float) -> None:
        events.append(("curve", (a, b, c, d, e, f)))

    def close() -> None:
        events.append(("close", ()))

    def seac(base: int, accent: int, x: float, y: float) -> None:
        events.append(("seac", (base, accent, x, y)))

    assert not execute_type2_charstring(
        bytes([139, 139, 21, 149, 159, 169, 179, 189, 199, 8, 14]),
        local_subrs=(),
        global_subrs=(),
        move=move,
        line=line,
        curve=curve,
        flush_contour=close,
        has_current_point=lambda: active,
        seac=seac,
        random_value=lambda: 0.5,
    )
    assert events == [("move", (0, 0)), ("curve", (10, 20, 30, 40, 50, 60)), ("close", ())]

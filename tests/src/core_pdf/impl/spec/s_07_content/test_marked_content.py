# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_07_content.marked_content import MarkedContentEntry
from tests.helpers.extract_fakes import text_run
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf


def test_actual_text_collects_geometry_but_keeps_first_run_metadata() -> None:
    first = text_run(
        "first",
        10,
        20,
        30,
        40,
        seqno=7,
        font_name="first-font",
        baseline=(10, 20, 30, 20),
        provenance=(("source", "first"),),
        confidence=0.9,
        fill_color=(1.0, 0.0, 0.0),
        visible=False,
    )
    second = text_run(
        "second",
        5,
        15,
        35,
        50,
        seqno=8,
        font_name="second-font",
        baseline=(5, 15, 35, 15),
        provenance=(("source", "second"),),
        confidence=0.5,
    )
    first_decoder = object()
    entry = MarkedContentEntry(actual_text="replacement")
    entry.add_run(first, font_decoder=first_decoder, effective_font_height=12)
    entry.add_run(second, font_decoder=object(), effective_font_height=24)

    assert entry.run is first
    assert (first.x0, first.y0, first.x1, first.y1) == (5, 15, 35, 50)
    assert first.advance_bbox == (5, 15, 35, 50)
    assert first.baseline == (10, 20, 35, 15)
    assert first.confidence == 0.5
    assert first.text == "first"
    assert first.seqno == 7
    assert first.font_name == "first-font"
    assert first.provenance == (("source", "first"),)
    assert first.fill_color == (1.0, 0.0, 0.0)
    assert not first.visible
    assert entry.font_decoder is first_decoder
    assert entry.effective_font_height == 12


def test_actual_text_replaces_a_multi_style_span_once() -> None:
    content = (
        b"BT /F1 12 Tf 10 40 Td "
        b"/Span << /ActualText (replacement) /MCID 4 >> BDC "
        b"(A) Tj 1 0 0 rg /F1 18 Tf (B) Tj EMC ET"
    )
    with open_pdf(one_page_pdf(content)) as document:
        program = document.pages[0].get_page_program()

    assert [run.text for run in program.runs] == ["replacement"]
    assert [glyph.text for glyph in program.glyphs] == ["replacement"]
    run = program.runs[0]
    glyph = program.glyphs[0]
    assert run.font_size == 12
    assert run.fill_color == (0.0, 0.0, 0.0)
    assert run.seqno == 0
    assert run.ink_bbox == run.advance_bbox == glyph.advance_bbox
    assert run.baseline is not None
    assert run.baseline[2] > run.baseline[0]
    assert ("mcid", 4) in run.provenance
    assert ("unicode_source", "actual_text") in run.provenance
    assert glyph.unicode_source == "actual_text"
    assert run.confidence == glyph.confidence == 1.0
    assert not run.glyph_clusters


@pytest.mark.parametrize(
    ("marked", "expected_runs", "expected_glyphs"),
    [
        (b"/Span << /ActualText (unused) >> BDC EMC", [], []),
        (b"/Span << /ActualText () >> BDC (A) Tj EMC", [""], [""]),
        (
            b"/Span << /ActualText (outer) >> BDC "
            b"/Span << /ActualText (inner) >> BDC (A) Tj EMC (B) Tj EMC",
            ["innerouter"],
            ["inner", "outer"],
        ),
    ],
)
def test_actual_text_preserves_empty_and_nested_scope_behavior(
    marked: bytes, expected_runs: list[str], expected_glyphs: list[str]
) -> None:
    with open_pdf(one_page_pdf(b"BT /F1 12 Tf 10 40 Td " + marked + b" ET")) as document:
        program = document.pages[0].get_page_program()
    assert [run.text for run in program.runs] == expected_runs
    assert [glyph.text for glyph in program.glyphs] == expected_glyphs

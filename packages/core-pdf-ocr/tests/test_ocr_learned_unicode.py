# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import replace

import pytest
from ocr_test_helpers.extract_fakes import text_run

from core_pdf.impl._impl.model.glyphs import GlyphCluster, GlyphObservation
from core_pdf.impl._impl.model.runs import TextRun
from core_pdf_ocr.impl.extract.capture import (
    internal_apply_learned_unicode_to_run,
    internal_capture_from_program,
)
from tests.helpers.pdf_bytes import HELVETICA, assemble_pdf, one_page_pdf, open_pdf, stream_obj


def internal_run(
    parts: tuple[tuple[bytes, str], ...],
    decoder: object,
    *,
    expanded: bool = False,
) -> TextRun:
    clusters: list[GlyphCluster] = []
    for index, (code, text) in enumerate(parts):
        box = (float(index * 10), 0.0, float(index * 10 + 10), 10.0)
        glyphs = tuple(
            GlyphObservation(
                fragment,
                box,
                box,
                index,
                code_bytes=code,
                font_decoder=decoder,
                unicode_source="identity",
                confidence=0.2,
            )
            for fragment in (tuple(text) if expanded else (text,))
        )
        clusters.append(GlyphCluster(index, text, glyphs, box, box, None, 0.2))
    return text_run("".join(text for _, text in parts), glyph_clusters=tuple(clusters))


@pytest.mark.parametrize(
    ("parts", "mapping", "expected"),
    [
        (((b"a", "A"), (b"b", "A")), {b"b": "B"}, "AB"),
        (((b"a", "A"), (b"b", "A"), (b"c", "A")), {b"a": "B", b"c": "C"}, "BAC"),
        (((b"a", "fi"), (b"b", "fi")), {b"b": "B"}, "fiB"),
    ],
)
def test_learned_unicode_consumes_unmapped_glyphs_before_replacements(
    parts: tuple[tuple[bytes, str], ...], mapping: dict[bytes, str], expected: str
) -> None:
    decoder = object()
    run = internal_run(parts, decoder)

    result = internal_apply_learned_unicode_to_run(run, {decoder: mapping})

    assert result.text == expected
    assert run.text == "".join(text for _, text in parts)
    assert result.glyph_clusters is run.glyph_clusters


def test_learned_unicode_replaces_one_expanded_source_glyph_once() -> None:
    decoder = object()
    run = internal_run(((b"a", "fi"), (b"b", "fi")), decoder, expanded=True)

    result = internal_apply_learned_unicode_to_run(run, {decoder: {b"b": "B"}})

    assert result.text == "fiB"
    assert tuple(glyph.text for glyph in run.glyph_clusters[1].glyphs) == ("f", "i")


@pytest.mark.parametrize("source", ["actual_text", "structure_actual_text"])
def test_learned_unicode_preserves_authoritative_actual_text(source: str) -> None:
    decoder = object()
    run = internal_run(((b"a", "A"),), decoder).replace(provenance=(("unicode_source", source),))

    assert internal_apply_learned_unicode_to_run(run, {decoder: {b"a": "B"}}) is run


def test_learned_unicode_preserves_inserted_spacing() -> None:
    decoder = object()
    original = internal_run(((b"a", "A"), (b"b", "A")), decoder)
    run = original.replace(text="A A", glyph_clusters=original.glyph_clusters)

    assert internal_apply_learned_unicode_to_run(run, {decoder: {b"b": "B"}}).text == "A B"


@pytest.mark.parametrize("text", ["missing A", "A missing", "AXA"])
def test_learned_unicode_leaves_incomplete_cluster_alignment_unchanged(text: str) -> None:
    decoder = object()
    original = internal_run(((b"a", "A"), (b"b", "A")), decoder)
    run = original.replace(text=text, glyph_clusters=original.glyph_clusters)

    assert internal_apply_learned_unicode_to_run(run, {decoder: {b"a": "B"}}) is run


@pytest.mark.parametrize("text_matrix", [b"1 0 0 1", b"-1 0 0 -1"])
def test_capture_applies_only_aligned_unicode_and_preserves_original_program(
    text_matrix: bytes,
) -> None:
    font = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding << /Type /Encoding /BaseEncoding /WinAnsiEncoding "
        b"/Differences [65 /A /A] >> >>"
    )
    content = b"BT /F1 12 Tf " + text_matrix + b" 20 40 Tm [(A) 0 (B)] TJ ET"
    with open_pdf(one_page_pdf(content, font=font)) as document:
        page = document.pages[0]
        program = page.get_page_program()
        assert program.runs[0].text == "AA"
        decoder = program.glyphs[0].font_decoder
        captured = internal_capture_from_program(
            page, program, learned_unicode={decoder: {b"B": "B"}}
        )

    assert captured.observations.text == ("AB",)
    assert captured.program is program
    assert program.runs[0].text == "AA"
    assert tuple(glyph.text for glyph in program.glyphs) == ("A", "A")


@pytest.mark.parametrize("text_matrix", [b"1 0 0 1", b"-1 0 0 -1"])
def test_capture_declines_overlay_when_repeated_clusters_hide_a_prepended_operation(
    text_matrix: bytes,
) -> None:
    font = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding << /Type /Encoding /BaseEncoding /WinAnsiEncoding "
        b"/Differences [65 /A /A] >> >>"
    )
    content = b"BT /F1 12 Tf " + text_matrix + b" 20 40 Tm [(A) 1334 (B)] TJ ET"
    with open_pdf(one_page_pdf(content, font=font)) as document:
        page = document.pages[0]
        program = page.get_page_program()
        assert tuple(run.text for run in program.runs) == ("AA",)
        baseline = internal_capture_from_program(page, program)
        decoder = program.glyphs[0].font_decoder
        captured = internal_capture_from_program(
            page, program, learned_unicode={decoder: {b"B": "B"}}
        )

    assert captured.observations.text == baseline.observations.text == ("AA",)
    assert captured.evidence.glyphs == baseline.evidence.glyphs


@pytest.mark.parametrize("replacement", ["A", "B"])
def test_aligned_unicode_confirmation_updates_glyph_evidence(replacement: str) -> None:
    with open_pdf(one_page_pdf(b"BT /F1 12 Tf 10 40 Td (A) Tj ET")) as document:
        page = document.pages[0]
        program = page.get_page_program()
        decoder = program.glyphs[0].font_decoder
        run = internal_run(((b"a", "A"),), decoder)
        program = replace(
            program,
            body=replace(program.body, runs=(run,), glyphs=run.glyph_clusters[0].glyphs),
        )
        captured = internal_capture_from_program(
            page, program, learned_unicode={decoder: {b"a": replacement}}
        )

    assert captured.observations.text == (replacement,)
    assert captured.evidence.glyphs.heuristic_glyphs == 1
    assert captured.evidence.glyphs.unknown_glyphs == 0
    assert captured.evidence.glyphs.low_confidence_glyphs == 0
    assert captured.evidence.glyphs.semantic_characters == 1


def test_failed_alignment_does_not_upgrade_glyph_evidence() -> None:
    with open_pdf(one_page_pdf(b"BT /F1 12 Tf 10 40 Td (A) Tj ET")) as document:
        page = document.pages[0]
        program = page.get_page_program()
        decoder = program.glyphs[0].font_decoder
        original = internal_run(((b"a", "A"),), decoder)
        run = original.replace(text="unrelated", glyph_clusters=original.glyph_clusters)
        program = replace(
            program,
            body=replace(program.body, runs=(run,), glyphs=run.glyph_clusters[0].glyphs),
        )
        captured = internal_capture_from_program(
            page, program, learned_unicode={decoder: {b"a": "B"}}
        )

    assert captured.observations.text == ("unrelated",)
    assert captured.evidence.glyphs.unknown_glyphs == 1
    assert captured.evidence.glyphs.low_confidence_glyphs == 1
    assert captured.evidence.glyphs.semantic_characters == 0


@pytest.mark.parametrize("expanded_text", ["fi", " fi"])
def test_capture_counts_an_expanded_glyph_replacement_once_and_keeps_unaffected_evidence(
    expanded_text: str,
) -> None:
    with open_pdf(one_page_pdf(b"BT /F1 12 Tf 10 40 Td (A) Tj ET")) as document:
        page = document.pages[0]
        program = page.get_page_program()
        decoder = program.glyphs[0].font_decoder
        run = internal_run(((b"a", "fi"), (b"b", expanded_text)), decoder, expanded=True)
        glyphs = tuple(glyph for cluster in run.glyph_clusters for glyph in cluster.glyphs)
        # Retain a raw glyph omitted from the extractable runs, as happens with
        # clipped or duplicate layers, without upgrading it through the overlay.
        excluded = replace(glyphs[-1], seqno=10)
        program = replace(
            program,
            body=replace(program.body, runs=(run,), glyphs=(*glyphs, excluded)),
        )
        captured = internal_capture_from_program(
            page, program, learned_unicode={decoder: {b"b": "B"}}
        )

    assert captured.observations.text == ("fiB",)
    evidence = captured.evidence.glyphs
    assert evidence.glyph_count == 4
    assert evidence.heuristic_glyphs == 1
    assert evidence.unknown_glyphs == 3
    assert evidence.low_confidence_glyphs == 3
    assert evidence.semantic_characters == 1


@pytest.mark.parametrize("structure_actual_text", [False, True])
def test_capture_preserves_actual_text_over_learned_font_mapping(
    structure_actual_text: bool,
) -> None:
    properties = b"/MCID 0" if structure_actual_text else b"/ActualText (A)"
    content = b"BT /F1 12 Tf 10 40 Td /Span << " + properties + b" >> BDC (A) Tj EMC ET"
    data = assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 6 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R /StructParents 0 >>",
            stream_obj(content),
            HELVETICA,
            b"<< /Type /StructTreeRoot /K [7 0 R] /ParentTree << /Nums [0 [7 0 R]] >> >>",
            b"<< /Type /StructElem /S /Span /P 6 0 R /Pg 3 0 R /K 0 /ActualText (A) >>",
        ]
    )
    with open_pdf(data) as document:
        page = document.pages[0]
        program = page.get_page_program()
        baseline = internal_capture_from_program(page, program)
        decoder = program.glyphs[0].font_decoder
        captured = internal_capture_from_program(
            page, program, learned_unicode={decoder: {b"A": "B"}}
        )

    assert captured.observations.text == ("A",)
    assert captured.evidence.glyphs == baseline.evidence.glyphs
    assert captured.evidence.glyphs.actual_text_characters == 1

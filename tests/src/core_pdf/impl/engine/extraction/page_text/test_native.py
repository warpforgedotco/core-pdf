from pathlib import Path
from typing import Any, cast

from core_layout.impl.layout.geometry_quality import (
    LayoutGeometrySummary,
    layout_geometry_should_trigger_ocr,
)
from core_layout.impl.layout.models import TextRun
from core_ocr.impl.native_text import (
    native_invisible_text_layer_has_fragmented_geometry,
    native_invisible_text_layer_is_trustworthy,
    native_text_runs_inside_page_bounds,
    should_try_rendered_glyph_repair,
    should_try_rendered_glyph_text,
    text_run_has_glyph_bitmap_repair_candidate,
)

from core_pdf.impl.engine.extraction.document import PdfDocument
from core_pdf.impl.engine.extraction.page_text.engine import build_page_extraction_result

TESTS_DIR = Path(__file__).parents[6]
SAMPLE_PDF = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "global-AIDS-strategy-p74-75-p001.pdf"


def text_run(text: str, x0: float, y0: float, x1: float, y1: float) -> TextRun:
    return TextRun(
        text,
        x0,
        y0,
        x1,
        y1,
        x0,
        y0,
        10.0,
        4.0,
        0,
        0,
        0,
    )


def test_page_bounds_keep_long_text_with_bad_horizontal_font_metrics() -> None:
    run = text_run(
        "A valid line whose reported width exceeds the page",
        60.0,
        100.0,
        1_200.0,
        112.0,
    )

    assert native_text_runs_inside_page_bounds(
        [run],
        (0.0, 0.0, 612.0, 792.0),
    ) == [run]


def test_page_bounds_still_drop_text_starting_outside_page() -> None:
    run = text_run(
        "A line entirely beyond the declared page",
        620.0,
        100.0,
        1_200.0,
        112.0,
    )

    assert not native_text_runs_inside_page_bounds(
        [run],
        (0.0, 0.0, 612.0, 792.0),
    )


def test_page_bounds_keep_zero_height_text_with_an_in_page_baseline() -> None:
    run = text_run("Valid text without vertical font metrics", 60.0, 100.0, 260.0, 100.0)

    assert native_text_runs_inside_page_bounds(
        [run],
        (0.0, 0.0, 612.0, 792.0),
    ) == [run]


def test_dense_native_text_is_not_rebuilt_from_rendered_glyphs() -> None:
    noisy_multicolumn_text = "index entry ........ 12 " * 200

    assert not should_try_rendered_glyph_text(noisy_multicolumn_text)


def test_uninterpretable_dense_native_text_can_use_rendered_glyphs() -> None:
    damaged_text = ("index entry ........ 12 " * 200) + "\ufffd"

    assert should_try_rendered_glyph_text(damaged_text)


def test_glyph_repair_preflight_requires_an_actionable_bitmap_label() -> None:
    ordinary = text_run("A", 0.0, 0.0, 10.0, 10.0)
    ordinary.confidence = 0.2
    suspicious = text_run("\ufffd", 0.0, 0.0, 10.0, 10.0)

    assert not text_run_has_glyph_bitmap_repair_candidate(
        ordinary,
        repair_contextual_punctuation=False,
    )
    assert text_run_has_glyph_bitmap_repair_candidate(
        suspicious,
        repair_contextual_punctuation=False,
    )
    assert not should_try_rendered_glyph_repair([ordinary], "A")
    assert should_try_rendered_glyph_repair([suspicious], "\ufffd")


def test_glyph_repair_preflight_keeps_contextual_punctuation_mode() -> None:
    punctuation = text_run("(", 0.0, 0.0, 10.0, 10.0)
    punctuation.confidence = 0.2

    assert not text_run_has_glyph_bitmap_repair_candidate(
        punctuation,
        repair_contextual_punctuation=False,
    )
    assert text_run_has_glyph_bitmap_repair_candidate(
        punctuation,
        repair_contextual_punctuation=True,
    )


def test_native_extraction_defers_ocr_geometry_summary_but_keeps_public_diagnostic() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])
        assert page.extract_text().strip()
        assert page.extraction_cache is not None
        assert "native_layout_geometry_summary" not in page.extraction_cache
        assert "page_region_classification" not in page.extraction_cache

        summary = page.extract_geometry_summary()
        assert summary["text_run_count"] > 0
        assert summary["line_count"] > 0


def test_structured_page_result_computes_deferred_region_classification() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])
        assert page.extract_text().strip()
        assert "page_region_classification" not in page.extraction_cache

        result = build_page_extraction_result(page)

        assert result.page_class == "native_text"
        classification = page.extraction_cache["page_region_classification"]
        assert classification.kind == "unknown"
        assert classification.signals["native_line_count"] > 0


def test_single_glyph_capture_shares_observation_with_cluster_and_preserves_provenance() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])
        state = page.capture_text_state()
        cluster = next(item for item in state.glyph_clusters if item.kind == "single_glyph")
        observation = cluster.glyphs[0]

        assert any(item is observation for item in state.glyphs)
        assert cluster.advance_bbox == observation.advance_bbox
        assert cluster.ink_bbox == observation.ink_bbox
        assert cluster.confidence == observation.confidence
        assert cluster.provenance is observation.provenance
        assert dict(observation.provenance)["source"] == "native_glyph"
        assert type(dict(observation.provenance)["decoded_glyph_index"]) is int


def test_sparse_geometry_issues_do_not_trigger_ocr_for_substantial_text() -> None:
    summary = LayoutGeometrySummary(
        issue_count=20,
        error_count=6,
        warning_count=14,
        repairable_count=20,
        text_run_count=619,
        line_count=37,
        issue_codes=(("glyph_cluster_text_mismatch", 6),),
        suspicion_score=68.0,
    )

    assert not layout_geometry_should_trigger_ocr(summary, text_tokens=324)


def test_isolated_glyph_mismatches_do_not_trigger_full_page_ocr() -> None:
    summary = LayoutGeometrySummary(
        issue_count=27,
        error_count=27,
        warning_count=0,
        repairable_count=27,
        text_run_count=425,
        line_count=90,
        issue_codes=(("glyph_cluster_text_mismatch", 27),),
        suspicion_score=148.5,
    )

    assert not layout_geometry_should_trigger_ocr(summary, text_tokens=783)


def test_dense_geometry_issues_still_trigger_ocr() -> None:
    summary = LayoutGeometrySummary(
        issue_count=40,
        error_count=6,
        warning_count=34,
        repairable_count=40,
        text_run_count=100,
        line_count=20,
        issue_codes=(("glyph_cluster_text_mismatch", 6),),
        suspicion_score=68.0,
    )

    assert layout_geometry_should_trigger_ocr(summary, text_tokens=324)


def test_fragmented_invisible_layer_is_not_trusted() -> None:
    runs = [
        TextRun(
            "x",
            float(index),
            10.0,
            float(index + 1),
            20.0,
            float(index),
            10.0,
            10.0,
            4.0,
            index,
            index,
            0,
            visible=False,
            provenance=(("text_render_mode", 3),),
        )
        for index in range(600)
    ]
    summary = LayoutGeometrySummary(
        issue_count=0,
        error_count=0,
        warning_count=0,
        repairable_count=0,
        text_run_count=600,
        line_count=20,
        issue_codes=(),
        suspicion_score=0.0,
    )
    text = "receipt item total " * 50

    assert native_invisible_text_layer_has_fragmented_geometry(runs, text, summary)
    assert not native_invisible_text_layer_is_trustworthy(runs, text, summary)


def test_word_runs_are_not_mistaken_for_fragmented_invisible_glyphs() -> None:
    runs = [
        TextRun(
            "word",
            float(index),
            10.0,
            float(index + 4),
            20.0,
            float(index),
            10.0,
            10.0,
            4.0,
            index,
            index,
            0,
            visible=False,
            provenance=(("text_render_mode", 3),),
        )
        for index in range(600)
    ]
    summary = LayoutGeometrySummary(
        issue_count=600,
        error_count=0,
        warning_count=600,
        repairable_count=600,
        text_run_count=600,
        line_count=20,
        issue_codes=(("low_confidence_repairable_glyph", 600),),
        suspicion_score=1_500.0,
    )

    assert not native_invisible_text_layer_has_fragmented_geometry(
        runs,
        "receipt item total " * 50,
        summary,
    )

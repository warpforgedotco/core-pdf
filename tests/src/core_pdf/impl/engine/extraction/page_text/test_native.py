from core_pdf.impl.engine.extraction.page_text.native import (
    native_invisible_text_layer_has_fragmented_geometry,
    native_invisible_text_layer_is_trustworthy,
    native_text_runs_inside_page_bounds,
)
from core_pdf.impl.engine.layout.geometry_quality import (
    LayoutGeometrySummary,
    layout_geometry_should_trigger_ocr,
)
from core_pdf.impl.engine.layout.models import TextRun


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

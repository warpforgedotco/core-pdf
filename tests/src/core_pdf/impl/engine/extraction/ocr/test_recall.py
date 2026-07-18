from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from core_ocr.impl import postprocess, selection
from core_ocr.impl.candidates import OcrCandidate, OcrPageTextResult
from core_ocr.impl.policy import (
    fragmented_invisible_text_layer_should_yield_to_ocr,
    should_replace_dominant_image_native_text_with_ocr,
    sparse_drawing_schematic_should_yield_to_ocr,
)
from core_ocr.impl.types import OcrImage, OcrTextResult

from core_pdf.impl.engine.extraction.common import page_geometry
from core_pdf.impl.engine.extraction.common.observation_resolver import ResolvedTextLine
from core_pdf.impl.engine.extraction.page_text import mixin


def test_sparse_vector_page_triggers_full_page_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        postprocess,
        "page_has_vector_stroke_text_candidates",
        lambda _page: True,
    )
    monkeypatch.setattr(
        postprocess,
        "native_text_layer_looks_reliable_enough",
        lambda *_args, **_kwargs: False,
    )

    assert postprocess.should_ocr_fallback(cast(Any, object()), "A B C")


def test_fully_covered_table_fusion_lines_are_pruned() -> None:
    geometry = tuple(
        ResolvedTextLine(
            f"G{index}",
            cast(
                page_geometry.PageObservation,
                page_geometry.page_observation_from_bbox(
                    (float(index), 10.0, float(index + 1), 20.0),
                    source="figure_ocr_regions",
                    kind="ocr_textline",
                    text=f"G{index}",
                    confidence=90,
                ),
            ),
        )
        for index in range(30)
    )
    fusion = ResolvedTextLine(
        "G0 G1 G2 G3 G4 G5 G6 G7 G8 G9",
        page_geometry.PageObservation(
            kind="ocr_textline",
            source="table_fusion_text",
            bbox=None,
            advance_bbox=None,
            ink_bbox=None,
            text="G0 G1 G2 G3 G4 G5 G6 G7 G8 G9",
        ),
    )
    assert all(line.observation is not None for line in geometry)
    assert postprocess.prune_fully_covered_fusion_lines(geometry + (fusion,)) == geometry


def test_vector_page_with_substantial_native_title_skips_full_page_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        postprocess,
        "page_has_vector_stroke_text_candidates",
        lambda _page: True,
    )
    monkeypatch.setattr(
        postprocess,
        "native_text_layer_looks_reliable_enough",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        postprocess,
        "layout_geometry_should_trigger_ocr",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        postprocess.ocr_text_analysis,
        "sparse_text_looks_noisy",
        lambda _text: False,
    )
    monkeypatch.setattr(
        postprocess.ocr_page_analysis,
        "has_dominant_page_image",
        lambda _page: False,
    )
    monkeypatch.setattr(
        postprocess.ocr_page_analysis,
        "has_uninterpretable_type3_fonts",
        lambda _page: False,
    )

    native_title = (
        "SPUR REPORT TRANSPORTATION The Future of Transportation Harnessing private "
        "mobility services to support the public good JULY 2020"
    )

    assert not postprocess.should_ocr_fallback(cast(Any, object()), native_title)


def test_sparse_drawing_schematic_yields_to_cleaner_ocr() -> None:
    native_profile = SimpleNamespace(
        drawing_line_count=4_323,
        candidate_schematic_signals=149,
    )
    ocr_profile = SimpleNamespace(confidence=60)

    assert sparse_drawing_schematic_should_yield_to_ocr(
        "A B C D",
        "R1 100k R2 10k GND OUT IN VCC " * 8,
        text_tokens=4,
        ocr_tokens=64,
        native_profile=cast(Any, native_profile),
        ocr_profile=cast(Any, ocr_profile),
    )


def test_fragmented_invisible_layer_yields_to_clean_full_page_ocr() -> None:
    assert fragmented_invisible_text_layer_should_yield_to_ocr(
        "broken hidden receipt words " * 30,
        "hotel receipt date room charge tax total " * 25,
        92,
        native_layer_is_fragmented=True,
    )


def test_fragmented_invisible_layer_rejects_low_confidence_ocr() -> None:
    assert not fragmented_invisible_text_layer_should_yield_to_ocr(
        "broken hidden receipt words " * 30,
        "hotel receipt date room charge tax total " * 25,
        62,
        native_layer_is_fragmented=True,
    )


def test_clean_ocr_does_not_replace_unfragmented_invisible_layer() -> None:
    assert not fragmented_invisible_text_layer_should_yield_to_ocr(
        "native technical table values " * 30,
        "technical table values volume temperature pressure " * 25,
        92,
        native_layer_is_fragmented=False,
    )


def test_image_only_complex_layout_gets_alternate_ocr_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = SimpleNamespace(
        get_page_profile=lambda: SimpleNamespace(
            recommended_strategy="image_or_ocr",
            has_text_showing_ops=False,
        )
    )
    image = OcrImage(b"", 100, 100, 0, 0, source="full_page_encoded_image")
    result = OcrTextResult(
        "poster text",
        70,
        line_rows=tuple({} for _ in range(20)),
        word_rows=tuple({} for _ in range(100)),
    )
    candidate = OcrCandidate("full_page_simple", result)
    monkeypatch.setattr(mixin, "extracted_text_token_count", lambda _text: 486)
    monkeypatch.setattr(mixin, "text_ocr_quality_score", lambda _text: 0.22)

    assert mixin.should_try_image_only_layout_ocr_candidate(
        cast(Any, page),
        image,
        candidate,
    )


def test_non_image_page_does_not_get_alternate_layout_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = SimpleNamespace(
        get_page_profile=lambda: SimpleNamespace(
            recommended_strategy="image_or_ocr",
            has_text_showing_ops=False,
        )
    )
    image = OcrImage(b"", 100, 100, 0, 0, source="rendered_page_300dpi")
    candidate = OcrCandidate(
        "rendered_page_300dpi",
        OcrTextResult(
            "page text",
            70,
            line_rows=tuple({} for _ in range(20)),
            word_rows=tuple({} for _ in range(100)),
        ),
    )
    monkeypatch.setattr(mixin, "extracted_text_token_count", lambda _text: 486)
    monkeypatch.setattr(mixin, "text_ocr_quality_score", lambda _text: 0.22)

    assert not mixin.should_try_image_only_layout_ocr_candidate(
        cast(Any, page),
        image,
        candidate,
    )


def test_dense_dominant_image_gets_high_resolution_sparse_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "\n".join(f"row {index} 100 200 300 400 500 600" for index in range(40))
    result = OcrTextResult(
        text,
        68,
        line_rows=tuple({} for _ in range(40)),
        word_rows=tuple({} for _ in range(280)),
    )
    candidate = OcrCandidate("full_page_simple", result)
    image = OcrImage(
        b"\xff" * (1_600 * 1_200),
        1_600,
        1_200,
        1,
        1_600,
        source="full_page_rendered_crop",
        resolution=300,
    )
    monkeypatch.setattr(
        mixin.ocr_page_analysis,
        "has_dominant_page_image",
        lambda _page: True,
    )

    assert mixin.should_try_dense_image_sparse_ocr_candidate(
        cast(Any, object()),
        image,
        candidate,
    )
    scaled = mixin.dense_image_sparse_ocr_image(image)
    assert scaled is not None
    assert max(scaled.target_width or 0, scaled.target_height or 0) == 8_192
    assert (scaled.target_width or 0) * (scaled.target_height or 0) <= 55_000_000


def test_cleaner_high_resolution_sparse_candidate_wins_dense_near_tie() -> None:
    primary_text = "\n".join(f"row {index} 100 200 300 400 500 600 extra" for index in range(40))
    sparse_text = "\n".join(f"row {index} 100 200 300 400 500 600" for index in range(40))
    primary = OcrCandidate(
        "full_page_simple",
        OcrTextResult(primary_text, 70),
    )
    sparse = OcrCandidate(
        "full_page_high_resolution_sparse",
        OcrTextResult(sparse_text, 82),
    )

    assert selection.select_ocr_candidate([primary, sparse]) is sparse


def test_sparse_native_footer_yields_to_substantial_dominant_image_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mixin.ocr_page_analysis,
        "has_dominant_page_image",
        lambda _page: True,
    )
    native = "Source: https://example.test/document"
    ocr = "\n".join(f"measurement row {index} value {100 + index}.5 valid" for index in range(35))

    assert should_replace_dominant_image_native_text_with_ocr(
        cast(Any, object()),
        native,
        ocr,
    )


def test_substantial_numeric_page_ocr_wins_over_partial_figure_ocr() -> None:
    broad_candidate = OcrCandidate(
        "full_page_simple",
        OcrTextResult("row 100 200 300 400 500 600 " * 45, 93),
    )
    figure_candidate = OcrCandidate(
        "figure_ocr_regions",
        OcrTextResult("row 100 200 300 400 " * 30, 94),
    )

    assert mixin.broad_page_ocr_should_win_over_figure_ocr(
        OcrPageTextResult(broad_candidate.result.text, broad_candidate),
        OcrPageTextResult(figure_candidate.result.text, figure_candidate),
    )


def test_substantial_prose_page_ocr_wins_over_partial_figure_ocr() -> None:
    broad_candidate = OcrCandidate(
        "full_page_simple",
        OcrTextResult("message body contains complete context " * 60, 85),
    )
    figure_candidate = OcrCandidate(
        "figure_ocr_regions",
        OcrTextResult("message body contains complete context " * 52, 90),
    )

    assert mixin.broad_page_ocr_should_win_over_figure_ocr(
        OcrPageTextResult(broad_candidate.result.text, broad_candidate),
        OcrPageTextResult(figure_candidate.result.text, figure_candidate),
    )


def test_dense_sparse_layout_render_cannot_drop_repeated_table_cells() -> None:
    candidate_text = " ".join(["N"] * 120 + ["0"] * 120 + ["200"] * 120)
    rendered_text = " ".join(["N"] * 80 + ["0"] * 80 + ["200"] * 80)
    candidate = OcrCandidate(
        "full_page_high_resolution_sparse",
        OcrTextResult(candidate_text, 82),
    )

    assert mixin.dense_sparse_layout_render_drops_material_text(
        candidate,
        candidate_text,
        rendered_text,
    )


def test_clean_full_page_ocr_can_preserve_substantial_raw_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = OcrCandidate(
        "full_page_simple",
        OcrTextResult("clean page text " * 120, 90),
    )
    monkeypatch.setattr(mixin, "text_ocr_quality_score", lambda _text: 0.08)
    monkeypatch.setattr(
        mixin.ocr_text_analysis,
        "scanned_ocr_artifact_score",
        lambda _text: 0.02,
    )

    assert mixin.clean_full_page_ocr_should_preserve_raw_text(
        candidate,
        candidate.result.text,
    )


def test_dense_sparse_loss_marks_raw_text_as_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = " ".join(["N"] * 120 + ["0"] * 120 + ["200"] * 120)
    candidate = OcrCandidate(
        "full_page_high_resolution_sparse",
        OcrTextResult(text, 82),
    )
    monkeypatch.setattr(mixin.ocr_postprocess, "ocr_is_enabled", lambda: True)
    monkeypatch.setattr(mixin.ocr_rendering, "ocr_timeout_seconds", lambda: None)
    monkeypatch.setattr(
        mixin,
        "collect_ocr_candidates",
        lambda *_args, **_kwargs: [candidate],
    )
    monkeypatch.setattr(
        mixin,
        "repair_ocr_output_lines_with_alternate_candidates",
        lambda *_args, **_kwargs: (),
    )

    result = mixin.extract_ocr_page_result(cast(Any, object()))

    assert result.preserve_raw_text
    assert result.text == text
    assert not result.output_lines


def test_dense_table_rows_restore_missing_repeated_categories() -> None:
    selected = OcrCandidate(
        "full_page_high_resolution_sparse",
        OcrTextResult("Monthly max 140.6 151.8", 82),
    )
    rows = OcrCandidate(
        "dense_table_rows",
        OcrTextResult("Max Off Off Off Off Off Off 140.6 151.8", 74),
    )

    supplemented = mixin.dense_table_categorical_token_supplement(
        selected.result.text,
        selected,
        (selected, rows),
    )

    assert supplemented.splitlines()[-1] == "Off Off Off Off Off Off"


def test_dense_table_row_rectangles_ignore_header_and_keep_wide_lower_row() -> None:
    image = OcrImage(b"", 1_000, 1_000, 0, 0)
    header: dict[str, object] = {
        "text": "Reporting of monitored emissions for June 2024 with descriptive title",
        "left": 20,
        "top": 100,
        "width": 900,
        "height": 30,
    }
    table_row: dict[str, object] = {
        "text": "Max Off Off Off Off 140.6 151.8 148.6 146.7 151.4 155.7 153.7",
        "left": 100,
        "top": 600,
        "width": 800,
        "height": 24,
    }

    rectangles = mixin.dense_image_table_row_rectangles(
        image,
        OcrTextResult("", 70, line_rows=(header, table_row)),
    )

    assert rectangles == [(88, 596, 912, 628)]


def test_image_only_layout_supplements_high_confidence_top_line() -> None:
    page = SimpleNamespace(
        media_box=(0.0, 0.0, 100.0, 100.0),
        get_page_profile=lambda: SimpleNamespace(
            recommended_strategy="image_or_ocr",
            has_text_showing_ops=False,
        ),
    )
    body_observation = page_geometry.page_observation_from_bbox(
        (5.0, 50.0, 95.0, 60.0),
        source="full_page_simple",
        kind="ocr_textline",
        text="Known body",
        confidence=90,
    )
    assert body_observation is not None
    lines = (ResolvedTextLine("Known body", body_observation),)
    primary = OcrCandidate("full_page_simple", OcrTextResult("Known body", 90))
    verification = OcrCandidate(
        "verification_full_page_simple_psm4",
        OcrTextResult(
            "POW-WOW",
            95,
            line_rows=(
                {
                    "text": "POW-WOW",
                    "conf": 95,
                    "page_bbox": (20.0, 80.0, 80.0, 90.0),
                },
            ),
        ),
    )
    broad = OcrPageTextResult(
        "Known body",
        primary,
        (primary,),
        verification_candidates=(verification,),
    )

    supplemented = mixin.supplement_image_only_layout_top_lines(
        cast(Any, page),
        lines,
        broad_page_result=broad,
    )

    assert [line.text for line in supplemented] == ["POW-WOW", "Known body"]


def test_image_only_layout_supplements_prominent_consensus_title() -> None:
    page = SimpleNamespace(
        media_box=(0.0, 0.0, 100.0, 100.0),
        get_page_profile=lambda: SimpleNamespace(
            recommended_strategy="image_or_ocr",
            has_text_showing_ops=False,
        ),
    )
    body_observation = page_geometry.page_observation_from_bbox(
        (5.0, 50.0, 95.0, 60.0),
        source="full_page_simple",
        kind="ocr_textline",
        text="Known body",
        confidence=90,
    )
    assert body_observation is not None
    lines = (ResolvedTextLine("Known body", body_observation),)
    primary = OcrCandidate(
        "full_page_simple",
        OcrTextResult("Known body\nPOW-WOW", 70),
    )
    verification = OcrCandidate(
        "verification_full_page_simple_psm4",
        OcrTextResult(
            "POW-WOW",
            80,
            line_rows=(
                {
                    "text": "POW-WOW",
                    "conf": 46,
                    "page_bbox": (10.0, 82.0, 90.0, 96.0),
                },
            ),
        ),
    )
    broad = OcrPageTextResult(
        primary.result.text,
        primary,
        (primary,),
        verification_candidates=(verification,),
    )

    supplemented = mixin.supplement_image_only_layout_top_lines(
        cast(Any, page),
        lines,
        broad_page_result=broad,
    )

    assert [line.text for line in supplemented] == ["POW-WOW", "Known body"]

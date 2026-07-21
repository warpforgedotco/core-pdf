from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

from core_ocr.impl import coordinator, policy
from core_ocr.impl.candidates import OcrCandidate, OcrPageTextResult
from core_ocr.impl.coordinator import (
    should_expand_weak_full_page_ocr_candidates,
    should_preserve_sparse_text_table_ocr_result,
)
from core_ocr.impl.policy import (
    OcrCandidateGeometryProfile,
    PageTextGeometryProfile,
    should_replace_dominant_image_native_text_with_ocr,
    tiny_native_text_should_yield_to_ocr,
)
from core_ocr.impl.types import OcrTextResult


def test_portrait_raster_formula_noise_prefers_dense_table_route(
    monkeypatch,
) -> None:
    geometry = PageTextGeometryProfile(
        page_width=600,
        page_height=900,
        native_run_count=0,
        native_line_count=0,
        wide_line_ratio=0.0,
        short_line_ratio=0.0,
        centered_line_ratio=0.0,
        numeric_line_ratio=0.0,
        left_anchor_count=0,
        right_anchor_count=0,
        estimated_column_count=0,
        native_aligned_column_count=0,
        candidate_aligned_column_count=2,
        candidate_table_signals=10,
        candidate_schematic_signals=220,
        drawing_line_count=0,
        horizontal_rule_count=0,
        vertical_rule_count=0,
        dominant_image=True,
        occupied_area_ratio=0.0,
    )
    monkeypatch.setattr(policy, "page_text_geometry_profile", lambda *args, **kwargs: geometry)
    monkeypatch.setattr(policy, "formula_heavy_ocr_text", lambda text: True)

    classification = policy.classify_page_region("noisy formula text")

    assert classification.kind == "dense_table"


def test_landscape_raster_formula_noise_keeps_technical_route(
    monkeypatch,
) -> None:
    geometry = PageTextGeometryProfile(
        page_width=900,
        page_height=600,
        native_run_count=0,
        native_line_count=0,
        wide_line_ratio=0.0,
        short_line_ratio=0.0,
        centered_line_ratio=0.0,
        numeric_line_ratio=0.0,
        left_anchor_count=0,
        right_anchor_count=0,
        estimated_column_count=0,
        native_aligned_column_count=0,
        candidate_aligned_column_count=0,
        candidate_table_signals=0,
        candidate_schematic_signals=0,
        drawing_line_count=0,
        horizontal_rule_count=0,
        vertical_rule_count=0,
        dominant_image=True,
        occupied_area_ratio=0.0,
    )
    monkeypatch.setattr(policy, "page_text_geometry_profile", lambda *args, **kwargs: geometry)
    monkeypatch.setattr(policy, "formula_heavy_ocr_text", lambda text: True)

    classification = policy.classify_page_region("noisy formula text")

    assert classification.kind == "patent_formula"


def test_two_line_corrupt_native_layer_yields_to_ocr() -> None:
    native = PageTextGeometryProfile(
        page_width=612,
        page_height=792,
        native_run_count=3,
        native_line_count=2,
        wide_line_ratio=0.0,
        short_line_ratio=1.0,
        centered_line_ratio=0.0,
        numeric_line_ratio=0.5,
        left_anchor_count=0,
        right_anchor_count=0,
        estimated_column_count=1,
        native_aligned_column_count=0,
        candidate_aligned_column_count=0,
        candidate_table_signals=0,
        candidate_schematic_signals=0,
        drawing_line_count=0,
        horizontal_rule_count=0,
        vertical_rule_count=0,
        dominant_image=False,
        occupied_area_ratio=0.02,
    )
    ocr = OcrCandidateGeometryProfile(
        line_count=58,
        word_count=158,
        aligned_column_count=1,
        occupied_area_ratio=0.10,
        wide_line_ratio=0.1,
        short_line_ratio=0.3,
        line_coverage_score=0.7,
        confidence=84,
        text_quality=0.27,
        artifact_score=0.16,
        gibberish_score=0.0,
    )

    assert tiny_native_text_should_yield_to_ocr(
        "(1 (1 -h",
        "ARCHITECTURE ANALYSIS " * 30,
        text_tokens=3,
        ocr_tokens=90,
        native_profile=native,
        ocr_profile=ocr,
    )


def test_medium_corrupt_native_image_layer_yields_to_ocr(
    monkeypatch,
) -> None:
    monkeypatch.setattr(policy.ocr_page_analysis, "has_dominant_page_image", lambda page: True)
    native_text = "i " * 79
    ocr_text = "Source industry document reference " * 100

    assert should_replace_dominant_image_native_text_with_ocr(
        object(),
        native_text,
        ocr_text,
    )


@dataclass
class _NativeTextPage:
    recommended_strategy: str = "native_text"

    def get_page_profile(self) -> Any:
        return self


@dataclass
class _ImagePage:
    recommended_strategy: str = "image"

    def get_page_profile(self) -> Any:
        return self


def test_sparse_native_table_ocr_preserves_selected_raw_result() -> None:
    candidate = OcrCandidate("full_page_simple", OcrTextResult("clean table " * 190, 69))
    result = OcrPageTextResult(candidate.result.text, candidate=candidate)

    assert should_preserve_sparse_text_table_ocr_result(
        cast(Any, _NativeTextPage()),
        "native table " * 90,
        result,
        "ocr_replace_general",
    )


def test_dense_low_confidence_full_page_ocr_expands_layout_search(monkeypatch) -> None:
    monkeypatch.setattr(coordinator, "text_ocr_quality_score", lambda text: 0.10)
    candidate = OcrCandidate("full_page_simple", OcrTextResult("table " * 1_000, 61))

    assert should_expand_weak_full_page_ocr_candidates(
        cast(Any, _NativeTextPage()),
        cast(Any, SimpleNamespace(source="full_page_image")),
        candidate,
    )


def test_empty_image_page_preserves_bounded_full_page_ocr() -> None:
    candidate = OcrCandidate("full_page_simple", OcrTextResult("label 12 " * 200, 60))
    result = OcrPageTextResult(candidate.result.text, candidate=candidate)

    assert should_preserve_sparse_text_table_ocr_result(
        cast(Any, _ImagePage()),
        "",
        result,
        "ocr_replace_dominant_image",
    )

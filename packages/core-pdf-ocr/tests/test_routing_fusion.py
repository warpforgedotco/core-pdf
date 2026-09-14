from dataclasses import replace
from typing import Any

import pytest

from core_pdf.impl._impl.extract.contracts import ObservationBatch, TextQualityStats
from core_pdf_ocr.impl.extract.contracts import (
    FusionPolicy,
    OcrPass,
    OcrPassScope,
    PageAnalysis,
    PagePlanReason,
    PageRoute,
    StrokedVectorTextEvidence,
    WorkPlan,
)
from core_pdf_ocr.impl.extract.observations import fuse_observations, plan_page


@pytest.mark.parametrize(
    ("overrides", "route", "reason"),
    [
        ({}, PageRoute.OCR, PagePlanReason.NATIVE_TEXT_UNAVAILABLE),
        (
            {"native_characters": 20, "visible_native_characters": 20},
            PageRoute.NATIVE,
            PagePlanReason.CLEAN_SHORT_NATIVE_TEXT,
        ),
        (
            {"native_characters": 80, "visible_native_characters": 80},
            PageRoute.NATIVE,
            PagePlanReason.HEALTHY_NATIVE_TEXT,
        ),
        (
            {"native_characters": 40, "visible_native_characters": 40, "suspicious_characters": 1},
            PageRoute.NATIVE,
            PagePlanReason.USABLE_NATIVE_TEXT,
        ),
        ({"vector_text_trusted": True}, PageRoute.NATIVE, PagePlanReason.NEWSTROKE_VECTOR_TEXT),
        (
            {"trusted_hidden_text": True},
            PageRoute.NATIVE,
            PagePlanReason.TRUSTED_HIDDEN_NATIVE_TEXT,
        ),
        (
            {"native_characters": 200, "visible_native_characters": 10},
            PageRoute.OCR,
            PagePlanReason.UNPAINTED_NATIVE_TEXT_LAYER,
        ),
        (
            {"stroked_vector_text": StrokedVectorTextEvidence(trusted=True)},
            PageRoute.OCR,
            PagePlanReason.STROKED_VECTOR_TEXT,
        ),
        (
            {
                "stroked_vector_text": StrokedVectorTextEvidence(trusted=True),
                "native_characters": 10,
                "visible_native_characters": 10,
            },
            PageRoute.HYBRID,
            PagePlanReason.STROKED_VECTOR_TEXT,
        ),
        ({"uncovered_vector_area": 30_000}, PageRoute.OCR, PagePlanReason.UNCOVERED_VECTOR_TEXT),
        (
            {"native_characters": 10, "visible_native_characters": 10, "image_count": 1},
            PageRoute.HYBRID,
            PagePlanReason.NATIVE_TEXT_NEEDS_AUGMENTATION,
        ),
        (
            {
                "native_characters": 100,
                "visible_native_characters": 100,
                "text_quality": TextQualityStats(short_token_ratio=1, wordlike_ratio=0.1),
            },
            PageRoute.OCR,
            PagePlanReason.NATIVE_TEXT_CORRUPT,
        ),
    ],
)
def test_routing_uses_page_evidence(
    ocr_capture: PageAnalysis, overrides: dict[str, Any], route: PageRoute, reason: PagePlanReason
) -> None:
    capture = replace(ocr_capture, evidence=replace(ocr_capture.evidence, **overrides))
    plan = plan_page(capture)
    assert plan.route is route
    assert plan.reason is reason
    assert bool(plan.ocr_passes) is (route is not PageRoute.NATIVE)


def test_corrupt_mapping_takes_priority_over_vector_trust(ocr_capture: PageAnalysis) -> None:
    evidence = replace(
        ocr_capture.evidence,
        native_characters=24,
        visible_native_characters=24,
        text_quality=TextQualityStats(short_token_ratio=1, wordlike_ratio=0.19),
        vector_text_trusted=True,
    )
    assert (
        plan_page(replace(ocr_capture, evidence=evidence)).reason
        is PagePlanReason.NATIVE_TEXT_CORRUPT
    )
    assert (
        plan_page(replace(ocr_capture, evidence=replace(evidence, vector_complexity=150))).reason
        is PagePlanReason.NEWSTROKE_VECTOR_TEXT
    )


def batch(
    text: tuple[str, ...],
    *,
    source: int = 0,
    confidence: tuple[float, ...] | None = None,
    offset: float = 0,
) -> ObservationBatch:
    return ObservationBatch.from_columns(
        text,
        ((offset, i * 20, offset + 10, i * 20 + 10) for i in range(len(text))),
        source=source,
        confidence=confidence,
    )


@pytest.mark.parametrize("route", list(PageRoute))
def test_fusion_empty_inputs_and_route_identity(route: PageRoute) -> None:
    native = batch(("native",))
    ocr = batch(("recognized",), source=1, confidence=(99,))
    plan = WorkPlan(route)
    assert fuse_observations(native, ObservationBatch.empty(), plan) is native
    empty = ObservationBatch.empty()
    assert fuse_observations(empty, ocr, plan) is (ocr if route is not PageRoute.NATIVE else empty)


@pytest.mark.parametrize(
    ("policy", "threshold"),
    [
        (FusionPolicy.DEFAULT, 45),
        (FusionPolicy.UNCOVERED_VECTOR, 30),
        (FusionPolicy.NOISY_NATIVE, 90),
    ],
)
def test_fusion_confidence_thresholds_preserve_native_text(
    policy: FusionPolicy, threshold: int
) -> None:
    native = batch(("Original sentence",))
    ocr = batch(
        ("Below threshold", "At threshold", "Unknown confidence"),
        source=1,
        confidence=(threshold - 0.01, threshold, float("nan")),
        offset=100,
    )
    result = fuse_observations(native, ocr, WorkPlan(PageRoute.HYBRID, fusion_policy=policy))
    assert result.text == ("Original sentence", "At threshold")
    assert tuple(result.source) == (0, 1)


def test_fusion_filters_duplicates_overlap_and_nontext_image_supplements() -> None:
    native = batch(("Original sentence",))
    ocr = ObservationBatch.from_columns(
        ("Original sentence", "Original sentence", "new overlapping", "...", "Novel text"),
        ((0, 0, 10, 10), (100, 0, 110, 10), (0, 0, 10, 10), (120, 0, 130, 10), (140, 0, 150, 10)),
        source=1,
        confidence=(99, 99, 99, 99, 99),
    )
    normal = fuse_observations(native, ocr, WorkPlan(PageRoute.HYBRID))
    assert normal.text == ("Original sentence", "...", "Novel text")
    plan = WorkPlan(
        PageRoute.HYBRID, ocr_passes=(OcrPass("image", OcrPassScope.IMAGE_REGIONS, 1, (3,)),)
    )
    assert fuse_observations(native, ocr, plan).text == (
        "Original sentence",
        "Original sentence",
        "Novel text",
    )


def test_sparse_native_fusion_replaces_fragmentary_text() -> None:
    native = batch(("x",))
    ocr = batch(("One", "Two", "Three", "Four"), source=1, confidence=(99,) * 4, offset=100)
    assert (
        fuse_observations(
            native, ocr, WorkPlan(PageRoute.HYBRID, fusion_policy=FusionPolicy.SPARSE_NATIVE)
        )
        is ocr
    )


@pytest.mark.parametrize(
    ("characters", "coverage", "images", "full_page", "area", "reason"),
    [
        (1000, 0.10, 1, True, 20000, PagePlanReason.FULL_PAGE_IMAGE_NATIVE_TEXT),
        (1000, 0.15, 1, False, 20000, PagePlanReason.MOSTLY_COVERED_NATIVE_TEXT),
        (1500, 0.20, 0, False, 50000, PagePlanReason.NATIVE_TEXT_WITHOUT_IMAGES),
        (3000, 0.18, 1, False, 50000, PagePlanReason.DENSE_NATIVE_TEXT),
    ],
)
def test_uncovered_artwork_preserves_usable_native_text_when_page_evidence_is_strong(
    ocr_capture, characters, coverage, images, full_page, area, reason
) -> None:
    evidence = replace(
        ocr_capture.evidence,
        native_characters=characters,
        visible_native_characters=characters,
        text_coverage=coverage,
        image_count=images,
        full_page_image=full_page,
        uncovered_vector_area=area,
    )
    result = plan_page(replace(ocr_capture, evidence=evidence))
    assert result.reason is reason
    assert result.route is PageRoute.NATIVE
    assert result.ocr_passes == ()


@pytest.mark.parametrize("authoritative", [False, True])
def test_well_mapped_vector_text_avoids_ocr_for_authoritative_or_heuristic_language(
    ocr_capture, authoritative
) -> None:
    from core_pdf.impl._impl.extract.contracts import GlyphEvidence

    glyphs = GlyphEvidence(
        glyph_count=200,
        authoritative_glyphs=200 if authoritative else 0,
        heuristic_glyphs=0 if authoritative else 200,
    )
    evidence = replace(
        ocr_capture.evidence,
        native_characters=200,
        visible_native_characters=200,
        text_coverage=0.10,
        uncovered_vector_area=20000,
        glyphs=glyphs,
        text_quality=TextQualityStats(wordlike_ratio=0.50 if authoritative else 0.65),
    )
    result = plan_page(replace(ocr_capture, evidence=evidence))
    assert result.reason is PagePlanReason.GLYPH_TRUSTED_VECTOR_TEXT
    assert result.ocr_passes == ()


@pytest.mark.parametrize(
    ("characters", "coverage", "rotated", "reason"),
    [
        (1000, 0.15, 200, PagePlanReason.GLYPH_TRUSTED_ROTATED_TEXT),
        (500, 0.10, 80, PagePlanReason.MINOR_ROTATED_NATIVE_TEXT),
        (500, 0.10, 81, PagePlanReason.ROTATED_NATIVE_TEXT),
    ],
)
def test_rotation_route_distinguishes_minor_labels_from_page_orientation(
    ocr_capture, characters, coverage, rotated, reason
) -> None:
    observations = ObservationBatch.from_columns(
        ("a" * rotated, "b" * (characters - rotated)),
        ((0, 0, 10, 100), (20, 0, 300, 20)),
        source=0,
        rotation=(90, 0),
    )
    evidence = replace(
        ocr_capture.evidence,
        native_characters=characters,
        visible_native_characters=characters,
        text_coverage=coverage,
    )
    result = plan_page(replace(ocr_capture, observations=observations, evidence=evidence))
    assert result.reason is reason
    if reason is PagePlanReason.ROTATED_NATIVE_TEXT:
        assert result.route is PageRoute.HYBRID
        assert tuple(operation.name for operation in result.ocr_passes) == (
            "orientation-page",
            "orientation-page-fallback",
        )
    else:
        assert result.route is PageRoute.NATIVE
        assert result.ocr_passes == ()

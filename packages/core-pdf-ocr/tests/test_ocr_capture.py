from __future__ import annotations

from ocr_test_helpers.extract_fakes import capture as make_capture
from ocr_test_helpers.extract_fakes import drawing, text_run

from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedLine,
    CapturedPath,
    CapturedSubpath,
)
from core_pdf_ocr import PdfDocument
from core_pdf_ocr.impl.extract.capture import (
    capture_page,
    internal_vector_complexity,
)
from core_pdf_ocr.impl.extract.contracts import (
    GlyphEvidence,
    OcrPassScope,
    PageEvidence,
    PageRoute,
    TextQualityStats,
    WorkPlan,
)
from core_pdf_ocr.impl.extract.observations import plan_page
from tests.helpers.paths import SCORE_BENCH


def run(
    text: str,
    *,
    depth: int = 0,
    clip: tuple[float, float, float, float] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> TextRun:
    provenance = (("clip_bbox", clip),) if clip is not None else ()
    x0, y0, x1, y1 = bbox or clip or (0.0, 0.0, 100.0, 100.0)
    return text_run(text, x0, y0, x1, y1, xobject_depth=depth, provenance=provenance)


def plan_for(
    page_evidence: PageEvidence,
    *,
    width: float = 600.0,
    height: float = 800.0,
    rotation: int = 0,
    drawings: tuple[CapturedDrawing, ...] = (),
) -> WorkPlan:
    """Route a page that carries only ``page_evidence`` (and optional drawings)."""
    return plan_page(
        make_capture(
            page_evidence, width=width, height=height, rotation=rotation, drawings=drawings
        )
    )


def evidence(
    *,
    characters: int,
    visible_characters: int,
    suspicious_characters: int = 0,
    image_count: int = 0,
    image_area_ratio: float = 0.0,
    vector_complexity: int = 0,
    page_area: float = 480_000.0,
    uncovered_vector_area: float | None = None,
    text_quality: TextQualityStats | None = None,
    all_text_quality: TextQualityStats | None = None,
    glyphs: GlyphEvidence | None = None,
    text_coverage: float = 0.0,
    trusted_hidden_text: bool = False,
    full_page_image: bool = False,
) -> PageEvidence:
    return PageEvidence(
        page_area=page_area,
        native_characters=characters,
        visible_native_characters=visible_characters,
        suspicious_characters=suspicious_characters,
        image_count=image_count,
        image_area_ratio=image_area_ratio,
        vector_complexity=vector_complexity,
        uncovered_vector_area=uncovered_vector_area,
        text_quality=text_quality or TextQualityStats(),
        all_text_quality=all_text_quality or TextQualityStats(),
        glyphs=glyphs or GlyphEvidence(),
        text_coverage=text_coverage,
        trusted_hidden_text=trusted_hidden_text,
        full_page_image=full_page_image,
    )


def test_vector_complexity_ignores_graphics_state_control_records() -> None:
    drawings = tuple(
        drawing(kind, (0.0, 0.0, 1.0, 1.0))
        for kind in ("state-push", "clip", "stroke", "state-pop", "image")
    )

    lines = (CapturedLine(0, 0, 1, 0), CapturedLine(1, 0, 1, 1))
    assert internal_vector_complexity(drawings, lines) == 5


def test_image_only_program_still_routes_ocr() -> None:
    fixture = SCORE_BENCH / "153rd-Omaha-Pow-Wow-p001.pdf"
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        capture = capture_page(page)
        plan = plan_page(capture)

    assert capture.evidence.image_count > 0
    assert plan.route.value == "ocr"


def test_clean_native_plan_has_no_ocr() -> None:
    native = plan_for(
        evidence(characters=200, visible_characters=200),
    )

    assert native.route is PageRoute.NATIVE
    assert not native.ocr_passes


def test_newstroke_vector_diagram_is_decoded_without_ocr() -> None:
    fixture = SCORE_BENCH / "esp32_s3_circuit_schematic.pdf"
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        capture = capture_page(page)
        plan = plan_page(capture)
        structured_text = document.extract().text
        tables = document.extract().table_view.tables

    assert capture.evidence.vector_complexity > 0
    assert capture.evidence.vector_text_trusted is True
    assert capture.evidence.vector_text_characters >= 2_000
    assert capture.evidence.vector_text_segment_coverage >= 0.80
    decoded = "\n".join(run.text for run in capture.program.runs)
    assert "MCU ESP32 S3" in decoded
    assert "ICSP UART Header" in decoded
    assert plan.route is PageRoute.NATIVE
    assert plan.reason == "newstroke-vector-text"
    assert not plan.ocr_passes
    assert "MCU ESP32 S3" in structured_text
    assert not tables


def test_route_selects_embedded_image_supplement() -> None:
    plan = plan_for(
        evidence(
            characters=200,
            visible_characters=200,
            image_count=1,
            image_area_ratio=0.2,
        ),
    )

    assert plan.reason == "embedded-image-text-supplement"
    assert plan.image_regions_only is True
    assert plan.ocr_passes[0].modes == (12,)
    assert len(plan.ocr_passes) == 1


def test_route_uses_layout_analysis_for_ultra_complex_line_art() -> None:
    plan = plan_for(
        evidence(
            characters=0,
            visible_characters=0,
            vector_complexity=150_000,
            page_area=960_000.0,
        ),
        width=1200.0,
        height=800.0,
    )

    primary = plan.ocr_passes[0]
    assert primary.modes == (3,)
    assert primary.scale == 4.0
    assert primary.scope is OcrPassScope.PAGE
    assert primary.minimum_confidence == 90.0
    assert plan.ocr_passes[1].scope is OcrPassScope.WEAK_REGIONS


def test_route_adds_region_recovery_for_uncovered_vector_text() -> None:
    plan = plan_for(
        evidence(
            characters=6,
            visible_characters=6,
            vector_complexity=200_000,
            uncovered_vector_area=100_000.0,
        ),
        width=1200.0,
        height=800.0,
    )

    assert plan.reason == "uncovered-vector-text"
    assert plan.ocr_passes[0].name == "schematic-regions"
    assert plan.ocr_passes[0].scope is OcrPassScope.WEAK_REGIONS
    assert plan.ocr_passes[0].tiles == 8
    assert plan.ocr_passes[0].region_columns == 4
    assert plan.ocr_passes[0].max_regions == 8
    assert plan.ocr_passes[0].minimum_confidence == 90.0
    assert plan.ocr_passes[0].seed_with_native
    assert plan.ocr_passes[1].name == "primary-page"


def test_route_skips_ocr_for_native_text_over_simple_rectangles() -> None:
    rectangle = drawing(
        "fill",
        (0.0, 0.0, 8.0, 8.0),
        path=CapturedPath(
            [CapturedSubpath([(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)], closed=True)]
        ),
    )

    plan = plan_for(
        evidence(
            characters=200,
            visible_characters=200,
            vector_complexity=2_400,
            uncovered_vector_area=32_000.0,
        ),
        drawings=(rectangle,) * 32,
    )

    assert plan.route is PageRoute.NATIVE
    assert plan.reason == "native-text-with-rectangular-vectors"
    assert not plan.ocr_passes


def test_route_skips_ocr_for_glyph_trusted_vector_text() -> None:
    plan = plan_for(
        evidence(
            characters=300,
            visible_characters=300,
            vector_complexity=20_000,
            uncovered_vector_area=100_000.0,
            text_coverage=0.15,
            text_quality=TextQualityStats(token_count=50, wordlike_ratio=0.75),
            glyphs=GlyphEvidence(
                glyph_count=300,
                semantic_characters=300,
                authoritative_glyphs=300,
            ),
        ),
    )

    assert plan.route is PageRoute.NATIVE
    assert plan.reason == "glyph-trusted-vector-text"
    assert not plan.ocr_passes


def test_identity_glyph_identifiers_do_not_suppress_vector_ocr() -> None:
    plan = plan_for(
        evidence(
            characters=300,
            visible_characters=300,
            vector_complexity=20_000,
            uncovered_vector_area=100_000.0,
            text_coverage=0.15,
            text_quality=TextQualityStats(token_count=50, wordlike_ratio=0.75),
            glyphs=GlyphEvidence(glyph_count=300, unknown_glyphs=300),
        ),
    )

    assert plan.reason == "uncovered-vector-text"
    assert plan.ocr_passes
    assert all(ocr_pass.include_native_text for ocr_pass in plan.ocr_passes)


def test_route_uses_binary_clean_ocr_for_noisy_native_text() -> None:
    plan = plan_for(
        evidence(
            characters=180,
            visible_characters=180,
            vector_complexity=500,
            text_quality=TextQualityStats(
                token_count=60,
                wordlike_ratio=0.10,
                short_token_ratio=0.55,
                symbol_ratio=0.20,
                non_ascii_ratio=0.04,
                digit_token_ratio=0.35,
            ),
        ),
    )

    assert plan.route is PageRoute.HYBRID
    assert plan.reason == "noisy-native-text"
    assert plan.ocr_passes[0].scope is OcrPassScope.WEAK_REGIONS
    assert plan.ocr_passes[0].preprocess == "binary-clean"
    assert plan.ocr_passes[0].minimum_confidence == 60.0
    assert plan.ocr_passes[1].preprocess == "binary-clean"


def test_route_keeps_regular_native_text_despite_some_short_tokens() -> None:
    plan = plan_for(
        evidence(
            characters=220,
            visible_characters=220,
            vector_complexity=500,
            text_quality=TextQualityStats(
                token_count=42,
                wordlike_ratio=0.55,
                short_token_ratio=0.25,
                symbol_ratio=0.08,
                digit_token_ratio=0.18,
            ),
        ),
    )

    assert plan.reason == "healthy-native-text"


def test_route_requires_stronger_evidence_for_ocr_replacement() -> None:
    plan = plan_for(
        evidence(
            characters=0,
            visible_characters=0,
            image_count=1,
            image_area_ratio=1.0,
        ),
    )

    assert plan.route is PageRoute.OCR
    assert tuple(ocr_pass.name for ocr_pass in plan.ocr_passes) == (
        "primary-page",
        "fallback-page",
    )
    assert plan.ocr_passes[0].minimum_confidence == 55.0


def test_hidden_text_route_requires_material_adaptive_gain() -> None:
    plan = plan_for(
        evidence(
            characters=140,
            visible_characters=10,
            image_count=1,
        ),
    )

    assert plan.reason == "unpainted-native-text-layer"
    adaptive = next(ocr_pass for ocr_pass in plan.ocr_passes if ocr_pass.name == "adaptive-page")
    assert adaptive.run_if_characters_below == 32
    assert all(ocr_pass.minimum_confidence == 60.0 for ocr_pass in plan.ocr_passes)
    assert adaptive.minimum_utility_gain == 1.20


def test_clean_numeric_hidden_layer_schedules_verification_before_full_ocr() -> None:
    plan = plan_for(
        evidence(
            characters=800,
            visible_characters=8,
            image_count=1,
            image_area_ratio=1.0,
            full_page_image=True,
            all_text_quality=TextQualityStats(
                token_count=180,
                wordlike_ratio=0.45,
                short_token_ratio=0.20,
                symbol_ratio=0.10,
                digit_token_ratio=0.30,
            ),
            glyphs=GlyphEvidence(glyph_count=800, heuristic_glyphs=800),
        ),
    )

    assert plan.reason == "unpainted-native-text-layer"
    assert plan.verify_hidden_text is True


def test_corrupt_numeric_hidden_layer_skips_verification() -> None:
    plan = plan_for(
        evidence(
            characters=800,
            visible_characters=8,
            image_count=1,
            image_area_ratio=1.0,
            full_page_image=True,
            all_text_quality=TextQualityStats(token_count=180, digit_token_ratio=0.30),
            glyphs=GlyphEvidence(
                glyph_count=800,
                unknown_glyphs=760,
                low_confidence_glyphs=760,
            ),
        ),
    )

    assert plan.reason == "unpainted-native-text-layer"
    assert plan.verify_hidden_text is False


def test_trusted_hidden_text_uses_extraction_layer_without_ocr() -> None:
    plan = plan_for(
        evidence(
            characters=200,
            visible_characters=200,
            image_count=1,
            trusted_hidden_text=True,
        ),
    )

    assert plan.route is PageRoute.NATIVE
    assert plan.reason == "trusted-hidden-native-text"
    assert not plan.ocr_passes


def test_native_unavailable_limits_low_yield_page_fallback() -> None:
    plan = plan_for(
        evidence(characters=0, visible_characters=0),
    )

    assert plan.reason == "native-text-unavailable"
    assert plan.ocr_passes[0].modes == (11,)
    fallback = next(ocr_pass for ocr_pass in plan.ocr_passes if ocr_pass.name == "fallback-page")
    assert fallback.modes == (6,)
    assert fallback.run_if_characters_below == 3


def test_hybrid_augmentation_uses_raster_text_confidence_floor() -> None:
    plan = plan_for(
        evidence(
            characters=20,
            visible_characters=10,
            image_count=1,
            image_area_ratio=0.80,
        ),
    )

    assert plan.reason == "native-text-needs-augmentation"
    assert all(ocr_pass.minimum_confidence == 60.0 for ocr_pass in plan.ocr_passes)


def test_route_uses_lower_scale_when_ultra_complex_page_contains_rasters() -> None:
    plan = plan_for(
        evidence(
            characters=0,
            visible_characters=0,
            image_count=1,
            image_area_ratio=0.2,
            vector_complexity=150_000,
        ),
    )

    assert plan.ocr_passes[0].scale == 3.5


def test_rotated_native_route_has_low_yield_page_fallback() -> None:
    plan = plan_for(
        evidence(characters=3, visible_characters=3, image_count=2),
        rotation=90,
    )

    assert plan.reason == "rotated-native-text"
    primary = next(ocr_pass for ocr_pass in plan.ocr_passes if ocr_pass.name == "orientation-page")
    fallback = next(
        ocr_pass for ocr_pass in plan.ocr_passes if ocr_pass.name == "orientation-page-fallback"
    )
    assert primary.minimum_characters_for_rescue == fallback.run_if_characters_below
    assert fallback.run_if_characters_below == 300
    assert fallback.region_first is False


def test_vector_evidence_counts_paint_but_not_scope_markers() -> None:
    drawings = tuple(
        CapturedDrawing(index, None, None, kind=kind)
        for index, kind in enumerate(("clip", "state-push", "stroke", "fill", "group-end"))
    )

    assert internal_vector_complexity(drawings, (CapturedLine(0, 0, 10, 10),)) == 7

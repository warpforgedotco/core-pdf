from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from core_pdf import PdfDocument
from core_pdf.impl.engine.layout.models import TextRun
from core_pdf.impl.engine.parse import (
    CapturedPage,
    GlyphEvidence,
    OcrPassScope,
    PageEvidence,
    PageRoute,
    ParseReport,
    TextQualityStats,
)
from core_pdf.impl.engine.parse.capture import (
    capture_page,
    internal_apply_structure_actual_text,
    internal_capture_from_program,
    internal_extractable_runs,
    internal_vector_complexity,
)
from core_pdf.impl.engine.parse.route import plan_page


def run(
    text: str,
    *,
    depth: int = 0,
    clip: tuple[float, float, float, float] | None = None,
) -> SimpleNamespace:
    provenance = (("clip_bbox", clip),) if clip is not None else ()
    return SimpleNamespace(
        text=text,
        xobject_depth=depth,
        provenance=provenance,
        inside_active_clip=True,
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


def test_capture_discards_duplicate_nested_text_layer() -> None:
    text = " ".join(f"token{index}" for index in range(30))
    page_run = run(text)
    nested_run = run(text, depth=1)
    distinct_nested_run = run(" ".join(f"other{index}" for index in range(30)), depth=2)

    assert internal_extractable_runs(cast(Any, (page_run, nested_run, distinct_nested_run))) == (
        page_run,
        distinct_nested_run,
    )


def test_capture_discards_duplicate_alternate_clip_layer() -> None:
    text = " ".join(f"token{index}" for index in range(30))
    page_run = run(text, clip=(0.0, 0.0, 100.0, 100.0))
    alternate_run = run(text, clip=(10.0, 10.0, 90.0, 90.0))
    distinct_clipped_run = run(
        " ".join(f"other{index}" for index in range(30)),
        clip=(20.0, 20.0, 80.0, 80.0),
    )

    assert internal_extractable_runs(
        cast(Any, (page_run, alternate_run, distinct_clipped_run))
    ) == (page_run, distinct_clipped_run)


def test_structure_actual_text_replaces_mcid_text_before_routing() -> None:
    source = TextRun(
        "broken",
        0.0,
        0.0,
        10.0,
        10.0,
        0.0,
        0.0,
        12.0,
        4.0,
        0,
        0,
        0,
        provenance=(("mcid", 0),),
    )
    page = SimpleNamespace(structure=(SimpleNamespace(actual_text="correct"),))

    result = internal_apply_structure_actual_text(page, (source,))

    assert result[0].text == "correct"
    assert ("unicode_source", "structure_actual_text") in result[0].provenance


def test_vector_complexity_ignores_graphics_state_control_records() -> None:
    drawings = tuple(
        SimpleNamespace(kind=kind)
        for kind in ("state-push", "clip", "stroke", "state-pop", "image")
    )

    assert internal_vector_complexity(drawings, (object(), object())) == 5


def test_native_text_capture_reuses_canonical_program() -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "Employee_Health_Benefits_Assess-p006.pdf"
    )
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        program = page.get_page_program()
        first = internal_capture_from_program(page, program)
        second = capture_page(page)

    assert first is second
    assert first.program is program
    assert first.evidence.image_count == 0


def test_image_only_program_still_routes_ocr() -> None:
    fixture = (
        Path(__file__).parent / "fixtures" / "SCORE-Bench" / "src" / "153rd-Omaha-Pow-Wow-p001.pdf"
    )
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        capture = capture_page(page)
        plan = plan_page(capture)

    assert capture.evidence.image_count > 0
    assert plan.route.value == "ocr"


def test_clean_native_plan_has_no_ocr() -> None:
    capture = SimpleNamespace(
        evidence=evidence(characters=200, visible_characters=200),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=600.0, height=800.0),
    )
    native = plan_page(cast(CapturedPage, capture))

    assert native.route is PageRoute.NATIVE
    assert not native.ocr_passes


def test_newstroke_vector_diagram_is_decoded_without_ocr() -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "esp32_s3_circuit_schematic.pdf"
    )
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        capture = capture_page(page)
        plan = plan_page(capture)
        structured_text = document.extract().text
        tables = document.extract().table_view.tables
        cache = page.extraction_cache
        assert cache is not None
        metrics = cast(ParseReport, cache["parse_report_v1"]).metrics

    assert capture.evidence.vector_complexity > 0
    assert capture.evidence.vector_text_trusted is True
    assert capture.evidence.vector_text_characters >= 2_000
    assert capture.evidence.vector_text_segment_coverage >= 0.80
    decoded = "\n".join(run.text for run in capture.runs)
    assert "MCU ESP32 S3" in decoded
    assert "ICSP UART Header" in decoded
    assert plan.route is PageRoute.NATIVE
    assert plan.reason == "newstroke-vector-text"
    assert not plan.ocr_passes
    assert "MCU ESP32 S3" in structured_text
    assert not tables
    assert metrics["ocr_raster_pixels"] == 0


def test_route_selects_embedded_image_supplement() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
            characters=200,
            visible_characters=200,
            image_count=1,
            image_area_ratio=0.2,
        ),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.reason == "embedded-image-text-supplement"
    assert plan.image_regions_only is True
    assert plan.ocr_passes[0].modes == (12,)
    assert len(plan.ocr_passes) == 1


def test_route_uses_layout_analysis_for_ultra_complex_line_art() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
            characters=0,
            visible_characters=0,
            vector_complexity=150_000,
            page_area=960_000.0,
        ),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=1200.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    primary = plan.ocr_passes[0]
    assert primary.modes == (3,)
    assert primary.scale == 4.0
    assert primary.scope is OcrPassScope.PAGE
    assert primary.minimum_confidence == 90.0
    assert plan.ocr_passes[1].scope is OcrPassScope.WEAK_REGIONS


def test_route_adds_region_recovery_for_uncovered_vector_text() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
            characters=6,
            visible_characters=6,
            vector_complexity=200_000,
            uncovered_vector_area=100_000.0,
        ),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=1200.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

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
    rectangle = SimpleNamespace(
        kind="fill",
        path=SimpleNamespace(
            subpaths=(
                SimpleNamespace(
                    closed=True,
                    points=((0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)),
                ),
            ),
        ),
    )
    capture = SimpleNamespace(
        evidence=evidence(
            characters=200,
            visible_characters=200,
            vector_complexity=2_400,
            uncovered_vector_area=32_000.0,
        ),
        drawings=(rectangle,) * 32,
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.route is PageRoute.NATIVE
    assert plan.reason == "native-text-with-rectangular-vectors"
    assert not plan.ocr_passes


def test_route_skips_ocr_for_glyph_trusted_vector_text() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
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
        observations=SimpleNamespace(rotation=(0,)),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.route is PageRoute.NATIVE
    assert plan.reason == "glyph-trusted-vector-text"
    assert not plan.ocr_passes


def test_identity_glyph_identifiers_do_not_suppress_vector_ocr() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
            characters=300,
            visible_characters=300,
            vector_complexity=20_000,
            uncovered_vector_area=100_000.0,
            text_coverage=0.15,
            text_quality=TextQualityStats(token_count=50, wordlike_ratio=0.75),
            glyphs=GlyphEvidence(glyph_count=300, unknown_glyphs=300),
        ),
        observations=SimpleNamespace(rotation=(0,)),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.reason == "uncovered-vector-text"
    assert plan.ocr_passes
    assert all(ocr_pass.include_native_text for ocr_pass in plan.ocr_passes)


def test_route_uses_binary_clean_ocr_for_noisy_native_text() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
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
        observations=SimpleNamespace(rotation=(0,)),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.route is PageRoute.HYBRID
    assert plan.reason == "noisy-native-text"
    assert plan.ocr_passes[0].scope is OcrPassScope.WEAK_REGIONS
    assert plan.ocr_passes[0].preprocess == "binary-clean"
    assert plan.ocr_passes[0].minimum_confidence == 60.0
    assert plan.ocr_passes[1].preprocess == "binary-clean"


def test_route_keeps_regular_native_text_despite_some_short_tokens() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
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
        observations=SimpleNamespace(rotation=(0,)),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.reason == "healthy-native-text"


def test_route_requires_stronger_evidence_for_ocr_replacement() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
            characters=0,
            visible_characters=0,
            image_count=1,
            image_area_ratio=1.0,
        ),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.route is PageRoute.OCR
    assert tuple(ocr_pass.name for ocr_pass in plan.ocr_passes) == (
        "primary-page",
        "fallback-page",
    )
    assert plan.ocr_passes[0].minimum_confidence == 55.0


def test_hidden_text_route_requires_material_adaptive_gain() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
            characters=140,
            visible_characters=10,
            image_count=1,
        ),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.reason == "unpainted-native-text-layer"
    adaptive = next(ocr_pass for ocr_pass in plan.ocr_passes if ocr_pass.name == "adaptive-page")
    assert adaptive.run_if_characters_below == 32
    assert all(ocr_pass.minimum_confidence == 60.0 for ocr_pass in plan.ocr_passes)
    assert adaptive.minimum_utility_gain == 1.20


def test_clean_numeric_hidden_layer_schedules_verification_before_full_ocr() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
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
        observations=SimpleNamespace(rotation=(0,)),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.reason == "unpainted-native-text-layer"
    assert plan.verify_hidden_text is True


def test_corrupt_numeric_hidden_layer_skips_verification() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
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
        observations=SimpleNamespace(rotation=(0,)),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.reason == "unpainted-native-text-layer"
    assert plan.verify_hidden_text is False


def test_trusted_hidden_text_uses_extraction_layer_without_ocr() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
            characters=200,
            visible_characters=200,
            image_count=1,
            trusted_hidden_text=True,
        ),
        observations=SimpleNamespace(rotation=(0,)),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.route is PageRoute.NATIVE
    assert plan.reason == "trusted-hidden-native-text"
    assert not plan.ocr_passes


def test_native_unavailable_limits_low_yield_page_fallback() -> None:
    capture = SimpleNamespace(
        evidence=evidence(characters=0, visible_characters=0),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.reason == "native-text-unavailable"
    assert plan.ocr_passes[0].modes == (11,)
    fallback = next(ocr_pass for ocr_pass in plan.ocr_passes if ocr_pass.name == "fallback-page")
    assert fallback.modes == (6,)
    assert fallback.run_if_characters_below == 3


def test_hybrid_augmentation_uses_raster_text_confidence_floor() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
            characters=20,
            visible_characters=10,
            image_count=1,
            image_area_ratio=0.80,
        ),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.reason == "native-text-needs-augmentation"
    assert all(ocr_pass.minimum_confidence == 60.0 for ocr_pass in plan.ocr_passes)


def test_route_uses_lower_scale_when_ultra_complex_page_contains_rasters() -> None:
    capture = SimpleNamespace(
        evidence=evidence(
            characters=0,
            visible_characters=0,
            image_count=1,
            image_area_ratio=0.2,
            vector_complexity=150_000,
        ),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.ocr_passes[0].scale == 3.5


def test_rotated_native_route_has_low_yield_page_fallback() -> None:
    capture = SimpleNamespace(
        evidence=evidence(characters=3, visible_characters=3, image_count=2),
        observations=SimpleNamespace(rotation=(90,)),
        drawings=(),
        grid_lines=(),
        page=SimpleNamespace(width=600.0, height=800.0),
    )

    plan = plan_page(cast(CapturedPage, capture))

    assert plan.reason == "rotated-native-text"
    primary = next(ocr_pass for ocr_pass in plan.ocr_passes if ocr_pass.name == "orientation-page")
    fallback = next(
        ocr_pass for ocr_pass in plan.ocr_passes if ocr_pass.name == "orientation-page-fallback"
    )
    assert primary.minimum_characters_for_rescue == fallback.run_if_characters_below
    assert fallback.run_if_characters_below == 300
    assert fallback.region_first is False

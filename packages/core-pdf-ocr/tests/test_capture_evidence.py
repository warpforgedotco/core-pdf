from copy import replace

import pytest

from core_pdf.impl.capture_program import CapturedProgram, PageProgram
from core_pdf.impl.capture_records import (
    CapturedDrawing,
    CapturedLine,
    CapturedLines,
    CapturedPath,
    CapturedSubpath,
    DrawingKind,
    ShadingPattern,
)
from core_pdf.impl.extract_contracts import ObservationBatch
from core_pdf_ocr.impl.extract import capture
from core_pdf_ocr.impl.extract.contracts import PageAnalysis, StrokedVectorTextEvidence


def stroke(index: int, *, size: float = 2, spread: bool = True) -> CapturedDrawing:
    x = index % 20 * (20 if spread else 0)
    y = index // 20 * (20 if spread else 0)
    return CapturedDrawing(
        index,
        None,
        None,
        kind="stroke",
        stroke_color=(0.0,),
        line_width=0.5,
        path=CapturedPath([CapturedSubpath([(x, y), (x + size, y + size)])]),
    )


def test_patterned_strokes_do_not_supply_solid_vector_text_evidence() -> None:
    solid = stroke(0)
    patterned = replace(solid, stroke_pattern=ShadingPattern({}))
    assert capture.stroked_vector_style(solid) is not None
    assert capture.stroked_vector_style(patterned) is None


def test_vector_complexity_counts_paints_and_segments_but_not_control_records() -> None:
    kinds: tuple[DrawingKind, ...] = (
        "stroke",
        "fill",
        "fillstroke",
        "scope-begin",
        "scope-end",
        "image",
    )
    drawings = tuple(CapturedDrawing(i, None, None, kind=kind) for i, kind in enumerate(kinds))
    lines = CapturedLines((CapturedLine(0, 0, 1, 1),) * 10)
    assert capture.vector_complexity(drawings, lines) == 19
    assert capture.vector_complexity((), CapturedLines()) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "fill"),
        ("path", None),
        ("stroke_color", None),
        ("stroke_opacity", 0),
        ("line_width", 0),
        ("line_width", 1.51),
        ("dash_pattern", ([1.0], 0.0)),
    ],
)
def test_stroke_style_rejects_ineligible_paints(field: str, value: object) -> None:
    assert capture.stroked_vector_style(replace(stroke(0), **{field: value})) is None


def test_stroke_style_normalizes_default_opacity_and_empty_dash() -> None:
    drawing = stroke(0)
    key = capture.stroked_vector_style(drawing)
    assert key is not None
    assert key[:4] == ("stroke", (0.0,), 1.0, 0.5)
    assert capture.stroked_vector_style(replace(drawing, dash_pattern=([], 0))) is not None


def test_distributed_stroke_evidence_selects_supported_styles_and_render_sized_paths() -> None:
    drawings = tuple(stroke(i) for i in range(300))
    secondary = tuple(replace(stroke(i), stroke_color=(1.0,)) for i in range(8))
    rare = tuple(replace(stroke(i), stroke_color=(0.5,)) for i in range(7))
    result = capture.stroked_vector_text_evidence(
        (*drawings, *secondary, *rare, stroke(0, size=5)),
        page_width=600,
        page_height=800,
    )
    assert result.trusted
    assert result.candidate_paths == 309
    assert result.drawing_indexes == (*range(308), 315)
    assert result.bbox == (0, 0, 382, 282)


@pytest.mark.parametrize(
    ("count", "size", "spread", "rotation", "width"),
    [
        (179, 2, True, 0, 600),
        (299, 2, True, 0, 600),
        (300, 2, True, 90, 600),
        (300, 2, True, 0, 0),
        (300, 2, False, 0, 600),
        (300, 6, True, 0, 600),
    ],
)
def test_stroke_trust_requires_density_distribution_and_unrotated_page(
    count: int,
    size: float,
    spread: bool,
    rotation: int,
    width: float,
) -> None:
    drawings = tuple(stroke(i, size=size, spread=spread) for i in range(count))
    assert (
        capture.stroked_vector_text_evidence(
            drawings,
            page_width=width,
            page_height=800,
            rotation=rotation,
        )
        == StrokedVectorTextEvidence()
    )


def test_wires_can_dilute_compact_stroke_evidence_below_trust() -> None:
    drawings = tuple(stroke(i) for i in range(300)) + tuple(stroke(i, size=100) for i in range(201))
    assert not capture.stroked_vector_text_evidence(
        drawings, page_width=600, page_height=800
    ).trusted


def test_missing_and_degenerate_path_bounds_do_not_contribute_evidence() -> None:
    drawings = (
        CapturedDrawing(0, None, None),
        replace(stroke(0), path=CapturedPath()),
        stroke(0, size=0),
    ) * 100
    assert (
        capture.stroked_vector_text_evidence(drawings, page_width=600, page_height=800)
        == StrokedVectorTextEvidence()
    )


def test_uncovered_area_subtracts_native_overlap_in_bounded_batches() -> None:
    drawings = tuple(
        CapturedDrawing(i, None, None, kind="fill", bbox=(0, 0, 10, 10)) for i in range(180)
    )
    native = ObservationBatch.from_columns(("native",), ((0, 0, 5, 10),), source=0)
    assert capture.uncovered_vector_area(drawings, native) == 9000
    duplicated = ObservationBatch.concatenate(native, native, native)
    assert capture.uncovered_vector_area(drawings, duplicated) == 0
    assert capture.uncovered_vector_area(drawings, native, page_area=100) == 0


def test_uncovered_area_skips_controls_missing_bounds_and_zero_area() -> None:
    native = ObservationBatch.from_columns(("native",), ((0, 0, 5, 10),), source=0)
    assert capture.uncovered_vector_area((), native) is None
    assert capture.uncovered_vector_area((stroke(0),), native) is None
    assert capture.uncovered_vector_area((stroke(0),) * 180, ObservationBatch.empty()) is None
    controls = (CapturedDrawing(0, None, None, kind="scope-begin"),) * 180
    assert capture.uncovered_vector_area(controls, native) is None
    drawings = (
        stroke(0),
        CapturedDrawing(0, None, None),
        CapturedDrawing(0, None, None, bbox=(0, 0, 0, 10)),
    ) * 60
    assert capture.uncovered_vector_area(drawings, native) == 0


@pytest.mark.parametrize(
    ("strokes", "fills", "compact", "expected"),
    [
        (9999, 0, True, False),
        (10000, 0, True, True),
        (10000, 600, True, False),
        (10000, 0, False, False),
    ],
)
def test_high_resolution_routing_requires_many_compact_strokes(
    ocr_capture: PageAnalysis,
    strokes: int,
    fills: int,
    compact: bool,
    expected: bool,
) -> None:
    drawings = (
        (stroke(0, size=2 if compact else 10),) * strokes
        + (CapturedDrawing(0, None, None, kind="fill"),) * fills
        + (CapturedDrawing(0, None, None, kind="scope-begin"),)
    )
    page = replace(
        ocr_capture,
        program=PageProgram(CapturedProgram(drawings=drawings)),
        evidence=replace(ocr_capture.evidence, vector_complexity=100000),
    )
    assert capture.requires_high_resolution_vector_ocr(page) is expected
    assert not capture.requires_high_resolution_vector_ocr(
        replace(page, evidence=replace(page.evidence, image_count=1))
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("page", "native_characters", 99),
        ("page", "painted_native_characters", 300),
        ("page", "full_page_image", False),
        ("page", "suspicious_characters", 11),
        ("glyphs", "glyph_count", 99),
        ("glyphs", "authoritative_glyphs", 980),
        ("glyphs", "low_confidence_glyphs", 11),
        ("quality", "token_count", 99),
        ("quality", "digit_token_ratio", 0.17),
        ("quality", "symbol_ratio", 0.31),
        ("quality", "short_token_ratio", 1),
    ],
)
def test_numeric_hidden_layer_verification_requires_clean_mapped_scan_evidence(
    ocr_capture: PageAnalysis,
    section: str,
    field: str,
    value: object,
) -> None:
    from core_pdf.impl.extract_contracts import GlyphEvidence, TextQualityStats

    evidence = replace(
        ocr_capture.evidence,
        native_characters=1000,
        painted_native_characters=0,
        full_page_image=True,
        glyphs=GlyphEvidence(glyph_count=1000, authoritative_glyphs=1000),
        all_text_quality=TextQualityStats(token_count=100, digit_token_ratio=0.18),
    )
    assert capture.hidden_text_needs_verification(evidence)
    if section == "page":
        changed = replace(evidence, **{field: value})
    elif section == "glyphs":
        changed = replace(evidence, glyphs=replace(evidence.glyphs, **{field: value}))
    else:
        changed = replace(
            evidence, all_text_quality=replace(evidence.all_text_quality, **{field: value})
        )
    assert not capture.hidden_text_needs_verification(changed)


def test_template_text_promotion_preserves_program_drawings_and_updates_evidence(
    ocr_capture: PageAnalysis,
) -> None:
    from core_pdf.impl.runs import TextRun
    from core_pdf_ocr.impl.extract.ocr.newstroke import NewstrokeDecode

    run = TextRun("R123", 10, 20, 30, 25, 10, 25, 5, 2, 0, 0, 0)
    original = replace(ocr_capture, program=PageProgram(CapturedProgram(drawings=(stroke(0),))))
    result = capture.capture_with_newstroke_text(
        original, NewstrokeDecode(runs=(run,), candidate_segments=10000, matched_segments=9900)
    )
    assert result.observations.text == ("R123",)
    assert result.program.runs == (run,)
    assert result.program.drawings == original.program.drawings
    assert result.evidence.vector_text_trusted
    assert result.evidence.native_characters == result.evidence.visible_native_characters == 4
    assert result.evidence.text_coverage == pytest.approx(100 / 480000)
    assert original.observations.text == ()


def test_promoted_hidden_text_prefers_normalized_observation_references(
    ocr_capture: PageAnalysis,
) -> None:
    from core_pdf.impl.runs import TextRun

    raw = TextRun("old", 10, 20, 30, 25, 10, 25, 5, 2, 0, 0, 0, visible=False)
    normalized = replace(raw, text="new")
    original = replace(ocr_capture, program=PageProgram(CapturedProgram(runs=(raw,))))
    assert capture.promoted_hidden_observations(original).text == ("old",)
    normalized_batch = capture.observations_from_runs((normalized,))
    result = capture.promoted_hidden_observations(replace(original, observations=normalized_batch))
    assert result.text == ("new",)
    assert result.bbox.tolist() == [[10, 20, 30, 25]]
    assert result.visible.tolist() == [True]

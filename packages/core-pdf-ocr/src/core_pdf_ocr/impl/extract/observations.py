# SPDX-License-Identifier: AGPL-3.0-only
"""Plan page extraction and fuse native and recognized observations."""

from __future__ import annotations

import numpy

from core_pdf.impl._impl.model.text import compact_text, text_tokens
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing
from core_pdf_ocr.impl.extract.capture import (
    internal_hidden_text_needs_verification,
    internal_requires_high_resolution_vector_ocr,
)
from core_pdf_ocr.impl.extract.contracts import (
    MAX_OCR_PIXELS,
    PRIMARY_OCR_PIXELS,
    PSM_AUTO,
    PSM_SPARSE_TEXT,
    PSM_SPARSE_TEXT_OSD,
    FusionPolicy,
    ObservationBatch,
    OcrPass,
    OcrPassScope,
    PageAnalysis,
    PageEvidence,
    PagePlanReason,
    PageRoute,
    WorkPlan,
)
from core_pdf_ocr.impl.extract.quality import internal_candidate

# Precision-first extraction thresholds.  Raster text below these confidence
# levels is more likely to be a layout artifact than a useful observation on
# the document classes routed through these passes.
# Scanned pages behind an unpainted text layer are usually degraded scans,
# where Tesseract reports mid-range confidence on perfectly legible digits.
# A 90.0 floor silently discarded most of a typewritten table's numbers;
# 60.0 recovers them while the duplicate/overlap filters downstream absorb
# the extra noise (measured: recall +0.8pt, precision -0.2pt bench-wide).
HIDDEN_TEXT_MIN_CONFIDENCE = 60.0
# Same reasoning as the hidden-text floor: scan-routed pages lose legible
# words to mid-range Tesseract confidence, and the downstream duplicate and
# overlap filters absorb what little noise the lower floor admits.
RASTER_TEXT_MIN_CONFIDENCE = 60.0
VECTOR_TEXT_MIN_CONFIDENCE = 90.0
NATIVE_UNAVAILABLE_MIN_CONFIDENCE = 90.0
# Printer-converted vector labels are filtered one word at a time, so a lower
# floor retains valid identifiers without admitting an entire uncertain line.
STROKED_VECTOR_WORD_MIN_CONFIDENCE = 80.0

# Confidence a recognized pass must reach before it may displace or
# supplement noisy native text. Deliberately stricter than the recognition
# pass floor: a low pass floor only admits more candidate words, while this
# gate decides whether recognized text replaces text the document carries.
FUSION_NOISY_NATIVE_MIN_CONFIDENCE = 90.0

COVERAGE_CHUNK = 256

# Upper bound on elements materialized per vectorized overlap chunk.
COVERAGE_VECTORIZED_ELEMENTS = 1_000_000


def maximum_candidate_coverage(
    candidate_boxes: numpy.ndarray,
    native_boxes: numpy.ndarray,
) -> numpy.ndarray:
    """Return each candidate's maximum covered-area ratio in bounded chunks."""
    if not len(candidate_boxes) or not len(native_boxes):
        return numpy.zeros(len(candidate_boxes), dtype=numpy.float32)
    output = numpy.zeros(len(candidate_boxes), dtype=numpy.float32)
    native_x0 = native_boxes[:, 0][None, :]
    native_y0 = native_boxes[:, 1][None, :]
    native_x1 = native_boxes[:, 2][None, :]
    native_y1 = native_boxes[:, 3][None, :]
    chunk_size = min(
        COVERAGE_CHUNK,
        max(1, COVERAGE_VECTORIZED_ELEMENTS // len(native_boxes)),
    )
    for start in range(0, len(candidate_boxes), chunk_size):
        stop = min(len(candidate_boxes), start + chunk_size)
        boxes = candidate_boxes[start:stop]
        widths = numpy.maximum(
            0.0,
            numpy.minimum(boxes[:, None, 2], native_x1)
            - numpy.maximum(boxes[:, None, 0], native_x0),
        )
        heights = numpy.maximum(
            0.0,
            numpy.minimum(boxes[:, None, 3], native_y1)
            - numpy.maximum(boxes[:, None, 1], native_y0),
        )
        areas = numpy.maximum(
            1.0,
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        )
        numpy.multiply(widths, heights, out=widths)
        output[start:stop] = numpy.max(widths, axis=1) / areas
    return output


def internal_ocr_scale(capture: PageAnalysis, *, schematic: bool, vector_complexity: int) -> float:
    if not schematic or vector_complexity < 4_000:
        return 3.0
    if vector_complexity < 150_000:
        return 5.0
    # Raster-backed mixed pages already contain sampled detail; pure vector pages
    # need another half pixel per point to preserve their smallest labels.
    return 3.5 if capture.evidence.image_count else 4.0


def internal_vector_text_scale(capture: PageAnalysis, vector_complexity: int) -> float:
    """Choose a higher raster scale for text embedded in vector artwork.

    Charts and diagrams often use small glyphs painted alongside thousands of
    vector paths.  The regular page scale is sufficient for prose, but loses
    those labels before OCR can associate them with the artwork.
    """
    return max(
        4.0, internal_ocr_scale(capture, schematic=True, vector_complexity=vector_complexity)
    )


def internal_schematic_page(
    vector_complexity: int,
    text_density: float,
    text_coverage: float,
) -> bool:
    """A vector-heavy page whose visible text is too sparse to read like prose."""
    return vector_complexity >= 180 and (text_density < 0.0015 or text_coverage < 0.05)


def internal_rotated_native_characters(capture: PageAnalysis) -> int:
    observations = capture.observations
    return sum(
        len(text.strip())
        for text, rotation in zip(observations.text, observations.rotation, strict=True)
        if int(rotation) % 360
    )


def internal_noisy_native_text(evidence: PageEvidence) -> bool:
    quality = evidence.text_quality
    if evidence.visible_native_characters < 24 or quality.token_count < 12:
        return False
    if quality.noise_score < 0.12:
        return False
    if evidence.suspicious_ratio >= 0.25:
        return False
    structure_signal = (
        evidence.vector_complexity >= 180
        or evidence.text_coverage < 0.08
        or (evidence.uncovered_vector_area or 0.0) >= evidence.page_area * 0.02
    )
    token_signal = (
        quality.short_token_ratio >= 0.40
        or quality.digit_token_ratio >= 0.30
        or quality.symbol_ratio >= 0.18
        or quality.non_ascii_ratio >= 0.03
    )
    language_signal = quality.wordlike_ratio < 0.35
    return structure_signal and token_signal and language_signal


def internal_native_mapping_is_usable(evidence: PageEvidence) -> bool:
    glyphs = evidence.glyphs
    if glyphs.actual_text_characters >= max(1, int(evidence.native_characters * 0.80)):
        return True
    # Some synthetic/API callers do not expose glyph observations.  Preserve
    # their text-only contract; real captured pages always carry glyph evidence.
    if not glyphs.glyph_count:
        return True
    return (
        glyphs.mapped_ratio >= 0.95
        and glyphs.unknown_ratio <= 0.05
        and glyphs.unsupported_ratio <= 0.02
        and glyphs.low_confidence_ratio <= 0.02
    )


def internal_drawing_is_simple_rectangle(drawing: CapturedDrawing) -> bool:
    path = drawing.path
    return path is not None and path.axis_aligned_rect() is not None


def internal_has_only_simple_vector_rectangles(capture: PageAnalysis) -> bool:
    drawings = tuple(
        drawing
        for drawing in capture.program.drawings
        if drawing.kind in {"fill", "fillstroke", "stroke"}
    )
    return len(drawings) >= 32 and all(
        internal_drawing_is_simple_rectangle(drawing) for drawing in drawings
    )


def internal_vector_native_text_is_trusted(evidence: PageEvidence) -> bool:
    glyphs = evidence.glyphs
    if evidence.visible_native_characters < 200 or not glyphs.glyph_count:
        return False
    quality = evidence.text_quality
    authoritative_language = glyphs.authoritative_ratio >= 0.90 and quality.wordlike_ratio >= 0.50
    heuristic_language = quality.wordlike_ratio >= 0.65
    return (
        glyphs.mapped_ratio >= 0.99
        and glyphs.low_confidence_ratio <= 0.01
        and glyphs.unsupported_ratio <= 0.01
        and glyphs.inflation(evidence.visible_native_characters) <= 2.5
        and evidence.text_coverage >= 0.09
        and quality.noise_score <= 0.02
        and (authoritative_language or heuristic_language)
    )


def internal_fallback_pass(
    *,
    schematic: bool,
    scale: float,
    modes: tuple[int, ...],
    minimum_confidence: float,
    run_if_characters_below: int,
    include_native_text: bool,
) -> OcrPass:
    """The second-chance pass an OCR route ends with.

    Whether the page is schematic settles the pass's name, its scope and its
    tiling together -- a schematic retries weak regions in four tiles, anything
    else retries the whole page once -- so those move as one decision rather
    than three parallel conditionals repeated at each call site. Confidence and
    the native-text and character thresholds are what actually differ by route.
    """
    return OcrPass(
        "fallback-regions" if schematic else "fallback-page",
        OcrPassScope.WEAK_REGIONS if schematic else OcrPassScope.PAGE,
        scale,
        modes,
        tiles=4 if schematic else 1,
        minimum_confidence=minimum_confidence,
        run_if_characters_below=run_if_characters_below,
        region_first=False,
        include_native_text=include_native_text,
    )


def plan_page(capture: PageAnalysis) -> WorkPlan:
    evidence = capture.evidence
    total_characters = evidence.native_characters
    characters = evidence.visible_native_characters
    suspicious_ratio = evidence.suspicious_ratio
    vector_complexity = evidence.vector_complexity
    text_density = evidence.visible_text_density
    text_coverage = evidence.text_coverage
    observations = capture.observations
    rotated_native = any(int(value) % 360 for value in observations.rotation)
    quality = evidence.text_quality
    corrupt_mapping = (
        characters >= 24
        and quality.noise_score >= 0.20
        and quality.wordlike_ratio < 0.20
        and vector_complexity < 150
    )
    if corrupt_mapping:
        schematic = internal_schematic_page(vector_complexity, text_density, text_coverage)
        image_modes = (PSM_AUTO,)
        scale = 6.0
        weak_threshold = 300 if schematic else 1_000
        high_resolution_vector = schematic and internal_requires_high_resolution_vector_ocr(capture)
        return WorkPlan(
            PageRoute.OCR,
            reason=PagePlanReason.NATIVE_TEXT_CORRUPT,
            ocr_passes=(
                OcrPass(
                    "primary-page",
                    OcrPassScope.PAGE,
                    8.0 if high_resolution_vector else scale,
                    image_modes,
                    minimum_confidence=(
                        STROKED_VECTOR_WORD_MIN_CONFIDENCE
                        if high_resolution_vector
                        else 45.0
                        if evidence.image_count
                        else NATIVE_UNAVAILABLE_MIN_CONFIDENCE
                    ),
                    adaptive_scale=not high_resolution_vector,
                    character_confidence_threshold=(
                        55.0 if schematic and not high_resolution_vector else None
                    ),
                    region_first=True,
                    pixel_budget=(MAX_OCR_PIXELS if high_resolution_vector else PRIMARY_OCR_PIXELS),
                    include_native_text=True,
                    recognize_words=high_resolution_vector,
                    parallel_tiles=2,
                ),
                internal_fallback_pass(
                    schematic=schematic,
                    scale=scale,
                    modes=(6,),
                    minimum_confidence=(NATIVE_UNAVAILABLE_MIN_CONFIDENCE if schematic else 45.0),
                    run_if_characters_below=weak_threshold,
                    include_native_text=True,
                ),
            ),
            allow_direct_image_ocr=False,
            augment_page_candidates=True,
        )
    if evidence.vector_text_trusted:
        return WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.NEWSTROKE_VECTOR_TEXT)
    if evidence.trusted_hidden_text:
        return WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.TRUSTED_HIDDEN_NATIVE_TEXT)
    if evidence.hidden_text_layer:
        hidden_text_scale = 2.0 if evidence.full_page_image and 20 <= characters < 32 else 3.0
        return WorkPlan(
            PageRoute.OCR,
            reason=PagePlanReason.UNPAINTED_NATIVE_TEXT_LAYER,
            ocr_passes=(
                OcrPass(
                    "primary-page",
                    OcrPassScope.PAGE,
                    hidden_text_scale,
                    (PSM_SPARSE_TEXT,),
                    minimum_confidence=HIDDEN_TEXT_MIN_CONFIDENCE,
                    adaptive_scale=True,
                    region_first=True,
                    pixel_budget=PRIMARY_OCR_PIXELS,
                ),
                OcrPass(
                    "fallback-regions",
                    OcrPassScope.WEAK_REGIONS,
                    hidden_text_scale,
                    (6,),
                    tiles=4,
                    minimum_confidence=HIDDEN_TEXT_MIN_CONFIDENCE,
                    # Sparse-mode primary OCR drops narrow columns outright
                    # (single-digit row labels, tick columns), so a page can
                    # read "well" by character count while missing a column.
                    # Run the weak-region sweep even on well-read pages, and
                    # recognize words so the per-word coverage test can admit
                    # the genuinely novel words while rejecting re-reads of
                    # text the primary already found (line-level supplements
                    # sneak whole duplicate rows past the coverage check).
                    run_if_characters_below=1500,
                    recognize_words=True,
                ),
                OcrPass(
                    "adaptive-page",
                    OcrPassScope.PAGE,
                    hidden_text_scale,
                    (PSM_SPARSE_TEXT,),
                    minimum_confidence=HIDDEN_TEXT_MIN_CONFIDENCE,
                    run_if_characters_below=32,
                    minimum_utility_gain=1.20,
                    region_first=False,
                ),
            ),
            verify_hidden_text=internal_hidden_text_needs_verification(evidence),
        )

    if evidence.stroked_vector_text.trusted:
        return WorkPlan(
            PageRoute.HYBRID if characters else PageRoute.OCR,
            reason=PagePlanReason.STROKED_VECTOR_TEXT,
            ocr_passes=(
                OcrPass(
                    "stroked-vector-text",
                    OcrPassScope.STROKED_VECTOR_TEXT,
                    6.0,
                    (PSM_SPARSE_TEXT,),
                    minimum_confidence=STROKED_VECTOR_WORD_MIN_CONFIDENCE,
                    region_first=False,
                    pixel_budget=MAX_OCR_PIXELS,
                ),
            ),
        )

    if evidence.uncovered_vector_area is not None and evidence.uncovered_vector_area >= 20_000.0:
        if (
            characters >= 32
            and evidence.image_count == 0
            and suspicious_ratio <= 0.02
            and internal_native_mapping_is_usable(evidence)
            and internal_has_only_simple_vector_rectangles(capture)
        ):
            return WorkPlan(
                PageRoute.NATIVE,
                reason=PagePlanReason.NATIVE_TEXT_WITH_RECTANGULAR_VECTORS,
            )
        if internal_vector_native_text_is_trusted(evidence):
            return WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.GLYPH_TRUSTED_VECTOR_TEXT)
        if (
            evidence.full_page_image
            and characters >= 1_000
            and internal_native_mapping_is_usable(evidence)
        ):
            return WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.FULL_PAGE_IMAGE_NATIVE_TEXT)
        if (
            characters >= 1_000
            and evidence.text_coverage >= 0.15
            and evidence.uncovered_vector_area / max(1.0, evidence.page_area) < 0.08
            and suspicious_ratio <= 0.02
            and internal_native_mapping_is_usable(evidence)
        ):
            return WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.MOSTLY_COVERED_NATIVE_TEXT)
        if (
            characters >= 1_500
            and evidence.image_count == 0
            and evidence.text_coverage >= 0.20
            and suspicious_ratio <= 0.02
            and internal_native_mapping_is_usable(evidence)
        ):
            return WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.NATIVE_TEXT_WITHOUT_IMAGES)
        if (
            characters >= 3_000
            and evidence.text_coverage >= 0.18
            and suspicious_ratio <= 0.02
            and internal_native_mapping_is_usable(evidence)
        ):
            return WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.DENSE_NATIVE_TEXT)
        return WorkPlan(
            PageRoute.HYBRID if characters else PageRoute.OCR,
            reason=PagePlanReason.UNCOVERED_VECTOR_TEXT,
            ocr_passes=(
                OcrPass(
                    "schematic-regions",
                    OcrPassScope.WEAK_REGIONS,
                    (
                        min(6.0, internal_vector_text_scale(capture, vector_complexity) * 1.2)
                        if characters >= 8
                        else internal_vector_text_scale(capture, vector_complexity)
                    ),
                    (PSM_SPARSE_TEXT,),
                    tiles=8,
                    region_columns=4,
                    max_regions=8,
                    minimum_confidence=VECTOR_TEXT_MIN_CONFIDENCE,
                    seed_with_native=True,
                    include_native_text=True,
                ),
                OcrPass(
                    "primary-page",
                    OcrPassScope.PAGE,
                    internal_vector_text_scale(capture, vector_complexity),
                    (PSM_AUTO,),
                    minimum_confidence=VECTOR_TEXT_MIN_CONFIDENCE,
                    run_if_additions_below=4,
                    adaptive_scale=True,
                    character_confidence_threshold=55.0,
                    region_first=True,
                    include_native_text=True,
                ),
                OcrPass(
                    "schematic-regions-fallback",
                    OcrPassScope.WEAK_REGIONS,
                    internal_vector_text_scale(capture, vector_complexity),
                    (6,),
                    tiles=8,
                    region_columns=4,
                    max_regions=8,
                    minimum_confidence=VECTOR_TEXT_MIN_CONFIDENCE,
                    run_if_additions_below=0,
                    seed_with_native=True,
                    region_first=True,
                    include_native_text=True,
                ),
            ),
            fusion_policy=FusionPolicy.UNCOVERED_VECTOR,
        )

    if internal_noisy_native_text(evidence):
        scale = min(
            4.5,
            max(
                3.0,
                internal_ocr_scale(capture, schematic=True, vector_complexity=vector_complexity),
            ),
        )
        return WorkPlan(
            PageRoute.HYBRID,
            reason=PagePlanReason.NOISY_NATIVE_TEXT,
            ocr_passes=(
                OcrPass(
                    "clean-regions",
                    OcrPassScope.WEAK_REGIONS,
                    scale,
                    (PSM_SPARSE_TEXT,),
                    tiles=6,
                    region_columns=4,
                    max_regions=8,
                    minimum_confidence=RASTER_TEXT_MIN_CONFIDENCE,
                    run_if_additions_below=4,
                    seed_with_native=True,
                    preprocess="binary-clean",
                    pixel_budget=PRIMARY_OCR_PIXELS,
                    include_native_text=True,
                ),
                OcrPass(
                    "clean-page",
                    OcrPassScope.PAGE,
                    scale,
                    (PSM_SPARSE_TEXT,),
                    minimum_confidence=RASTER_TEXT_MIN_CONFIDENCE,
                    run_if_additions_below=2,
                    minimum_utility_gain=1.05,
                    adaptive_scale=True,
                    region_first=False,
                    preprocess="binary-clean",
                    include_native_text=True,
                ),
            ),
            fusion_policy=FusionPolicy.NOISY_NATIVE,
        )

    image_text_supplement = (
        0.10 <= evidence.image_area_ratio < 0.65
        and evidence.image_count <= 4
        and characters >= 32
        and suspicious_ratio <= 0.05
    )
    if image_text_supplement:
        return WorkPlan(
            PageRoute.HYBRID,
            reason=PagePlanReason.EMBEDDED_IMAGE_TEXT_SUPPLEMENT,
            ocr_passes=(
                OcrPass(
                    "image-regions",
                    OcrPassScope.IMAGE_REGIONS,
                    2.0,
                    (PSM_SPARSE_TEXT_OSD,),
                    minimum_confidence=HIDDEN_TEXT_MIN_CONFIDENCE,
                    adaptive_scale=True,
                    pixel_budget=PRIMARY_OCR_PIXELS,
                ),
            ),
        )

    if rotated_native:
        rotated_characters = internal_rotated_native_characters(capture)
        if (
            characters >= 1_000
            and text_coverage >= 0.15
            and internal_native_mapping_is_usable(evidence)
        ):
            return WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.GLYPH_TRUSTED_ROTATED_TEXT)
        if (
            characters >= 500
            and suspicious_ratio <= 0.05
            and rotated_characters <= max(80, int(characters * 0.03))
        ):
            return WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.MINOR_ROTATED_NATIVE_TEXT)
        return WorkPlan(
            PageRoute.HYBRID,
            reason=PagePlanReason.ROTATED_NATIVE_TEXT,
            ocr_passes=(
                OcrPass(
                    "orientation-page",
                    OcrPassScope.PAGE,
                    2.0,
                    (PSM_SPARSE_TEXT_OSD,),
                    minimum_confidence=RASTER_TEXT_MIN_CONFIDENCE,
                    adaptive_scale=True,
                    minimum_characters_for_rescue=300,
                    region_first=True,
                    pixel_budget=PRIMARY_OCR_PIXELS,
                    include_native_text=True,
                ),
                OcrPass(
                    "orientation-page-fallback",
                    OcrPassScope.PAGE,
                    2.0,
                    (PSM_SPARSE_TEXT,),
                    minimum_confidence=RASTER_TEXT_MIN_CONFIDENCE,
                    run_if_characters_below=300,
                    minimum_utility_gain=1.05,
                    region_first=False,
                    include_native_text=True,
                ),
            ),
        )

    mapping_usable = internal_native_mapping_is_usable(evidence)
    if characters >= 80 and suspicious_ratio <= 0.02 and mapping_usable:
        return WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.HEALTHY_NATIVE_TEXT)
    if (
        characters >= 32
        and suspicious_ratio <= 0.05
        and evidence.image_count == 0
        and mapping_usable
    ):
        return WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.USABLE_NATIVE_TEXT)
    if (
        characters > 0
        and characters == total_characters
        and suspicious_ratio == 0.0
        and evidence.image_count == 0
        and vector_complexity < 30
        and mapping_usable
    ):
        return WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.CLEAN_SHORT_NATIVE_TEXT)

    schematic = internal_schematic_page(vector_complexity, text_density, text_coverage)
    mode = PSM_AUTO if (schematic and vector_complexity >= 150_000) else PSM_SPARSE_TEXT
    image_modes = (mode,)
    scale = internal_ocr_scale(capture, schematic=schematic, vector_complexity=vector_complexity)
    if characters == 0 or suspicious_ratio >= 0.25:
        weak_threshold = 300 if schematic else 3
        high_resolution_vector = schematic and internal_requires_high_resolution_vector_ocr(capture)
        return WorkPlan(
            PageRoute.OCR,
            reason=PagePlanReason.NATIVE_TEXT_UNAVAILABLE,
            ocr_passes=(
                OcrPass(
                    "primary-page",
                    OcrPassScope.PAGE,
                    8.0 if high_resolution_vector else scale,
                    image_modes,
                    minimum_confidence=(
                        STROKED_VECTOR_WORD_MIN_CONFIDENCE
                        if high_resolution_vector
                        else (
                            55.0
                            if (characters < 10 and evidence.image_count > 0)
                            else NATIVE_UNAVAILABLE_MIN_CONFIDENCE
                        )
                    ),
                    adaptive_scale=not high_resolution_vector,
                    character_confidence_threshold=(
                        55.0 if schematic and not high_resolution_vector else None
                    ),
                    region_first=(
                        characters >= 10
                        or bool(evidence.stroked_vector_text.drawing_indexes)
                        or evidence.image_count >= 2
                    ),
                    pixel_budget=(MAX_OCR_PIXELS if high_resolution_vector else PRIMARY_OCR_PIXELS),
                    include_native_text=bool(characters),
                    recognize_words=high_resolution_vector,
                    parallel_tiles=2,
                ),
                internal_fallback_pass(
                    schematic=schematic,
                    scale=scale,
                    modes=(6,),
                    minimum_confidence=(
                        55.0
                        if (characters < 10 and evidence.image_count > 0)
                        else NATIVE_UNAVAILABLE_MIN_CONFIDENCE
                    ),
                    run_if_characters_below=weak_threshold,
                    include_native_text=bool(characters),
                ),
            ),
        )
    return WorkPlan(
        PageRoute.HYBRID,
        reason=PagePlanReason.NATIVE_TEXT_NEEDS_AUGMENTATION,
        ocr_passes=(
            OcrPass(
                "primary-page",
                OcrPassScope.PAGE,
                scale,
                image_modes if (schematic or evidence.image_count > 0) else (PSM_SPARSE_TEXT,),
                minimum_confidence=(55.0 if characters < 10 else RASTER_TEXT_MIN_CONFIDENCE),
                adaptive_scale=True,
                region_first=characters >= 10 or evidence.image_count >= 2,
                pixel_budget=PRIMARY_OCR_PIXELS,
                include_native_text=True,
            ),
            internal_fallback_pass(
                schematic=schematic,
                scale=scale,
                modes=(
                    image_modes if (schematic or evidence.image_count > 0) else (PSM_SPARSE_TEXT,)
                ),
                minimum_confidence=(55.0 if characters < 10 else RASTER_TEXT_MIN_CONFIDENCE),
                run_if_characters_below=300 if schematic else 32,
                include_native_text=True,
            ),
        ),
        fusion_policy=FusionPolicy.SPARSE_NATIVE,
    )


def internal_duplicate_of_native_text(
    native_compact: str,
    native_tokens: frozenset[str],
    ocr_text: str,
) -> bool:
    """Detect raster OCR that repeats the page's native text.

    Vector pages can have different coordinate systems after rasterization, so
    geometry alone cannot identify duplicate OCR. A compact text containment
    check is deliberately limited to reasonably long observations to avoid
    discarding short schematic labels such as ``R1`` or ``+5V``.
    """
    compact = compact_text(ocr_text)
    # Short observations get no containment shortcut; they fall through to the
    # token check so that labels such as ``R1`` are not discarded.
    if len(compact) >= 8 and compact in native_compact:
        return True
    tokens = text_tokens(ocr_text)
    return bool(tokens) and all(token in native_tokens for token in tokens)


def fuse_observations(
    native: ObservationBatch,
    ocr: ObservationBatch,
    plan: WorkPlan,
) -> ObservationBatch:
    route = plan.route
    if route is PageRoute.NATIVE or not len(ocr):
        return native
    if route is PageRoute.OCR or not len(native):
        return ocr
    if (
        plan.fusion_policy is FusionPolicy.SPARSE_NATIVE
        and len(native) < 16
        and len(ocr) >= len(native) * 4
        and sum(character.isalnum() for text in native.text for character in text) <= 4
    ):
        return ocr
    if plan.fusion_policy is FusionPolicy.NOISY_NATIVE:
        native_candidate = internal_candidate(-1, native)
        ocr_candidate = internal_candidate(-1, ocr)
        if (
            len(ocr) >= 4
            and ocr_candidate.metrics.mean_confidence >= FUSION_NOISY_NATIVE_MIN_CONFIDENCE
            and ocr_candidate.metrics.utility >= native_candidate.metrics.utility * 1.05
            and ocr_candidate.metrics.alphanumeric_characters
            >= native_candidate.metrics.alphanumeric_characters * 0.80
        ):
            return ocr

    minimum_confidence = (
        75.0
        if plan.image_regions_only
        else FUSION_NOISY_NATIVE_MIN_CONFIDENCE
        if plan.fusion_policy is FusionPolicy.NOISY_NATIVE
        else 30.0
        if plan.fusion_policy is FusionPolicy.UNCOVERED_VECTOR
        else 45.0
    )
    confidence_mask = ocr.confidence >= minimum_confidence
    if plan.image_regions_only or plan.fusion_policy is FusionPolicy.UNCOVERED_VECTOR:
        alphanumeric_mask = numpy.fromiter(
            (sum(character.isalnum() for character in text) >= 1 for text in ocr.text),
            dtype=numpy.bool_,
            count=len(ocr),
        )
    else:
        alphanumeric_mask = numpy.ones(len(ocr), dtype=numpy.bool_)
    # Image supplements share page coordinates with native text, so overlap
    # identifies duplicates without discarding repeated labels elsewhere.
    if not plan.image_regions_only:
        native_compact = "".join(compact_text(text) for text in native.text)
        native_tokens = frozenset(token for text in native.text for token in text_tokens(text))
        duplicate_mask = numpy.fromiter(
            (
                internal_duplicate_of_native_text(native_compact, native_tokens, text)
                for text in ocr.text
            ),
            dtype=numpy.bool_,
            count=len(ocr),
        )
    else:
        duplicate_mask = numpy.zeros(len(ocr), dtype=numpy.bool_)
    overlap = maximum_candidate_coverage(ocr.bbox, native.bbox)
    additions = confidence_mask & alphanumeric_mask & ~duplicate_mask & (overlap < 0.30)
    return ObservationBatch.concatenate_selected(native, ocr, additions)

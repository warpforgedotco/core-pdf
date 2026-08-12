# SPDX-License-Identifier: AGPL-3.0-only
"""Decide how a page should be extracted, producing a WorkPlan."""

from __future__ import annotations

from typing import cast

from core_pdf.impl.engine.parse.capture import (
    internal_hidden_text_needs_verification,
    internal_requires_high_resolution_vector_ocr,
)
from core_pdf.impl.engine.parse.model import (
    MAX_OCR_PIXELS,
    PRIMARY_OCR_PIXELS,
    CapturedPage,
    OcrPass,
    OcrPassScope,
    PageEvidence,
    PageRoute,
    WorkPlan,
)

PSM_AUTO = 3
PSM_SPARSE_TEXT = 11
PSM_SPARSE_TEXT_OSD = 12

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


def internal_ocr_scale(capture: CapturedPage, *, schematic: bool, vector_complexity: int) -> float:
    if not schematic or vector_complexity < 4_000:
        return 3.0
    if vector_complexity < 150_000:
        return 5.0
    # Raster-backed mixed pages already contain sampled detail; pure vector pages
    # need another half pixel per point to preserve their smallest labels.
    return 3.5 if capture.evidence.image_count else 4.0


def internal_vector_text_scale(capture: CapturedPage, vector_complexity: int) -> float:
    """Choose a higher raster scale for text embedded in vector artwork.

    Charts and diagrams often use small glyphs painted alongside thousands of
    vector paths.  The regular page scale is sufficient for prose, but loses
    those labels before OCR can associate them with the artwork.
    """
    return max(
        4.0, internal_ocr_scale(capture, schematic=True, vector_complexity=vector_complexity)
    )


def internal_rotated_native_characters(capture: CapturedPage) -> int:
    observations = getattr(capture, "observations", None)
    if observations is None or not hasattr(observations, "text"):
        return 0
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


def internal_drawing_is_simple_rectangle(drawing: object) -> bool:
    path = getattr(drawing, "path", None)
    subpaths: object = getattr(path, "subpaths", None)
    if not isinstance(subpaths, (list, tuple)) or len(subpaths) != 1:
        return False
    subpath = subpaths[0]
    points: object = getattr(subpath, "points", None)
    if (
        not isinstance(points, (list, tuple))
        or not getattr(subpath, "closed", False)
        or len(points) != 4
    ):
        return False
    rectangle_points = cast("list[tuple[float, float]] | tuple[tuple[float, float], ...]", points)
    xs = {point[0] for point in rectangle_points}
    ys = {point[1] for point in rectangle_points}
    if len(xs) != 2 or len(ys) != 2:
        return False
    return all(
        left[0] == right[0] or left[1] == right[1]
        for left, right in zip(
            rectangle_points,
            (*rectangle_points[1:], rectangle_points[0]),
            strict=True,
        )
    )


def internal_has_only_simple_vector_rectangles(capture: CapturedPage) -> bool:
    drawings = tuple(
        drawing
        for drawing in getattr(capture, "drawings", ())
        if getattr(drawing, "kind", None) in {"fill", "fillstroke", "stroke"}
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


def internal_base_plan_page(capture: CapturedPage) -> WorkPlan:
    evidence = capture.evidence
    total_characters = evidence.native_characters
    characters = evidence.visible_native_characters
    suspicious_ratio = evidence.suspicious_ratio
    vector_complexity = evidence.vector_complexity
    text_density = evidence.visible_text_density
    text_coverage = evidence.text_coverage
    observations = getattr(capture, "observations", None)
    rotated_native = bool(
        observations is not None and any(int(value) % 360 for value in observations.rotation)
    )
    quality = evidence.text_quality
    corrupt_mapping = (
        characters >= 24
        and quality.noise_score >= 0.20
        and quality.wordlike_ratio < 0.20
        and vector_complexity < 150
    )
    if corrupt_mapping:
        schematic = vector_complexity >= 180 and text_density < 0.0015
        schematic = schematic or (text_coverage < 0.05 and vector_complexity >= 180)
        image_modes = (PSM_AUTO,)
        scale = 6.0
        weak_threshold = 300 if schematic else 1_000
        high_resolution_vector = schematic and internal_requires_high_resolution_vector_ocr(capture)
        return WorkPlan(
            PageRoute.OCR,
            reason="native-text-corrupt",
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
                OcrPass(
                    "fallback-regions" if schematic else "fallback-page",
                    OcrPassScope.WEAK_REGIONS if schematic else OcrPassScope.PAGE,
                    scale,
                    (6,),
                    tiles=4 if schematic else 1,
                    minimum_confidence=(NATIVE_UNAVAILABLE_MIN_CONFIDENCE if schematic else 45.0),
                    run_if_characters_below=weak_threshold,
                    region_first=False,
                    include_native_text=True,
                ),
            ),
        )
    if evidence.vector_text_trusted:
        return WorkPlan(PageRoute.NATIVE, reason="newstroke-vector-text")
    if evidence.trusted_hidden_text:
        return WorkPlan(PageRoute.NATIVE, reason="trusted-hidden-native-text")
    if evidence.hidden_text_layer:
        hidden_text_scale = 2.0 if evidence.full_page_image and 20 <= characters < 32 else 3.0
        return WorkPlan(
            PageRoute.OCR,
            reason="unpainted-native-text-layer",
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
            reason="stroked-vector-text",
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
            return WorkPlan(PageRoute.NATIVE, reason="native-text-with-rectangular-vectors")
        if internal_vector_native_text_is_trusted(evidence):
            return WorkPlan(PageRoute.NATIVE, reason="glyph-trusted-vector-text")
        if (
            evidence.full_page_image
            and characters >= 1_000
            and internal_native_mapping_is_usable(evidence)
        ):
            return WorkPlan(PageRoute.NATIVE, reason="full-page-image-native-text")
        if (
            characters >= 1_000
            and evidence.text_coverage >= 0.15
            and evidence.uncovered_vector_area / max(1.0, evidence.page_area) < 0.08
            and suspicious_ratio <= 0.02
            and internal_native_mapping_is_usable(evidence)
        ):
            return WorkPlan(PageRoute.NATIVE, reason="mostly-covered-native-text")
        if (
            characters >= 1_500
            and evidence.image_count == 0
            and evidence.text_coverage >= 0.20
            and suspicious_ratio <= 0.02
            and internal_native_mapping_is_usable(evidence)
        ):
            return WorkPlan(PageRoute.NATIVE, reason="native-text-without-images")
        if (
            characters >= 3_000
            and evidence.text_coverage >= 0.18
            and suspicious_ratio <= 0.02
            and internal_native_mapping_is_usable(evidence)
        ):
            return WorkPlan(PageRoute.NATIVE, reason="dense-native-text")
        return WorkPlan(
            PageRoute.HYBRID if characters else PageRoute.OCR,
            reason="uncovered-vector-text",
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
            reason="noisy-native-text",
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
            reason="embedded-image-text-supplement",
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
            return WorkPlan(PageRoute.NATIVE, reason="glyph-trusted-rotated-text")
        if (
            characters >= 500
            and suspicious_ratio <= 0.05
            and rotated_characters <= max(80, int(characters * 0.03))
        ):
            return WorkPlan(PageRoute.NATIVE, reason="minor-rotated-native-text")
        return WorkPlan(
            PageRoute.HYBRID,
            reason="rotated-native-text",
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
        return WorkPlan(PageRoute.NATIVE, reason="healthy-native-text")
    if (
        characters >= 32
        and suspicious_ratio <= 0.05
        and evidence.image_count == 0
        and mapping_usable
    ):
        return WorkPlan(PageRoute.NATIVE, reason="usable-native-text")
    if (
        characters > 0
        and characters == total_characters
        and suspicious_ratio == 0.0
        and evidence.image_count == 0
        and vector_complexity < 30
        and mapping_usable
    ):
        return WorkPlan(PageRoute.NATIVE, reason="clean-short-native-text")

    schematic = vector_complexity >= 180 and text_density < 0.0015
    schematic = schematic or (text_coverage < 0.05 and vector_complexity >= 180)
    mode = PSM_AUTO if (schematic and vector_complexity >= 150_000) else PSM_SPARSE_TEXT
    image_modes = (mode,)
    scale = internal_ocr_scale(capture, schematic=schematic, vector_complexity=vector_complexity)
    if characters == 0 or suspicious_ratio >= 0.25:
        weak_threshold = 300 if schematic else 3
        high_resolution_vector = schematic and internal_requires_high_resolution_vector_ocr(capture)
        return WorkPlan(
            PageRoute.OCR,
            reason="native-text-unavailable",
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
                OcrPass(
                    "fallback-regions" if schematic else "fallback-page",
                    OcrPassScope.WEAK_REGIONS if schematic else OcrPassScope.PAGE,
                    scale,
                    (6,),
                    tiles=4 if schematic else 1,
                    minimum_confidence=(
                        55.0
                        if (characters < 10 and evidence.image_count > 0)
                        else NATIVE_UNAVAILABLE_MIN_CONFIDENCE
                    ),
                    run_if_characters_below=weak_threshold,
                    region_first=False,
                    include_native_text=bool(characters),
                ),
            ),
        )
    return WorkPlan(
        PageRoute.HYBRID,
        reason="native-text-needs-augmentation",
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
            OcrPass(
                "fallback-regions" if schematic else "fallback-page",
                OcrPassScope.WEAK_REGIONS if schematic else OcrPassScope.PAGE,
                scale,
                image_modes if (schematic or evidence.image_count > 0) else (PSM_SPARSE_TEXT,),
                tiles=4 if schematic else 1,
                minimum_confidence=(55.0 if characters < 10 else RASTER_TEXT_MIN_CONFIDENCE),
                run_if_characters_below=300 if schematic else 32,
                region_first=False,
                include_native_text=True,
            ),
        ),
    )


def plan_page(capture: CapturedPage) -> WorkPlan:
    return internal_base_plan_page(capture)

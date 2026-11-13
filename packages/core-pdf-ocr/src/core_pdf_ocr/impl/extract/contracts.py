# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from enum import StrEnum
from typing import Any

from core_pdf.impl.base.types import GeneratedRecord, RecordType, field
from core_pdf.impl.extract import contracts as extract_contracts
from core_pdf.impl.extract.contracts import ObservationBatch

PSM_AUTO = 3
PSM_SPARSE_TEXT = 11
PSM_SPARSE_TEXT_OSD = 12
VECTOR_PAINT_KINDS = frozenset({"fill", "fillstroke", "shading", "stroke"})
PRIMARY_OCR_PIXELS = 6_000_000
OCR_PREFLIGHT_PIXELS = 1_000_000
HIDDEN_TEXT_VERIFY_PIXELS = 2_000_000
MAX_OCR_PIXELS = 16_000_000
OCR_RESCUE_MIN_WEAK_INK_RATIO = 0.03
OCR_RESCUE_SATURATED_MEAN_INK = 0.85
OCR_RESCUE_MIN_CONFIDENCE = 95.0
OCR_RESCUE_DENSE_MIN_CHARACTERS: int = 2_000
OCR_RESCUE_DENSE_MIN_CONFIDENCE: float = 92.0
OCR_RESCUE_LARGE_TEXT_HEIGHT = 32.0
OCR_PARALLEL_TILE_MIN_VECTOR_COMPLEXITY = 100_000
HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE = 80.0
HIDDEN_TEXT_VERIFY_MIN_MATCHED_TOKENS = 24
HIDDEN_TEXT_VERIFY_MIN_TOKEN_OVERLAP = 0.72
HIDDEN_TEXT_VERIFY_MIN_SPATIAL_OVERLAP = 0.55


class PageRoute(StrEnum):
    NATIVE = "native"
    HYBRID = "hybrid"
    OCR = "ocr"


class OcrPassScope(StrEnum):
    PAGE = "page"
    TILES = "tiles"
    WEAK_REGIONS = "weak-regions"
    IMAGE_REGIONS = "image-regions"
    STROKED_VECTOR_TEXT = "stroked-vector-text"


class PagePlanReason(StrEnum):
    UNSPECIFIED = "unspecified"
    NATIVE_TEXT_CORRUPT = "native-text-corrupt"
    NEWSTROKE_VECTOR_TEXT = "newstroke-vector-text"
    TRUSTED_HIDDEN_NATIVE_TEXT = "trusted-hidden-native-text"
    UNPAINTED_NATIVE_TEXT_LAYER = "unpainted-native-text-layer"
    STROKED_VECTOR_TEXT = "stroked-vector-text"
    NATIVE_TEXT_WITH_RECTANGULAR_VECTORS = "native-text-with-rectangular-vectors"
    GLYPH_TRUSTED_VECTOR_TEXT = "glyph-trusted-vector-text"
    FULL_PAGE_IMAGE_NATIVE_TEXT = "full-page-image-native-text"
    MOSTLY_COVERED_NATIVE_TEXT = "mostly-covered-native-text"
    NATIVE_TEXT_WITHOUT_IMAGES = "native-text-without-images"
    DENSE_NATIVE_TEXT = "dense-native-text"
    UNCOVERED_VECTOR_TEXT = "uncovered-vector-text"
    NOISY_NATIVE_TEXT = "noisy-native-text"
    EMBEDDED_IMAGE_TEXT_SUPPLEMENT = "embedded-image-text-supplement"
    GLYPH_TRUSTED_ROTATED_TEXT = "glyph-trusted-rotated-text"
    MINOR_ROTATED_NATIVE_TEXT = "minor-rotated-native-text"
    ROTATED_NATIVE_TEXT = "rotated-native-text"
    HEALTHY_NATIVE_TEXT = "healthy-native-text"
    USABLE_NATIVE_TEXT = "usable-native-text"
    CLEAN_SHORT_NATIVE_TEXT = "clean-short-native-text"
    NATIVE_TEXT_UNAVAILABLE = "native-text-unavailable"
    NATIVE_TEXT_NEEDS_AUGMENTATION = "native-text-needs-augmentation"


class FusionPolicy(StrEnum):
    DEFAULT = "default"
    SPARSE_NATIVE = "sparse-native"
    NOISY_NATIVE = "noisy-native"
    UNCOVERED_VECTOR = "uncovered-vector"


class StrokedVectorTextEvidence(GeneratedRecord):
    trusted: bool = False
    drawing_indexes: tuple[int, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    candidate_paths: int = 0


class PageEvidence(extract_contracts.PageEvidence, metaclass=RecordType):
    vector_complexity: int = 0
    image_filters: tuple[str, ...] = ()
    uncovered_vector_area: float | None = None
    vector_text_trusted: bool = False
    stroked_vector_text: StrokedVectorTextEvidence = field(
        default_factory=StrokedVectorTextEvidence
    )


class PageAnalysis(extract_contracts.PageAnalysis):
    __slots__ = ()

    evidence: PageEvidence


class OcrPass(GeneratedRecord):
    name: str
    scope: OcrPassScope
    scale: float
    modes: tuple[int, ...]
    tiles: int = 1
    parallel_tiles: int = 1
    region_columns: int = 2
    max_regions: int = 3
    minimum_confidence: float = 20.0
    run_if_characters_below: int | None = None
    minimum_utility_gain: float = 1.1
    adaptive_scale: bool = False
    minimum_characters_for_rescue: int = 0
    character_confidence_threshold: float | None = None
    run_if_additions_below: int | None = None
    seed_with_native: bool = False
    region_first: bool = True
    preprocess: str = "none"
    pixel_budget: int = MAX_OCR_PIXELS
    include_native_text: bool = False
    recognize_words: bool = False
    collect_symbols: bool = False


class WorkPlan(GeneratedRecord):
    route: PageRoute
    reason: PagePlanReason = PagePlanReason.UNSPECIFIED
    ocr_passes: tuple[OcrPass, ...] = ()
    verify_hidden_text: bool = False
    fusion_policy: FusionPolicy = FusionPolicy.DEFAULT
    allow_direct_image_ocr: bool = True
    augment_page_candidates: bool = False

    @property
    def image_regions_only(self) -> bool:
        return bool(self.ocr_passes) and all(
            ocr_pass.scope is OcrPassScope.IMAGE_REGIONS for ocr_pass in self.ocr_passes
        )


class RecognitionResult(GeneratedRecord):
    observations: ObservationBatch
    stroked_vector_alphabet: tuple[tuple[Any, str], ...] = ()

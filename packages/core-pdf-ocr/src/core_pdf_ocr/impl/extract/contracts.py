# SPDX-License-Identifier: AGPL-3.0-only
"""Recognition contracts and evidence layered over native extraction records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

from core_pdf.impl.extract.contracts import (
    FULL_PAGE_IMAGE_COVERAGE as FULL_PAGE_IMAGE_COVERAGE,
)
from core_pdf.impl.extract.contracts import (
    BoolArray as BoolArray,
)
from core_pdf.impl.extract.contracts import (
    ByteArray as ByteArray,
)
from core_pdf.impl.extract.contracts import (
    FloatArray as FloatArray,
)
from core_pdf.impl.extract.contracts import (
    GlyphEvidence as GlyphEvidence,
)
from core_pdf.impl.extract.contracts import (
    IntArray as IntArray,
)
from core_pdf.impl.extract.contracts import (
    ObservationBatch as ObservationBatch,
)
from core_pdf.impl.extract.contracts import (
    PageAnalysis as NativePageAnalysis,
)
from core_pdf.impl.extract.contracts import (
    PageEvidence as NativePageEvidence,
)
from core_pdf.impl.extract.contracts import (
    ParsedBlock as ParsedBlock,
)
from core_pdf.impl.extract.contracts import (
    ParsedLine as ParsedLine,
)
from core_pdf.impl.extract.contracts import (
    ReadingOrderEvidence as ReadingOrderEvidence,
)
from core_pdf.impl.extract.contracts import (
    TextQualityStats as TextQualityStats,
)
from core_pdf.impl.extract.contracts import (
    internal_bbox_tuple as internal_bbox_tuple,
)

PSM_AUTO = 3
PSM_SPARSE_TEXT = 11
PSM_SPARSE_TEXT_OSD = 12
VECTOR_PAINT_KINDS = frozenset({"fill", "fillstroke", "shading", "stroke"})
PRIMARY_OCR_PIXELS = 6_000_000
OCR_PREFLIGHT_PIXELS = 1_000_000
HIDDEN_TEXT_VERIFY_PIXELS = 2_000_000
MAX_OCR_PIXELS = 16_000_000
MAX_OCR_RASTER_BYTES = MAX_OCR_PIXELS * 4
OCR_RESCUE_MIN_WEAK_INK_RATIO = 0.03
OCR_RESCUE_SATURATED_MEAN_INK = 0.85
OCR_RESCUE_MIN_CONFIDENCE = 95.0
internal_OCR_RESCUE_DENSE_MIN_CHARACTERS: int = 2_000
internal_OCR_RESCUE_DENSE_MIN_CONFIDENCE: float = 92.0
OCR_RESCUE_LARGE_TEXT_HEIGHT = 32.0
OCR_PARALLEL_TILE_MIN_VECTOR_COMPLEXITY = 100_000
HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE = 80.0
HIDDEN_TEXT_VERIFY_MIN_MATCHED_TOKENS = 24
HIDDEN_TEXT_VERIFY_MIN_TOKEN_OVERLAP = 0.72
HIDDEN_TEXT_VERIFY_MIN_SPATIAL_OVERLAP = 0.55


class ObservationSource(IntEnum):
    NATIVE = 0
    OCR = 1
    STRUCTURE = 2


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
    """Stable explanations for why the router chose a page plan."""

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
    """How hybrid OCR observations interact with the native text layer."""

    DEFAULT = "default"
    SPARSE_NATIVE = "sparse-native"
    NOISY_NATIVE = "noisy-native"
    UNCOVERED_VECTOR = "uncovered-vector"


@dataclass(frozen=True, slots=True)
class StrokedVectorTextEvidence:
    """Compact path families that are likely to be flattened single-line text."""

    trusted: bool = False
    drawing_indexes: tuple[int, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    candidate_paths: int = 0


@dataclass(frozen=True, slots=True)
class PageEvidence(NativePageEvidence):
    """Native page evidence enriched with recognition-specific vector signals."""

    vector_complexity: int = 0
    image_filters: tuple[str, ...] = ()
    uncovered_vector_area: float | None = None
    vector_text_characters: int = 0
    vector_text_candidate_segments: int = 0
    vector_text_matched_segments: int = 0
    vector_text_sequences: int = 0
    vector_text_maximum_error: float = 0.0
    vector_text_trusted: bool = False
    stroked_vector_text: StrokedVectorTextEvidence = field(
        default_factory=StrokedVectorTextEvidence
    )

    @property
    def vector_text_segment_coverage(self) -> float:
        return self.vector_text_matched_segments / max(1, self.vector_text_candidate_segments)


@dataclass(frozen=True, slots=True)
class PageAnalysis(NativePageAnalysis):
    evidence: PageEvidence


@dataclass(frozen=True, slots=True)
class OcrPass:
    """One independently measurable OCR operation over a declared raster scope."""

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
    minimum_utility_gain: float = 1.10
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


@dataclass(frozen=True, slots=True)
class WorkPlan:
    route: PageRoute
    reason: PagePlanReason = PagePlanReason.UNSPECIFIED
    ocr_passes: tuple[OcrPass, ...] = ()
    verify_hidden_text: bool = False
    fusion_policy: FusionPolicy = FusionPolicy.DEFAULT
    allow_direct_image_ocr: bool = True
    augment_page_candidates: bool = False

    def __post_init__(self) -> None:
        # Keep direct construction ergonomic for tests and downstream internal
        # callers while guaranteeing that the stored contract is typed.
        if not isinstance(self.reason, PagePlanReason):
            object.__setattr__(self, "reason", PagePlanReason(self.reason))

    @property
    def image_regions_only(self) -> bool:
        return bool(self.ocr_passes) and all(
            ocr_pass.scope is OcrPassScope.IMAGE_REGIONS for ocr_pass in self.ocr_passes
        )


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    observations: ObservationBatch
    stroked_vector_alphabet: tuple[tuple[Any, str], ...] = ()

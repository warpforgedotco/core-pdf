# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any, ClassVar

from core_pdf.impl.extract_contracts import (
    GlyphEvidence,
    ObservationBatch,
    TextQualityStats,
)
from core_pdf.impl.extract_contracts import (
    PageAnalysis as NativePageAnalysis,
)
from core_pdf.impl.extract_contracts import (
    PageEvidence as NativePageEvidence,
)
from core_pdf.impl.types import Record, frozen_setattr

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
OCR_RESCUE_DENSE_MIN_CHARACTERS: int = 2_000
OCR_RESCUE_DENSE_MIN_CONFIDENCE: float = 92.0
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


class StrokedVectorTextEvidence(Record):
    __slots__ = ("trusted", "drawing_indexes", "bbox", "candidate_paths")

    trusted: bool
    drawing_indexes: tuple[int, ...]
    bbox: tuple[float, float, float, float] | None
    candidate_paths: int

    __fields__: ClassVar[tuple[str, ...]] = (
        "trusted",
        "drawing_indexes",
        "bbox",
        "candidate_paths",
    )
    __match_args__ = ("trusted", "drawing_indexes", "bbox", "candidate_paths")

    def __init__(
        self,
        trusted: bool = False,
        drawing_indexes: tuple[int, ...] = (),
        bbox: tuple[float, float, float, float] | None = None,
        candidate_paths: int = 0,
    ) -> None:
        frozen_setattr(self, "trusted", trusted)
        frozen_setattr(self, "drawing_indexes", drawing_indexes)
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "candidate_paths", candidate_paths)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.trusted == other.trusted
            and self.drawing_indexes == other.drawing_indexes
            and self.bbox == other.bbox
            and self.candidate_paths == other.candidate_paths
        )

    def __hash__(self) -> int:
        return hash((self.trusted, self.drawing_indexes, self.bbox, self.candidate_paths))


class PageEvidence(NativePageEvidence):
    __slots__ = (
        "vector_complexity",
        "image_filters",
        "uncovered_vector_area",
        "vector_text_candidate_segments",
        "vector_text_matched_segments",
        "vector_text_trusted",
        "stroked_vector_text",
    )

    vector_complexity: int
    image_filters: tuple[str, ...]
    uncovered_vector_area: float | None
    vector_text_candidate_segments: int
    vector_text_matched_segments: int
    vector_text_trusted: bool
    stroked_vector_text: StrokedVectorTextEvidence

    __fields__: ClassVar[tuple[str, ...]] = (
        "page_area",
        "native_characters",
        "visible_native_characters",
        "suspicious_characters",
        "image_count",
        "image_area_ratio",
        "image_boxes",
        "text_coverage",
        "full_page_image",
        "text_quality",
        "all_text_quality",
        "glyphs",
        "painted_native_characters",
        "trusted_hidden_text",
        "vector_complexity",
        "image_filters",
        "uncovered_vector_area",
        "vector_text_candidate_segments",
        "vector_text_matched_segments",
        "vector_text_trusted",
        "stroked_vector_text",
    )
    __match_args__ = (
        "page_area",
        "native_characters",
        "visible_native_characters",
        "suspicious_characters",
        "image_count",
        "image_area_ratio",
        "image_boxes",
        "text_coverage",
        "full_page_image",
        "text_quality",
        "all_text_quality",
        "glyphs",
        "painted_native_characters",
        "trusted_hidden_text",
        "vector_complexity",
        "image_filters",
        "uncovered_vector_area",
        "vector_text_candidate_segments",
        "vector_text_matched_segments",
        "vector_text_trusted",
        "stroked_vector_text",
    )

    def __init__(
        self,
        page_area: float,
        native_characters: int,
        visible_native_characters: int,
        suspicious_characters: int,
        image_count: int,
        image_area_ratio: float,
        image_boxes: tuple[tuple[float, float, float, float], ...] = (),
        text_coverage: float = 0.0,
        full_page_image: bool = False,
        text_quality: TextQualityStats | None = None,
        all_text_quality: TextQualityStats | None = None,
        glyphs: GlyphEvidence | None = None,
        painted_native_characters: int | None = None,
        trusted_hidden_text: bool = False,
        vector_complexity: int = 0,
        image_filters: tuple[str, ...] = (),
        uncovered_vector_area: float | None = None,
        vector_text_candidate_segments: int = 0,
        vector_text_matched_segments: int = 0,
        vector_text_trusted: bool = False,
        stroked_vector_text: StrokedVectorTextEvidence | None = None,
    ) -> None:
        frozen_setattr(self, "page_area", page_area)
        frozen_setattr(self, "native_characters", native_characters)
        frozen_setattr(self, "visible_native_characters", visible_native_characters)
        frozen_setattr(self, "suspicious_characters", suspicious_characters)
        frozen_setattr(self, "image_count", image_count)
        frozen_setattr(self, "image_area_ratio", image_area_ratio)
        frozen_setattr(self, "image_boxes", image_boxes)
        frozen_setattr(self, "text_coverage", text_coverage)
        frozen_setattr(self, "full_page_image", full_page_image)
        frozen_setattr(
            self, "text_quality", TextQualityStats() if text_quality is None else text_quality
        )
        frozen_setattr(
            self,
            "all_text_quality",
            TextQualityStats() if all_text_quality is None else all_text_quality,
        )
        frozen_setattr(self, "glyphs", GlyphEvidence() if glyphs is None else glyphs)
        frozen_setattr(self, "painted_native_characters", painted_native_characters)
        frozen_setattr(self, "trusted_hidden_text", trusted_hidden_text)
        frozen_setattr(self, "vector_complexity", vector_complexity)
        frozen_setattr(self, "image_filters", image_filters)
        frozen_setattr(self, "uncovered_vector_area", uncovered_vector_area)
        frozen_setattr(self, "vector_text_candidate_segments", vector_text_candidate_segments)
        frozen_setattr(self, "vector_text_matched_segments", vector_text_matched_segments)
        frozen_setattr(self, "vector_text_trusted", vector_text_trusted)
        frozen_setattr(
            self,
            "stroked_vector_text",
            StrokedVectorTextEvidence() if stroked_vector_text is None else stroked_vector_text,
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.page_area == other.page_area
            and self.native_characters == other.native_characters
            and self.visible_native_characters == other.visible_native_characters
            and self.suspicious_characters == other.suspicious_characters
            and self.image_count == other.image_count
            and self.image_area_ratio == other.image_area_ratio
            and self.image_boxes == other.image_boxes
            and self.text_coverage == other.text_coverage
            and self.full_page_image == other.full_page_image
            and self.text_quality == other.text_quality
            and self.all_text_quality == other.all_text_quality
            and self.glyphs == other.glyphs
            and self.painted_native_characters == other.painted_native_characters
            and self.trusted_hidden_text == other.trusted_hidden_text
            and self.vector_complexity == other.vector_complexity
            and self.image_filters == other.image_filters
            and self.uncovered_vector_area == other.uncovered_vector_area
            and self.vector_text_candidate_segments == other.vector_text_candidate_segments
            and self.vector_text_matched_segments == other.vector_text_matched_segments
            and self.vector_text_trusted == other.vector_text_trusted
            and self.stroked_vector_text == other.stroked_vector_text
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.page_area,
                self.native_characters,
                self.visible_native_characters,
                self.suspicious_characters,
                self.image_count,
                self.image_area_ratio,
                self.image_boxes,
                self.text_coverage,
                self.full_page_image,
                self.text_quality,
                self.all_text_quality,
                self.glyphs,
                self.painted_native_characters,
                self.trusted_hidden_text,
                self.vector_complexity,
                self.image_filters,
                self.uncovered_vector_area,
                self.vector_text_candidate_segments,
                self.vector_text_matched_segments,
                self.vector_text_trusted,
                self.stroked_vector_text,
            )
        )


class PageAnalysis(NativePageAnalysis):
    """Core's page analysis, carrying the OCR evidence.

    Its fields, constructor, equality, hash, repr and replace are core's.
    """

    __slots__ = ()

    evidence: PageEvidence


class OcrPass(Record):
    __slots__ = (
        "name",
        "scope",
        "scale",
        "modes",
        "tiles",
        "parallel_tiles",
        "region_columns",
        "max_regions",
        "minimum_confidence",
        "run_if_characters_below",
        "minimum_utility_gain",
        "adaptive_scale",
        "minimum_characters_for_rescue",
        "character_confidence_threshold",
        "run_if_additions_below",
        "seed_with_native",
        "region_first",
        "preprocess",
        "pixel_budget",
        "include_native_text",
        "recognize_words",
        "collect_symbols",
    )

    name: str
    scope: OcrPassScope
    scale: float
    modes: tuple[int, ...]
    tiles: int
    parallel_tiles: int
    region_columns: int
    max_regions: int
    minimum_confidence: float
    run_if_characters_below: int | None
    minimum_utility_gain: float
    adaptive_scale: bool
    minimum_characters_for_rescue: int
    character_confidence_threshold: float | None
    run_if_additions_below: int | None
    seed_with_native: bool
    region_first: bool
    preprocess: str
    pixel_budget: int
    include_native_text: bool
    recognize_words: bool
    collect_symbols: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "name",
        "scope",
        "scale",
        "modes",
        "tiles",
        "parallel_tiles",
        "region_columns",
        "max_regions",
        "minimum_confidence",
        "run_if_characters_below",
        "minimum_utility_gain",
        "adaptive_scale",
        "minimum_characters_for_rescue",
        "character_confidence_threshold",
        "run_if_additions_below",
        "seed_with_native",
        "region_first",
        "preprocess",
        "pixel_budget",
        "include_native_text",
        "recognize_words",
        "collect_symbols",
    )
    __match_args__ = (
        "name",
        "scope",
        "scale",
        "modes",
        "tiles",
        "parallel_tiles",
        "region_columns",
        "max_regions",
        "minimum_confidence",
        "run_if_characters_below",
        "minimum_utility_gain",
        "adaptive_scale",
        "minimum_characters_for_rescue",
        "character_confidence_threshold",
        "run_if_additions_below",
        "seed_with_native",
        "region_first",
        "preprocess",
        "pixel_budget",
        "include_native_text",
        "recognize_words",
        "collect_symbols",
    )

    def __init__(
        self,
        name: str,
        scope: OcrPassScope,
        scale: float,
        modes: tuple[int, ...],
        tiles: int = 1,
        parallel_tiles: int = 1,
        region_columns: int = 2,
        max_regions: int = 3,
        minimum_confidence: float = 20.0,
        run_if_characters_below: int | None = None,
        minimum_utility_gain: float = 1.1,
        adaptive_scale: bool = False,
        minimum_characters_for_rescue: int = 0,
        character_confidence_threshold: float | None = None,
        run_if_additions_below: int | None = None,
        seed_with_native: bool = False,
        region_first: bool = True,
        preprocess: str = "none",
        pixel_budget: int = MAX_OCR_PIXELS,
        include_native_text: bool = False,
        recognize_words: bool = False,
        collect_symbols: bool = False,
    ) -> None:
        frozen_setattr(self, "name", name)
        frozen_setattr(self, "scope", scope)
        frozen_setattr(self, "scale", scale)
        frozen_setattr(self, "modes", modes)
        frozen_setattr(self, "tiles", tiles)
        frozen_setattr(self, "parallel_tiles", parallel_tiles)
        frozen_setattr(self, "region_columns", region_columns)
        frozen_setattr(self, "max_regions", max_regions)
        frozen_setattr(self, "minimum_confidence", minimum_confidence)
        frozen_setattr(self, "run_if_characters_below", run_if_characters_below)
        frozen_setattr(self, "minimum_utility_gain", minimum_utility_gain)
        frozen_setattr(self, "adaptive_scale", adaptive_scale)
        frozen_setattr(self, "minimum_characters_for_rescue", minimum_characters_for_rescue)
        frozen_setattr(self, "character_confidence_threshold", character_confidence_threshold)
        frozen_setattr(self, "run_if_additions_below", run_if_additions_below)
        frozen_setattr(self, "seed_with_native", seed_with_native)
        frozen_setattr(self, "region_first", region_first)
        frozen_setattr(self, "preprocess", preprocess)
        frozen_setattr(self, "pixel_budget", pixel_budget)
        frozen_setattr(self, "include_native_text", include_native_text)
        frozen_setattr(self, "recognize_words", recognize_words)
        frozen_setattr(self, "collect_symbols", collect_symbols)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.name == other.name
            and self.scope == other.scope
            and self.scale == other.scale
            and self.modes == other.modes
            and self.tiles == other.tiles
            and self.parallel_tiles == other.parallel_tiles
            and self.region_columns == other.region_columns
            and self.max_regions == other.max_regions
            and self.minimum_confidence == other.minimum_confidence
            and self.run_if_characters_below == other.run_if_characters_below
            and self.minimum_utility_gain == other.minimum_utility_gain
            and self.adaptive_scale == other.adaptive_scale
            and self.minimum_characters_for_rescue == other.minimum_characters_for_rescue
            and self.character_confidence_threshold == other.character_confidence_threshold
            and self.run_if_additions_below == other.run_if_additions_below
            and self.seed_with_native == other.seed_with_native
            and self.region_first == other.region_first
            and self.preprocess == other.preprocess
            and self.pixel_budget == other.pixel_budget
            and self.include_native_text == other.include_native_text
            and self.recognize_words == other.recognize_words
            and self.collect_symbols == other.collect_symbols
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.scope,
                self.scale,
                self.modes,
                self.tiles,
                self.parallel_tiles,
                self.region_columns,
                self.max_regions,
                self.minimum_confidence,
                self.run_if_characters_below,
                self.minimum_utility_gain,
                self.adaptive_scale,
                self.minimum_characters_for_rescue,
                self.character_confidence_threshold,
                self.run_if_additions_below,
                self.seed_with_native,
                self.region_first,
                self.preprocess,
                self.pixel_budget,
                self.include_native_text,
                self.recognize_words,
                self.collect_symbols,
            )
        )


class WorkPlan(Record):
    __slots__ = (
        "route",
        "reason",
        "ocr_passes",
        "verify_hidden_text",
        "fusion_policy",
        "allow_direct_image_ocr",
        "augment_page_candidates",
    )

    route: PageRoute
    reason: PagePlanReason
    ocr_passes: tuple[OcrPass, ...]
    verify_hidden_text: bool
    fusion_policy: FusionPolicy
    allow_direct_image_ocr: bool
    augment_page_candidates: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "route",
        "reason",
        "ocr_passes",
        "verify_hidden_text",
        "fusion_policy",
        "allow_direct_image_ocr",
        "augment_page_candidates",
    )
    __match_args__ = (
        "route",
        "reason",
        "ocr_passes",
        "verify_hidden_text",
        "fusion_policy",
        "allow_direct_image_ocr",
        "augment_page_candidates",
    )

    def __init__(
        self,
        route: PageRoute,
        reason: PagePlanReason = PagePlanReason.UNSPECIFIED,
        ocr_passes: tuple[OcrPass, ...] = (),
        verify_hidden_text: bool = False,
        fusion_policy: FusionPolicy = FusionPolicy.DEFAULT,
        allow_direct_image_ocr: bool = True,
        augment_page_candidates: bool = False,
    ) -> None:
        frozen_setattr(self, "route", route)
        frozen_setattr(self, "reason", reason)
        frozen_setattr(self, "ocr_passes", ocr_passes)
        frozen_setattr(self, "verify_hidden_text", verify_hidden_text)
        frozen_setattr(self, "fusion_policy", fusion_policy)
        frozen_setattr(self, "allow_direct_image_ocr", allow_direct_image_ocr)
        frozen_setattr(self, "augment_page_candidates", augment_page_candidates)
        self._post_init()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.route == other.route
            and self.reason == other.reason
            and self.ocr_passes == other.ocr_passes
            and self.verify_hidden_text == other.verify_hidden_text
            and self.fusion_policy == other.fusion_policy
            and self.allow_direct_image_ocr == other.allow_direct_image_ocr
            and self.augment_page_candidates == other.augment_page_candidates
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.route,
                self.reason,
                self.ocr_passes,
                self.verify_hidden_text,
                self.fusion_policy,
                self.allow_direct_image_ocr,
                self.augment_page_candidates,
            )
        )

    def _post_init(self) -> None:
        if not isinstance(self.reason, PagePlanReason):
            object.__setattr__(self, "reason", PagePlanReason(self.reason))

    @property
    def image_regions_only(self) -> bool:
        return bool(self.ocr_passes) and all(
            ocr_pass.scope is OcrPassScope.IMAGE_REGIONS for ocr_pass in self.ocr_passes
        )


class RecognitionResult(Record):
    __slots__ = ("observations", "stroked_vector_alphabet")

    observations: ObservationBatch
    stroked_vector_alphabet: tuple[tuple[Any, str], ...]

    __fields__: ClassVar[tuple[str, ...]] = ("observations", "stroked_vector_alphabet")
    __match_args__ = ("observations", "stroked_vector_alphabet")

    def __init__(
        self,
        observations: ObservationBatch,
        stroked_vector_alphabet: tuple[tuple[Any, str], ...] = (),
    ) -> None:
        frozen_setattr(self, "observations", observations)
        frozen_setattr(self, "stroked_vector_alphabet", stroked_vector_alphabet)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.observations == other.observations
            and self.stroked_vector_alphabet == other.stroked_vector_alphabet
        )

    def __hash__(self) -> int:
        return hash((self.observations, self.stroked_vector_alphabet))

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Self

from core_pdf.impl._impl.extract.contracts import (
    GlyphEvidence,
    ObservationBatch,
    TextQualityStats,
)
from core_pdf.impl._impl.extract.contracts import (
    PageAnalysis as NativePageAnalysis,
)
from core_pdf.impl._impl.extract.contracts import (
    PageEvidence as NativePageEvidence,
)

if TYPE_CHECKING:
    from core_pdf.impl._impl.capture.program import PageProgram
    from core_pdf.impl._impl.document.page import PdfPage
    from core_pdf.impl._impl.document.records import RawAnnotation, RawFormField

internal_frozen_setattr = object.__setattr__


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


class StrokedVectorTextEvidence:
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
        internal_frozen_setattr(self, "trusted", trusted)
        internal_frozen_setattr(self, "drawing_indexes", drawing_indexes)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "candidate_paths", candidate_paths)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"trusted={self.trusted!r}, "
            f"drawing_indexes={self.drawing_indexes!r}, "
            f"bbox={self.bbox!r}, "
            f"candidate_paths={self.candidate_paths!r}"
            ")"
        )

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        trusted = changes.pop("trusted", self.trusted)
        drawing_indexes = changes.pop("drawing_indexes", self.drawing_indexes)
        bbox = changes.pop("bbox", self.bbox)
        candidate_paths = changes.pop("candidate_paths", self.candidate_paths)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(trusted, drawing_indexes, bbox, candidate_paths)


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
        internal_frozen_setattr(self, "page_area", page_area)
        internal_frozen_setattr(self, "native_characters", native_characters)
        internal_frozen_setattr(self, "visible_native_characters", visible_native_characters)
        internal_frozen_setattr(self, "suspicious_characters", suspicious_characters)
        internal_frozen_setattr(self, "image_count", image_count)
        internal_frozen_setattr(self, "image_area_ratio", image_area_ratio)
        internal_frozen_setattr(self, "image_boxes", image_boxes)
        internal_frozen_setattr(self, "text_coverage", text_coverage)
        internal_frozen_setattr(self, "full_page_image", full_page_image)
        internal_frozen_setattr(
            self, "text_quality", TextQualityStats() if text_quality is None else text_quality
        )
        internal_frozen_setattr(
            self,
            "all_text_quality",
            TextQualityStats() if all_text_quality is None else all_text_quality,
        )
        internal_frozen_setattr(self, "glyphs", GlyphEvidence() if glyphs is None else glyphs)
        internal_frozen_setattr(self, "painted_native_characters", painted_native_characters)
        internal_frozen_setattr(self, "trusted_hidden_text", trusted_hidden_text)
        internal_frozen_setattr(self, "vector_complexity", vector_complexity)
        internal_frozen_setattr(self, "image_filters", image_filters)
        internal_frozen_setattr(self, "uncovered_vector_area", uncovered_vector_area)
        internal_frozen_setattr(
            self, "vector_text_candidate_segments", vector_text_candidate_segments
        )
        internal_frozen_setattr(self, "vector_text_matched_segments", vector_text_matched_segments)
        internal_frozen_setattr(self, "vector_text_trusted", vector_text_trusted)
        internal_frozen_setattr(
            self,
            "stroked_vector_text",
            StrokedVectorTextEvidence() if stroked_vector_text is None else stroked_vector_text,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"page_area={self.page_area!r}, "
            f"native_characters={self.native_characters!r}, "
            f"visible_native_characters={self.visible_native_characters!r}, "
            f"suspicious_characters={self.suspicious_characters!r}, "
            f"image_count={self.image_count!r}, "
            f"image_area_ratio={self.image_area_ratio!r}, "
            f"image_boxes={self.image_boxes!r}, "
            f"text_coverage={self.text_coverage!r}, "
            f"full_page_image={self.full_page_image!r}, "
            f"text_quality={self.text_quality!r}, "
            f"all_text_quality={self.all_text_quality!r}, "
            f"glyphs={self.glyphs!r}, "
            f"painted_native_characters={self.painted_native_characters!r}, "
            f"trusted_hidden_text={self.trusted_hidden_text!r}, "
            f"vector_complexity={self.vector_complexity!r}, "
            f"image_filters={self.image_filters!r}, "
            f"uncovered_vector_area={self.uncovered_vector_area!r}, "
            f"vector_text_candidate_segments={self.vector_text_candidate_segments!r}, "
            f"vector_text_matched_segments={self.vector_text_matched_segments!r}, "
            f"vector_text_trusted={self.vector_text_trusted!r}, "
            f"stroked_vector_text={self.stroked_vector_text!r}"
            ")"
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

    def __replace__(self, /, **changes: Any) -> Self:
        page_area = changes.pop("page_area", self.page_area)
        native_characters = changes.pop("native_characters", self.native_characters)
        visible_native_characters = changes.pop(
            "visible_native_characters", self.visible_native_characters
        )
        suspicious_characters = changes.pop("suspicious_characters", self.suspicious_characters)
        image_count = changes.pop("image_count", self.image_count)
        image_area_ratio = changes.pop("image_area_ratio", self.image_area_ratio)
        image_boxes = changes.pop("image_boxes", self.image_boxes)
        text_coverage = changes.pop("text_coverage", self.text_coverage)
        full_page_image = changes.pop("full_page_image", self.full_page_image)
        text_quality = changes.pop("text_quality", self.text_quality)
        all_text_quality = changes.pop("all_text_quality", self.all_text_quality)
        glyphs = changes.pop("glyphs", self.glyphs)
        painted_native_characters = changes.pop(
            "painted_native_characters", self.painted_native_characters
        )
        trusted_hidden_text = changes.pop("trusted_hidden_text", self.trusted_hidden_text)
        vector_complexity = changes.pop("vector_complexity", self.vector_complexity)
        image_filters = changes.pop("image_filters", self.image_filters)
        uncovered_vector_area = changes.pop("uncovered_vector_area", self.uncovered_vector_area)
        vector_text_candidate_segments = changes.pop(
            "vector_text_candidate_segments", self.vector_text_candidate_segments
        )
        vector_text_matched_segments = changes.pop(
            "vector_text_matched_segments", self.vector_text_matched_segments
        )
        vector_text_trusted = changes.pop("vector_text_trusted", self.vector_text_trusted)
        stroked_vector_text = changes.pop("stroked_vector_text", self.stroked_vector_text)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            page_area,
            native_characters,
            visible_native_characters,
            suspicious_characters,
            image_count,
            image_area_ratio,
            image_boxes,
            text_coverage,
            full_page_image,
            text_quality,
            all_text_quality,
            glyphs,
            painted_native_characters,
            trusted_hidden_text,
            vector_complexity,
            image_filters,
            uncovered_vector_area,
            vector_text_candidate_segments,
            vector_text_matched_segments,
            vector_text_trusted,
            stroked_vector_text,
        )

    @property
    def vector_text_segment_coverage(self) -> float:
        return self.vector_text_matched_segments / max(1, self.vector_text_candidate_segments)


class PageAnalysis(NativePageAnalysis):
    __slots__ = ()

    evidence: PageEvidence

    __fields__: ClassVar[tuple[str, ...]] = (
        "page",
        "width",
        "height",
        "rotation",
        "fields",
        "annotations",
        "program",
        "observations",
        "evidence",
    )
    __match_args__ = (
        "page",
        "width",
        "height",
        "rotation",
        "fields",
        "annotations",
        "program",
        "observations",
        "evidence",
    )

    def __init__(
        self,
        page: PdfPage,
        width: float,
        height: float,
        rotation: int,
        fields: tuple[RawFormField, ...],
        annotations: tuple[RawAnnotation, ...],
        program: PageProgram,
        observations: ObservationBatch,
        evidence: PageEvidence,
    ) -> None:
        internal_frozen_setattr(self, "page", page)
        internal_frozen_setattr(self, "width", width)
        internal_frozen_setattr(self, "height", height)
        internal_frozen_setattr(self, "rotation", rotation)
        internal_frozen_setattr(self, "fields", fields)
        internal_frozen_setattr(self, "annotations", annotations)
        internal_frozen_setattr(self, "program", program)
        internal_frozen_setattr(self, "observations", observations)
        internal_frozen_setattr(self, "evidence", evidence)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"page={self.page!r}, "
            f"width={self.width!r}, "
            f"height={self.height!r}, "
            f"rotation={self.rotation!r}, "
            f"fields={self.fields!r}, "
            f"annotations={self.annotations!r}, "
            f"program={self.program!r}, "
            f"observations={self.observations!r}, "
            f"evidence={self.evidence!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.page == other.page
            and self.width == other.width
            and self.height == other.height
            and self.rotation == other.rotation
            and self.fields == other.fields
            and self.annotations == other.annotations
            and self.program == other.program
            and self.observations == other.observations
            and self.evidence == other.evidence
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.page,
                self.width,
                self.height,
                self.rotation,
                self.fields,
                self.annotations,
                self.program,
                self.observations,
                self.evidence,
            )
        )

    def __replace__(self, /, **changes: Any) -> Self:
        page = changes.pop("page", self.page)
        width = changes.pop("width", self.width)
        height = changes.pop("height", self.height)
        rotation = changes.pop("rotation", self.rotation)
        fields = changes.pop("fields", self.fields)
        annotations = changes.pop("annotations", self.annotations)
        program = changes.pop("program", self.program)
        observations = changes.pop("observations", self.observations)
        evidence = changes.pop("evidence", self.evidence)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            page,
            width,
            height,
            rotation,
            fields,
            annotations,
            program,
            observations,
            evidence,
        )


class OcrPass:
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
        internal_frozen_setattr(self, "name", name)
        internal_frozen_setattr(self, "scope", scope)
        internal_frozen_setattr(self, "scale", scale)
        internal_frozen_setattr(self, "modes", modes)
        internal_frozen_setattr(self, "tiles", tiles)
        internal_frozen_setattr(self, "parallel_tiles", parallel_tiles)
        internal_frozen_setattr(self, "region_columns", region_columns)
        internal_frozen_setattr(self, "max_regions", max_regions)
        internal_frozen_setattr(self, "minimum_confidence", minimum_confidence)
        internal_frozen_setattr(self, "run_if_characters_below", run_if_characters_below)
        internal_frozen_setattr(self, "minimum_utility_gain", minimum_utility_gain)
        internal_frozen_setattr(self, "adaptive_scale", adaptive_scale)
        internal_frozen_setattr(
            self, "minimum_characters_for_rescue", minimum_characters_for_rescue
        )
        internal_frozen_setattr(
            self, "character_confidence_threshold", character_confidence_threshold
        )
        internal_frozen_setattr(self, "run_if_additions_below", run_if_additions_below)
        internal_frozen_setattr(self, "seed_with_native", seed_with_native)
        internal_frozen_setattr(self, "region_first", region_first)
        internal_frozen_setattr(self, "preprocess", preprocess)
        internal_frozen_setattr(self, "pixel_budget", pixel_budget)
        internal_frozen_setattr(self, "include_native_text", include_native_text)
        internal_frozen_setattr(self, "recognize_words", recognize_words)
        internal_frozen_setattr(self, "collect_symbols", collect_symbols)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"name={self.name!r}, "
            f"scope={self.scope!r}, "
            f"scale={self.scale!r}, "
            f"modes={self.modes!r}, "
            f"tiles={self.tiles!r}, "
            f"parallel_tiles={self.parallel_tiles!r}, "
            f"region_columns={self.region_columns!r}, "
            f"max_regions={self.max_regions!r}, "
            f"minimum_confidence={self.minimum_confidence!r}, "
            f"run_if_characters_below={self.run_if_characters_below!r}, "
            f"minimum_utility_gain={self.minimum_utility_gain!r}, "
            f"adaptive_scale={self.adaptive_scale!r}, "
            f"minimum_characters_for_rescue={self.minimum_characters_for_rescue!r}, "
            f"character_confidence_threshold={self.character_confidence_threshold!r}, "
            f"run_if_additions_below={self.run_if_additions_below!r}, "
            f"seed_with_native={self.seed_with_native!r}, "
            f"region_first={self.region_first!r}, "
            f"preprocess={self.preprocess!r}, "
            f"pixel_budget={self.pixel_budget!r}, "
            f"include_native_text={self.include_native_text!r}, "
            f"recognize_words={self.recognize_words!r}, "
            f"collect_symbols={self.collect_symbols!r}"
            ")"
        )

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        name = changes.pop("name", self.name)
        scope = changes.pop("scope", self.scope)
        scale = changes.pop("scale", self.scale)
        modes = changes.pop("modes", self.modes)
        tiles = changes.pop("tiles", self.tiles)
        parallel_tiles = changes.pop("parallel_tiles", self.parallel_tiles)
        region_columns = changes.pop("region_columns", self.region_columns)
        max_regions = changes.pop("max_regions", self.max_regions)
        minimum_confidence = changes.pop("minimum_confidence", self.minimum_confidence)
        run_if_characters_below = changes.pop(
            "run_if_characters_below", self.run_if_characters_below
        )
        minimum_utility_gain = changes.pop("minimum_utility_gain", self.minimum_utility_gain)
        adaptive_scale = changes.pop("adaptive_scale", self.adaptive_scale)
        minimum_characters_for_rescue = changes.pop(
            "minimum_characters_for_rescue", self.minimum_characters_for_rescue
        )
        character_confidence_threshold = changes.pop(
            "character_confidence_threshold", self.character_confidence_threshold
        )
        run_if_additions_below = changes.pop("run_if_additions_below", self.run_if_additions_below)
        seed_with_native = changes.pop("seed_with_native", self.seed_with_native)
        region_first = changes.pop("region_first", self.region_first)
        preprocess = changes.pop("preprocess", self.preprocess)
        pixel_budget = changes.pop("pixel_budget", self.pixel_budget)
        include_native_text = changes.pop("include_native_text", self.include_native_text)
        recognize_words = changes.pop("recognize_words", self.recognize_words)
        collect_symbols = changes.pop("collect_symbols", self.collect_symbols)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            name,
            scope,
            scale,
            modes,
            tiles,
            parallel_tiles,
            region_columns,
            max_regions,
            minimum_confidence,
            run_if_characters_below,
            minimum_utility_gain,
            adaptive_scale,
            minimum_characters_for_rescue,
            character_confidence_threshold,
            run_if_additions_below,
            seed_with_native,
            region_first,
            preprocess,
            pixel_budget,
            include_native_text,
            recognize_words,
            collect_symbols,
        )


class WorkPlan:
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
        internal_frozen_setattr(self, "route", route)
        internal_frozen_setattr(self, "reason", reason)
        internal_frozen_setattr(self, "ocr_passes", ocr_passes)
        internal_frozen_setattr(self, "verify_hidden_text", verify_hidden_text)
        internal_frozen_setattr(self, "fusion_policy", fusion_policy)
        internal_frozen_setattr(self, "allow_direct_image_ocr", allow_direct_image_ocr)
        internal_frozen_setattr(self, "augment_page_candidates", augment_page_candidates)
        self._post_init()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"route={self.route!r}, "
            f"reason={self.reason!r}, "
            f"ocr_passes={self.ocr_passes!r}, "
            f"verify_hidden_text={self.verify_hidden_text!r}, "
            f"fusion_policy={self.fusion_policy!r}, "
            f"allow_direct_image_ocr={self.allow_direct_image_ocr!r}, "
            f"augment_page_candidates={self.augment_page_candidates!r}"
            ")"
        )

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        route = changes.pop("route", self.route)
        reason = changes.pop("reason", self.reason)
        ocr_passes = changes.pop("ocr_passes", self.ocr_passes)
        verify_hidden_text = changes.pop("verify_hidden_text", self.verify_hidden_text)
        fusion_policy = changes.pop("fusion_policy", self.fusion_policy)
        allow_direct_image_ocr = changes.pop("allow_direct_image_ocr", self.allow_direct_image_ocr)
        augment_page_candidates = changes.pop(
            "augment_page_candidates", self.augment_page_candidates
        )
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            route,
            reason,
            ocr_passes,
            verify_hidden_text,
            fusion_policy,
            allow_direct_image_ocr,
            augment_page_candidates,
        )

    def _post_init(self) -> None:
        if not isinstance(self.reason, PagePlanReason):
            object.__setattr__(self, "reason", PagePlanReason(self.reason))

    @property
    def image_regions_only(self) -> bool:
        return bool(self.ocr_passes) and all(
            ocr_pass.scope is OcrPassScope.IMAGE_REGIONS for ocr_pass in self.ocr_passes
        )


class RecognitionResult:
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
        internal_frozen_setattr(self, "observations", observations)
        internal_frozen_setattr(self, "stroked_vector_alphabet", stroked_vector_alphabet)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"observations={self.observations!r}, "
            f"stroked_vector_alphabet={self.stroked_vector_alphabet!r}"
            ")"
        )

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        observations = changes.pop("observations", self.observations)
        stroked_vector_alphabet = changes.pop(
            "stroked_vector_alphabet", self.stroked_vector_alphabet
        )
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(observations, stroked_vector_alphabet)

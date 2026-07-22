# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from functools import lru_cache
from statistics import median
from typing import TYPE_CHECKING, Any, Callable, Iterable, Protocol, cast

from core_layout.impl.layout import word_frequencies

from core_ocr.impl import candidate_generation as ocr_candidate_generation
from core_ocr.impl import candidates as ocr_candidates
from core_ocr.impl import execution as ocr_execution
from core_ocr.impl import full_page as ocr_full_page
from core_ocr.impl import layout as ocr_layout
from core_ocr.impl import line_reconciliation as ocr_line_reconciliation
from core_ocr.impl import page_analysis as ocr_page_analysis
from core_ocr.impl import postprocess as ocr_postprocess
from core_ocr.impl import rendering as ocr_rendering
from core_ocr.impl import schematic as ocr_schematic
from core_ocr.impl import selection as ocr_selection
from core_ocr.impl import session as ocr_session_runtime
from core_ocr.impl import table_regions as ocr_table_regions
from core_ocr.impl import text_analysis as ocr_text_analysis
from core_ocr.impl import tiling as ocr_tiling
from core_ocr.impl.backend import TesseractCtypesBackend
from core_ocr.impl.output import (
    append_resolved_supplement_lines,
    best_effort_resolved_text_lines,
    ocr_result_output_lines,
    vector_stroke_result_output_lines,
)
from core_ocr.impl.policy import (
    classify_page_region,
    fragmented_invisible_text_layer_should_yield_to_ocr,
    should_preserve_substantial_text_table_native_text,
    should_replace_dominant_image_native_text_with_ocr,
    should_replace_noisy_native_text_with_compact_ocr,
    should_replace_symbol_encoded_text_with_ocr,
    should_replace_text_with_ocr,
)
from core_ocr.impl.services import service_function, service_module
from core_ocr.impl.text_analysis import (
    extracted_text_token_count,
    normalized_text_tokens,
    numeric_token_ratio,
    sparse_text_looks_noisy,
    text_ocr_quality_score,
)
from core_ocr.impl.types import (
    TESSERACT_RIL_BLOCK,
    TESSERACT_RIL_TEXTLINE,
    TESSERACT_RIL_WORD,
    OcrComponentBox,
    OcrImage,
    OcrTextResult,
    leptonica_pix_size_is_supported,
    ocr_float_value,
    ocr_int_value,
)
from core_ocr.impl.vector_text import (
    VectorStrokeOcrResult,
    vector_stroke_ocr_result_with_timeout,
)

ExtractionCache = Any
ExtractionCacheMapping = Any
PdfDict = Any
PdfStream = Any
TextSpan = Any
RenderOptions = service_module("render_options")
ImageColorManager = service_module("image_color_manager")
MarkdownRenderer = service_module("markdown_renderer")
LayoutAnalyzer = service_module("layout_analyzer")
observation_resolver = service_module("observation_resolver")
page_geometry = service_module("page_geometry")
page_profile = service_module("page_profile")
native_text = service_module("native_text")
apply_flate = service_function("apply_flate")
decode_stream_data = service_function("decode_stream_data")
compose_page = service_function("compose_page")
detect_grid = service_function("detect_grid")
image_filter_names = service_function("image_filter_names")
lookup_dict_key = service_function("lookup_dict_key")
pdf_int = service_function("pdf_int")
render_page_observation_lines = service_function("render_page_observation_lines")
render_resolved_text_lines = service_function("render_resolved_text_lines")
page_extraction_decision = service_function("page_extraction_decision")
create_extraction_cache = service_function("create_extraction_cache")
layout_geometry_summary_record = service_function("layout_geometry_summary_record")
page_layout_geometry_summary = service_function("page_layout_geometry_summary")
try_extract_native_text_fast = service_function("try_extract_native_text_fast")
native_text_runs_for_extraction = service_function("native_text_runs_for_extraction")
native_text_runs_inside_page_bounds = service_function("native_text_runs_inside_page_bounds")
native_text_runs_inside_visible_row_bands = service_function(
    "native_text_runs_inside_visible_row_bands"
)
select_native_text_layout = service_function("select_native_text_layout")
should_try_rendered_glyph_repair = service_function("should_try_rendered_glyph_repair")
apply_rendered_glyph_repair_to_native_text = service_function(
    "apply_rendered_glyph_repair_to_native_text"
)
native_layout_geometry_summary_for_runs = service_function(
    "native_layout_geometry_summary_for_runs"
)
native_invisible_text_layer_has_fragmented_geometry = service_function(
    "native_invisible_text_layer_has_fragmented_geometry"
)
native_invisible_text_layer_is_trustworthy = service_function(
    "native_invisible_text_layer_is_trustworthy"
)

if TYPE_CHECKING:
    from core_layout.impl.layout.models import LayoutLine, TextRun

    from core_ocr.impl.candidates import OcrCandidate, OcrPageTextResult

OCR_FALLBACK_DPI = ocr_execution.OCR_DEFAULT_DPI
OCR_FALLBACK_PAGE_SEGMENTATION_MODE = ocr_execution.OCR_DEFAULT_PAGE_SEGMENTATION_MODE
OCR_FALLBACK_IMAGE_AREA_RATIO = 0.50
OCR_FIGURE_RENDER_PADDING_POINTS = 10.0
OCR_FIGURE_RENDER_DPI = 400
OCR_FIGURE_AUTO_PAGE_SEGMENTATION_MODE = 3
OCR_FIGURE_SOURCE_PAGE_SEGMENTATION_MODES = (
    OCR_FIGURE_AUTO_PAGE_SEGMENTATION_MODE,
    ocr_execution.OCR_DEFAULT_PAGE_SEGMENTATION_MODE,
    ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE,
)
OCR_FIGURE_SUBREGION_SCALE = 5
OCR_FIGURE_SUBREGION_MAX_TARGET_PIXELS = 30_000_000
OCR_FIGURE_IMAGE_VIEW_SCALE = 2
OCR_FIGURE_IMAGE_VIEW_MAX_TARGET_PIXELS = 30_000_000
OCR_FIGURE_MAX_SUBREGIONS = 6
OCR_FIGURE_MAX_STACK_SUBREGIONS = 18
OCR_FIGURE_MAX_LABEL_CLUSTER_SUBREGIONS = 12
OCR_FIGURE_MAX_CALLOUT_CLUSTER_SUBREGIONS = 8
OCR_FIGURE_MAX_CALLOUT_NEIGHBORHOOD_SUBREGIONS = 8
OCR_FIGURE_MAX_GRID_SUBREGIONS = 28
OCR_FIGURE_MAX_PIXEL_SUBREGIONS = 12
OCR_FIGURE_MAX_TOTAL_SUBREGIONS = 32
OCR_VECTOR_DIAGRAM_TILE_MAX_SIDE_PIXELS = 1_400
OCR_FIGURE_GRID_COLUMNS = (2, 4)
OCR_FIGURE_GRID_ROWS = (2, 4)
OCR_FIGURE_GRID_OVERLAP_RATIO = 0.12
OCR_FIGURE_BASE_VARIABLES = ocr_table_regions.OCR_TESSERACT_TABLE_VARIABLES
OCR_FIGURE_OTSU_VARIABLES = {
    **ocr_table_regions.OCR_TESSERACT_TABLE_VARIABLES,
    "thresholding_method": "1",
}
OCR_FIGURE_SAUVOLA_VARIABLES = {
    **ocr_table_regions.OCR_TESSERACT_TABLE_VARIABLES,
    "thresholding_method": "2",
    "thresholding_window_size": "0.33",
    "thresholding_kfactor": "0.34",
}
OCR_FIGURE_WHOLE_IMAGE_PROFILES = (
    ("base", OCR_FIGURE_BASE_VARIABLES),
    ("otsu", OCR_FIGURE_OTSU_VARIABLES),
    ("sauvola", OCR_FIGURE_SAUVOLA_VARIABLES),
)
OCR_FIGURE_SUBREGION_PROFILES = OCR_FIGURE_WHOLE_IMAGE_PROFILES
OCR_FIGURE_SUBREGION_PAGE_SEGMENTATION_MODES = (
    OCR_FIGURE_AUTO_PAGE_SEGMENTATION_MODE,
    ocr_execution.OCR_DEFAULT_PAGE_SEGMENTATION_MODE,
    ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE,
)
OCR_FIGURE_FULL_PAGE_IMAGE_ONLY_WHOLE_IMAGE_PROFILES = (("base", OCR_FIGURE_BASE_VARIABLES),)
OCR_FIGURE_FULL_PAGE_IMAGE_ONLY_PAGE_SEGMENTATION_MODES = (
    OCR_FIGURE_AUTO_PAGE_SEGMENTATION_MODE,
    ocr_execution.OCR_DEFAULT_PAGE_SEGMENTATION_MODE,
)
OCR_EMBEDDED_IMAGE_TEXT_MIN_PIXELS = 30_000
OCR_EMBEDDED_IMAGE_TEXT_MAX_PIXELS = 250_000
OCR_EMBEDDED_IMAGE_TEXT_MAX_AREA_RATIO = 0.025
OCR_EMBEDDED_IMAGE_TEXT_MIN_PIXEL_DENSITY = 8.0
OCR_EMBEDDED_IMAGE_TEXT_MIN_ASPECT_RATIO = 0.55
OCR_EMBEDDED_IMAGE_TEXT_MAX_ASPECT_RATIO = 1.85
OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_PIXELS = 750_000
OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_AREA_RATIO = 0.085
OCR_EMBEDDED_IMAGE_TEXT_WIDE_MIN_ASPECT_RATIO = 2.5
OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_ASPECT_RATIO = 8.0
OCR_EMBEDDED_IMAGE_TEXT_DPI = 500
OCR_EMBEDDED_IMAGE_TEXT_LINEAR_SCALE = 4
OCR_EMBEDDED_IMAGE_TEXT_LINEAR_MAX_TARGET_PIXELS = 16_000_000
OCR_EMBEDDED_IMAGE_TEXT_MAX_LINEAR_REGIONS = 8
OCR_EMBEDDED_IMAGE_TEXT_MIN_LINEAR_REGION_PIXELS = 256
OCR_EMBEDDED_IMAGE_TEXT_LINEAR_REGION_PADDING = 6
OCR_EMBEDDED_IMAGE_TEXT_VARIABLES = {
    **OCR_FIGURE_BASE_VARIABLES,
    "preserve_interword_spaces": "1",
    "tessedit_char_whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz& -",
}
OCR_EMBEDDED_IMAGE_TEXT_RING_CONFIGS = (
    ("top_ring", 170.0, 370.0, 0.64, 0.95, 1400, 104),
    ("top_ring_core", 180.0, 360.0, 0.64, 0.95, 1200, 104),
    ("top_ring_outer", 180.0, 360.0, 0.72, 0.98, 1200, 104),
    ("bottom_ring", 10.0, 190.0, 0.64, 0.95, 1400, 104),
    ("bottom_ring_reverse", 190.0, 10.0, 0.64, 0.95, 1400, 104),
    ("bottom_ring_core", 0.0, 180.0, 0.64, 0.95, 1200, 104),
    ("bottom_ring_outer", 0.0, 180.0, 0.72, 0.98, 1200, 104),
    ("full_ring", 0.0, 360.0, 0.64, 0.95, 1800, 104),
    ("full_ring_reverse", 360.0, 0.0, 0.64, 0.95, 1800, 104),
)
OCR_EMBEDDED_IMAGE_TEXT_RING_PSMS = (
    ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE,
    ocr_execution.OCR_DEFAULT_PAGE_SEGMENTATION_MODE,
    13,
)
OCR_EMBEDDED_IMAGE_TEXT_LINEAR_PSMS = (
    ocr_execution.OCR_DEFAULT_PAGE_SEGMENTATION_MODE,
    7,
    ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE,
)
OCR_ORIENTATION_ENSEMBLE_MAX_PIXELS = 16_000_000
OCR_LARGE_FULL_PAGE_IMAGE_MAX_RENDER_DPI = 400
OCR_FULL_PAGE_PRIMARY_MAX_PIXELS = 24_000_000
OCR_FULL_PAGE_FIGURE_MAX_PIXELS = 30_000_000
OCR_FULL_PAGE_FIGURE_FOLLOWUP_MAX_PIXELS = 27_000_000
OCR_DENSE_IMAGE_SPARSE_MAX_SIDE = 8_192
OCR_DENSE_IMAGE_SPARSE_MAX_PIXELS = 55_000_000
OCR_DENSE_IMAGE_SPARSE_MIN_TOKENS = 250
OCR_DENSE_IMAGE_SPARSE_MAX_TOKENS = 2_200
OCR_DENSE_IMAGE_SPARSE_MIN_WORDS = 120
OCR_DENSE_IMAGE_SPARSE_MIN_LINES = 12
OCR_IMAGE_ONLY_LAYOUT_RETRY_MIN_TOKENS = 260
OCR_IMAGE_ONLY_LAYOUT_RETRY_MAX_TOKENS = 650
OCR_IMAGE_ONLY_LAYOUT_RETRY_MAX_CONFIDENCE = 78
OCR_IMAGE_ONLY_LAYOUT_RETRY_MIN_QUALITY = 0.18
OCR_IMAGE_ONLY_LAYOUT_RETRY_MIN_LINES = 20
OCR_IMAGE_ONLY_LAYOUT_RETRY_MIN_WORDS = 100
OCR_IMAGE_ONLY_LAYOUT_SUPPLEMENT_MIN_CONFIDENCE = 85
OCR_IMAGE_ONLY_LAYOUT_SUPPLEMENT_MAX_TOKENS = 18
OCR_IMAGE_ONLY_LAYOUT_SUPPLEMENT_TOP_RATIO = 0.40
OCR_IMAGE_ONLY_LAYOUT_SUPPLEMENT_MAX_LINES = 4
OCR_IMAGE_ONLY_LAYOUT_PROMINENT_MIN_CONFIDENCE = 40
OCR_IMAGE_ONLY_LAYOUT_PROMINENT_TOP_RATIO = 0.20
OCR_IMAGE_ONLY_LAYOUT_PROMINENT_MIN_WIDTH_RATIO = 0.35
OCR_IMAGE_ONLY_LAYOUT_PROMINENT_MAX_TOKENS = 6
OCR_TABLE_FUSION_MAX_REPLACEMENTS = 16
OCR_TABLE_FUSION_MAX_ADDITIONS = 24
OCR_TABLE_FUSION_MAX_REJECTIONS = 48
OCR_MULTI_COLUMN_BAND_MIN_LINES = 8
OCR_MULTI_COLUMN_BAND_MAX_REGION_COUNT = 2
OCR_MULTI_COLUMN_BAND_MIN_TEXTLINES = 24
OCR_MULTI_COLUMN_BAND_MAX_TEXTLINES = 140
OCR_MULTI_COLUMN_BAND_MAX_TOKENS = 720
OCR_MULTI_COLUMN_BAND_MAX_PAGE_AREA_RATIO = 0.18
OCR_MULTI_COLUMN_BAND_MAX_HEIGHT_RATIO = 0.24
OCR_RENDER_CONSENSUS_MIN_TOKENS = 120
OCR_RENDER_CONSENSUS_MIN_COVERAGE = 0.60
OCR_CROSS_SOURCE_CONSENSUS_MIN_COVERAGE = 0.85
OCR_CROSS_SOURCE_MAX_RENDER_PIXELS = 30_000_000
OCR_CROSS_SOURCE_MAX_SELECTED_CONFIDENCE = 89
OCR_CROSS_SOURCE_NATIVE_MIN_QUALITY = 0.18
OCR_RENDER_CONSENSUS_MIN_OUTPUT_OVERLAP = 0.65
OCR_RENDER_CONSENSUS_FAMILY_SUPPORT_RATIO = 2.0 / 3.0
OCR_RENDER_CONSENSUS_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class EmbeddedImageTextRegion:
    bbox: tuple[float, float, float, float]
    item_index: int
    source_kind: str
    signals: dict[str, float | int | str | bool]


@dataclass(frozen=True)
class EmbeddedImageTextLine:
    text: str
    candidate: OcrCandidate
    region: EmbeddedImageTextRegion
    page_bbox: tuple[float, float, float, float] | None
    confidence: int | None


@dataclass(frozen=True)
class EmbeddedImageTextSpanRecord:
    text: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: int | None


@dataclass(frozen=True)
class DominantImageLabelObservation:
    """A spatially anchored label observation from one OCR candidate."""

    text: str
    numeric_token: str | None
    bbox: tuple[float, float, float, float]
    source: str
    confidence: float | None


NONSPACE_TOKEN_RE = re.compile(r"\S+")
ALNUM_RE = re.compile(r"[^\W_]")
NONSPACE_RE = re.compile(r"\S")
DIGIT_RE = re.compile(r"\d")
UNINTERPRETABLE_TEXT_RE = re.compile("[\ue000-\uf8ff\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\x7f\xad]")
OCR_ARTIFACT_CHARS = frozenset("~_=|¦¬^°•·`“”‘’")
VECTOR_SPATIAL_TEXT_RE = re.compile(r"[A-Za-z0-9_+\-]")
VECTOR_SPATIAL_ALLOWED_PUNCTUATION = frozenset("+-._/")


@dataclass(frozen=True)
class NativeOcrRegion:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass(frozen=True)
class TableOcrFusionLine:
    text: str
    observation: page_geometry.PageObservation | None = None
    confidence: int | None = None


@dataclass(frozen=True)
class TableOcrFusionResult:
    text: str
    output_lines: tuple[observation_resolver.ResolvedTextLine, ...] = ()


@dataclass(frozen=True)
class FigureTokenEvidence:
    token: str
    text: str
    bbox: tuple[float, float, float, float]
    source: str
    confidence: float | None


@dataclass(frozen=True)
class FigureFragmentCluster:
    fragment: str
    count: int
    bbox: tuple[float, float, float, float]
    source: str
    confidence: float | None


@dataclass(frozen=True)
class FigureFragmentFusionSupport:
    fragment: str
    count: int
    bbox: tuple[float, float, float, float]
    source: str
    confidence: float | None
    alpha_tokens: tuple[str, ...]
    numeric_tokens: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class FigureBandSlot:
    anchor_token: str
    anchor_bbox: tuple[float, float, float, float]
    gap_bbox: tuple[float, float, float, float]
    numeric_tokens: tuple[str, ...]
    numeric_bbox: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class FigureBandSlotPlan:
    alpha_clusters: dict[
        str,
        tuple[tuple[int, tuple[float, float, float, float], str, float | None], ...],
    ] = field(default_factory=dict)
    numeric_clusters: dict[
        str,
        tuple[tuple[int, tuple[float, float, float, float], str, float | None], ...],
    ] = field(default_factory=dict)
    slots: tuple[FigureBandSlot, ...] = ()


@dataclass(frozen=True)
class FigureBandSlotEvidence:
    slot: FigureBandSlot
    fragments: tuple[str, ...]
    ordered_fragments: tuple[str, ...]
    words: tuple[str, ...]
    sources: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class FigureBandSlotHypothesis:
    token: str
    score: float
    fragment_matches: tuple[str, ...]
    word_matches: tuple[str, ...]


@dataclass(frozen=True)
class FigureRegionGeometryPlan:
    micro_band_boxes: tuple[tuple[float, float, float, float], ...] = ()
    micro_fragment_boxes: tuple[tuple[float, float, float, float], ...] = ()


@dataclass(frozen=True)
class FigureRegionGeometryEvidence:
    micro_band_candidates: tuple[OcrCandidate, ...] = ()
    band_slots: tuple[FigureBandSlot, ...] = ()
    band_slot_candidates: tuple[OcrCandidate, ...] = ()
    band_slot_window_candidates: tuple[OcrCandidate, ...] = ()
    micro_fragment_candidates: tuple[OcrCandidate, ...] = ()

    def all_candidates(self) -> tuple[OcrCandidate, ...]:
        return (
            self.micro_band_candidates
            + self.band_slot_candidates
            + self.band_slot_window_candidates
            + self.micro_fragment_candidates
        )


@dataclass(frozen=True)
class FigureFragmentAnalysis:
    vocabulary: set[str] = field(default_factory=set)
    raw_clusters: dict[str, tuple[FigureFragmentCluster, ...]] = field(default_factory=dict)
    slot_plan: FigureBandSlotPlan | None = None
    fusion_support: dict[str, tuple[FigureFragmentFusionSupport, ...]] = field(default_factory=dict)
    slot_evidence: tuple[FigureBandSlotEvidence, ...] = ()


class PageExtractionHost(Protocol):
    document: Any
    state: Any
    text_lines: list[LayoutLine] | None
    text_spans: list[TextSpan] | None
    extraction_cache: ExtractionCache | None

    @property
    def chars(self) -> list[TextRun]: ...

    @property
    def rotation(self) -> int: ...

    @property
    def media_box(self) -> tuple[float, float, float, float] | None: ...

    @property
    def resources(self) -> PdfDict: ...

    @property
    def content_streams(self) -> tuple[PdfStream, ...]: ...

    def get_page_profile(self) -> page_profile.PageProfile: ...

    def get_graphics(self) -> Any: ...

    def capture_text_state(self) -> Any: ...

    def get_text_and_graphics_state(self) -> Any: ...

    def get_text_lines(self) -> list[LayoutLine]: ...

    def get_drawings(self) -> list[dict[str, Any]]: ...

    def get_grid_lines(self) -> list[Any]: ...

    def get_text_spans(self) -> list[TextSpan]: ...

    def extract_lines(self, *, include_words: bool = False) -> list[dict[str, Any]]: ...

    def extract_images(
        self,
        *,
        include_inline: bool = True,
        include_xobjects: bool = True,
    ) -> list[dict[str, Any]]: ...

    def get_annotations(self) -> list[Any]: ...

    def render(self, options: RenderOptions | None = None) -> Any: ...

    def extract_text(self) -> str: ...


@dataclass(frozen=True)
class PageExtractionSnapshot:
    text: str
    resolved_output_lines: tuple[observation_resolver.ResolvedTextLine, ...]


def cache_page_extraction_snapshot(
    cache: ExtractionCache,
    text: str,
    resolved_output_lines: Iterable[observation_resolver.ResolvedTextLine],
) -> str:
    lines = tuple(resolved_output_lines)
    cache["resolved_output_lines"] = lines
    cache["page_extraction_snapshot"] = PageExtractionSnapshot(text, lines)
    return text


def fuse_table_ocr_candidates(
    text: str,
    candidates: Iterable[OcrCandidate],
    *,
    selected_candidate: OcrCandidate | None = None,
) -> TableOcrFusionResult:
    if not text.strip():
        return TableOcrFusionResult(text)
    lines = text.splitlines()
    base_records = table_ocr_base_line_records(lines, selected_candidate)
    if selected_candidate is not None and selected_candidate.name.endswith("_tiled"):
        return TableOcrFusionResult(
            text,
            table_ocr_resolved_text_lines_from_records(base_records),
        )
    raw_table_candidates = [
        candidate
        for candidate in candidates
        if candidate.name in ocr_table_regions.OCR_TABLE_CANDIDATE_NAMES and candidate.result.text
    ]
    if not raw_table_candidates:
        return TableOcrFusionResult(
            text,
            table_ocr_resolved_text_lines_from_records(base_records),
        )
    table_candidates: list[OcrCandidate] = []
    for candidate in raw_table_candidates:
        reason = table_ocr_candidate_fusion_rejection_reason(candidate)
        if reason is None:
            table_candidates.append(candidate)
    if not table_candidates:
        return TableOcrFusionResult(
            text,
            table_ocr_resolved_text_lines_from_records(base_records),
        )
    seen_tokens = set(normalized_text_tokens(text))
    addition_records: list[TableOcrFusionLine] = []
    replacements = 0
    added = 0
    for candidate in sorted(table_candidates, key=table_ocr_candidate_fusion_order):
        for table_record in table_ocr_candidate_line_records(candidate):
            line = ocr_table_regions.normalize_table_region_ocr_text(table_record.text)
            observation = (
                replace(table_record.observation, text=line)
                if table_record.observation is not None
                else None
            )
            table_record = TableOcrFusionLine(
                line,
                observation,
                table_record.confidence,
            )
            plausibility_reason = table_ocr_fusion_plausibility_rejection_reason(
                line,
                candidate.name,
            )
            if plausibility_reason is not None:
                continue
            decision = table_ocr_fusion_line_decision(
                lines,
                seen_tokens,
                line,
                candidate.name,
                base_records=base_records,
                table_record=table_record,
            )
            if decision["action"] == "replace":
                if replacements >= OCR_TABLE_FUSION_MAX_REPLACEMENTS:
                    continue
                index = cast(int, decision["target_index"])
                lines[index] = line
                base_observation = base_records[index].observation
                if base_observation is not None:
                    base_observation = replace(
                        base_observation,
                        text=line,
                        confidence=page_geometry.numeric_confidence(
                            table_record.confidence,
                        )
                        or base_observation.confidence,
                    )
                base_records[index] = TableOcrFusionLine(
                    line,
                    base_observation,
                    table_record.confidence,
                )
                replacements += 1
            elif decision["action"] == "append":
                if added >= OCR_TABLE_FUSION_MAX_ADDITIONS:
                    continue
                addition_records.append(table_record)
                added += 1
            else:
                continue
            line_tokens = normalized_text_tokens(line)
            seen_tokens.update(line_tokens)
    if replacements == 0 and added == 0:
        return TableOcrFusionResult(
            table_ocr_text_from_records(base_records),
            table_ocr_resolved_text_lines_from_records(base_records),
        )
    output_records = [*base_records, *addition_records]
    return TableOcrFusionResult(
        table_ocr_text_from_records(output_records),
        table_ocr_resolved_text_lines_from_records(output_records),
    )


def table_ocr_text_from_records(records: list[TableOcrFusionLine]) -> str:
    return "\n".join(record.text for record in records).rstrip()


def table_ocr_resolved_text_lines_from_records(
    records: list[TableOcrFusionLine],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    output_lines: list[observation_resolver.ResolvedTextLine] = []
    for line_index, record in enumerate(records):
        stripped = record.text.strip()
        if not stripped:
            continue
        observation = table_ocr_record_observation(record, line_index=line_index)
        output_lines.append(
            observation_resolver.ResolvedTextLine(
                stripped,
                observation,
                break_before=1,
                contributing_observations=(observation,),
            )
        )
    return observation_resolver.resolve_text_lines(output_lines)


def repair_ocr_output_lines_with_alternate_candidates(
    output_lines: tuple[observation_resolver.ResolvedTextLine, ...],
    candidates: Iterable[OcrCandidate],
    *,
    selected_candidate: OcrCandidate | None,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not output_lines or selected_candidate is None:
        return output_lines
    repaired = list(output_lines)
    for candidate in candidates:
        if (
            candidate is selected_candidate
            or not ocr_output_line_repair_candidate(candidate)
            or ocr_output_line_repair_candidate_is_less_refined_sibling(
                candidate,
                selected_candidate,
            )
        ):
            continue
        repaired = repair_ocr_output_lines_with_candidate(repaired, candidate)
    return tuple(repaired)


def resolved_line_content_record(
    line: observation_resolver.ResolvedTextLine,
    *,
    line_index: int,
) -> dict[str, Any]:
    observation = line.observation
    record: dict[str, Any] = {
        "text": line.text,
        "bbox": observation.bbox,
        "advance_bbox": observation.advance_bbox,
        "ink_bbox": observation.ink_bbox,
        "x0": observation.bbox[0] if observation.bbox is not None else None,
        "y0": observation.bbox[1] if observation.bbox is not None else None,
        "x1": observation.bbox[2] if observation.bbox is not None else None,
        "y1": observation.bbox[3] if observation.bbox is not None else None,
        "line_index": line_index,
        "break_before": line.break_before,
        "source": observation.source,
        "observation_kind": observation.kind,
        "confidence": observation.confidence,
        "baseline": observation.baseline,
        "provenance": list(observation.provenance),
        "contributing_sources": [
            contribution.source for contribution in line.contributing_observations
        ],
    }
    return record


def ocr_output_line_repair_candidate(candidate: OcrCandidate) -> bool:
    source_name = ocr_selection.ocr_variant_source_name(candidate.name)
    if not ocr_selection.broad_page_candidate_name(source_name):
        return False
    if candidate.region_count != 0:
        return False
    return bool(candidate.result.line_rows)


def ocr_output_line_repair_candidate_is_less_refined_sibling(
    candidate: OcrCandidate,
    selected_candidate: OcrCandidate,
) -> bool:
    if ocr_selection.ocr_variant_source_name(
        candidate.name
    ) != ocr_selection.ocr_variant_source_name(selected_candidate.name):
        return False
    return ocr_candidate_refinement_rank(candidate.name) < ocr_candidate_refinement_rank(
        selected_candidate.name
    )


def ocr_candidate_refinement_rank(name: str) -> int:
    if name.endswith("_word_refined"):
        return 3
    if name.endswith("_reconciled_layout"):
        return 2
    if name.endswith("_word_layout"):
        return 1
    return 0


def repair_ocr_output_lines_with_candidate(
    output_lines: list[observation_resolver.ResolvedTextLine],
    candidate: OcrCandidate,
) -> list[observation_resolver.ResolvedTextLine]:
    candidate_records = table_ocr_candidate_line_records(candidate)
    if not candidate_records:
        return output_lines
    repaired = list(output_lines)
    used_indexes: set[int] = set()
    for line_index, current in enumerate(repaired):
        candidate_index, geometry_score = best_alternate_ocr_line_record_match(
            current,
            candidate_records,
            used_indexes,
        )
        if candidate_index is None:
            continue
        candidate_record = candidate_records[candidate_index]
        current_confidence = page_geometry.numeric_confidence(current.observation.confidence)
        if not should_replace_ocr_output_line_with_alternate(
            current.text,
            candidate_record.text,
            current_confidence,
            candidate_record.confidence,
            geometry_score=geometry_score,
        ):
            continue
        observation = table_ocr_record_observation(candidate_record, line_index=line_index)
        repaired[line_index] = replace(
            current,
            text=candidate_record.text,
            observation=observation,
            contributing_observations=(observation,),
        )
        used_indexes.add(candidate_index)
    return repaired


def best_alternate_ocr_line_record_match(
    current: observation_resolver.ResolvedTextLine,
    candidate_records: list[TableOcrFusionLine],
    used_indexes: set[int],
) -> tuple[int | None, float]:
    if current.observation.bbox is None:
        return (None, 0.0)
    best_index: int | None = None
    best_score = 0.0
    for index, record in enumerate(candidate_records):
        if index in used_indexes or record.observation is None:
            continue
        score = page_geometry.observation_geometry_match_score(
            current.observation,
            record.observation,
        )
        if score > best_score:
            best_index = index
            best_score = score
    if best_score < 0.76:
        return (None, best_score)
    return (best_index, best_score)


def should_replace_ocr_output_line_with_alternate(
    current_text: str,
    candidate_text: str,
    current_confidence: float | None,
    candidate_confidence: int | None,
    *,
    geometry_score: float,
) -> bool:
    current = current_text.strip()
    candidate = candidate_text.strip()
    if not current or not candidate or current == candidate:
        return False
    if geometry_score < 0.76:
        return False
    current_tokens = normalized_text_tokens(current)
    candidate_tokens = normalized_text_tokens(candidate)
    if len(candidate_tokens) < max(2, int(len(current_tokens) * 0.55)):
        return False
    if len(candidate_tokens) > max(len(current_tokens) + 16, int(len(current_tokens) * 1.9)):
        return False
    current_quality = text_ocr_quality_score(current)
    candidate_quality = text_ocr_quality_score(candidate)
    if candidate_quality > min(0.48, current_quality + 0.06):
        return False
    if alternate_ocr_line_drops_connector_payload(
        current,
        current_tokens,
        candidate_tokens,
    ):
        return False
    current_conf = current_confidence if current_confidence is not None else 0.0
    candidate_conf = float(candidate_confidence) if candidate_confidence is not None else 0.0
    if alternate_ocr_line_extends_short_numeric_token(
        current_tokens,
        candidate_tokens,
        current_confidence=current_conf,
        candidate_confidence=candidate_conf,
    ):
        return False
    if lower_confidence_token_substitution_is_untrusted(
        current_tokens,
        candidate_tokens,
        current_conf,
        candidate_conf,
        quality_gain=current_quality - candidate_quality,
    ):
        return False
    current_short_ratio = short_token_ratio(current_tokens)
    candidate_short_ratio = short_token_ratio(candidate_tokens)
    current_punct_ratio = ocr_output_line_punctuation_ratio(current)
    candidate_punct_ratio = ocr_output_line_punctuation_ratio(candidate)
    current_compact = compact_alnum_text(current)
    candidate_compact = compact_alnum_text(candidate)
    compact_similarity = (
        SequenceMatcher(None, current_compact, candidate_compact).ratio()
        if current_compact and candidate_compact
        else 0.0
    )
    if compact_similarity >= 0.74:
        if candidate_quality + 0.02 < current_quality:
            return True
        return (
            candidate_conf >= current_conf + 8.0
            and candidate_quality <= current_quality + 0.01
            and candidate_punct_ratio + 0.05 < current_punct_ratio
        )
    current_is_noisy = current_short_ratio >= 0.72 and current_punct_ratio >= 0.22
    if current_is_noisy:
        return (
            candidate_quality + 0.03 < current_quality
            and candidate_short_ratio + 0.12 < current_short_ratio
            and candidate_conf >= current_conf - 5.0
        )
    if table_ocr_line_looks_weak(current):
        return (
            candidate_quality + 0.03 < current_quality
            and candidate_short_ratio + 0.10 < current_short_ratio
            and candidate_conf >= current_conf
        )
    return False


def alternate_ocr_line_drops_connector_payload(
    current_text: str,
    current_tokens: list[str],
    candidate_tokens: list[str],
) -> bool:
    if not any(ch in current_text for ch in "-‐‑‒–—−"):
        return False
    current_counts = Counter(current_tokens)
    candidate_counts = Counter(candidate_tokens)
    if not candidate_counts or not all(
        count <= current_counts[token] for token, count in candidate_counts.items()
    ):
        return False
    if current_counts == candidate_counts:
        return False
    return any(
        count > candidate_counts[token] and alternate_ocr_line_connector_payload_token(token)
        for token, count in current_counts.items()
    )


def alternate_ocr_line_connector_payload_token(token: str) -> bool:
    if token == "_":
        return False
    if token.isdigit():
        return True
    return token.isalpha() and len(token) >= 2


def alternate_ocr_line_extends_short_numeric_token(
    current_tokens: list[str],
    candidate_tokens: list[str],
    *,
    current_confidence: float,
    candidate_confidence: float,
) -> bool:
    if len(current_tokens) != len(candidate_tokens):
        return False
    changed = [
        (current, candidate)
        for current, candidate in zip(current_tokens, candidate_tokens, strict=False)
        if current != candidate
    ]
    if len(changed) != 1:
        return False
    current, candidate = changed[0]
    if not (current.isdigit() and candidate.isdigit()):
        return False
    if not (len(current) <= 2 and candidate.startswith(current)):
        return False
    if len(candidate) <= len(current):
        return False
    return current_confidence >= 50.0 or candidate_confidence <= current_confidence + 5.0


def lower_confidence_token_substitution_is_untrusted(
    current_tokens: list[str],
    candidate_tokens: list[str],
    current_confidence: float,
    candidate_confidence: float,
    *,
    quality_gain: float,
) -> bool:
    if candidate_confidence >= current_confidence:
        return False
    if current_tokens == candidate_tokens:
        return False
    if len(current_tokens) != len(candidate_tokens):
        return False
    return not quality_gain >= 0.08


def short_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 1.0
    return sum(1 for token in tokens if len(token) <= 2) / len(tokens)


def ocr_output_line_punctuation_ratio(text: str) -> float:
    nonspace = [ch for ch in text if not ch.isspace()]
    if not nonspace:
        return 1.0
    noisy = 0
    for ch in nonspace:
        if ch.isalnum():
            continue
        noisy += 1
    return noisy / len(nonspace)


def compact_alnum_text(text: str) -> str:
    return "".join(ch.casefold() for ch in text if ch.isalnum())


def table_ocr_record_observation(
    record: TableOcrFusionLine,
    *,
    line_index: int,
) -> page_geometry.PageObservation:
    if record.observation is not None:
        return replace(
            record.observation,
            text=record.text.strip(),
            provenance=(
                *record.observation.provenance,
                *page_geometry.provenance_tuple(line_index=line_index),
            ),
        )
    return page_geometry.PageObservation(
        kind="table_ocr_line",
        source="table_fusion_text",
        text=record.text.strip(),
        confidence=page_geometry.numeric_confidence(record.confidence),
        provenance=page_geometry.provenance_tuple(line_index=line_index),
    )


def table_ocr_candidate_is_fusible(candidate: OcrCandidate) -> bool:
    return table_ocr_candidate_fusion_rejection_reason(candidate) is None


def table_ocr_candidate_fusion_rejection_reason(
    candidate: OcrCandidate,
) -> str | None:
    if candidate.name not in ocr_table_regions.OCR_TABLE_CANDIDATE_NAMES:
        return "not_table_candidate"
    if not candidate.result.text:
        return "empty_candidate"
    tokens = extracted_text_token_count(candidate.result.text)
    if candidate.name == "table_cells_rotated":
        if tokens < 2:
            return "too_few_tokens"
        if text_ocr_quality_score(candidate.result.text) > 0.62:
            return "low_quality"
        return None
    if candidate.name == "table_cell_consensus":
        if tokens < 8:
            return "too_few_tokens"
        confidence = candidate.result.confidence
        if confidence is not None and confidence < 25:
            return "low_confidence"
        if text_ocr_quality_score(candidate.result.text) > 0.55:
            return "low_quality"
        return None
    if tokens < 8:
        return "too_few_tokens"
    confidence = candidate.result.confidence
    if confidence is not None and confidence < 30:
        return "low_confidence"
    if candidate.name == "table_rows" and confidence is not None and confidence < 55:
        return "low_row_confidence"
    if text_ocr_quality_score(candidate.result.text) > 0.42:
        return "low_quality"
    return None


def table_ocr_candidate_fusion_order(candidate: OcrCandidate) -> tuple[int, float]:
    priority = {
        "table_cell_consensus": 0,
        "table_rows": 1,
        "table_cells": 2,
        "table_cells_rotated": 3,
    }.get(candidate.name, 9)
    return (priority, -ocr_selection.ocr_candidate_score(candidate))


def table_ocr_base_line_records(
    lines: list[str],
    selected_candidate: OcrCandidate | None,
) -> list[TableOcrFusionLine]:
    if selected_candidate is None:
        return [TableOcrFusionLine(line) for line in lines]
    row_records = table_ocr_row_fusion_lines(
        selected_candidate.result.line_rows,
        source=selected_candidate.name,
    )
    if len(row_records) == len(lines):
        return [
            TableOcrFusionLine(line, row_record.observation, row_record.confidence)
            for line, row_record in zip(lines, row_records, strict=False)
        ]
    used_rows: set[int] = set()
    records: list[TableOcrFusionLine] = []
    for line in lines:
        row_index = matching_table_ocr_row_record_index(line, row_records, used_rows)
        if row_index is None:
            records.append(TableOcrFusionLine(line))
            continue
        used_rows.add(row_index)
        row_record = row_records[row_index]
        records.append(TableOcrFusionLine(line, row_record.observation, row_record.confidence))
    return records


def table_ocr_candidate_line_records(
    candidate: OcrCandidate,
) -> list[TableOcrFusionLine]:
    row_records = table_ocr_row_fusion_lines(
        candidate.result.line_rows,
        source=candidate.name,
    )
    if row_records:
        return row_records
    return [TableOcrFusionLine(line) for line in candidate.result.text.splitlines() if line.strip()]


def table_ocr_row_fusion_lines(
    rows: tuple[dict[str, Any], ...],
    *,
    source: str,
) -> list[TableOcrFusionLine]:
    records: list[TableOcrFusionLine] = []
    for row_index, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        confidence = table_ocr_row_confidence(row)
        records.append(
            TableOcrFusionLine(
                text,
                table_ocr_row_page_observation(
                    row,
                    source=source,
                    row_index=row_index,
                    text=text,
                    confidence=confidence,
                ),
                confidence,
            )
        )
    return records


def matching_table_ocr_row_record_index(
    line: str,
    records: list[TableOcrFusionLine],
    used_rows: set[int],
) -> int | None:
    line_tokens = normalized_text_tokens(line)
    if not line_tokens:
        return None
    for index, record in enumerate(records):
        if index in used_rows:
            continue
        if normalized_text_tokens(record.text) == line_tokens:
            return index
    return None


def table_ocr_row_page_bbox(
    row: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    bbox = row.get("page_bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def table_ocr_row_confidence(row: dict[str, Any]) -> int | None:
    confidence = row.get("conf")
    try:
        return int(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        return None


def table_ocr_row_page_observation(
    row: dict[str, Any],
    *,
    source: str,
    row_index: int,
    text: str,
    confidence: int | None,
) -> page_geometry.PageObservation | None:
    return page_geometry.page_observation_from_bbox(
        table_ocr_row_page_bbox(row),
        source=source,
        kind="table_ocr_line",
        text=text,
        confidence=page_geometry.numeric_confidence(confidence),
        provenance={"row_index": row_index},
    )


def table_ocr_fusion_line_decision(
    lines: list[str],
    seen_tokens: set[str],
    line: str,
    candidate_name: str,
    *,
    base_records: list[TableOcrFusionLine] | None = None,
    table_record: TableOcrFusionLine | None = None,
) -> dict[str, Any]:
    line_tokens = normalized_text_tokens(line)
    new_tokens = sum(1 for token in line_tokens if token not in seen_tokens)
    target_index: int | None = None
    match_score = 0.0
    geometry_score = 0.0
    if candidate_name != "table_cells_rotated":
        if base_records is not None and table_record is not None:
            target_index, geometry_score = best_table_ocr_fusion_geometry_line_match(
                base_records,
                table_record,
            )
        if target_index is not None:
            match_score = table_ocr_text_line_match_score(lines[target_index], line)
            if table_ocr_should_replace_line(
                lines[target_index],
                line,
                candidate_name,
                match_score=max(match_score, geometry_score),
                new_tokens=new_tokens,
                match_kind="geometry",
                geometry_score=geometry_score,
            ):
                return {
                    "action": "replace",
                    "target_index": target_index,
                    "match_kind": "geometry",
                    "match_score": round(match_score, 4),
                    "geometry_score": round(geometry_score, 4),
                    "new_tokens": new_tokens,
                }
            return {
                "action": "skip",
                "reason": "geometry_matched_no_replace",
                "target_index": target_index,
                "match_kind": "geometry",
                "match_score": round(match_score, 4),
                "geometry_score": round(geometry_score, 4),
                "new_tokens": new_tokens,
            }
        target_index, match_score = best_table_ocr_fusion_line_match(lines, line)
        if target_index is not None and table_ocr_should_replace_line(
            lines[target_index],
            line,
            candidate_name,
            match_score=match_score,
            new_tokens=new_tokens,
            match_kind="text",
        ):
            return {
                "action": "replace",
                "target_index": target_index,
                "match_kind": "text",
                "match_score": round(match_score, 4),
                "geometry_score": None,
                "new_tokens": new_tokens,
            }
    if table_ocr_should_append_line(
        line,
        candidate_name,
        new_tokens=new_tokens,
        seen_tokens=seen_tokens,
        base_lines=lines,
        table_record=table_record,
    ):
        return {"action": "append", "new_tokens": new_tokens}
    return {
        "action": "skip",
        "reason": "no_replace_or_append",
        "target_index": target_index,
        "match_kind": "text" if target_index is not None else None,
        "match_score": round(match_score, 4),
        "geometry_score": None,
        "new_tokens": new_tokens,
    }


def table_ocr_fusion_line_is_plausible(line: str, candidate_name: str) -> bool:
    return table_ocr_fusion_plausibility_rejection_reason(line, candidate_name) is None


def table_ocr_fusion_plausibility_rejection_reason(
    line: str,
    candidate_name: str,
) -> str | None:
    tokens = normalized_text_tokens(line)
    if not tokens:
        return "empty_line"
    quality = text_ocr_quality_score(line)
    if candidate_name == "table_cells_rotated":
        if len(tokens) < 2:
            return "too_few_tokens"
        if quality > 0.62:
            return "low_quality"
        return None
    if len(tokens) < 3:
        return "too_few_tokens"
    if quality > 0.44:
        return "low_quality"
    if len(tokens) > 56:
        return "too_many_tokens"
    alnum = sum(1 for _ in ALNUM_RE.finditer(line))
    if alnum < 6:
        return "too_little_text"
    return None


def best_table_ocr_fusion_line_match(
    lines: list[str],
    table_line: str,
) -> tuple[int | None, float]:
    table_tokens = normalized_text_tokens(table_line)
    if not table_tokens:
        return (None, 0.0)
    best_index: int | None = None
    best_score = 0.0
    for index, line in enumerate(lines):
        score = table_ocr_text_line_match_score(line, table_line)
        if score > best_score:
            best_index = index
            best_score = score
    if best_score < 0.42:
        return (None, best_score)
    return (best_index, best_score)


def best_table_ocr_fusion_geometry_line_match(
    base_records: list[TableOcrFusionLine],
    table_record: TableOcrFusionLine,
) -> tuple[int | None, float]:
    if table_record.observation is None:
        return (None, 0.0)
    best_index: int | None = None
    best_score = 0.0
    for index, base_record in enumerate(base_records):
        if base_record.observation is None:
            continue
        score = page_geometry.observation_geometry_match_score(
            base_record.observation,
            table_record.observation,
        )
        if score > best_score:
            best_index = index
            best_score = score
    if best_score < 0.62:
        return (None, best_score)
    return (best_index, best_score)


def table_ocr_text_line_match_score(base_line: str, table_line: str) -> float:
    table_tokens = normalized_text_tokens(table_line)
    line_tokens = normalized_text_tokens(base_line)
    table_token_set = set(table_tokens)
    line_token_set = set(line_tokens)
    if not table_token_set or not line_token_set:
        return 0.0
    common = table_token_set.intersection(line_token_set)
    table_anchors = ocr_table_regions.table_ocr_line_anchor_tokens(table_tokens)
    line_anchors = ocr_table_regions.table_ocr_line_anchor_tokens(line_tokens)
    anchor_common = table_anchors.intersection(line_anchors)
    if not common and not anchor_common:
        return 0.0
    coverage = len(common) / max(1, min(len(table_token_set), len(line_token_set)))
    anchor_bonus = min(0.45, len(anchor_common) * 0.15)
    if table_tokens[:1] and table_tokens[:1] == line_tokens[:1]:
        anchor_bonus += 0.20
    return coverage + anchor_bonus


def table_ocr_should_replace_line(
    base_line: str,
    table_line: str,
    candidate_name: str,
    *,
    match_score: float,
    new_tokens: int,
    match_kind: str = "text",
    geometry_score: float = 0.0,
) -> bool:
    if candidate_name == "table_cells_rotated":
        return False
    base_tokens = normalized_text_tokens(base_line)
    table_tokens = normalized_text_tokens(table_line)
    if len(table_tokens) < 3:
        return False
    if len(base_tokens) < 2 and match_kind != "geometry":
        return False
    if len(table_tokens) > max(len(base_tokens) * 2.25, len(base_tokens) + 14):
        return False
    base_quality = text_ocr_quality_score(base_line)
    table_quality = text_ocr_quality_score(table_line)
    if match_kind == "geometry" and geometry_score >= 0.72:
        if table_quality > min(0.42, base_quality + 0.16):
            return False
        if table_ocr_line_looks_weak(base_line) and len(table_tokens) >= len(base_tokens):
            return True
        if new_tokens >= max(2, int(len(table_tokens) * 0.35)) and table_quality <= 0.28:
            return True
        if len(table_tokens) >= len(base_tokens) + 2 and table_quality + 0.06 < base_quality:
            return True
    if match_score >= 0.90 and len(table_tokens) >= len(base_tokens) + 2 and table_quality <= 0.28:
        return True
    if table_quality > min(0.42, base_quality + 0.10):
        return False
    if table_ocr_line_looks_weak(base_line) and len(table_tokens) >= len(base_tokens):
        return True
    if match_score >= 0.72 and len(table_tokens) >= len(base_tokens) + 2:
        return True
    return bool(new_tokens >= 2 and match_score >= 0.55 and table_quality + 0.045 < base_quality)


def table_ocr_line_looks_weak(line: str) -> bool:
    tokens = normalized_text_tokens(line)
    if len(tokens) <= 2:
        return True
    if len(tokens) <= 4 and any(any(ch.isdigit() for ch in token) for token in tokens):
        return True
    if text_ocr_quality_score(line) >= 0.34:
        return True
    one_char_tokens = sum(1 for token in tokens if len(token) == 1)
    return one_char_tokens / max(1, len(tokens)) >= 0.28


def table_ocr_should_append_line(
    line: str,
    candidate_name: str,
    *,
    new_tokens: int,
    seen_tokens: set[str],
    base_lines: list[str],
    table_record: TableOcrFusionLine | None = None,
) -> bool:
    tokens = normalized_text_tokens(line)
    if not tokens:
        return False
    quality = text_ocr_quality_score(line)
    if candidate_name == "table_cell_consensus":
        # Consensus cell OCR is only reliable when it can repair an existing row.
        # Without a geometry/text match, novel tokens can come from nearby figures.
        return False
    if candidate_name == "table_cells" and not table_ocr_base_lines_are_table_dominant(base_lines):
        return False
    if candidate_name == "table_cells_rotated":
        if len(tokens) < 2 or quality > 0.62:
            return False
        return new_tokens >= max(1, len(tokens) // 2)
    if len(tokens) < 4 or quality > 0.40:
        return False
    if len(tokens) > 48:
        return False
    if not ocr_table_regions.table_ocr_line_has_table_signal(line, tokens):
        return False
    needed = max(2, int(len(tokens) * 0.42))
    if new_tokens < needed:
        return False
    token_set = set(tokens)
    return not len(token_set.difference(seen_tokens)) < 2


def table_ocr_base_lines_are_table_dominant(lines: list[str]) -> bool:
    signal_lines = 0
    content_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = normalized_text_tokens(stripped)
        if not tokens:
            continue
        content_lines += 1
        digit_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
        compact_numeric_row = len(tokens) <= 14 and digit_tokens >= 2
        monetary_row = "$" in stripped
        if monetary_row or compact_numeric_row:
            signal_lines += 1
    if content_lines == 0:
        return False
    if content_lines == 1:
        return signal_lines >= 1
    if content_lines <= 3:
        return signal_lines >= 2
    return signal_lines * 2 >= content_lines


def extract_vector_stroke_page_text(page: PageExtractionHost) -> str:
    return extract_vector_stroke_page_result(page).text


def extract_vector_stroke_page_result(
    page: PageExtractionHost,
) -> VectorStrokeOcrResult:
    return vector_stroke_ocr_result_with_timeout(
        page,
        ocr_rendering.ocr_timeout_seconds(),
    )


FORM_BLANK_PLACEHOLDER = "___"
FORM_BLANK_MAX_SUPPLEMENT = 36


def supplement_form_blank_fields(page: PageExtractionHost, text: str) -> str:
    supplement_count = form_blank_field_supplement_count(page, text)
    if supplement_count <= 0:
        return text
    supplement = " ".join([FORM_BLANK_PLACEHOLDER] * supplement_count)
    if not text:
        return supplement
    return text.rstrip() + "\n" + supplement


def form_blank_field_supplement_count(page: PageExtractionHost, text: str) -> int:
    if not form_blank_supplement_text_is_eligible(text):
        return 0
    try:
        graphics = page.get_graphics()
    except Exception:
        return 0
    lines = list(getattr(graphics, "lines", ()) or ())
    if len(lines) < 12:
        return 0
    existing = text.count(FORM_BLANK_PLACEHOLDER)
    empty_cells, total_cells = empty_form_grid_cell_count(page, lines)
    underlines = isolated_form_underline_count(lines)
    if total_cells and empty_cells < max(3, total_cells // 5):
        empty_cells = 0
    count = max(empty_cells, underlines)
    if count <= existing:
        return 0
    return min(FORM_BLANK_MAX_SUPPLEMENT, count - existing)


def form_blank_supplement_text_is_eligible(text: str) -> bool:
    if not text:
        return False
    tokens = normalized_text_tokens(text)
    if len(tokens) < 40:
        return False
    return numeric_token_ratio(text) < 0.35


def empty_form_grid_cell_count(page: PageExtractionHost, lines: list[Any]) -> tuple[int, int]:
    grid = detect_grid(lines)
    if grid is None:
        return 0, 0
    if len(grid.cols) < 3 or len(grid.rows) < 3:
        return 0, 0
    chars = [
        char
        for char in getattr(page, "chars", ())
        if getattr(char, "text", "").strip() and not getattr(char, "stripped_text", "").isspace()
    ]
    count = 0
    total = 0
    for row_top, row_bottom in zip(grid.rows, grid.rows[1:]):
        top = max(row_top, row_bottom)
        bottom = min(row_top, row_bottom)
        height = top - bottom
        if height < 6.0 or height > 80.0:
            continue
        for left, right in zip(grid.cols, grid.cols[1:]):
            width = right - left
            if width < 24.0 or width > 360.0:
                continue
            total += 1
            if form_cell_has_text(chars, left, bottom, right, top):
                continue
            count += 1
            if count >= FORM_BLANK_MAX_SUPPLEMENT:
                return count, total
    return count, total


def form_cell_has_text(
    chars: list[Any], left: float, bottom: float, right: float, top: float
) -> bool:
    x_padding = min(3.0, max(0.5, (right - left) * 0.04))
    y_padding = min(2.0, max(0.5, (top - bottom) * 0.08))
    for char in chars:
        mid_x = float(getattr(char, "mid_x", (char.x0 + char.x1) * 0.5))
        mid_y = float(getattr(char, "mid_y", (char.y0 + char.y1) * 0.5))
        if (
            left + x_padding <= mid_x <= right - x_padding
            and bottom + y_padding <= mid_y <= top - y_padding
        ):
            return True
    return False


def isolated_form_underline_count(lines: list[Any]) -> int:
    horizontals: list[tuple[float, float, float]] = []
    verticals: list[tuple[float, float, float]] = []
    for line in lines:
        try:
            x0, x1 = sorted((float(line.x0), float(line.x1)))
            y0, y1 = sorted((float(line.y0), float(line.y1)))
            line_width = float(getattr(line, "line_width", 1.0))
        except (TypeError, ValueError):
            continue
        width = x1 - x0
        height = y1 - y0
        if width >= 24.0 and height <= 1.25 and line_width <= 2.0:
            horizontals.append((x0, (y0 + y1) * 0.5, x1))
        elif height >= 10.0 and width <= 1.25:
            verticals.append(((x0 + x1) * 0.5, y0, y1))
    count = 0
    seen: list[tuple[float, float, float]] = []
    for x0, y, x1 in horizontals:
        width = x1 - x0
        if not (24.0 <= width <= 260.0):
            continue
        if any(abs(y - old_y) <= 1.0 and abs(x0 - old_x0) <= 2.0 for old_x0, old_y, _ in seen):
            continue
        intersections = sum(
            1 for x, y0, y1 in verticals if x0 - 1.0 <= x <= x1 + 1.0 and y0 - 1.0 <= y <= y1 + 1.0
        )
        if intersections == 0:
            seen.append((x0, y, x1))
            count += 1
            if count >= FORM_BLANK_MAX_SUPPLEMENT:
                return count
    return count


def broad_page_ocr_should_win_over_figure_ocr(
    broad_result: OcrPageTextResult | None,
    figure_result: OcrPageTextResult,
) -> bool:
    """Keep a substantial numeric page OCR result over a cleaner partial crop."""
    if broad_result is None or broad_result.candidate is None:
        return False
    if not broad_result.candidate.name.startswith("full_page"):
        return False
    broad_text = broad_result.text
    figure_text = figure_result.text
    broad_tokens = extracted_text_token_count(broad_text)
    figure_tokens = extracted_text_token_count(figure_text)
    broad_is_numeric = numeric_token_ratio(broad_text) >= 0.20 or (
        ocr_text_analysis.text_has_many_digit_lines(broad_text)
    )
    minimum_broad_tokens = (
        max(180, int(figure_tokens * 1.25)) if broad_is_numeric else max(300, figure_tokens + 40)
    )
    if broad_tokens < minimum_broad_tokens:
        return False
    return (
        text_ocr_quality_score(broad_text) <= 0.24
        and ocr_text_analysis.scanned_ocr_artifact_score(broad_text) <= 0.20
    )


def extract_figure_ocr_page_result(
    page: PageExtractionHost,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
    broad_candidate: OcrCandidate | None = None,
) -> OcrPageTextResult:
    regions = ocr_page_analysis.figure_ocr_regions(page)
    if not regions:
        return ocr_candidates.OcrPageTextResult("")

    timeout = ocr_rendering.ocr_timeout_seconds()
    candidates: list[OcrCandidate] = []
    selected_candidates: list[OcrCandidate] = []
    rendered_page_image: OcrImage | None = None
    for region in regions:
        image = figure_region_source_image(page, region)
        if image is None:
            if rendered_page_image is None:
                rendered_page_image = ocr_rendering.render_page_for_ocr_at_dpi(
                    page,
                    dpi=OCR_FIGURE_RENDER_DPI,
                    source="figure_rendered_page",
                )
            image = figure_region_rendered_image(page, rendered_page_image, region)
        if image is not None and figure_region_is_image_only_full_page(region):
            image = optimized_full_page_ocr_image(
                image,
                max_pixels=OCR_FULL_PAGE_FIGURE_MAX_PIXELS,
                source_suffix="scaled_figure",
                allow_any_source=True,
            )
        if image is None:
            continue

        if figure_should_use_fixed_grid_subregions(region):
            image_views = figure_ocr_image_variants(image, include_dark=False)
        else:
            image_views = [("source", image)]
        fixed_grid_subregions = figure_should_use_fixed_grid_subregions(region)
        region_candidates: list[OcrCandidate] = []
        if (
            figure_region_is_image_only_full_page(region)
            and broad_candidate is not None
            and broad_candidate.name == "full_page_simple"
        ):
            reused_candidate = replace(
                broad_candidate,
                name=f"figure_region_{region.item_index}_reused_full_page_simple",
            )
            reused_candidate = figure_candidate_with_layout_text(
                reused_candidate,
                region.bbox,
            )
            candidates.append(reused_candidate)
            region_candidates.append(reused_candidate)
            geometry_candidate = figure_geometry_band_candidate(
                reused_candidate,
                region,
            )
            if geometry_candidate is not None and figure_geometry_candidate_is_material_improvement(
                geometry_candidate,
                reused_candidate,
            ):
                candidates.append(geometry_candidate)
                region_candidates.append(geometry_candidate)
        for image_view_name, image_view in image_views:
            for profile_name, variables in figure_whole_image_profiles_for_region(region):
                profile_candidate_count = len(region_candidates)
                psms = list(figure_source_page_segmentation_modes_for_region(region))
                if (
                    figure_region_is_image_only_full_page(region)
                    and broad_candidate is not None
                    and broad_candidate.name == "full_page_simple"
                ):
                    psms = [
                        psm
                        for psm in psms
                        if psm != ocr_execution.OCR_DEFAULT_PAGE_SEGMENTATION_MODE
                    ]
                if not psms:
                    continue
                ocr_image_view = image_view
                if (
                    figure_region_is_image_only_full_page(region)
                    and broad_candidate is not None
                    and broad_candidate.name == "full_page_simple"
                    and psms == [OCR_FIGURE_AUTO_PAGE_SEGMENTATION_MODE]
                ):
                    ocr_image_view = optimized_full_page_ocr_image(
                        image_view,
                        max_pixels=OCR_FULL_PAGE_FIGURE_FOLLOWUP_MAX_PIXELS,
                        source_suffix="scaled_figure_followup",
                        allow_any_source=True,
                    )
                ocr_results = (
                    ocr_session.image_to_text_results(
                        ocr_image_view,
                        psms=psms,
                        variables=variables,
                    )
                    if ocr_session is not None
                    else ocr_execution.ocr_image_to_text_results_with_psms_timeout(
                        ocr_image_view,
                        psms=psms,
                        variables=variables,
                        timeout=timeout,
                    )
                )
                for psm, ocr_result in zip(psms, ocr_results, strict=False):
                    candidate = ocr_candidate_generation.ocr_candidate_from_image(
                        f"figure_region_{region.item_index}_{image_view_name}_"
                        f"{profile_name}_psm{psm}",
                        ocr_result,
                        ocr_image_view,
                    )
                    candidate = figure_candidate_with_layout_text(candidate, region.bbox)
                    candidates.append(candidate)
                    region_candidates.append(candidate)
                if (
                    fixed_grid_subregions
                    and profile_name == "base"
                    and len(region_candidates) > profile_candidate_count
                    and not figure_fixed_grid_subregion_retry_is_needed(
                        region,
                        region_candidates[profile_candidate_count:],
                    )
                ):
                    break

        if figure_region_is_image_only_full_page(region) and broad_candidate is not None:
            geometry_plan = figure_region_geometry_plan(
                broad_candidate,
                region_bbox=region.bbox,
            )
            geometry_evidence = collect_figure_region_geometry_evidence(
                page,
                region,
                broad_candidate,
                timeout,
                ocr_session=ocr_session,
                roi_plan=geometry_plan,
            )
            geometry_candidate_groups = (
                geometry_evidence.micro_band_candidates,
                geometry_evidence.band_slot_candidates,
                geometry_evidence.band_slot_window_candidates,
                geometry_evidence.micro_fragment_candidates,
            )
            for group_candidates in geometry_candidate_groups:
                for candidate in group_candidates:
                    candidates.append(candidate)

        should_try_full_page_subregions = figure_should_try_full_page_subregion_recovery(
            region,
            region_candidates,
        )
        include_grid_subregions = figure_should_include_grid_subregions(
            region,
            region_candidates,
        )
        if not figure_region_is_image_only_full_page(region) or should_try_full_page_subregions:
            should_try_fixed_grid_subregions = True
            if fixed_grid_subregions:
                should_try_fixed_grid_subregions = figure_fixed_grid_subregion_retry_is_needed(
                    region,
                    region_candidates,
                )
            subregion_candidates: list[OcrCandidate] = []
            if should_try_fixed_grid_subregions:
                subregion_candidates = figure_subregion_ocr_candidates(
                    region,
                    image_views,
                    region_candidates,
                    timeout,
                    include_grid=include_grid_subregions,
                    ocr_session=ocr_session,
                )
                for candidate in subregion_candidates:
                    candidates.append(candidate)
                    region_candidates.append(candidate)

            if not fixed_grid_subregions:
                callout_cluster_candidates = []
                if (
                    not figure_region_is_image_only_full_page(region)
                    or should_try_full_page_subregions
                ):
                    callout_cluster_candidates = figure_callout_cluster_ocr_candidates(
                        page,
                        region,
                        image_views,
                        region_candidates,
                        timeout,
                        ocr_session=ocr_session,
                    )
                for candidate in callout_cluster_candidates:
                    candidates.append(candidate)
                    region_candidates.append(candidate)

            if not fixed_grid_subregions and not figure_region_is_image_only_full_page(region):
                pixel_subregion_candidates = figure_pixel_subregion_ocr_candidates(
                    region,
                    image_views,
                    timeout,
                    ocr_session=ocr_session,
                )
                for candidate in pixel_subregion_candidates:
                    candidates.append(candidate)
                    region_candidates.append(candidate)

        selected = fused_figure_ocr_candidate(region, region_candidates)
        if selected is not None:
            candidates.append(selected)
            selected_candidates.append(selected)

    result = combined_figure_ocr_page_result(selected_candidates, candidates)
    return result


def extract_embedded_image_text_ocr_page_result(
    page: PageExtractionHost,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
) -> OcrPageTextResult:
    regions = embedded_image_text_regions(page)
    if not regions:
        return ocr_candidates.OcrPageTextResult("")

    timeout = ocr_rendering.ocr_timeout_seconds()
    candidates: list[OcrCandidate] = []
    line_candidates: list[EmbeddedImageTextLine] = []
    for region in regions:
        image = embedded_image_text_region_source_image(page, region)
        if image is None:
            continue
        for candidate in embedded_image_text_region_candidates(
            region,
            image,
            timeout,
            ocr_session=ocr_session,
        ):
            candidates.append(candidate)
            for line in embedded_image_text_selected_lines(candidate, region):
                line_candidates.append(line)

    selected_lines = embedded_image_text_best_lines(line_candidates)

    text = "\n".join(line.text for line in selected_lines)
    confidence = embedded_image_text_confidence(candidates)
    line_rows = tuple(
        embedded_image_text_line_row(line, index) for index, line in enumerate(selected_lines)
    )
    result_text = OcrTextResult(text, confidence, line_rows=line_rows)
    embedded_candidate = (
        ocr_candidates.OcrCandidate(
            "embedded_image_text",
            result_text,
            region_count=len({line.region.item_index for line in selected_lines}),
            page_bbox=union_page_bboxes(
                line.page_bbox for line in selected_lines if line.page_bbox is not None
            ),
        )
        if text
        else None
    )
    result = ocr_candidates.OcrPageTextResult(
        text,
        embedded_candidate,
        tuple(candidates),
    )
    return result


def embedded_image_text_regions(
    page: PageExtractionHost,
) -> tuple[EmbeddedImageTextRegion, ...]:
    if getattr(page, "rotation", 0) % 360 != 0:
        return ()
    cache = getattr(page, "extraction_cache", None)
    cache_key = "embedded_image_text_regions"
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    rendered = ocr_page_analysis.rendered_page_for_ocr_analysis(page)
    page_area = max(1.0, float(rendered.width) * float(rendered.height))
    regions: list[EmbeddedImageTextRegion] = []
    for item_index, item in enumerate(rendered.display_list.items):
        if item.kind not in {"image", "inline-image"}:
            continue
        metadata = item.data.get("image_metadata")
        if not isinstance(metadata, dict):
            continue
        pixels = int(metadata.get("pixels") or 0)
        if not (
            OCR_EMBEDDED_IMAGE_TEXT_MIN_PIXELS <= pixels <= OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_PIXELS
        ):
            continue
        box = page_geometry.normalize_rect(item.data.get("bbox"))
        if box is None:
            continue
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width <= 0.0 or height <= 0.0:
            continue
        area = width * height
        area_ratio = area / page_area
        pixel_density = pixels / max(area, 1.0)
        aspect = width / height
        if not embedded_image_text_region_is_eligible(
            pixels,
            width=width,
            height=height,
            area_ratio=area_ratio,
            pixel_density=pixel_density,
        ):
            continue
        regions.append(
            EmbeddedImageTextRegion(
                box,
                item_index,
                str(item.kind),
                {
                    "pixels": pixels,
                    "area_ratio": round(area_ratio, 5),
                    "pixel_density": round(pixel_density, 2),
                    "aspect_ratio": round(aspect, 3),
                },
            )
        )
    result = tuple(sorted(regions, key=lambda region: (-region.bbox[3], region.bbox[0])))
    if cache is not None:
        cache[cache_key] = result
    return result


def embedded_image_text_region_is_eligible(
    pixels: int,
    *,
    width: float,
    height: float,
    area_ratio: float,
    pixel_density: float,
) -> bool:
    if width <= 0.0 or height <= 0.0:
        return False
    if pixel_density < OCR_EMBEDDED_IMAGE_TEXT_MIN_PIXEL_DENSITY:
        return False
    aspect = width / height
    if (
        OCR_EMBEDDED_IMAGE_TEXT_MIN_PIXELS <= pixels <= OCR_EMBEDDED_IMAGE_TEXT_MAX_PIXELS
        and area_ratio <= OCR_EMBEDDED_IMAGE_TEXT_MAX_AREA_RATIO
        and OCR_EMBEDDED_IMAGE_TEXT_MIN_ASPECT_RATIO
        <= aspect
        <= OCR_EMBEDDED_IMAGE_TEXT_MAX_ASPECT_RATIO
    ):
        return True
    return (
        OCR_EMBEDDED_IMAGE_TEXT_MIN_PIXELS <= pixels <= OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_PIXELS
        and area_ratio <= OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_AREA_RATIO
        and OCR_EMBEDDED_IMAGE_TEXT_WIDE_MIN_ASPECT_RATIO
        <= aspect
        <= OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_ASPECT_RATIO
    )


def embedded_image_text_region_source_image(
    page: PageExtractionHost,
    region: EmbeddedImageTextRegion,
) -> OcrImage | None:
    rendered = ocr_page_analysis.rendered_page_for_ocr_analysis(page)
    items = getattr(rendered.display_list, "items", ())
    if region.item_index < 0 or region.item_index >= len(items):
        return None
    return ocr_image_from_rendered_image_item(
        items[region.item_index],
        encoded_source="embedded_image_text_encoded_image",
        rgb_source="embedded_image_text_rgb_image",
        prefer_decoded=True,
        cache=page.extraction_cache,
        cache_key=("rendered_page_item", region.item_index, "embedded_image_text"),
    )


def embedded_image_text_region_candidates(
    region: EmbeddedImageTextRegion,
    image: OcrImage,
    timeout: float | None,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
) -> list[OcrCandidate]:
    candidates: list[OcrCandidate] = []
    for views in embedded_image_text_view_batches(image):
        for name, view in views:
            psms = list(embedded_image_text_view_page_segmentation_modes(name))
            ocr_results = (
                ocr_session.image_to_text_results(
                    view,
                    psms=psms,
                    variables=OCR_EMBEDDED_IMAGE_TEXT_VARIABLES,
                )
                if ocr_session is not None
                else ocr_execution.ocr_image_to_text_results_with_psms_timeout(
                    view,
                    psms=psms,
                    variables=OCR_EMBEDDED_IMAGE_TEXT_VARIABLES,
                    timeout=timeout,
                )
            )
            for psm, ocr_result in zip(psms, ocr_results, strict=False):
                candidate = ocr_candidate_generation.ocr_candidate_from_image(
                    f"embedded_image_text_{region.item_index}_{name}_psm{psm}",
                    ocr_result,
                    view,
                )
                candidates.append(candidate)
        if embedded_image_text_region_candidates_are_sufficient(region, candidates):
            break
    return candidates


def embedded_image_text_view_page_segmentation_modes(name: str) -> tuple[int, ...]:
    if name.startswith(("linear_", "center_", "wide_")):
        return OCR_EMBEDDED_IMAGE_TEXT_LINEAR_PSMS
    return OCR_EMBEDDED_IMAGE_TEXT_RING_PSMS


def embedded_image_text_region_candidates_are_sufficient(
    region: EmbeddedImageTextRegion,
    candidates: list[OcrCandidate],
) -> bool:
    lines: list[EmbeddedImageTextLine] = []
    for candidate in candidates:
        lines.extend(embedded_image_text_selected_lines(candidate, region))
    selected = embedded_image_text_best_lines(lines)
    families = {embedded_image_text_candidate_family(line.candidate.name) for line in selected}
    if not {"top", "bottom"} <= families:
        return False
    top_bottom_lines = [
        line
        for line in selected
        if embedded_image_text_candidate_family(line.candidate.name) in {"top", "bottom"}
    ]
    if len(top_bottom_lines) < 2:
        return False
    return all(embedded_image_text_line_score(line) >= 18.0 for line in top_bottom_lines)


def embedded_image_text_view_batches(
    image: OcrImage,
) -> list[list[tuple[str, OcrImage]]]:
    linear_center_views: list[tuple[str, OcrImage]] = embedded_image_text_wide_views(image)
    linear_center_views.extend(embedded_image_text_linear_views(image))
    linear_center_views.extend(embedded_image_text_center_views(image))
    top_views: list[tuple[str, OcrImage]] = []
    bottom_views: list[tuple[str, OcrImage]] = []
    full_views: list[tuple[str, OcrImage]] = []
    fallback_views: list[tuple[str, OcrImage]] = []
    max_radius = min(image.width, image.height) * 0.49
    for (
        name,
        start,
        end,
        inner,
        outer,
        width,
        height,
    ) in OCR_EMBEDDED_IMAGE_TEXT_RING_CONFIGS:
        view = polar_unwrap_embedded_image_text_ring(
            image,
            start_degrees=start,
            end_degrees=end,
            inner_radius=max_radius * inner,
            outer_radius=max_radius * outer,
            width=width,
            height=height,
            source=f"{image.source}_{name}",
        )
        if view is not None:
            if name.startswith("top_"):
                top_views.append((name, view))
            elif name.startswith("bottom_"):
                if "reverse" in name:
                    fallback_views.append((name, view))
                    fallback_views.extend(
                        embedded_image_text_ring_orientation_views(
                            name,
                            view,
                            suffixes=("hflip", "vflip", "rot180"),
                        )
                    )
                else:
                    bottom_views.append((name, view))
                    bottom_views.extend(
                        embedded_image_text_ring_orientation_views(
                            name,
                            view,
                            suffixes=("rot180",),
                        )
                    )
                    fallback_views.extend(
                        embedded_image_text_ring_orientation_views(
                            name,
                            view,
                            suffixes=("hflip", "vflip"),
                        )
                    )
            elif name.startswith("full_"):
                full_views.append((name, view))
            else:
                fallback_views.append((name, view))
    batches = [
        linear_center_views,
        top_views,
        bottom_views,
        full_views,
        fallback_views,
    ]
    return [batch for batch in batches if batch]


def embedded_image_text_wide_views(image: OcrImage) -> list[tuple[str, OcrImage]]:
    if image.width <= 0 or image.height <= 0:
        return []
    aspect = image.width / image.height
    if not (
        OCR_EMBEDDED_IMAGE_TEXT_WIDE_MIN_ASPECT_RATIO
        <= aspect
        <= OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_ASPECT_RATIO
    ):
        return []
    scaled = scaled_embedded_image_text_linear_view(
        image,
        source=f"{image.source}_wide_full_scale{OCR_EMBEDDED_IMAGE_TEXT_LINEAR_SCALE}",
    )
    return [("wide_full", scaled)] if scaled is not None else []


def embedded_image_text_ring_orientation_views(
    name: str,
    view: OcrImage,
    *,
    suffixes: tuple[str, ...],
) -> list[tuple[str, OcrImage]]:
    variants: list[tuple[str, OcrImage]] = []
    for suffix in suffixes:
        transformed = transform_ocr_image_pixels(
            view,
            source=f"{view.source}_{suffix}",
            transform=suffix,
        )
        if transformed is not None:
            variants.append((f"{name}_{suffix}", transformed))
    return variants


def embedded_image_text_center_views(image: OcrImage) -> list[tuple[str, OcrImage]]:
    views: list[tuple[str, OcrImage]] = []
    for name, left, top, right, bottom in (
        ("center_core", 0.28, 0.28, 0.72, 0.72),
        ("center_wide", 0.18, 0.18, 0.82, 0.82),
    ):
        region = NativeOcrRegion(
            max(0, int(image.width * left)),
            max(0, int(image.height * top)),
            min(image.width, int(math.ceil(image.width * right))),
            min(image.height, int(math.ceil(image.height * bottom))),
        )
        if region.width < 8 or region.height < 8:
            continue
        crop = crop_ocr_image_pixel_region(
            image,
            region,
            source=f"{image.source}_{name}",
        )
        if crop is None:
            continue
        scaled = scaled_embedded_image_text_linear_view(
            crop,
            source=f"{crop.source}_scale{OCR_EMBEDDED_IMAGE_TEXT_LINEAR_SCALE}",
        )
        if scaled is not None:
            views.append((name, scaled))
    return views


def embedded_image_text_linear_views(image: OcrImage) -> list[tuple[str, OcrImage]]:
    views: list[tuple[str, OcrImage]] = []
    for index, box in enumerate(
        ocr_image_text_cluster_boxes(
            image,
            max_regions=OCR_EMBEDDED_IMAGE_TEXT_MAX_LINEAR_REGIONS,
        ),
        start=1,
    ):
        if ocr_image_region_is_whole_image(box, image):
            continue
        crop = crop_ocr_image_pixel_region(
            image,
            box,
            source=f"{image.source}_linear_region_{index}",
        )
        if crop is None:
            continue
        scaled = scaled_embedded_image_text_linear_view(
            crop,
            source=f"{crop.source}_scale{OCR_EMBEDDED_IMAGE_TEXT_LINEAR_SCALE}",
        )
        if scaled is None:
            continue
        views.append((f"linear_region_{index}", scaled))
    return views


def scaled_embedded_image_text_linear_view(
    image: OcrImage,
    *,
    source: str,
) -> OcrImage | None:
    if image.width <= 0 or image.height <= 0:
        return None
    target_width = image.width * OCR_EMBEDDED_IMAGE_TEXT_LINEAR_SCALE
    target_height = image.height * OCR_EMBEDDED_IMAGE_TEXT_LINEAR_SCALE
    if target_width * target_height > OCR_EMBEDDED_IMAGE_TEXT_LINEAR_MAX_TARGET_PIXELS:
        return None
    if not leptonica_pix_size_is_supported(target_width, target_height):
        return None
    return replace(
        image,
        source=source,
        target_width=target_width,
        target_height=target_height,
        resolution=OCR_EMBEDDED_IMAGE_TEXT_DPI,
    )


def transform_ocr_image_pixels(
    image: OcrImage,
    *,
    source: str,
    transform: str,
) -> OcrImage | None:
    if image.bytes_per_pixel <= 0 or not image.data:
        return None
    if image.width <= 0 or image.height <= 0:
        return None
    if image.bytes_per_line < image.width * image.bytes_per_pixel:
        return None
    target = bytearray(image.width * image.height * image.bytes_per_pixel)
    target_row_bytes = image.width * image.bytes_per_pixel
    for y in range(image.height):
        for x in range(image.width):
            if transform == "hflip":
                source_x = image.width - 1 - x
                source_y = y
            elif transform == "vflip":
                source_x = x
                source_y = image.height - 1 - y
            elif transform == "rot180":
                source_x = image.width - 1 - x
                source_y = image.height - 1 - y
            else:
                return None
            source_offset = source_y * image.bytes_per_line + source_x * image.bytes_per_pixel
            target_offset = y * target_row_bytes + x * image.bytes_per_pixel
            target[target_offset : target_offset + image.bytes_per_pixel] = image.data[
                source_offset : source_offset + image.bytes_per_pixel
            ]
    return OcrImage(
        bytes(target),
        image.width,
        image.height,
        image.bytes_per_pixel,
        target_row_bytes,
        source=source,
        resolution=image.resolution,
        page_bbox=image.page_bbox,
        page_clockwise_quarter_turns=image.page_clockwise_quarter_turns,
    )


def polar_unwrap_embedded_image_text_ring(
    image: OcrImage,
    *,
    start_degrees: float,
    end_degrees: float,
    inner_radius: float,
    outer_radius: float,
    width: int,
    height: int,
    source: str,
) -> OcrImage | None:
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return None
    if image.width <= 1 or image.height <= 1 or width <= 0 or height <= 0:
        return None
    if image.bytes_per_line < image.width * image.bytes_per_pixel:
        return None
    center_x = (image.width - 1) * 0.5
    center_y = (image.height - 1) * 0.5
    data = bytearray(width * height * 3)
    out = 0
    for target_y in range(height):
        radius_t = 1.0 - (target_y / max(1, height - 1))
        radius = inner_radius + (outer_radius - inner_radius) * radius_t
        for target_x in range(width):
            angle_t = target_x / max(1, width - 1)
            angle = math.radians(start_degrees + (end_degrees - start_degrees) * angle_t)
            red, green, blue = embedded_image_sample_rgb(
                image,
                center_x + math.cos(angle) * radius,
                center_y + math.sin(angle) * radius,
            )
            gray = (red * 30 + green * 59 + blue * 11) // 100
            value = max(0, gray - 35) if gray < 180 else min(255, gray + 15)
            data[out] = value
            data[out + 1] = value
            data[out + 2] = value
            out += 3
    return OcrImage(
        bytes(data),
        width,
        height,
        3,
        width * 3,
        source=source,
        resolution=OCR_EMBEDDED_IMAGE_TEXT_DPI,
        page_bbox=image.page_bbox,
        page_clockwise_quarter_turns=image.page_clockwise_quarter_turns,
    )


def embedded_image_sample_rgb(
    image: OcrImage,
    x: float,
    y: float,
) -> tuple[int, int, int]:
    if x < 0.0 or y < 0.0 or x >= image.width - 1 or y >= image.height - 1:
        return (255, 255, 255)
    source_x = int(x)
    source_y = int(y)
    offset = source_y * image.bytes_per_line + source_x * image.bytes_per_pixel
    if image.bytes_per_pixel == 1:
        value = image.data[offset]
        return (value, value, value)
    red = image.data[offset]
    green = image.data[offset + 1]
    blue = image.data[offset + 2]
    if image.bytes_per_pixel == 4:
        alpha = image.data[offset + 3]
        if alpha < 255:
            inverse_alpha = 255 - alpha
            red = (red * alpha + 255 * inverse_alpha) // 255
            green = (green * alpha + 255 * inverse_alpha) // 255
            blue = (blue * alpha + 255 * inverse_alpha) // 255
    return (red, green, blue)


def embedded_image_text_span_lines(candidate: OcrCandidate) -> list[str]:
    if embedded_image_text_candidate_is_wide_logo(candidate):
        return embedded_image_text_wide_logo_span_lines(candidate)
    rows = [row for row in candidate.result.word_rows if isinstance(row, dict)]
    lines: list[str] = []
    current: list[str] = []
    current_key: tuple[int, int, int] | None = None
    allow_single_acronym = "_center_" in candidate.name
    for row in rows:
        row_key = (
            ocr_int_value(row.get("block_num", 0) or 0),
            ocr_int_value(row.get("par_num", 0) or 0),
            ocr_int_value(row.get("line_num", 0) or 0),
        )
        words = embedded_image_text_words(row, candidate.image_width)
        if row_key != current_key:
            embedded_image_flush_span(
                current,
                lines,
                allow_single_acronym=allow_single_acronym,
            )
            current = []
            current_key = row_key
        if words is None:
            embedded_image_flush_span(
                current,
                lines,
                allow_single_acronym=allow_single_acronym,
            )
            current = []
            continue
        current.extend(words)
    embedded_image_flush_span(
        current,
        lines,
        allow_single_acronym=allow_single_acronym,
    )
    return lines


def embedded_image_text_candidate_is_wide_logo(candidate: OcrCandidate) -> bool:
    return "_wide_full_" in candidate.name and candidate.name.endswith(
        f"_psm{ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE}"
    )


def embedded_image_text_wide_logo_span_lines(candidate: OcrCandidate) -> list[str]:
    records = embedded_image_text_wide_logo_span_records(candidate)
    if not records:
        return []
    image_width = candidate.image_width or max(record.right for record in records)
    image_height = candidate.image_height or max(record.bottom for record in records)
    logo_lines = [
        record.text
        for record in records
        if embedded_image_text_record_is_left_logo_acronym(
            record,
            image_width=image_width,
            image_height=image_height,
        )
    ]
    text_records = [
        record
        for record in records
        if not embedded_image_text_record_is_left_logo_acronym(
            record,
            image_width=image_width,
            image_height=image_height,
        )
    ]
    columns = embedded_image_text_wide_logo_columns(text_records, image_width)
    lines = list(dict.fromkeys(logo_lines))
    for column in columns:
        words: list[str] = []
        for record in sorted(column, key=lambda value: (value.top, value.left)):
            words.extend(ocr_text_analysis.normalized_text_tokens(record.text))
        if not words:
            continue
        line = " ".join(word.upper() for word in words)
        if line not in lines:
            lines.append(line)
    return lines


def embedded_image_text_wide_logo_span_records(
    candidate: OcrCandidate,
) -> list[EmbeddedImageTextSpanRecord]:
    rows = [row for row in candidate.result.word_rows if isinstance(row, dict)]
    records: list[EmbeddedImageTextSpanRecord] = []
    current_words: list[str] = []
    current_rows: list[dict[str, Any]] = []
    current_key: tuple[int, int, int] | None = None
    for row in rows:
        row_key = (
            ocr_int_value(row.get("block_num", 0) or 0),
            ocr_int_value(row.get("par_num", 0) or 0),
            ocr_int_value(row.get("line_num", 0) or 0),
        )
        words = embedded_image_text_words(
            row,
            candidate.image_width,
            allow_titlecase=True,
        )
        if row_key != current_key:
            embedded_image_flush_span_record(
                current_words,
                current_rows,
                records,
                allow_single_acronym=True,
            )
            current_words = []
            current_rows = []
            current_key = row_key
        if words is None:
            embedded_image_flush_span_record(
                current_words,
                current_rows,
                records,
                allow_single_acronym=True,
            )
            current_words = []
            current_rows = []
            continue
        current_words.extend(words)
        current_rows.append(row)
    embedded_image_flush_span_record(
        current_words,
        current_rows,
        records,
        allow_single_acronym=True,
    )
    return records


def embedded_image_flush_span_record(
    words: list[str],
    rows: list[dict[str, Any]],
    records: list[EmbeddedImageTextSpanRecord],
    *,
    allow_single_acronym: bool,
) -> None:
    words = embedded_image_text_repair_fragmented_words(words)
    if not embedded_image_text_span_is_useful(
        words,
        allow_single_acronym=allow_single_acronym,
        allow_single_known_word=True,
    ):
        return
    bounds = embedded_image_text_word_rows_bounds(rows)
    if bounds is None:
        return
    confidences = [embedded_image_text_row_confidence(row) for row in rows]
    numeric_confidences = [value for value in confidences if value is not None]
    confidence = int(round(median(numeric_confidences))) if numeric_confidences else None
    left, top, right, bottom = bounds
    records.append(
        EmbeddedImageTextSpanRecord(
            " ".join(words),
            left,
            top,
            right,
            bottom,
            confidence,
        )
    )


def embedded_image_text_word_rows_bounds(
    rows: list[dict[str, Any]],
) -> tuple[int, int, int, int] | None:
    bounds: list[tuple[int, int, int, int]] = []
    for row in rows:
        try:
            left = int(row.get("left", 0))
            top = int(row.get("top", 0))
            width = int(row.get("width", 0))
            height = int(row.get("height", 0))
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        bounds.append((left, top, left + width, top + height))
    if not bounds:
        return None
    return (
        min(bound[0] for bound in bounds),
        min(bound[1] for bound in bounds),
        max(bound[2] for bound in bounds),
        max(bound[3] for bound in bounds),
    )


def embedded_image_text_row_confidence(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("conf", 0))
    except (TypeError, ValueError):
        return None


def embedded_image_text_record_is_left_logo_acronym(
    record: EmbeddedImageTextSpanRecord,
    *,
    image_width: int,
    image_height: int,
) -> bool:
    tokens = ocr_text_analysis.normalized_text_tokens(record.text)
    if len(tokens) != 1:
        return False
    token = tokens[0].upper()
    if not embedded_image_text_word_is_plausible_acronym(token, record.confidence or 0):
        return False
    return record.left <= image_width * 0.35 and record.top <= image_height * 0.75


def embedded_image_text_wide_logo_columns(
    records: list[EmbeddedImageTextSpanRecord],
    image_width: int,
) -> list[list[EmbeddedImageTextSpanRecord]]:
    if not records:
        return []
    tolerance = max(24.0, image_width * 0.08)
    columns: list[list[EmbeddedImageTextSpanRecord]] = []
    anchors: list[float] = []
    for record in sorted(records, key=lambda value: (value.left, value.top)):
        best_index: int | None = None
        best_distance = tolerance
        for index, anchor in enumerate(anchors):
            distance = abs(record.left - anchor)
            if distance <= best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            columns.append([record])
            anchors.append(float(record.left))
            continue
        columns[best_index].append(record)
        anchors[best_index] = sum(item.left for item in columns[best_index]) / len(
            columns[best_index]
        )
    return [
        column
        for _, column in sorted(
            zip(anchors, columns, strict=False),
            key=lambda pair: pair[0],
        )
    ]


def embedded_image_text_selected_lines(
    candidate: OcrCandidate,
    region: EmbeddedImageTextRegion,
) -> list[EmbeddedImageTextLine]:
    page_bbox = candidate.page_bbox or region.bbox
    confidence = candidate.result.confidence
    return [
        EmbeddedImageTextLine(
            text=line,
            candidate=candidate,
            region=region,
            page_bbox=page_bbox,
            confidence=confidence,
        )
        for line in embedded_image_text_span_lines(candidate)
    ]


def embedded_image_text_best_lines(
    lines: list[EmbeddedImageTextLine],
) -> list[EmbeddedImageTextLine]:
    grouped: dict[tuple[int, str, str], list[EmbeddedImageTextLine]] = {}
    for line in lines:
        tokens = ocr_text_analysis.normalized_text_tokens(line.text)
        if not tokens:
            continue
        family = embedded_image_text_candidate_family(line.candidate.name)
        key = (
            line.region.item_index,
            family,
            line.text.casefold() if family == "wide" else "",
        )
        grouped.setdefault(key, []).append(line)
    selected = [
        max(group, key=embedded_image_text_line_score) for key, group in sorted(grouped.items())
    ]
    selected = embedded_image_text_drop_subset_lines(selected)
    return sorted(
        selected,
        key=lambda line: (
            -line.region.bbox[3],
            line.region.bbox[0],
            embedded_image_text_candidate_family_order(
                embedded_image_text_candidate_family(line.candidate.name)
            ),
        ),
    )


def embedded_image_text_candidate_family(name: str) -> str:
    if "_wide_full_" in name:
        return "wide"
    if "_center_" in name:
        return "center"
    if "_top_ring" in name:
        return "top"
    if "_bottom_ring" in name:
        return "bottom"
    if "_linear_region" in name:
        return "linear"
    if "_full_ring" in name:
        return "full"
    return "other"


def embedded_image_text_candidate_family_order(family: str) -> int:
    order = {
        "wide": 0,
        "top": 1,
        "center": 2,
        "bottom": 3,
        "linear": 4,
        "full": 5,
        "other": 6,
    }
    return order.get(family, 99)


def embedded_image_text_line_score(line: EmbeddedImageTextLine) -> float:
    tokens = ocr_text_analysis.normalized_text_tokens(line.text)
    confidence = line.confidence if line.confidence is not None else 45
    score = len(tokens) * 8.0 + min(95, confidence) * 0.06
    for token in tokens:
        upper = token.upper()
        if len(token) >= 4 and embedded_image_text_word_is_known(upper):
            score += 2.5
        if len(token) <= 2 and token not in {"of"}:
            score -= 4.0
        if len(token) >= 5 and not embedded_image_text_word_is_known(upper):
            score -= 3.0
    if embedded_image_text_candidate_family(line.candidate.name) == "center":
        score += 4.0
    score -= ocr_text_analysis.text_ocr_quality_score(line.text) * 18.0
    return score


def embedded_image_text_drop_subset_lines(
    lines: list[EmbeddedImageTextLine],
) -> list[EmbeddedImageTextLine]:
    kept: list[EmbeddedImageTextLine] = []
    for line in sorted(lines, key=embedded_image_text_line_score, reverse=True):
        tokens = set(ocr_text_analysis.normalized_text_tokens(line.text))
        if not tokens:
            continue
        if any(
            line.region.item_index == existing.region.item_index
            and tokens <= set(ocr_text_analysis.normalized_text_tokens(existing.text))
            for existing in kept
        ):
            continue
        kept.append(line)
    return kept


def embedded_image_text_line_row(
    line: EmbeddedImageTextLine,
    index: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "text": line.text,
        "conf": line.confidence if line.confidence is not None else 0,
        "left": 0,
        "top": index,
        "width": 1,
        "height": 1,
    }
    if line.page_bbox is not None:
        x0, y0, x1, y1 = line.page_bbox
        row.update(
            {
                "left": int(round(x0)),
                "top": int(round(y0)),
                "width": max(1, int(round(x1 - x0))),
                "height": max(1, int(round(y1 - y0))),
                "page_bbox": line.page_bbox,
            }
        )
    return row


def embedded_image_flush_span(
    words: list[str],
    lines: list[str],
    *,
    allow_single_acronym: bool = False,
) -> None:
    words = embedded_image_text_repair_fragmented_words(words)
    if not embedded_image_text_span_is_useful(
        words,
        allow_single_acronym=allow_single_acronym,
    ):
        return
    line = " ".join(words).replace(" & ", " & ")
    if line not in lines:
        lines.append(line)


def embedded_image_text_repair_fragmented_words(words: list[str]) -> list[str]:
    repaired: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if index + 1 < len(words):
            joined = word + words[index + 1]
            completion = embedded_image_text_common_prefix_completion(joined)
            if completion is not None:
                repaired.append(completion)
                index += 2
                continue
        completion = embedded_image_text_common_prefix_completion(word)
        if completion is not None and completion != word:
            repaired.append(completion)
        else:
            repaired.append(word)
        index += 1
    return repaired


@lru_cache(maxsize=4096)
def embedded_image_text_common_prefix_completion(prefix: str) -> str | None:
    if not prefix.isalpha() or len(prefix) < 7:
        return None
    normalized = prefix.casefold()
    if word_frequencies.is_common_word(normalized, max_rank=40_000):
        return prefix.upper()
    best_word: str | None = None
    best_rank: int | None = None
    for word, frequency in word_frequencies.english_word_frequency_prefix_items(normalized):
        if frequency.rank > 40_000:
            continue
        if not word.startswith(normalized):
            continue
        if len(word) - len(normalized) > 6:
            continue
        if len(normalized) / len(word) < 0.68:
            continue
        if best_rank is None or frequency.rank < best_rank:
            best_word = word
            best_rank = frequency.rank
    return best_word.upper() if best_word is not None else None


@lru_cache(maxsize=4096)
def dominant_image_common_prefix_completion(prefix: str) -> str | None:
    if not prefix.isalpha() or len(prefix) < 3:
        return None
    normalized = prefix.casefold()
    if len(normalized) >= 5 and word_frequencies.is_common_word(normalized, max_rank=20_000):
        return prefix.upper()
    best_word: str | None = None
    best_key: tuple[int, int, int] | None = None
    for word, frequency in word_frequencies.english_word_frequency_prefix_items(normalized):
        if frequency.rank > 20_000:
            continue
        if not word.startswith(normalized):
            continue
        if len(word) - len(normalized) > 4:
            continue
        if len(normalized) / len(word) < 0.45:
            continue
        if len(normalized) <= 3 and (len(word) < 6 or word.endswith("s")):
            continue
        key = (
            0 if len(normalized) >= 4 else -len(word),
            0 if not word.endswith("s") else 1,
            frequency.rank,
        )
        if best_key is None or key < best_key:
            best_word = word
            best_key = key
    return best_word.upper() if best_word is not None else None


def embedded_image_text_words(
    row: dict[str, Any],
    image_width: int | None,
    *,
    allow_titlecase: bool = False,
) -> tuple[str, ...] | None:
    text = str(row.get("text", "")).strip()
    if not text:
        return None
    try:
        confidence = int(row.get("conf", 0))
    except (TypeError, ValueError):
        confidence = 0
    if confidence < 70:
        return None
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in {"&", "-"})
    if not cleaned:
        return None
    if cleaned == "&":
        return (cleaned,)
    letters = [ch for ch in cleaned if ch.isalpha()]
    if not letters:
        return None
    uppercase = sum(1 for ch in letters if ch.isupper())
    if not allow_titlecase and uppercase / len(letters) < 0.72:
        return None
    if len(cleaned) <= 2 and cleaned.upper() != "OF":
        return None
    if embedded_image_text_short_word_touches_strip_edge(
        cleaned,
        row,
        image_width,
    ):
        return None
    upper = cleaned.upper()
    if embedded_image_text_word_is_plausible_acronym(upper, confidence):
        return (upper,)
    if embedded_image_text_word_is_known(upper):
        return (upper,)
    split = split_embedded_image_compound_word(upper)
    if split is not None:
        return split
    return (upper,)


def embedded_image_text_short_word_touches_strip_edge(
    word: str,
    row: dict[str, Any],
    image_width: int | None,
) -> bool:
    if image_width is None or len(word) > 3:
        return False
    try:
        left = int(row.get("left", 0))
        width = int(row.get("width", 0))
    except (TypeError, ValueError):
        return False
    if width <= 0:
        return False
    right = left + width
    return left <= 4 or right >= image_width - 4


def embedded_image_text_word_is_plausible_acronym(
    word: str,
    confidence: int,
) -> bool:
    if confidence < 78:
        return False
    if not word.isalpha() or not word.isupper():
        return False
    if not 2 <= len(word) <= 6:
        return False
    unique = set(word)
    if len(unique) <= 1:
        return False
    return max(word.count(ch) for ch in unique) / len(word) <= 0.55


def embedded_image_text_word_is_known(word: str) -> bool:
    if word == "OF":
        return True
    if not word.isalpha():
        return False
    return word_frequencies.is_common_word(word, max_rank=120_000)


def split_embedded_image_compound_word(word: str) -> tuple[str, str] | None:
    if not word.isalpha() or len(word) < 8:
        return None
    best: tuple[int, str, str] | None = None
    for index in range(2, len(word) - 1):
        left = word[:index]
        right = word[index:]
        if not embedded_image_text_word_is_known(left):
            continue
        if not embedded_image_text_word_is_known(right):
            continue
        score = min(len(left), len(right))
        if best is None or score > best[0]:
            best = (score, left, right)
    if best is None:
        return None
    return (best[1], best[2])


def embedded_image_text_span_is_useful(
    words: list[str],
    *,
    allow_single_acronym: bool = False,
    allow_single_known_word: bool = False,
) -> bool:
    if not words:
        return False
    alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
    if (
        allow_single_acronym
        and len(alpha_words) == 1
        and (
            embedded_image_text_word_is_known(alpha_words[0])
            or embedded_image_text_word_is_plausible_acronym(alpha_words[0], 90)
        )
        and 2 <= len(alpha_words[0]) <= 6
    ):
        return True
    if (
        allow_single_known_word
        and len(alpha_words) == 1
        and len(alpha_words[0]) >= 4
        and embedded_image_text_word_is_known(alpha_words[0])
    ):
        return True
    if len(alpha_words) < 2:
        return False
    if not any(len(word) >= 4 for word in alpha_words):
        return False
    long_unknown_words = [
        word
        for word in alpha_words
        if len(word) >= 5 and not word_frequencies.is_common_word(word, max_rank=120_000)
    ]
    if long_unknown_words:
        return False
    text = " ".join(words)
    return not ocr_text_analysis.text_ocr_quality_score(text) > 0.34


def embedded_image_text_confidence(candidates: list[OcrCandidate]) -> int | None:
    confidences = [
        candidate.result.confidence
        for candidate in candidates
        if candidate.result.confidence is not None
    ]
    if not confidences:
        return None
    return int(round(median(confidences)))


def figure_region_is_image_only_full_page(
    region: ocr_page_analysis.FigureOcrRegion,
) -> bool:
    signals = region.signals or {}
    return bool(signals.get("image_only_full_page_region"))


def page_has_image_only_full_page_figure_region(page: PageExtractionHost) -> bool:
    cache = page.extraction_cache
    if cache is not None:
        cached = cache.get("has_image_only_full_page_figure_region")
        if isinstance(cached, bool):
            return cached
    regions = ocr_page_analysis.figure_ocr_regions(page)
    value = any(figure_region_is_image_only_full_page(region) for region in regions)
    if cache is not None:
        cache["has_image_only_full_page_figure_region"] = value
    return value


def figure_whole_image_profiles_for_region(
    region: ocr_page_analysis.FigureOcrRegion,
) -> tuple[tuple[str, dict[str, str]], ...]:
    if figure_region_is_image_only_full_page(region):
        return OCR_FIGURE_FULL_PAGE_IMAGE_ONLY_WHOLE_IMAGE_PROFILES
    return OCR_FIGURE_WHOLE_IMAGE_PROFILES


def figure_source_page_segmentation_modes_for_region(
    region: ocr_page_analysis.FigureOcrRegion,
) -> tuple[int, ...]:
    if figure_region_is_image_only_full_page(region):
        return OCR_FIGURE_FULL_PAGE_IMAGE_ONLY_PAGE_SEGMENTATION_MODES
    return OCR_FIGURE_SOURCE_PAGE_SEGMENTATION_MODES


def figure_region_source_image(
    page: PageExtractionHost,
    region: ocr_page_analysis.FigureOcrRegion,
) -> OcrImage | None:
    rendered = ocr_page_analysis.rendered_page_for_ocr_analysis(page)
    items = getattr(rendered.display_list, "items", ())
    if region.item_index < 0 or region.item_index >= len(items):
        return None
    return ocr_image_from_rendered_image_item(
        items[region.item_index],
        encoded_source="figure_encoded_image",
        rgb_source="figure_rgb_image",
        cache=page.extraction_cache,
        cache_key=("rendered_page_item", region.item_index),
    )


def figure_region_rendered_image(
    page: PageExtractionHost,
    image: OcrImage | None,
    region: ocr_page_analysis.FigureOcrRegion,
) -> OcrImage | None:
    if image is None:
        return None
    return figure_crop_page_bbox_from_image(
        image,
        page_bbox=region.bbox,
        cache=page.extraction_cache,
        cache_key=("figure_rendered_region", id(image)),
    )


def figure_render_ocr_regions(
    page: PageExtractionHost,
    image: OcrImage,
    boxes: tuple[tuple[float, float, float, float], ...],
) -> list[NativeOcrRegion]:
    page_space = page_geometry.PageSpace.from_page(page)
    if page_space is None:
        return []
    geometry = page_geometry.ImageSpace.from_dimensions(
        image_width=image.width,
        image_height=image.height,
        image_resolution=image.resolution,
        page_bbox=page_space.bbox,
        source=image.source,
    )
    regions: list[NativeOcrRegion] = []
    for box in boxes:
        rectangle = page_geometry.page_bbox_to_image_pixel_bbox(
            box,
            geometry,
            padding=OCR_FIGURE_RENDER_PADDING_POINTS,
            clamp=True,
        )
        if rectangle is None:
            continue
        left, top, right, bottom = rectangle
        if right - left < 4 or bottom - top < 4:
            continue
        regions.append(NativeOcrRegion(left, top, right, bottom))
    return regions


def crop_ocr_image_region(
    image: OcrImage,
    region: NativeOcrRegion,
    *,
    source: str,
    page_bbox: tuple[float, float, float, float],
) -> OcrImage | None:
    if image.bytes_per_pixel <= 0 or not image.data:
        return None
    if region.width <= 0 or region.height <= 0:
        return None
    source_row_bytes = image.bytes_per_line
    target_row_bytes = region.width * image.bytes_per_pixel
    required = (region.y1 - 1) * source_row_bytes + region.x1 * image.bytes_per_pixel
    if len(image.data) < required:
        return None
    data = bytearray(target_row_bytes * region.height)
    for y in range(region.height):
        source_start = (region.y0 + y) * source_row_bytes + region.x0 * image.bytes_per_pixel
        source_stop = source_start + target_row_bytes
        target_start = y * target_row_bytes
        data[target_start : target_start + target_row_bytes] = image.data[source_start:source_stop]
    return OcrImage(
        bytes(data),
        region.width,
        region.height,
        image.bytes_per_pixel,
        target_row_bytes,
        source=source,
        resolution=image.resolution,
        page_bbox=page_geometry.normalize_rect(page_bbox),
    )


def crop_ocr_image_pixel_region(
    image: OcrImage,
    region: NativeOcrRegion,
    *,
    source: str,
) -> OcrImage | None:
    page_bbox = ocr_image_pixel_region_page_bbox(image, region)
    if page_bbox is None:
        page_bbox = image.page_bbox
    if page_bbox is None:
        return None
    return crop_ocr_image_region(
        image,
        region,
        source=source,
        page_bbox=page_bbox,
    )


def ocr_image_pixel_region_page_bbox(
    image: OcrImage,
    region: NativeOcrRegion,
) -> tuple[float, float, float, float] | None:
    geometry = page_geometry.ImageSpace.from_ocr_image(image, source=image.source)
    return page_geometry.image_bbox_to_page_bbox(
        (region.x0, region.y0, region.x1, region.y1),
        geometry,
    )


def ocr_image_region_is_whole_image(
    region: NativeOcrRegion,
    image: OcrImage,
) -> bool:
    if image.width <= 0 or image.height <= 0:
        return True
    area_ratio = region.area / max(1, image.width * image.height)
    return (
        area_ratio >= 0.82
        and region.width >= image.width * 0.85
        and region.height >= image.height * 0.85
    )


def ocr_image_text_cluster_boxes(
    image: OcrImage,
    *,
    max_regions: int,
) -> list[NativeOcrRegion]:
    mask = ocr_image_foreground_mask(image)
    if mask is None:
        return []
    row_spans = ocr_foreground_projection_spans(
        ocr_foreground_row_counts(mask, image.width, image.height),
        min_count=max(2, int(image.width * 0.006)),
        max_gap=max(2, int(image.height * 0.012)),
    )
    boxes: list[NativeOcrRegion] = []
    for y0, y1 in row_spans:
        if y1 - y0 < 4:
            continue
        column_counts = ocr_foreground_column_counts(mask, image.width, y0, y1)
        column_spans = ocr_foreground_projection_spans(
            column_counts,
            min_count=max(1, int((y1 - y0) * 0.045)),
            max_gap=max(6, int(image.width * 0.035)),
        )
        for x0, x1 in column_spans:
            box = ocr_refined_foreground_box(mask, image.width, x0, y0, x1, y1)
            if box is None:
                continue
            expanded = ocr_expand_image_region(
                box,
                image.width,
                image.height,
                padding=max(
                    OCR_EMBEDDED_IMAGE_TEXT_LINEAR_REGION_PADDING,
                    int(min(image.width, image.height) * 0.018),
                ),
            )
            if expanded.area < OCR_EMBEDDED_IMAGE_TEXT_MIN_LINEAR_REGION_PIXELS:
                continue
            boxes.append(expanded)
    if not boxes:
        return []
    boxes = ocr_merge_overlapping_image_regions(boxes)
    boxes = [box for box in boxes if box.area >= OCR_EMBEDDED_IMAGE_TEXT_MIN_LINEAR_REGION_PIXELS]
    boxes.sort(key=lambda box: (box.y0, box.x0, -box.area))
    return boxes[:max_regions]


def ocr_image_foreground_mask(image: OcrImage) -> bytearray | None:
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return None
    if image.width <= 0 or image.height <= 0:
        return None
    if image.bytes_per_line < image.width * image.bytes_per_pixel:
        return None
    required = (image.height - 1) * image.bytes_per_line + image.width * image.bytes_per_pixel
    if len(image.data) < required:
        return None
    data = image.data
    width = image.width
    height = image.height
    bytes_per_line = image.bytes_per_line
    bytes_per_pixel = image.bytes_per_pixel
    mask = bytearray(width * height)
    index = 0
    if bytes_per_pixel == 1:
        for y in range(height):
            offset = y * bytes_per_line
            row_stop = offset + width
            while offset < row_stop:
                if data[offset] <= 238:
                    mask[index] = 1
                index += 1
                offset += 1
        return mask
    for y in range(height):
        offset = y * bytes_per_line
        row_stop = offset + width * bytes_per_pixel
        while offset < row_stop:
            red = data[offset]
            green = data[offset + 1]
            blue = data[offset + 2]
            if bytes_per_pixel != 4 or data[offset + 3] > 16:
                gray = (red * 30 + green * 59 + blue * 11) // 100
                distance_from_white = max(255 - red, 255 - green, 255 - blue)
                saturation = max(red, green, blue) - min(red, green, blue)
                if gray <= 238 or distance_from_white >= 38 or (saturation >= 28 and gray <= 248):
                    mask[index] = 1
            index += 1
            offset += bytes_per_pixel
    return mask


def embedded_image_pixel_is_foreground(image: OcrImage, offset: int) -> bool:
    if image.bytes_per_pixel == 1:
        value = image.data[offset]
        return value <= 238
    red = image.data[offset]
    green = image.data[offset + 1]
    blue = image.data[offset + 2]
    if image.bytes_per_pixel == 4 and image.data[offset + 3] <= 16:
        return False
    gray = (red * 30 + green * 59 + blue * 11) // 100
    distance_from_white = max(255 - red, 255 - green, 255 - blue)
    saturation = max(red, green, blue) - min(red, green, blue)
    return gray <= 238 or distance_from_white >= 38 or (saturation >= 28 and gray <= 248)


def ocr_foreground_row_counts(
    mask: bytearray,
    width: int,
    height: int,
) -> list[int]:
    return [
        sum(mask[row_start : row_start + width]) for row_start in range(0, width * height, width)
    ]


def ocr_foreground_column_counts(
    mask: bytearray,
    width: int,
    y0: int,
    y1: int,
) -> list[int]:
    counts = [0] * width
    for y in range(y0, y1):
        row_start = y * width
        for x in range(width):
            counts[x] += mask[row_start + x]
    return counts


def ocr_foreground_projection_spans(
    counts: list[int],
    *,
    min_count: int,
    max_gap: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    last_foreground: int | None = None
    gap = 0
    for index, count in enumerate(counts):
        if count >= min_count:
            if start is None:
                start = index
            last_foreground = index
            gap = 0
            continue
        if start is None:
            continue
        gap += 1
        if gap > max_gap:
            if last_foreground is not None and last_foreground + 1 > start:
                spans.append((start, last_foreground + 1))
            start = None
            last_foreground = None
            gap = 0
    if start is not None and last_foreground is not None:
        spans.append((start, last_foreground + 1))
    return spans


def ocr_refined_foreground_box(
    mask: bytearray,
    width: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> NativeOcrRegion | None:
    min_x = x1
    min_y = y1
    max_x = x0
    max_y = y0
    foreground = 0
    for y in range(y0, y1):
        row_start = y * width
        for x in range(x0, x1):
            if not mask[row_start + x]:
                continue
            foreground += 1
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + 1)
            max_y = max(max_y, y + 1)
    if foreground <= 0 or max_x <= min_x or max_y <= min_y:
        return None
    return NativeOcrRegion(min_x, min_y, max_x, max_y)


def ocr_expand_image_region(
    region: NativeOcrRegion,
    width: int,
    height: int,
    *,
    padding: int,
) -> NativeOcrRegion:
    return NativeOcrRegion(
        max(0, region.x0 - padding),
        max(0, region.y0 - padding),
        min(width, region.x1 + padding),
        min(height, region.y1 + padding),
    )


def ocr_merge_overlapping_image_regions(
    boxes: list[NativeOcrRegion],
) -> list[NativeOcrRegion]:
    merged: list[NativeOcrRegion] = []
    for box in boxes:
        current = box
        changed = True
        while changed:
            changed = False
            remaining: list[NativeOcrRegion] = []
            for existing in merged:
                if ocr_image_regions_should_merge(current, existing):
                    current = NativeOcrRegion(
                        min(current.x0, existing.x0),
                        min(current.y0, existing.y0),
                        max(current.x1, existing.x1),
                        max(current.y1, existing.y1),
                    )
                    changed = True
                else:
                    remaining.append(existing)
            merged = remaining
        merged.append(current)
    return merged


def ocr_image_regions_should_merge(
    left: NativeOcrRegion,
    right: NativeOcrRegion,
) -> bool:
    horizontal_gap = max(0, max(left.x0, right.x0) - min(left.x1, right.x1))
    vertical_gap = max(0, max(left.y0, right.y0) - min(left.y1, right.y1))
    overlap_width = min(left.x1, right.x1) - max(left.x0, right.x0)
    overlap_height = min(left.y1, right.y1) - max(left.y0, right.y0)
    median_height = max(1, int((left.height + right.height) * 0.5))
    median_width = max(1, int((left.width + right.width) * 0.5))
    if overlap_height > 0 and horizontal_gap <= max(8, int(median_height * 0.75)):
        return True
    return overlap_width > 0 and vertical_gap <= max(6, int(median_width * 0.18))


def figure_ocr_image_variants(
    image: OcrImage,
    *,
    include_dark: bool = True,
) -> list[tuple[str, OcrImage]]:
    variants: list[tuple[str, OcrImage]] = [("source", image)]
    scaled = scaled_figure_image_view(
        image,
        source=f"{image.source}_scale{OCR_FIGURE_IMAGE_VIEW_SCALE}",
    )
    if scaled is not None:
        variants.append((f"scale{OCR_FIGURE_IMAGE_VIEW_SCALE}", scaled))
    if not include_dark:
        return variants
    darkened = ocr_candidate_generation.darken_ocr_image_min_3x3(
        image,
        source=f"{image.source}_dark",
        resolution=image.resolution or OCR_FIGURE_RENDER_DPI,
    )
    if darkened is not None:
        variants.append(("dark", darkened))
        scaled_darkened = scaled_figure_image_view(
            darkened,
            source=f"{darkened.source}_scale{OCR_FIGURE_IMAGE_VIEW_SCALE}",
        )
        if scaled_darkened is not None:
            variants.append((f"dark_scale{OCR_FIGURE_IMAGE_VIEW_SCALE}", scaled_darkened))
    return variants


def scaled_figure_image_view(image: OcrImage, *, source: str) -> OcrImage | None:
    if image.width <= 0 or image.height <= 0:
        return None
    target_width = image.width * OCR_FIGURE_IMAGE_VIEW_SCALE
    target_height = image.height * OCR_FIGURE_IMAGE_VIEW_SCALE
    current_target_width = image.target_width or image.width
    current_target_height = image.target_height or image.height
    if target_width <= current_target_width and target_height <= current_target_height:
        return None
    target_pixels = target_width * target_height
    if target_pixels > OCR_FIGURE_IMAGE_VIEW_MAX_TARGET_PIXELS:
        return None
    if not leptonica_pix_size_is_supported(target_width, target_height):
        return None
    return replace(
        image,
        source=source,
        target_width=target_width,
        target_height=target_height,
    )


def figure_subregion_image_views(
    image_views: list[tuple[str, OcrImage]],
) -> list[tuple[str, OcrImage]]:
    return [
        (name, image)
        for name, image in image_views
        if image.target_width is None and image.target_height is None
    ]


def figure_subregion_ocr_candidates(
    region: ocr_page_analysis.FigureOcrRegion,
    image_views: list[tuple[str, OcrImage]],
    base_candidates: list[OcrCandidate],
    timeout: float | None,
    *,
    include_grid: bool = False,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
) -> list[OcrCandidate]:
    base_candidate = fused_figure_ocr_candidate(region, base_candidates)
    if base_candidate is None:
        return []
    lines = figure_candidate_text_geometry_lines(base_candidate)
    boxes = figure_subregion_boxes(
        lines,
        region.bbox,
        include_grid=include_grid,
    )
    profiles = figure_subregion_profiles_for_base_candidate(
        base_candidate,
        source_candidates=base_candidates,
    )
    candidates: list[OcrCandidate] = []
    for image_view_name, image in figure_subregion_image_views(image_views):
        if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
            continue
        for index, box in enumerate(boxes, start=1):
            source_name = f"figure_region_{region.item_index}_{image_view_name}_subregion_{index}"
            scaled_crop, results_by_profile = figure_subregion_text_results(
                image,
                box,
                source=source_name,
                psms=list(OCR_FIGURE_SUBREGION_PAGE_SEGMENTATION_MODES),
                variables_by_profile=profiles,
                timeout=timeout,
                ocr_session=ocr_session,
            )
            if scaled_crop is None or results_by_profile is None:
                continue
            psms = list(OCR_FIGURE_SUBREGION_PAGE_SEGMENTATION_MODES)
            for profile_name, _variables in profiles:
                for psm, ocr_result in zip(
                    psms,
                    results_by_profile[profile_name],
                    strict=False,
                ):
                    candidate = ocr_candidate_generation.ocr_candidate_from_image(
                        f"{source_name}_{profile_name}_psm{psm}",
                        ocr_result,
                        scaled_crop,
                    )
                    candidate = figure_candidate_with_layout_text(candidate, box)
                    if not figure_subregion_candidate_is_useful(candidate):
                        continue
                    candidates.append(candidate)
    return candidates


def figure_subregion_profiles_for_base_candidate(
    candidate: OcrCandidate,
    *,
    source_candidates: Iterable[OcrCandidate] = (),
) -> tuple[tuple[str, dict[str, str]], ...]:
    if figure_candidate_supports_reduced_subregion_profiles(candidate) or any(
        figure_candidate_supports_reduced_subregion_profiles(source_candidate)
        for source_candidate in source_candidates
    ):
        return (
            ("base", OCR_FIGURE_BASE_VARIABLES),
            ("otsu", OCR_FIGURE_OTSU_VARIABLES),
        )
    return OCR_FIGURE_SUBREGION_PROFILES


def figure_candidate_supports_reduced_subregion_profiles(
    candidate: OcrCandidate,
) -> bool:
    result = candidate.result
    tokens = extracted_text_token_count(result.text)
    confidence = result.confidence if result.confidence is not None else 0
    return tokens >= 350 and confidence >= 85 and text_ocr_quality_score(result.text) <= 0.28


def figure_callout_cluster_ocr_candidates(
    page: PageExtractionHost,
    region: ocr_page_analysis.FigureOcrRegion,
    image_views: list[tuple[str, OcrImage]],
    base_candidates: list[OcrCandidate],
    timeout: float | None,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
) -> list[OcrCandidate]:
    base_candidate = fused_figure_ocr_candidate(region, base_candidates)
    if base_candidate is None:
        return []
    lines = figure_candidate_text_geometry_lines(base_candidate)
    boxes = figure_patent_callout_subregion_boxes(page, region, lines)
    if not boxes:
        return []
    candidates: list[OcrCandidate] = []
    for image_view_name, image in figure_subregion_image_views(image_views):
        if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
            continue
        for index, box in enumerate(boxes, start=1):
            source_name = (
                f"figure_region_{region.item_index}_{image_view_name}_callout_cluster_{index}"
            )
            scaled_crop, results_by_profile = figure_subregion_text_results(
                image,
                box,
                source=source_name,
                psms=list(OCR_FIGURE_SUBREGION_PAGE_SEGMENTATION_MODES),
                variables_by_profile=OCR_FIGURE_SUBREGION_PROFILES,
                timeout=timeout,
                ocr_session=ocr_session,
                cache=page.extraction_cache,
            )
            if scaled_crop is None or results_by_profile is None:
                continue
            psms = list(OCR_FIGURE_SUBREGION_PAGE_SEGMENTATION_MODES)
            for profile_name, _variables in OCR_FIGURE_SUBREGION_PROFILES:
                for psm, ocr_result in zip(
                    psms,
                    results_by_profile[profile_name],
                    strict=False,
                ):
                    candidate = ocr_candidate_generation.ocr_candidate_from_image(
                        f"{source_name}_{profile_name}_psm{psm}",
                        ocr_result,
                        scaled_crop,
                    )
                    candidate = figure_candidate_with_layout_text(candidate, box)
                    if not figure_subregion_candidate_is_useful(candidate):
                        continue
                    candidates.append(candidate)
    return candidates


def figure_rendered_micro_band_ocr_candidates(
    page: PageExtractionHost,
    region: ocr_page_analysis.FigureOcrRegion,
    broad_candidate: OcrCandidate,
    timeout: float | None,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
    roi_plan: FigureRegionGeometryPlan | None = None,
) -> list[OcrCandidate]:
    if not str(broad_candidate.name).startswith("rendered_page_"):
        return []
    boxes = (
        list(roi_plan.micro_band_boxes)
        if roi_plan is not None
        else figure_broad_candidate_micro_band_boxes(
            broad_candidate,
            region_bbox=region.bbox,
        )
    )
    if not boxes:
        return []
    image = ocr_rendering.render_page_for_ocr_at_dpi(
        page,
        dpi=OCR_FIGURE_RENDER_DPI,
        source=f"figure_region_{region.item_index}_micro_band_rendered_page",
    )
    if image is None:
        return []
    psms = [11]
    profiles = (
        ("base", OCR_FIGURE_BASE_VARIABLES),
        ("sauvola", OCR_FIGURE_SAUVOLA_VARIABLES),
    )
    candidates: list[OcrCandidate] = []
    for index, box in enumerate(boxes, start=1):
        source_name = f"figure_region_{region.item_index}_micro_band_{index}"
        scaled_crop, results_by_profile = figure_subregion_text_results(
            image,
            box,
            source=source_name,
            psms=psms,
            variables_by_profile=profiles,
            timeout=timeout,
            ocr_session=ocr_session,
            cache=page.extraction_cache,
        )
        if scaled_crop is None or results_by_profile is None:
            continue
        for profile_name, _variables in profiles:
            for psm, ocr_result in zip(psms, results_by_profile[profile_name], strict=False):
                candidate = ocr_candidate_generation.ocr_candidate_from_image(
                    f"{source_name}_{profile_name}_psm{psm}",
                    ocr_result,
                    scaled_crop,
                )
                candidate = figure_candidate_with_layout_text(candidate, box)
                if figure_micro_band_candidate_is_useful(candidate):
                    candidates.append(candidate)
    return candidates


def figure_rendered_micro_fragment_ocr_candidates(
    page: PageExtractionHost,
    region: ocr_page_analysis.FigureOcrRegion,
    broad_candidate: OcrCandidate,
    timeout: float | None,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
    roi_plan: FigureRegionGeometryPlan | None = None,
) -> list[OcrCandidate]:
    if not str(broad_candidate.name).startswith("rendered_page_"):
        return []
    boxes = (
        list(roi_plan.micro_fragment_boxes)
        if roi_plan is not None
        else figure_broad_candidate_micro_fragment_boxes(
            broad_candidate,
            region_bbox=region.bbox,
        )
    )
    if not boxes:
        return []
    image = ocr_rendering.render_page_for_ocr_at_dpi(
        page,
        dpi=OCR_FIGURE_RENDER_DPI,
        source=f"figure_region_{region.item_index}_micro_fragment_rendered_page",
    )
    if image is None:
        return []
    psms = [8]
    profiles = (
        ("base", OCR_FIGURE_BASE_VARIABLES),
        ("otsu", OCR_FIGURE_OTSU_VARIABLES),
        ("sauvola", OCR_FIGURE_SAUVOLA_VARIABLES),
    )
    candidates: list[OcrCandidate] = []
    for index, box in enumerate(boxes, start=1):
        source_name = f"figure_region_{region.item_index}_micro_fragment_{index}"
        scaled_crop, results_by_profile = figure_subregion_text_results(
            image,
            box,
            source=source_name,
            psms=psms,
            variables_by_profile=profiles,
            timeout=timeout,
            ocr_session=ocr_session,
            cache=page.extraction_cache,
        )
        if scaled_crop is None or results_by_profile is None:
            continue
        for profile_name, _variables in profiles:
            for psm, ocr_result in zip(psms, results_by_profile[profile_name], strict=False):
                candidate = ocr_candidate_generation.ocr_candidate_from_image(
                    f"{source_name}_{profile_name}_psm{psm}",
                    ocr_result,
                    scaled_crop,
                )
                candidate = figure_candidate_with_layout_text(candidate, box)
                if figure_micro_fragment_candidate_is_useful(candidate):
                    candidates.append(candidate)
    return candidates


def figure_rendered_band_slot_ocr_candidates(
    page: PageExtractionHost,
    region: ocr_page_analysis.FigureOcrRegion,
    micro_band_candidates: list[OcrCandidate],
    timeout: float | None,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
    slots: tuple[FigureBandSlot, ...] | None = None,
) -> list[OcrCandidate]:
    if not micro_band_candidates:
        return []
    planned_slots = (
        dominant_image_figure_band_slots(
            tuple(micro_band_candidates),
            label_region_bbox=region.bbox,
        )
        if slots is None
        else slots
    )
    if not planned_slots:
        return []
    image = ocr_rendering.render_page_for_ocr_at_dpi(
        page,
        dpi=OCR_FIGURE_RENDER_DPI,
        source=f"figure_region_{region.item_index}_band_slot_rendered_page",
    )
    if image is None:
        return []
    configurations = (
        ("base", [8], OCR_FIGURE_BASE_VARIABLES),
        ("otsu", [11], OCR_FIGURE_OTSU_VARIABLES),
    )
    candidates: list[OcrCandidate] = []
    seen_boxes: list[tuple[float, float, float, float]] = []
    for index, slot in enumerate(planned_slots, start=1):
        if any(dominant_image_label_boxes_match(slot.gap_bbox, seen) for seen in seen_boxes):
            continue
        seen_boxes.append(slot.gap_bbox)
        source_name = f"figure_region_{region.item_index}_band_slot_{index}"
        for profile_name, psms, variables in configurations:
            scaled_crop, results_by_profile = figure_subregion_text_results(
                image,
                slot.gap_bbox,
                source=source_name,
                psms=psms,
                variables_by_profile=((profile_name, variables),),
                timeout=timeout,
                ocr_session=ocr_session,
                cache=page.extraction_cache,
            )
            if scaled_crop is None or results_by_profile is None:
                continue
            for psm, ocr_result in zip(psms, results_by_profile[profile_name], strict=False):
                candidate = ocr_candidate_generation.ocr_candidate_from_image(
                    f"{source_name}_{profile_name}_psm{psm}",
                    ocr_result,
                    scaled_crop,
                )
                candidate = figure_candidate_with_layout_text(candidate, slot.gap_bbox)
                if figure_band_slot_candidate_is_useful(candidate):
                    candidates.append(candidate)
    return candidates


def figure_rendered_band_slot_subwindow_ocr_candidates(
    page: PageExtractionHost,
    region: ocr_page_analysis.FigureOcrRegion,
    micro_band_candidates: list[OcrCandidate],
    timeout: float | None,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
    slots: tuple[FigureBandSlot, ...] | None = None,
) -> list[OcrCandidate]:
    if not micro_band_candidates:
        return []
    planned_slots = (
        dominant_image_figure_band_slots(
            tuple(micro_band_candidates),
            label_region_bbox=region.bbox,
        )
        if slots is None
        else slots
    )
    if not planned_slots:
        return []
    image = ocr_rendering.render_page_for_ocr_at_dpi(
        page,
        dpi=OCR_FIGURE_RENDER_DPI,
        source=f"figure_region_{region.item_index}_band_slot_window_rendered_page",
    )
    if image is None:
        return []
    psms = [8]
    profiles = (("base", OCR_FIGURE_BASE_VARIABLES),)
    candidates: list[OcrCandidate] = []
    seen_boxes: list[tuple[float, float, float, float]] = []
    for slot_index, slot in enumerate(planned_slots, start=1):
        for window_index, box in enumerate(
            figure_band_slot_subwindow_boxes(slot),
            start=1,
        ):
            if any(dominant_image_label_boxes_match(box, seen) for seen in seen_boxes):
                continue
            seen_boxes.append(box)
            source_name = (
                f"figure_region_{region.item_index}_band_slot_{slot_index}_window_{window_index}"
            )
            scaled_crop, results_by_profile = figure_subregion_text_results(
                image,
                box,
                source=source_name,
                psms=psms,
                variables_by_profile=profiles,
                timeout=timeout,
                ocr_session=ocr_session,
                cache=page.extraction_cache,
            )
            if scaled_crop is None or results_by_profile is None:
                continue
            for profile_name, _variables in profiles:
                for psm, ocr_result in zip(psms, results_by_profile[profile_name], strict=False):
                    candidate = ocr_candidate_generation.ocr_candidate_from_image(
                        f"{source_name}_{profile_name}_psm{psm}",
                        ocr_result,
                        scaled_crop,
                    )
                    candidate = figure_candidate_with_layout_text(candidate, box)
                    if figure_band_slot_candidate_is_useful(candidate):
                        candidates.append(candidate)
    return candidates


def figure_broad_candidate_micro_band_boxes(
    broad_candidate: OcrCandidate,
    *,
    region_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    seed_lines = [
        ocr_page_analysis.text_geometry_line_from_bbox(
            item.text,
            item.bbox,
            None if item.confidence is None else int(round(item.confidence)),
            source=item.source,
            kind="figure_text_line",
        )
        for item in figure_candidate_token_evidence(broad_candidate)
        if figure_micro_band_seed_evidence_is_usable(item)
    ]
    if len(seed_lines) < 4:
        return []
    vertical_bands: list[list[ocr_page_analysis.TextGeometryLine]] = []
    for line in sorted(
        seed_lines,
        key=lambda item: (
            (
                (item.observation.bbox or (0.0, 0.0, 0.0, 0.0))[1]
                + (item.observation.bbox or (0.0, 0.0, 0.0, 0.0))[3]
            )
            * 0.5,
            (item.observation.bbox or (0.0, 0.0, 0.0, 0.0))[0],
        ),
    ):
        bbox = line.observation.bbox
        if bbox is None:
            continue
        center_y = (bbox[1] + bbox[3]) * 0.5
        target_band: list[ocr_page_analysis.TextGeometryLine] | None = None
        for band in vertical_bands:
            band_boxes = [
                item.observation.bbox for item in band if item.observation.bbox is not None
            ]
            if not band_boxes:
                continue
            band_center_y = sum((box[1] + box[3]) * 0.5 for box in band_boxes) / len(band_boxes)
            band_height = max(box[3] for box in band_boxes) - min(box[1] for box in band_boxes)
            line_height = bbox[3] - bbox[1]
            new_top = min(bbox[1], min(box[1] for box in band_boxes))
            new_bottom = max(bbox[3], max(box[3] for box in band_boxes))
            new_height = new_bottom - new_top
            if new_height <= max(72.0, line_height * 6.0) and abs(center_y - band_center_y) <= max(
                42.0, max(band_height, line_height) * 1.5
            ):
                target_band = band
                break
        if target_band is None:
            vertical_bands.append([line])
        else:
            target_band.append(line)
    boxes: list[tuple[float, float, float, float]] = []
    for band in vertical_bands:
        ordered = sorted(
            [line for line in band if line.observation.bbox is not None],
            key=lambda item: cast(tuple[float, float, float, float], item.observation.bbox)[0],
        )
        if len(ordered) < 4:
            continue
        segments: list[list[ocr_page_analysis.TextGeometryLine]] = []
        current: list[ocr_page_analysis.TextGeometryLine] = [ordered[0]]
        for previous, line in zip(ordered, ordered[1:], strict=False):
            previous_bbox = cast(tuple[float, float, float, float], previous.observation.bbox)
            bbox = cast(tuple[float, float, float, float], line.observation.bbox)
            previous_width = max(1.0, previous_bbox[2] - previous_bbox[0])
            width = max(1.0, bbox[2] - bbox[0])
            gap = bbox[0] - previous_bbox[2]
            if gap > max(48.0, max(previous_width, width) * 3.0):
                segments.append(current)
                current = [line]
                continue
            current.append(line)
        segments.append(current)
        for segment in segments:
            box = expanded_figure_micro_band_box(segment, region_bbox)
            if box is None:
                continue
            boxes.append(box)
    ranked = sorted(
        figure_unique_subregion_boxes(boxes),
        key=lambda box: (-page_geometry.rect_area(box), box[0], -box[3]),
    )
    return ranked[:4]


def figure_broad_candidate_micro_fragment_boxes(
    broad_candidate: OcrCandidate,
    *,
    region_bbox: tuple[float, float, float, float],
    band_boxes: Iterable[tuple[float, float, float, float]] | None = None,
) -> list[tuple[float, float, float, float]]:
    resolved_band_boxes = (
        list(band_boxes)
        if band_boxes is not None
        else figure_broad_candidate_micro_band_boxes(
            broad_candidate,
            region_bbox=region_bbox,
        )
    )
    if not resolved_band_boxes:
        return []
    seed_evidence = [
        item
        for item in figure_candidate_token_evidence(
            broad_candidate,
            token_extractor=dominant_image_alpha_fragment_token,
        )
        if item.token.isalpha() and 2 <= len(item.token) <= 4
    ]
    numeric_evidence = [
        item
        for item in figure_candidate_token_evidence(broad_candidate)
        if item.token.isdigit() and len(item.token) == 2
    ]
    boxes: list[tuple[float, float, float, float]] = []
    for band_box in resolved_band_boxes:
        band_numeric = [
            item for item in numeric_evidence if figure_box_lives_inside_band(item.bbox, band_box)
        ]
        if len(band_numeric) < 2:
            continue
        boxes.extend(
            figure_band_anchor_interstitial_fragment_boxes(
                broad_candidate,
                band_box=band_box,
                band_numeric=band_numeric,
                region_bbox=region_bbox,
            )
        )
        for item in seed_evidence:
            if not figure_box_lives_inside_band(item.bbox, band_box):
                continue
            right_numeric_boxes = [
                numeric.bbox
                for numeric in band_numeric
                if numeric.bbox[0] > item.bbox[2]
                and abs(
                    ((numeric.bbox[1] + numeric.bbox[3]) * 0.5)
                    - ((item.bbox[1] + item.bbox[3]) * 0.5)
                )
                <= max(40.0, (item.bbox[3] - item.bbox[1]) * 4.5)
                and numeric.bbox[0] - item.bbox[2]
                <= max(
                    180.0,
                    (band_box[2] - band_box[0]) * 0.55,
                )
            ]
            if len(right_numeric_boxes) < 2:
                continue
            right_numeric_boxes = sorted(right_numeric_boxes, key=lambda box: box[0])
            nearest_numeric_x0 = right_numeric_boxes[0][0]
            box = (
                max(region_bbox[0], item.bbox[0] - 8.0),
                max(region_bbox[1], item.bbox[1] - 8.0),
                min(
                    region_bbox[2],
                    min(
                        band_box[2],
                        max(item.bbox[2] + 36.0, nearest_numeric_x0 - 10.0),
                    ),
                ),
                min(region_bbox[3], item.bbox[3] + 8.0),
            )
            if page_geometry.rect_area(box) <= 0.0:
                continue
            width = box[2] - box[0]
            height = box[3] - box[1]
            if width < 24.0 or width > 112.0 or height < 12.0 or height > 32.0:
                continue
            boxes.append(box)
    ranked = sorted(
        figure_unique_subregion_boxes(boxes),
        key=lambda box: (box[0], -box[3]),
    )
    return ranked[:4]


def figure_region_geometry_plan(
    broad_candidate: OcrCandidate,
    *,
    region_bbox: tuple[float, float, float, float],
) -> FigureRegionGeometryPlan:
    micro_band_boxes = tuple(
        figure_broad_candidate_micro_band_boxes(
            broad_candidate,
            region_bbox=region_bbox,
        )
    )
    micro_fragment_boxes = tuple(
        figure_broad_candidate_micro_fragment_boxes(
            broad_candidate,
            region_bbox=region_bbox,
            band_boxes=micro_band_boxes,
        )
    )
    return FigureRegionGeometryPlan(
        micro_band_boxes=micro_band_boxes,
        micro_fragment_boxes=micro_fragment_boxes,
    )


def figure_band_anchor_interstitial_fragment_boxes(
    broad_candidate: OcrCandidate,
    *,
    band_box: tuple[float, float, float, float],
    band_numeric: list[FigureTokenEvidence],
    region_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    alpha_evidence = [
        item
        for item in figure_candidate_token_evidence(broad_candidate)
        if item.token.isalpha()
        and len(item.token) >= 4
        and figure_box_lives_inside_band(item.bbox, band_box)
    ]
    boxes: list[tuple[float, float, float, float]] = []
    for item in alpha_evidence:
        right_numeric = [
            numeric
            for numeric in band_numeric
            if numeric.bbox[0] > item.bbox[2]
            and abs(
                ((numeric.bbox[1] + numeric.bbox[3]) * 0.5) - ((item.bbox[1] + item.bbox[3]) * 0.5)
            )
            <= max(44.0, (item.bbox[3] - item.bbox[1]) * 5.0)
            and numeric.bbox[0] - item.bbox[2]
            <= max(
                220.0,
                (band_box[2] - band_box[0]) * 0.64,
            )
        ]
        if len(right_numeric) < 2:
            continue
        right_numeric = sorted(right_numeric, key=lambda evidence: evidence.bbox[0])
        nearest_numeric = right_numeric[0].bbox
        if nearest_numeric[0] - item.bbox[2] < 8.0:
            continue
        corridor_top = min(item.bbox[1], nearest_numeric[1]) - 4.0
        corridor_bottom = max(item.bbox[3], nearest_numeric[3]) + 4.0
        box = (
            max(region_bbox[0], item.bbox[2] - 1.0),
            max(region_bbox[1], corridor_top),
            min(region_bbox[2], min(band_box[2], nearest_numeric[0] + 3.0)),
            min(region_bbox[3], corridor_bottom),
        )
        width = box[2] - box[0]
        height = box[3] - box[1]
        if page_geometry.rect_area(box) <= 0.0:
            continue
        if width < 24.0 or width > 128.0 or height < 14.0 or height > 40.0:
            continue
        boxes.append(box)
    return boxes


def figure_patent_callout_subregion_boxes(
    page: PageExtractionHost,
    region: ocr_page_analysis.FigureOcrRegion,
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> list[tuple[float, float, float, float]]:
    boxes = figure_patent_callout_cluster_subregion_boxes(page, lines, region.bbox)
    boxes = figure_filter_overlarge_subregion_boxes(boxes, region.bbox)
    if boxes or not figure_region_is_image_only_full_page(region):
        return boxes
    label_lines = [line for line in lines if figure_line_looks_like_compact_label(line)]
    boxes = figure_patent_callout_neighborhood_subregion_boxes(label_lines, region.bbox)
    boxes = figure_filter_overlarge_subregion_boxes(boxes, region.bbox)
    if boxes:
        return boxes
    return figure_full_page_label_neighborhood_subregion_boxes(lines, region.bbox)


def figure_full_page_label_neighborhood_subregion_boxes(
    lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for line in lines:
        if not figure_line_looks_like_full_page_callout_seed(line):
            continue
        box = expanded_figure_patent_callout_neighborhood_box(line, region_bbox)
        if box is not None:
            boxes.append(box)
    ranked = sorted(
        figure_filter_overlarge_subregion_boxes(
            figure_unique_subregion_boxes(boxes),
            region_bbox,
        ),
        key=lambda box: (-page_geometry.rect_area(box), box[0], -box[3]),
    )
    return ranked[:OCR_FIGURE_MAX_CALLOUT_NEIGHBORHOOD_SUBREGIONS]


def figure_line_looks_like_full_page_callout_seed(
    line: ocr_page_analysis.TextGeometryLine,
) -> bool:
    if figure_line_looks_like_compact_label(line):
        return True
    text = line.text.strip()
    if not figure_text_line_is_useful(text, line.confidence):
        return False
    tokens = normalized_text_tokens(text)
    if not tokens or len(tokens) > 5:
        return False
    readable_alpha = sum(1 for token in tokens if token.isalpha() and len(token) >= 4)
    has_numeric = any(token.isdigit() for token in tokens)
    return readable_alpha >= 1 and has_numeric


def figure_filter_overlarge_subregion_boxes(
    boxes: list[tuple[float, float, float, float]],
    region_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    return [
        box
        for box in boxes
        if not figure_subregion_box_is_effectively_whole_region(box, region_bbox)
    ]


def figure_subregion_box_is_effectively_whole_region(
    box: tuple[float, float, float, float],
    region_bbox: tuple[float, float, float, float],
) -> bool:
    box_area = page_geometry.rect_area(box)
    region_area = page_geometry.rect_area(region_bbox)
    if box_area <= 0.0 or region_area <= 0.0:
        return True
    area_ratio = box_area / region_area
    if area_ratio >= 0.72:
        return True
    x_overlap = max(0.0, min(box[2], region_bbox[2]) - max(box[0], region_bbox[0]))
    y_overlap = max(0.0, min(box[3], region_bbox[3]) - max(box[1], region_bbox[1]))
    region_width = max(1.0, region_bbox[2] - region_bbox[0])
    region_height = max(1.0, region_bbox[3] - region_bbox[1])
    return x_overlap / region_width >= 0.92 and y_overlap / region_height >= 0.92


def figure_pixel_subregion_ocr_candidates(
    region: ocr_page_analysis.FigureOcrRegion,
    image_views: list[tuple[str, OcrImage]],
    timeout: float | None,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
) -> list[OcrCandidate]:
    candidates: list[OcrCandidate] = []
    for image_view_name, image in figure_subregion_image_views(image_views):
        boxes = ocr_image_text_cluster_boxes(
            image,
            max_regions=OCR_FIGURE_MAX_PIXEL_SUBREGIONS,
        )
        for index, box in enumerate(boxes, start=1):
            if ocr_image_region_is_whole_image(box, image):
                continue
            source_name = (
                f"figure_region_{region.item_index}_{image_view_name}_pixel_region_{index}"
            )
            scaled_crop, results_by_profile = figure_pixel_subregion_text_results(
                image,
                box,
                source=source_name,
                psms=list(OCR_FIGURE_SUBREGION_PAGE_SEGMENTATION_MODES),
                variables_by_profile=OCR_FIGURE_SUBREGION_PROFILES,
                timeout=timeout,
                ocr_session=ocr_session,
            )
            if scaled_crop is None or results_by_profile is None:
                continue
            psms = list(OCR_FIGURE_SUBREGION_PAGE_SEGMENTATION_MODES)
            for profile_name, _variables in OCR_FIGURE_SUBREGION_PROFILES:
                for psm, ocr_result in zip(
                    psms,
                    results_by_profile[profile_name],
                    strict=False,
                ):
                    candidate = ocr_candidate_generation.ocr_candidate_from_image(
                        f"{source_name}_{profile_name}_psm{psm}",
                        ocr_result,
                        scaled_crop,
                    )
                    candidate = figure_candidate_with_layout_text(
                        candidate,
                        scaled_crop.page_bbox or region.bbox,
                    )
                    if not figure_subregion_candidate_is_useful(candidate):
                        continue
                    candidates.append(candidate)
    return candidates


def figure_subregion_boxes(
    lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
    *,
    include_grid: bool = True,
) -> list[tuple[float, float, float, float]]:
    boxes = [
        *figure_column_subregion_boxes(lines, region_bbox),
        *figure_stack_subregion_boxes(lines, region_bbox),
        *figure_label_cluster_subregion_boxes(lines, region_bbox),
    ]
    if include_grid:
        boxes.extend(figure_grid_subregion_boxes(region_bbox))
    return figure_unique_subregion_boxes(boxes)[:OCR_FIGURE_MAX_TOTAL_SUBREGIONS]


def figure_column_subregion_boxes(
    lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    usable_lines = [
        line for line in lines if figure_text_line_is_useful(line.text, line.confidence)
    ]
    if len(usable_lines) < 4:
        return []
    region_width = max(1.0, region_bbox[2] - region_bbox[0])
    tolerance = max(14.0, min(32.0, region_width * 0.10))
    columns = figure_x_position_clusters(usable_lines, tolerance)
    boxes: list[tuple[float, float, float, float]] = []
    for column in columns:
        if len(column) < 4:
            continue
        column_bbox = page_geometry.observation_union_bbox(line.observation for line in column)
        if column_bbox is None:
            continue
        x0, y0, x1, y1 = column_bbox
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        padded = (
            max(region_bbox[0], x0 - max(8.0, width * 0.45)),
            max(region_bbox[1], y0 - max(6.0, height * 0.05)),
            min(region_bbox[2], x1 + max(18.0, width * 1.55)),
            min(region_bbox[3], y1 + max(6.0, height * 0.05)),
        )
        if page_geometry.rect_area(padded) <= 0.0:
            continue
        boxes.append(padded)
    boxes.sort(key=lambda box: box[0])
    return boxes[:OCR_FIGURE_MAX_SUBREGIONS]


def figure_stack_subregion_boxes(
    lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    usable_lines = [
        line for line in lines if figure_text_line_is_useful(line.text, line.confidence)
    ]
    if len(usable_lines) < 4:
        return []
    region_width = max(1.0, region_bbox[2] - region_bbox[0])
    tolerance = max(14.0, min(32.0, region_width * 0.10))
    columns = figure_x_position_clusters(usable_lines, tolerance)
    boxes: list[tuple[float, float, float, float]] = []
    for column in columns:
        if len(column) < 4:
            continue
        for group in figure_vertical_line_groups(column):
            if len(group) < 2:
                continue
            box = expanded_figure_subregion_box(group, region_bbox)
            if box is not None:
                boxes.append(box)
    boxes.sort(key=lambda box: (box[0], -box[3]))
    return boxes[:OCR_FIGURE_MAX_STACK_SUBREGIONS]


def figure_label_cluster_subregion_boxes(
    lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    label_lines = [line for line in lines if figure_line_looks_like_compact_label(line)]
    if len(label_lines) < 4:
        return []
    clusters: list[list[ocr_page_analysis.TextGeometryLine]] = []
    for line in sorted(label_lines, key=figure_text_line_reading_order_key):
        target: list[ocr_page_analysis.TextGeometryLine] | None = None
        for cluster in clusters:
            if any(figure_label_lines_share_cluster(line, existing) for existing in cluster):
                target = cluster
                break
        if target is None:
            clusters.append([line])
        else:
            target.append(line)
    boxes: list[tuple[float, float, float, float]] = []
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        box = expanded_figure_subregion_box(cluster, region_bbox)
        if box is None:
            continue
        boxes.append(expand_figure_label_cluster_box(box, region_bbox))
    boxes.sort(key=lambda box: (box[0], -box[3]))
    return boxes[:OCR_FIGURE_MAX_LABEL_CLUSTER_SUBREGIONS]


def figure_patent_callout_cluster_subregion_boxes(
    page: PageExtractionHost,
    lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    label_lines = [line for line in lines if figure_line_looks_like_compact_label(line)]
    if not figure_should_use_patent_callout_clusters(label_lines, region_bbox):
        return []
    clusters: list[list[ocr_page_analysis.TextGeometryLine]] = []
    for line in sorted(label_lines, key=figure_text_line_reading_order_key):
        target: list[ocr_page_analysis.TextGeometryLine] | None = None
        for cluster in clusters:
            if any(
                figure_patent_callout_lines_share_cluster(line, existing) for existing in cluster
            ):
                target = cluster
                break
        if target is None:
            clusters.append([line])
        else:
            target.append(line)
    boxes: list[tuple[float, float, float, float]] = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        box = expanded_figure_patent_callout_cluster_box(cluster, region_bbox)
        if box is not None:
            boxes.append(box)
    boxes.extend(figure_patent_callout_neighborhood_subregion_boxes(label_lines, region_bbox))
    return figure_filter_patent_callout_boxes(
        boxes,
        label_lines,
        region_bbox,
        page_geometry.page_grid_lines(page),
    )


def figure_line_looks_like_compact_label(
    line: ocr_page_analysis.TextGeometryLine,
) -> bool:
    text = line.text.strip()
    if not figure_text_line_is_useful(text, line.confidence):
        return False
    if figure_reference_precision_line_is_metadata(text):
        return False
    tokens = normalized_text_tokens(text)
    if not tokens or len(tokens) > 5:
        return False
    if all(token.isdigit() for token in tokens):
        return False
    readable_alpha = sum(1 for token in tokens if token.isalpha() and len(token) >= 4)
    return readable_alpha >= 1 and (
        any(token.isdigit() for token in tokens)
        or len(tokens) <= 3
        or any(token.isupper() for token in tokens if token.isalpha())
    )


def figure_line_looks_like_micro_band_seed(
    line: ocr_page_analysis.TextGeometryLine,
) -> bool:
    bbox = line.observation.bbox
    if bbox is None:
        return False
    text = line.text.strip()
    if not text:
        return False
    tokens = precision_label_candidate_tokens(text)
    if len(tokens) != 1:
        return False
    token = tokens[0]
    width = max(1.0, bbox[2] - bbox[0])
    height = max(1.0, bbox[3] - bbox[1])
    if width > 96.0 or height > 42.0:
        return False
    if token.isdigit():
        return len(token) == 2
    return token.isalpha() and len(token) >= 4


def figure_micro_band_seed_evidence_is_usable(item: FigureTokenEvidence) -> bool:
    bbox = item.bbox
    width = max(1.0, bbox[2] - bbox[0])
    height = max(1.0, bbox[3] - bbox[1])
    if width > 96.0 or height > 42.0:
        return False
    if item.token.isdigit():
        return len(item.token) == 2
    return item.token.isalpha() and len(item.token) >= 4


def figure_box_lives_inside_band(
    bbox: tuple[float, float, float, float],
    band_box: tuple[float, float, float, float],
) -> bool:
    center_x = (bbox[0] + bbox[2]) * 0.5
    center_y = (bbox[1] + bbox[3]) * 0.5
    pad_x = max(8.0, (band_box[2] - band_box[0]) * 0.04)
    pad_y = max(8.0, (band_box[3] - band_box[1]) * 0.16)
    return (
        band_box[0] - pad_x <= center_x <= band_box[2] + pad_x
        and band_box[1] - pad_y <= center_y <= band_box[3] + pad_y
    )


def figure_label_lines_share_cluster(
    left: ocr_page_analysis.TextGeometryLine,
    right: ocr_page_analysis.TextGeometryLine,
) -> bool:
    left_bbox = left.observation.bbox
    right_bbox = right.observation.bbox
    if left_bbox is None or right_bbox is None:
        return False
    left_height = max(1.0, page_geometry.observation_height(left.observation))
    right_height = max(1.0, page_geometry.observation_height(right.observation))
    left_width = max(1.0, page_geometry.observation_width(left.observation))
    right_width = max(1.0, page_geometry.observation_width(right.observation))
    center_dx = abs(
        page_geometry.observation_mid_x(left.observation)
        - page_geometry.observation_mid_x(right.observation)
    )
    center_dy = abs(
        page_geometry.observation_mid_y(left.observation)
        - page_geometry.observation_mid_y(right.observation)
    )
    if (
        center_dx <= max(left_width, right_width) * 2.8
        and center_dy <= max(left_height, right_height) * 3.4
    ):
        return True
    return (
        page_geometry.observation_geometry_match_score(
            left.observation,
            right.observation,
        )
        >= 0.18
    )


def figure_patent_callout_lines_share_cluster(
    left: ocr_page_analysis.TextGeometryLine,
    right: ocr_page_analysis.TextGeometryLine,
) -> bool:
    if figure_label_lines_share_cluster(left, right):
        return True
    left_bbox = left.observation.bbox
    right_bbox = right.observation.bbox
    if left_bbox is None or right_bbox is None:
        return False
    left_height = max(1.0, page_geometry.observation_height(left.observation))
    right_height = max(1.0, page_geometry.observation_height(right.observation))
    left_width = max(1.0, page_geometry.observation_width(left.observation))
    right_width = max(1.0, page_geometry.observation_width(right.observation))
    center_dx = abs(
        page_geometry.observation_mid_x(left.observation)
        - page_geometry.observation_mid_x(right.observation)
    )
    center_dy = abs(
        page_geometry.observation_mid_y(left.observation)
        - page_geometry.observation_mid_y(right.observation)
    )
    return (
        center_dx <= max(left_width, right_width) * 5.0
        and center_dy <= max(left_height, right_height) * 5.0
    )


def figure_micro_band_lines_share_cluster(
    left: ocr_page_analysis.TextGeometryLine,
    right: ocr_page_analysis.TextGeometryLine,
) -> bool:
    left_bbox = left.observation.bbox
    right_bbox = right.observation.bbox
    if left_bbox is None or right_bbox is None:
        return False
    left_height = max(1.0, page_geometry.observation_height(left.observation))
    right_height = max(1.0, page_geometry.observation_height(right.observation))
    left_width = max(1.0, page_geometry.observation_width(left.observation))
    right_width = max(1.0, page_geometry.observation_width(right.observation))
    center_dx = abs(
        page_geometry.observation_mid_x(left.observation)
        - page_geometry.observation_mid_x(right.observation)
    )
    center_dy = abs(
        page_geometry.observation_mid_y(left.observation)
        - page_geometry.observation_mid_y(right.observation)
    )
    if center_dx > max(left_width, right_width) * 10.5:
        return False
    return center_dy <= max(left_height, right_height) * 5.5


def expand_figure_label_cluster_box(
    box: tuple[float, float, float, float],
    region_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    return (
        max(region_bbox[0], x0 - max(10.0, width * 0.18)),
        max(region_bbox[1], y0 - max(6.0, height * 0.12)),
        min(region_bbox[2], x1 + max(18.0, width * 0.28)),
        min(region_bbox[3], y1 + max(6.0, height * 0.12)),
    )


def expanded_figure_patent_callout_cluster_box(
    lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    line_bbox = page_geometry.observation_union_bbox(line.observation for line in lines)
    if line_bbox is None:
        return None
    region_width = max(1.0, region_bbox[2] - region_bbox[0])
    region_height = max(1.0, region_bbox[3] - region_bbox[1])
    x0, y0, x1, y1 = line_bbox
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    box = (
        max(region_bbox[0], x0 - max(12.0, width * 0.25)),
        max(region_bbox[1], y0 - max(10.0, height * 0.40)),
        min(region_bbox[2], x1 + max(28.0, width * 0.70, region_width * 0.08)),
        min(region_bbox[3], y1 + max(12.0, height * 0.45, region_height * 0.03)),
    )
    if page_geometry.rect_area(box) <= 0.0:
        return None
    return box


def expanded_figure_micro_band_box(
    lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    if len(lines) < 4:
        return None
    alpha_count = 0
    digit_count = 0
    line_bbox = page_geometry.observation_union_bbox(line.observation for line in lines)
    if line_bbox is None:
        return None
    for line in lines:
        tokens = precision_label_candidate_tokens(line.text)
        if len(tokens) != 1:
            continue
        token = tokens[0]
        if token.isdigit():
            digit_count += 1
        elif token.isalpha():
            alpha_count += 1
    if alpha_count < 1 or alpha_count > 2 or digit_count < 3:
        return None
    region_width = max(1.0, region_bbox[2] - region_bbox[0])
    region_height = max(1.0, region_bbox[3] - region_bbox[1])
    x0, y0, x1, y1 = line_bbox
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    if width < region_width * 0.10 or width > region_width * 0.55:
        return None
    if height > region_height * 0.16:
        return None
    if width < height * 2.0:
        return None
    box = (
        max(region_bbox[0], x0 - max(12.0, width * 0.10)),
        max(region_bbox[1], y0 - max(8.0, height * 0.18)),
        min(region_bbox[2], x1 + max(18.0, width * 0.20)),
        min(region_bbox[3], y1 + max(8.0, height * 0.22)),
    )
    return box if page_geometry.rect_area(box) > 0.0 else None


def figure_patent_callout_neighborhood_subregion_boxes(
    label_lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    if not label_lines:
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for line in sorted(label_lines, key=figure_text_line_reading_order_key):
        box = expanded_figure_patent_callout_neighborhood_box(line, region_bbox)
        if box is not None:
            boxes.append(box)
    ranked = sorted(
        figure_unique_subregion_boxes(boxes),
        key=lambda box: (-page_geometry.rect_area(box), box[0], -box[3]),
    )
    return ranked[:OCR_FIGURE_MAX_CALLOUT_NEIGHBORHOOD_SUBREGIONS]


def figure_filter_patent_callout_boxes(
    boxes: list[tuple[float, float, float, float]],
    label_lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
    vector_lines: tuple[Any, ...] = (),
) -> list[tuple[float, float, float, float]]:
    ranked: list[tuple[float, tuple[float, float, float, float]]] = []
    for box in figure_unique_subregion_boxes(boxes):
        support = figure_patent_callout_box_support_score(
            box,
            label_lines,
            region_bbox,
            vector_lines,
        )
        if support is None:
            continue
        ranked.append((support, box))
    ranked.sort(key=lambda item: (-item[0], item[1][0], -item[1][3]))
    limit = (
        OCR_FIGURE_MAX_CALLOUT_CLUSTER_SUBREGIONS + OCR_FIGURE_MAX_CALLOUT_NEIGHBORHOOD_SUBREGIONS
    )
    return [box for _score, box in ranked[:limit]]


def figure_patent_callout_box_support_score(
    box: tuple[float, float, float, float],
    label_lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
    vector_lines: tuple[Any, ...] = (),
) -> float | None:
    local_lines = [line for line in label_lines if figure_line_is_within_callout_box(line, box)]
    if not local_lines:
        return None
    descriptive_lines = sum(
        1 for line in local_lines if figure_callout_candidate_line_is_descriptive(line.text)
    )
    numeric_lines = sum(1 for line in local_lines if any(ch.isdigit() for ch in line.text))
    vector_support = figure_patent_callout_box_vector_support(
        box,
        local_lines,
        vector_lines,
    )
    region_area = max(1.0, page_geometry.rect_area(region_bbox))
    area_ratio = page_geometry.rect_area(box) / region_area
    if len(local_lines) == 1:
        if descriptive_lines == 0 or numeric_lines == 0 or area_ratio > 0.08 or vector_support <= 0:
            return None
    elif len(local_lines) == 2 and (descriptive_lines == 0 or vector_support <= 0):
        return None
    return (
        len(local_lines) * 10.0
        + descriptive_lines * 4.0
        + numeric_lines * 2.0
        + vector_support * 6.0
        - area_ratio * 80.0
    )


def figure_line_is_within_callout_box(
    line: ocr_page_analysis.TextGeometryLine,
    box: tuple[float, float, float, float],
) -> bool:
    bbox = line.observation.bbox
    if bbox is None:
        return False
    overlap = page_geometry.rect_intersection_area(bbox, box)
    if overlap <= 0.0:
        return False
    bbox_area = page_geometry.rect_area(bbox)
    return bbox_area > 0.0 and overlap / bbox_area >= 0.60


def figure_patent_callout_box_vector_support(
    box: tuple[float, float, float, float],
    local_lines: list[ocr_page_analysis.TextGeometryLine],
    vector_lines: tuple[Any, ...],
) -> int:
    if not local_lines or not vector_lines:
        return 0
    supported = 0
    for vector_line in vector_lines:
        segment = page_geometry.line_segment(vector_line)
        if segment is None:
            continue
        if not figure_vector_segment_intersects_box(box, segment):
            continue
        if any(
            figure_vector_segment_is_near_label_line(segment, label_line)
            for label_line in local_lines
        ):
            supported += 1
    return supported


def figure_vector_segment_intersects_box(
    box: tuple[float, float, float, float],
    segment: tuple[float, float, float, float],
) -> bool:
    x0, y0, x1, y1 = box
    sx0, sy0, sx1, sy1 = segment
    seg_box = (
        min(sx0, sx1),
        min(sy0, sy1),
        max(sx0, sx1),
        max(sy0, sy1),
    )
    expanded = (x0 - 8.0, y0 - 8.0, x1 + 8.0, y1 + 8.0)
    return page_geometry.rect_intersection_area(expanded, seg_box) > 0.0


def figure_vector_segment_is_near_label_line(
    segment: tuple[float, float, float, float],
    line: ocr_page_analysis.TextGeometryLine,
) -> bool:
    bbox = line.observation.bbox
    if bbox is None:
        return False
    line_height = max(1.0, page_geometry.observation_height(line.observation))
    threshold = max(10.0, line_height * 2.5)
    for x, y in ((segment[0], segment[1]), (segment[2], segment[3])):
        dx = 0.0 if bbox[0] <= x <= bbox[2] else min(abs(x - bbox[0]), abs(x - bbox[2]))
        dy = 0.0 if bbox[1] <= y <= bbox[3] else min(abs(y - bbox[1]), abs(y - bbox[3]))
        if dx <= threshold and dy <= threshold:
            return True
    return False


def expanded_figure_patent_callout_neighborhood_box(
    line: ocr_page_analysis.TextGeometryLine,
    region_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    bbox = line.observation.bbox
    if bbox is None:
        return None
    region_width = max(1.0, region_bbox[2] - region_bbox[0])
    region_height = max(1.0, region_bbox[3] - region_bbox[1])
    x0, y0, x1, y1 = bbox
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    box = (
        max(region_bbox[0], x0 - max(10.0, width * 0.18)),
        max(region_bbox[1], y0 - max(10.0, height * 0.55)),
        min(region_bbox[2], x1 + max(40.0, width * 0.95, region_width * 0.07)),
        min(region_bbox[3], y1 + max(12.0, height * 0.55, region_height * 0.025)),
    )
    if page_geometry.rect_area(box) <= 0.0:
        return None
    return box


def figure_should_use_patent_callout_clusters(
    label_lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
) -> bool:
    if len(label_lines) < 4:
        return False
    label_bbox = page_geometry.observation_union_bbox(line.observation for line in label_lines)
    if label_bbox is None:
        return False
    region_area = page_geometry.rect_area(region_bbox)
    label_area = page_geometry.rect_area(label_bbox)
    if region_area <= 0.0 or label_area <= 0.0:
        return False
    label_area_ratio = label_area / region_area
    digit_lines = sum(1 for line in label_lines if any(ch.isdigit() for ch in line.text))
    readable_lines = sum(
        1
        for line in label_lines
        if any(token.isalpha() and len(token) >= 4 for token in normalized_text_tokens(line.text))
    )
    region_width = max(1.0, region_bbox[2] - region_bbox[0])
    region_height = max(1.0, region_bbox[3] - region_bbox[1])
    spread_x = max(0.0, label_bbox[2] - label_bbox[0]) / region_width
    spread_y = max(0.0, label_bbox[3] - label_bbox[1]) / region_height
    dispersed_labels = spread_x >= 0.42 or (spread_x >= 0.28 and spread_y >= 0.24)
    return (
        digit_lines >= 3 and readable_lines >= 3 and (label_area_ratio >= 0.12 or dispersed_labels)
    )


def figure_grid_subregion_boxes(
    region_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    normalized_region = page_geometry.normalize_rect(region_bbox)
    if normalized_region is None:
        return []
    x0, y0, x1, y1 = normalized_region
    width = x1 - x0
    height = y1 - y0
    if width <= 0.0 or height <= 0.0:
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for columns in OCR_FIGURE_GRID_COLUMNS:
        boxes.extend(figure_grid_partition_boxes(region_bbox, columns=columns, rows=1))
    for rows in OCR_FIGURE_GRID_ROWS:
        boxes.extend(figure_grid_partition_boxes(region_bbox, columns=1, rows=rows))
    boxes.extend(figure_grid_partition_boxes(region_bbox, columns=4, rows=4))
    unique = figure_unique_subregion_boxes(
        [box for box in boxes if figure_grid_subregion_box_is_large_enough(box)]
    )
    return unique[:OCR_FIGURE_MAX_GRID_SUBREGIONS]


def figure_grid_partition_boxes(
    region_bbox: tuple[float, float, float, float],
    *,
    columns: int,
    rows: int,
) -> list[tuple[float, float, float, float]]:
    if columns <= 0 or rows <= 0:
        return []
    normalized_region = page_geometry.normalize_rect(region_bbox)
    if normalized_region is None:
        return []
    x0, y0, x1, y1 = normalized_region
    width = x1 - x0
    height = y1 - y0
    if width <= 0.0 or height <= 0.0:
        return []
    cell_width = width / columns
    cell_height = height / rows
    x_overlap = cell_width * OCR_FIGURE_GRID_OVERLAP_RATIO
    y_overlap = cell_height * OCR_FIGURE_GRID_OVERLAP_RATIO
    boxes: list[tuple[float, float, float, float]] = []
    for row in range(rows):
        for column in range(columns):
            cell_x0 = x0 + cell_width * column
            cell_x1 = cell_x0 + cell_width
            cell_y0 = y0 + cell_height * row
            cell_y1 = cell_y0 + cell_height
            boxes.append(
                (
                    max(x0, cell_x0 - x_overlap),
                    max(y0, cell_y0 - y_overlap),
                    min(x1, cell_x1 + x_overlap),
                    min(y1, cell_y1 + y_overlap),
                )
            )
    return boxes


def figure_grid_subregion_box_is_large_enough(
    box: tuple[float, float, float, float],
) -> bool:
    x0, y0, x1, y1 = box
    return x1 - x0 >= 12.0 and y1 - y0 >= 8.0


def figure_should_use_fixed_grid_subregions(
    region: ocr_page_analysis.FigureOcrRegion,
) -> bool:
    signals = region.signals or {}
    if not signals.get("high_density_region"):
        return False
    return (
        figure_region_signal_float(signals, "area_ratio") <= 0.18
        and figure_region_signal_float(signals, "pixel_density") >= 100.0
        and figure_region_signal_float(signals, "native_overlap") <= 0.05
    )


def figure_should_try_full_page_subregion_recovery(
    region: ocr_page_analysis.FigureOcrRegion,
    base_candidates: list[OcrCandidate],
) -> bool:
    if not figure_region_is_image_only_full_page(region):
        return False
    if not base_candidates:
        return False
    if not figure_full_page_recovery_density_is_sufficient(region):
        return False
    return figure_fixed_grid_subregion_retry_is_needed(region, base_candidates)


def figure_full_page_recovery_density_is_sufficient(
    region: ocr_page_analysis.FigureOcrRegion,
) -> bool:
    signals = region.signals or {}
    return (
        figure_region_signal_float(
            signals,
            "pixel_density",
        )
        >= ocr_page_analysis.OCR_FIGURE_MIN_PIXEL_DENSITY
    )


def figure_should_include_grid_subregions(
    region: ocr_page_analysis.FigureOcrRegion,
    base_candidates: list[OcrCandidate],
) -> bool:
    del base_candidates
    return figure_should_use_fixed_grid_subregions(region)


def figure_region_signal_float(signals: dict[str, Any], key: str) -> float:
    value = signals.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def figure_vertical_line_groups(
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> list[list[ocr_page_analysis.TextGeometryLine]]:
    sorted_lines = sorted(
        lines,
        key=lambda line: -page_geometry.observation_mid_y(line.observation),
    )
    heights = [page_geometry.observation_height(line.observation) for line in sorted_lines]
    positive_heights = [height for height in heights if height > 0]
    median_height = median(positive_heights) if positive_heights else 4.0
    split_gap = max(12.0, median_height * 4.0)
    groups: list[list[ocr_page_analysis.TextGeometryLine]] = []
    current: list[ocr_page_analysis.TextGeometryLine] = []
    previous_mid_y: float | None = None
    for line in sorted_lines:
        mid_y = page_geometry.observation_mid_y(line.observation)
        if previous_mid_y is not None and previous_mid_y - mid_y > split_gap:
            if current:
                groups.append(current)
            current = [line]
        else:
            current.append(line)
        previous_mid_y = mid_y
    if current:
        groups.append(current)
    return groups


def expanded_figure_subregion_box(
    lines: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    line_bbox = page_geometry.observation_union_bbox(line.observation for line in lines)
    if line_bbox is None:
        return None
    x0, y0, x1, y1 = line_bbox
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    box = (
        max(region_bbox[0], x0 - max(7.0, width * 0.35)),
        max(region_bbox[1], y0 - max(5.0, height * 0.16)),
        min(region_bbox[2], x1 + max(18.0, width * 1.35)),
        min(region_bbox[3], y1 + max(5.0, height * 0.16)),
    )
    if page_geometry.rect_area(box) <= 0.0:
        return None
    return box


def figure_unique_subregion_boxes(
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    unique: list[tuple[float, float, float, float]] = []
    for box in boxes:
        if any(figure_subregion_boxes_are_equivalent(box, existing) for existing in unique):
            continue
        unique.append(box)
    return unique


def figure_subregion_boxes_are_equivalent(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    left_area = page_geometry.rect_area(left)
    right_area = page_geometry.rect_area(right)
    if min(left_area, right_area) <= 0.0:
        return False
    overlap = page_geometry.rect_intersection_area(left, right)
    area_ratio = min(left_area, right_area) / max(left_area, right_area)
    return area_ratio >= 0.60 and overlap / min(left_area, right_area) >= 0.90


def figure_crop_page_bbox_from_image(
    image: OcrImage,
    page_bbox: tuple[float, float, float, float],
    *,
    cache: dict[Any, Any] | None = None,
    cache_key: Any | None = None,
) -> OcrImage | None:
    crop_cache_key = (
        ("figure_crop_page_bbox_from_image", cache_key, page_bbox)
        if cache is not None and cache_key is not None
        else None
    )
    if cache is not None and crop_cache_key is not None:
        cached = cache.get(crop_cache_key)
        if cached is None and crop_cache_key in cache:
            return None
        if isinstance(cached, OcrImage):
            return cached
    region = figure_page_bbox_image_region(
        image,
        page_bbox,
        cache=cache,
        cache_key=cache_key,
    )
    if region is None:
        if cache is not None and crop_cache_key is not None:
            cache[crop_cache_key] = None
        return None
    crop = ocr_execution.crop_ocr_image_region(
        image,
        (region.x0, region.y0, region.x1, region.y1),
    )
    if crop is None:
        if cache is not None and crop_cache_key is not None:
            cache[crop_cache_key] = None
        return None
    cropped = replace(
        crop,
        source="figure_subregion",
        page_bbox=page_geometry.normalize_rect(page_bbox),
    )
    if cache is not None and crop_cache_key is not None:
        cache[crop_cache_key] = cropped
    return cropped


def figure_page_bbox_image_region(
    image: OcrImage,
    page_bbox: tuple[float, float, float, float],
    *,
    cache: dict[Any, Any] | None = None,
    cache_key: Any | None = None,
) -> NativeOcrRegion | None:
    region_cache_key = (
        ("figure_page_bbox_image_region", cache_key, page_bbox)
        if cache is not None and cache_key is not None
        else None
    )
    if cache is not None and region_cache_key is not None:
        cached = cache.get(region_cache_key)
        if cached is None and region_cache_key in cache:
            return None
        if isinstance(cached, NativeOcrRegion):
            return cached
    geometry = page_geometry.ImageSpace.from_dimensions(
        image_width=image.width,
        image_height=image.height,
        image_resolution=image.resolution,
        page_bbox=image.page_bbox,
        clockwise_quarter_turns=image.page_clockwise_quarter_turns,
        source=image.source,
    )
    rectangle = page_geometry.page_bbox_to_image_pixel_bbox(
        page_bbox,
        geometry,
        clamp=False,
    )
    if rectangle is None:
        return None
    x0, y0, x1, y1 = rectangle
    x0 = max(0, min(image.width, x0))
    y0 = max(0, min(image.height, y0))
    x1 = max(x0, min(image.width, x1))
    y1 = max(y0, min(image.height, y1))
    if x1 <= x0 or y1 <= y0:
        if cache is not None and region_cache_key is not None:
            cache[region_cache_key] = None
        return None
    region = NativeOcrRegion(x0, y0, x1, y1)
    if cache is not None and region_cache_key is not None:
        cache[region_cache_key] = region
    return region


def figure_virtual_subregion_image(
    image: OcrImage,
    region: NativeOcrRegion,
    *,
    source: str,
    page_bbox: tuple[float, float, float, float],
) -> OcrImage:
    return OcrImage(
        data=b"",
        width=region.width,
        height=region.height,
        bytes_per_pixel=image.bytes_per_pixel,
        bytes_per_line=region.width * image.bytes_per_pixel,
        source=source,
        resolution=image.resolution,
        page_bbox=page_geometry.normalize_rect(page_bbox),
    )


def scaled_figure_subregion_image(image: OcrImage, *, source: str) -> OcrImage | None:
    if image.width < 32 or image.height < 32:
        return None
    target_width = image.width * OCR_FIGURE_SUBREGION_SCALE
    target_height = image.height * OCR_FIGURE_SUBREGION_SCALE
    target_pixels = target_width * target_height
    if target_pixels > OCR_FIGURE_SUBREGION_MAX_TARGET_PIXELS:
        return None
    if not leptonica_pix_size_is_supported(target_width, target_height):
        return None
    return replace(
        image,
        source=source,
        target_width=target_width,
        target_height=target_height,
    )


def figure_subregion_text_results(
    image: OcrImage,
    page_bbox: tuple[float, float, float, float],
    *,
    source: str,
    psms: list[int],
    variables_by_profile: tuple[tuple[str, dict[str, str]], ...],
    timeout: float | None,
    ocr_session: ocr_session_runtime.OcrPageSession | None,
    cache: dict[Any, Any] | None = None,
) -> tuple[OcrImage | None, dict[str, list[OcrTextResult]] | None]:
    region = figure_page_bbox_image_region(
        image,
        page_bbox,
        cache=cache,
        cache_key=id(image),
    )
    if region is None:
        return None, None
    crop = figure_virtual_subregion_image(
        image,
        region,
        source="figure_subregion",
        page_bbox=page_bbox,
    )
    scaled_crop = scaled_figure_subregion_image(crop, source=source)
    if scaled_crop is None:
        return None, None
    results_by_profile: dict[str, list[OcrTextResult]] = {}
    if ocr_session is not None:
        for profile_name, variables in variables_by_profile:
            ocr_results = ocr_session.image_subregion_to_text_results(
                image,
                (region.x0, region.y0, region.x1, region.y1),
                scaled_crop,
                psms=psms,
                variables=variables,
            )
            if ocr_results is None:
                results_by_profile.clear()
                break
            results_by_profile[profile_name] = ocr_results
        if results_by_profile:
            return scaled_crop, results_by_profile
    fallback_crop = figure_crop_page_bbox_from_image(image, page_bbox)
    if fallback_crop is None:
        return None, None
    fallback_scaled_crop = scaled_figure_subregion_image(fallback_crop, source=source)
    if fallback_scaled_crop is None:
        return None, None
    for profile_name, variables in variables_by_profile:
        results_by_profile[profile_name] = (
            ocr_session.image_to_text_results(
                fallback_scaled_crop,
                psms=psms,
                variables=variables,
            )
            if ocr_session is not None
            else ocr_execution.ocr_image_to_text_results_with_psms_timeout(
                fallback_scaled_crop,
                psms=psms,
                variables=variables,
                timeout=timeout,
            )
        )
    return fallback_scaled_crop, results_by_profile


def figure_pixel_subregion_text_results(
    image: OcrImage,
    region: NativeOcrRegion,
    *,
    source: str,
    psms: list[int],
    variables_by_profile: tuple[tuple[str, dict[str, str]], ...],
    timeout: float | None,
    ocr_session: ocr_session_runtime.OcrPageSession | None,
) -> tuple[OcrImage | None, dict[str, list[OcrTextResult]] | None]:
    page_bbox = ocr_image_pixel_region_page_bbox(image, region)
    if page_bbox is None:
        return None, None
    crop = figure_virtual_subregion_image(
        image,
        region,
        source=source,
        page_bbox=page_bbox,
    )
    scaled_crop = scaled_figure_subregion_image(
        crop,
        source=f"{source}_scale{OCR_FIGURE_SUBREGION_SCALE}",
    )
    if scaled_crop is None:
        return None, None
    results_by_profile: dict[str, list[OcrTextResult]] = {}
    if ocr_session is not None:
        for profile_name, variables in variables_by_profile:
            ocr_results = ocr_session.image_subregion_to_text_results(
                image,
                (region.x0, region.y0, region.x1, region.y1),
                scaled_crop,
                psms=psms,
                variables=variables,
            )
            if ocr_results is None:
                results_by_profile.clear()
                break
            results_by_profile[profile_name] = ocr_results
        if results_by_profile:
            return scaled_crop, results_by_profile
    fallback_crop = crop_ocr_image_pixel_region(image, region, source=source)
    if fallback_crop is None:
        return None, None
    fallback_scaled_crop = scaled_figure_subregion_image(
        fallback_crop,
        source=f"{source}_scale{OCR_FIGURE_SUBREGION_SCALE}",
    )
    if fallback_scaled_crop is None:
        return None, None
    for profile_name, variables in variables_by_profile:
        results_by_profile[profile_name] = (
            ocr_session.image_to_text_results(
                fallback_scaled_crop,
                psms=psms,
                variables=variables,
            )
            if ocr_session is not None
            else ocr_execution.ocr_image_to_text_results_with_psms_timeout(
                fallback_scaled_crop,
                psms=psms,
                variables=variables,
                timeout=timeout,
            )
        )
    return fallback_scaled_crop, results_by_profile


def figure_subregion_candidate_is_useful(candidate: OcrCandidate) -> bool:
    text = candidate.result.text.strip()
    if not text:
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    useful_lines = sum(1 for line in lines if figure_text_line_is_useful(line))
    if len(lines) == 1:
        return useful_lines == 1 and figure_subregion_single_line_candidate_is_useful(
            candidate,
            lines[0],
        )
    if len(lines) < 2:
        return False
    if useful_lines < 2:
        return False
    return figure_ocr_candidate_score(candidate) >= 12.0


def figure_fixed_grid_subregion_retry_is_needed(
    region: ocr_page_analysis.FigureOcrRegion,
    base_candidates: list[OcrCandidate],
) -> bool:
    base_candidate = fused_figure_ocr_candidate(region, base_candidates)
    if base_candidate is None:
        return True
    text = base_candidate.result.text.strip()
    if not text:
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    useful_lines = sum(1 for line in lines if figure_text_line_is_useful(line))
    token_count = extracted_text_token_count(text)
    confidence = base_candidate.result.confidence or 0
    quality = text_ocr_quality_score(text)
    score = figure_ocr_candidate_score(base_candidate)
    return not (
        useful_lines >= 8
        and token_count >= 28
        and confidence >= 75
        and quality <= 0.32
        and score >= 180.0
    )


def figure_subregion_single_line_candidate_is_useful(
    candidate: OcrCandidate,
    line: str,
) -> bool:
    is_callout_cluster = "callout_cluster_" in candidate.name
    if "subregion_" not in candidate.name and not is_callout_cluster:
        return False
    tokens = normalized_text_tokens(line)
    if not tokens:
        return False
    if figure_reference_precision_line_is_metadata(line):
        return False
    if all(token.isdigit() for token in tokens):
        return False
    readable_alpha = sum(1 for token in tokens if token.isalpha() and len(token) >= 4)
    if readable_alpha == 0:
        return False
    if is_callout_cluster and readable_alpha < 2:
        return False
    if len(tokens) > 5:
        return False
    if figure_text_noise_ratio(line) > 0.22:
        return False
    return figure_ocr_candidate_score(candidate) >= 6.0


def figure_micro_band_candidate_is_useful(candidate: OcrCandidate) -> bool:
    text = candidate.result.text.strip()
    if not text:
        return False
    tokens = precision_label_candidate_tokens(text)
    alpha_tokens = [token for token in tokens if token.isalpha() and len(token) >= 4]
    digit_tokens = [token for token in tokens if token.isdigit() and len(token) == 2]
    if len(alpha_tokens) < 1 or len(digit_tokens) < 2:
        return False
    if len(tokens) > 16:
        return False
    if text_ocr_quality_score(text) > 0.55:
        return False
    return figure_ocr_candidate_score(candidate) >= 6.0


def figure_micro_fragment_candidate_is_useful(candidate: OcrCandidate) -> bool:
    text = candidate.result.text.strip()
    fragment_tokens = {
        item.token
        for item in figure_candidate_token_evidence(
            candidate,
            rows="word_rows",
            token_extractor=dominant_image_alpha_fragment_token,
        )
        if 2 <= len(item.token) <= 5
    }
    if not text and not fragment_tokens:
        return False
    completion = dominant_image_micro_fragment_completion(text) if text else None
    if completion is None and not fragment_tokens:
        return False
    return text_ocr_quality_score(text) <= 0.75 if text else True


def figure_band_slot_candidate_is_useful(candidate: OcrCandidate) -> bool:
    text = candidate.result.text.strip()
    fragment_tokens = {
        item.token
        for item in figure_candidate_token_evidence(
            candidate,
            rows="word_rows",
            token_extractor=dominant_image_alpha_fragment_token,
        )
        if 2 <= len(item.token) <= 6
    }
    if fragment_tokens:
        return True
    if not text:
        return False
    tokens = precision_label_candidate_tokens(text)
    if not tokens:
        return False
    alpha_tokens = [token for token in tokens if token.isalpha()]
    return bool(alpha_tokens) and text_ocr_quality_score(text) <= 0.78


def figure_band_slot_subwindow_boxes(
    slot: FigureBandSlot,
) -> tuple[tuple[float, float, float, float], ...]:
    x0, y0, x1, y1 = slot.gap_bbox
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    anchor_center_y = (slot.anchor_bbox[1] + slot.anchor_bbox[3]) * 0.5
    numeric_center_y = (slot.numeric_bbox[1] + slot.numeric_bbox[3]) * 0.5
    lane_mid_y = (anchor_center_y + numeric_center_y) * 0.5
    upper_bottom = min(y1, max(y0 + height * 0.45, lane_mid_y + 3.0))
    lower_top = max(y0, min(y1 - height * 0.45, lane_mid_y - 3.0))
    windows = (
        (
            x0,
            y0,
            x0 + width * 0.58,
            upper_bottom,
        ),
        (
            x0 + width * 0.12,
            lower_top,
            x0 + width * 0.62,
            y1,
        ),
        (
            x0 + width * 0.18,
            y0 + height * 0.18,
            x0 + width * 0.54,
            y0 + height * 0.78,
        ),
        (
            x0 + width * 0.42,
            y0 + height * 0.10,
            x0 + width * 0.88,
            y0 + height * 0.62,
        ),
    )
    normalized: list[tuple[float, float, float, float]] = []
    for box in windows:
        normalized_box = page_geometry.normalize_rect(box)
        if normalized_box is None or page_geometry.rect_area(normalized_box) <= 0.0:
            continue
        normalized.append(normalized_box)
    return tuple(normalized)


def figure_should_try_band_slot_ocr(
    micro_band_candidates: list[OcrCandidate],
) -> bool:
    if not micro_band_candidates:
        return False
    anchor_tokens: set[str] = set()
    short_fragment_counts: Counter[str] = Counter()
    numeric_tokens: set[str] = set()
    for candidate in micro_band_candidates:
        candidate_short_fragments: set[str] = set()
        for item in figure_candidate_token_evidence(candidate):
            token = item.token
            if token.isdigit() and len(token) == 2:
                numeric_tokens.add(token)
                continue
            if token.isalpha() and len(token) >= 4:
                anchor_tokens.add(token)
                continue
            if token.isalpha() and 2 <= len(token) <= 3:
                candidate_short_fragments.add(token)
        for item in figure_candidate_token_evidence(
            candidate,
            rows="word_rows",
            token_extractor=dominant_image_alpha_fragment_token,
        ):
            token = item.token
            if token in anchor_tokens:
                continue
            if 2 <= len(token) <= 3:
                candidate_short_fragments.add(token)
        for raw_text in candidate.result.text.splitlines():
            raw_token = dominant_image_alpha_fragment_token(raw_text.strip())
            if raw_token is not None and 2 <= len(raw_token) <= 3:
                candidate_short_fragments.add(raw_token)
        for token in precision_label_candidate_tokens(candidate.result.text):
            if token.isdigit() and len(token) == 2:
                numeric_tokens.add(token)
                continue
            if token.isalpha() and len(token) >= 4:
                anchor_tokens.add(token)
                continue
            if token.isalpha() and 2 <= len(token) <= 3:
                candidate_short_fragments.add(token)
        for token in candidate_short_fragments:
            short_fragment_counts[token] += 1
    repeated_short_fragments = {
        token for token, count in short_fragment_counts.items() if count >= 2
    }
    return bool(anchor_tokens) and bool(numeric_tokens) and bool(repeated_short_fragments)


def figure_should_try_band_slot_subwindows(
    band_slot_candidates: list[OcrCandidate],
) -> bool:
    if not band_slot_candidates:
        return False
    for candidate in band_slot_candidates:
        for item in figure_candidate_token_evidence(
            candidate,
            rows="word_rows",
            token_extractor=dominant_image_alpha_fragment_token,
        ):
            if len(item.token) >= 3:
                return False
        text = candidate.result.text.strip()
        if text and any(
            token.isalpha() and len(token) >= 3 for token in precision_label_candidate_tokens(text)
        ):
            return False
    return True


def dominant_image_figure_band_slot_plan(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
    alpha_clusters: dict[
        str,
        tuple[tuple[int, tuple[float, float, float, float], str, float | None], ...],
    ]
    | None = None,
    numeric_clusters: dict[
        str,
        tuple[tuple[int, tuple[float, float, float, float], str, float | None], ...],
    ]
    | None = None,
) -> FigureBandSlotPlan:
    alpha_groups = (
        dominant_image_figure_micro_band_alpha_word_clusters(
            candidates,
            label_region_bbox=label_region_bbox,
        )
        if alpha_clusters is None
        else alpha_clusters
    )
    numeric_groups = (
        dominant_image_numeric_label_cluster_groups(
            candidates,
            label_region_bbox=label_region_bbox,
            pad_ratio_x=0.16,
            pad_ratio_y=0.16,
        )
        if numeric_clusters is None
        else numeric_clusters
    )
    slots: list[FigureBandSlot] = []
    for alpha_token, alpha_cluster_group in alpha_groups.items():
        if not alpha_cluster_group:
            continue
        alpha_cluster = alpha_cluster_group[0]
        anchor_bbox = alpha_cluster[1]
        right_numeric = [
            (token, cluster_group[0])
            for token, cluster_group in numeric_groups.items()
            if cluster_group
            and dominant_image_band_slot_numeric_supports_anchor(
                anchor_bbox,
                cluster_group[0][1],
            )
        ]
        if not right_numeric:
            continue
        right_numeric = sorted(right_numeric, key=lambda item: item[1][1][0])
        first_numeric_bbox = right_numeric[0][1][1]
        numeric_bbox = union_page_bboxes(cluster[1] for _token, cluster in right_numeric)
        if numeric_bbox is None:
            continue
        corridor_top = min(anchor_bbox[1], first_numeric_bbox[1]) - 4.0
        corridor_bottom = max(anchor_bbox[3], first_numeric_bbox[3]) + 4.0
        gap_bbox = (
            max(anchor_bbox[0], anchor_bbox[2] - 2.0),
            corridor_top,
            first_numeric_bbox[0] + 8.0,
            corridor_bottom,
        )
        normalized_gap_bbox = page_geometry.normalize_rect(gap_bbox)
        if normalized_gap_bbox is None or page_geometry.rect_area(normalized_gap_bbox) <= 0.0:
            continue
        score = alpha_cluster[0] * 2.0 + len(right_numeric) * 1.5
        slots.append(
            FigureBandSlot(
                anchor_token=alpha_token,
                anchor_bbox=anchor_bbox,
                gap_bbox=normalized_gap_bbox,
                numeric_tokens=tuple(token for token, _cluster in right_numeric),
                numeric_bbox=numeric_bbox,
                score=score,
            )
        )
    return FigureBandSlotPlan(
        alpha_clusters=alpha_groups,
        numeric_clusters=numeric_groups,
        slots=tuple(
            sorted(
                slots,
                key=lambda slot: (-slot.score, slot.gap_bbox[1], slot.gap_bbox[0]),
            )
        ),
    )


def collect_figure_region_geometry_evidence(
    page: PageExtractionHost,
    region: ocr_page_analysis.FigureOcrRegion,
    broad_candidate: OcrCandidate,
    timeout: float | None,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
    roi_plan: FigureRegionGeometryPlan | None = None,
) -> FigureRegionGeometryEvidence:
    if not str(broad_candidate.name).startswith("rendered_page_"):
        return FigureRegionGeometryEvidence()
    resolved_plan = (
        roi_plan
        if roi_plan is not None
        else figure_region_geometry_plan(
            broad_candidate,
            region_bbox=region.bbox,
        )
    )
    micro_band_candidates = tuple(
        figure_rendered_micro_band_ocr_candidates(
            page,
            region,
            broad_candidate,
            timeout,
            ocr_session=ocr_session,
            roi_plan=resolved_plan,
        )
    )
    band_slot_candidates: tuple[OcrCandidate, ...] = ()
    band_slot_window_candidates: tuple[OcrCandidate, ...] = ()
    band_slots: tuple[FigureBandSlot, ...] = ()
    if figure_should_try_band_slot_ocr(list(micro_band_candidates)):
        slot_plan = dominant_image_figure_band_slot_plan(
            micro_band_candidates,
            label_region_bbox=region.bbox,
        )
        band_slots = slot_plan.slots
        band_slot_candidates = tuple(
            figure_rendered_band_slot_ocr_candidates(
                page,
                region,
                list(micro_band_candidates),
                timeout,
                ocr_session=ocr_session,
                slots=band_slots,
            )
        )
        if figure_should_try_band_slot_subwindows(list(band_slot_candidates)):
            band_slot_window_candidates = tuple(
                figure_rendered_band_slot_subwindow_ocr_candidates(
                    page,
                    region,
                    list(micro_band_candidates),
                    timeout,
                    ocr_session=ocr_session,
                    slots=band_slots,
                )
            )
    micro_fragment_candidates = tuple(
        figure_rendered_micro_fragment_ocr_candidates(
            page,
            region,
            broad_candidate,
            timeout,
            ocr_session=ocr_session,
            roi_plan=resolved_plan,
        )
    )
    return FigureRegionGeometryEvidence(
        micro_band_candidates=micro_band_candidates,
        band_slots=band_slots,
        band_slot_candidates=band_slot_candidates,
        band_slot_window_candidates=band_slot_window_candidates,
        micro_fragment_candidates=micro_fragment_candidates,
    )


def figure_candidate_with_layout_text(
    candidate: OcrCandidate,
    region_bbox: tuple[float, float, float, float],
) -> OcrCandidate:
    lines = figure_text_geometry_lines_from_word_rows(
        candidate.result.word_rows,
        region_bbox,
    )
    if not lines:
        text_result = OcrTextResult(
            "",
            candidate.result.confidence,
            word_rows=candidate.result.word_rows,
            symbol_rows=candidate.result.symbol_rows,
            component_boxes=candidate.result.component_boxes,
            observations=candidate.result.observations,
        )
        return replace(candidate, result=text_result, region_count=1)
    line_rows = tuple(figure_text_geometry_line_row(line) for line in lines)
    text = "\n".join(line.text for line in lines if line.text.strip())
    confidence = figure_text_lines_confidence(lines, candidate.result.confidence)
    text_result = OcrTextResult(
        text,
        confidence,
        line_rows=line_rows,
        word_rows=candidate.result.word_rows,
        symbol_rows=candidate.result.symbol_rows,
        component_boxes=candidate.result.component_boxes,
        observations=candidate.result.observations,
    )
    return replace(candidate, result=text_result, region_count=1)


def figure_geometry_band_candidate(
    candidate: OcrCandidate,
    region: ocr_page_analysis.FigureOcrRegion,
) -> OcrCandidate | None:
    image_width = candidate.image_width
    image_height = candidate.image_height
    page_bbox = candidate.page_bbox
    if image_width is None or image_height is None or page_bbox is None:
        return None
    regions: list[NativeOcrRegion] = []
    for rows, level in (
        (candidate.result.line_rows, TESSERACT_RIL_TEXTLINE),
        (candidate.result.word_rows, TESSERACT_RIL_WORD),
    ):
        if len(rows) < OCR_MULTI_COLUMN_BAND_MIN_TEXTLINES:
            continue
        boxes = ocr_component_boxes_from_rows(rows, level=level)
        rects = two_column_text_rects_from_boxes(boxes, image_width, image_height)
        if len(rects) < OCR_MULTI_COLUMN_BAND_MIN_LINES * 2:
            continue
        regions = figure_geometry_band_regions_from_rects(
            rects,
            image_width,
            image_height,
        )
        if len(regions) >= 2:
            break
    if len(regions) < 2:
        return None
    geometry = page_geometry.ImageSpace.from_dimensions(
        image_width=image_width,
        image_height=image_height,
        image_resolution=candidate.image_resolution,
        page_bbox=page_bbox,
        source=candidate.name,
    )
    selected_lines: list[ocr_page_analysis.TextGeometryLine] = []
    selected_word_rows: list[dict[str, Any]] = []
    seen_rows: set[tuple[int, int, int, int, str]] = set()
    for band_region in regions:
        region_words = [
            row
            for row in candidate.result.word_rows
            if figure_word_row_center_in_region(row, band_region)
        ]
        if len(region_words) < 3:
            continue
        for row in region_words:
            try:
                left = ocr_int_value(row["left"])
                top = ocr_int_value(row["top"])
                width = ocr_int_value(row["width"])
                height = ocr_int_value(row["height"])
            except (KeyError, TypeError, ValueError):
                continue
            key = (left, top, width, height, str(row.get("text", "")))
            if key in seen_rows:
                continue
            seen_rows.add(key)
            selected_word_rows.append(row)
        region_page_bbox = page_geometry.image_bbox_to_page_bbox(
            (band_region.x0, band_region.y0, band_region.x1, band_region.y1),
            geometry,
        )
        if region_page_bbox is None:
            region_page_bbox = region.bbox
        selected_lines.extend(
            figure_text_geometry_lines_from_word_rows(region_words, region_page_bbox)
        )
    if len(selected_lines) < 4 or len(selected_word_rows) < 12:
        return None
    ordered_lines = order_figure_text_segments(selected_lines, region.bbox)
    text = "\n".join(line.text for line in ordered_lines if line.text.strip())
    if not text.strip():
        return None
    confidence = figure_text_lines_confidence(ordered_lines, candidate.result.confidence)
    line_rows = tuple(figure_text_geometry_line_row(line) for line in ordered_lines)
    return replace(
        candidate,
        name=f"figure_region_{region.item_index}_geometry_bands",
        result=OcrTextResult(
            text,
            confidence,
            line_rows=line_rows,
            word_rows=tuple(selected_word_rows),
            symbol_rows=(),
        ),
        region_count=1,
    )


def figure_geometry_candidate_is_material_improvement(
    candidate: OcrCandidate,
    baseline: OcrCandidate,
) -> bool:
    candidate_score = figure_ocr_candidate_score(candidate)
    baseline_score = figure_ocr_candidate_score(baseline)
    if candidate_score >= baseline_score + 24.0:
        return True
    candidate_gibberish = ocr_text_analysis.alphabetic_gibberish_score(candidate.result.text)
    baseline_gibberish = ocr_text_analysis.alphabetic_gibberish_score(baseline.result.text)
    if candidate_gibberish > baseline_gibberish - 0.03:
        return False
    candidate_lines = [line.strip() for line in candidate.result.text.splitlines() if line.strip()]
    baseline_lines = [line.strip() for line in baseline.result.text.splitlines() if line.strip()]
    candidate_useful = sum(1 for line in candidate_lines if figure_text_line_is_useful(line))
    baseline_useful = sum(1 for line in baseline_lines if figure_text_line_is_useful(line))
    return candidate_score >= baseline_score * 0.96 and candidate_useful >= baseline_useful


def figure_word_row_center_in_region(
    row: dict[str, Any],
    region: NativeOcrRegion,
) -> bool:
    try:
        left = int(row["left"])
        top = int(row["top"])
        width = int(row["width"])
        height = int(row["height"])
    except (KeyError, TypeError, ValueError):
        return False
    if width <= 0 or height <= 0:
        return False
    center_x = left + width * 0.5
    center_y = top + height * 0.5
    return region.x0 <= center_x < region.x1 and region.y0 <= center_y < region.y1


def figure_geometry_band_regions_from_rects(
    rects: list[tuple[int, int, int, int]],
    image_width: int,
    image_height: int,
) -> list[NativeOcrRegion]:
    if image_width < 900 or image_height < 900 or len(rects) < 24:
        return []
    groups = vertical_rect_groups(rects, image_height)
    regions: list[NativeOcrRegion] = []
    for group in groups:
        if len(group) < OCR_MULTI_COLUMN_BAND_MIN_LINES:
            continue
        band_bbox = rect_union(group)
        if band_bbox is None:
            continue
        band_width = band_bbox[2] - band_bbox[0]
        band_height = band_bbox[3] - band_bbox[1]
        if band_width < image_width * 0.45:
            continue
        if band_height < max(72.0, image_height * 0.03):
            continue
        translated = [
            (x0 - band_bbox[0], y0 - band_bbox[1], x1 - band_bbox[0], y1 - band_bbox[1])
            for x0, y0, x1, y1 in group
        ]
        if not rects_support_two_column_gutter(translated, band_width):
            continue
        split = two_column_split_from_rects(translated, band_width)
        if split is None:
            continue
        band_regions = band_ocr_regions_from_split(
            band_bbox,
            split=band_bbox[0] + split,
            image_width=image_width,
            image_height=image_height,
        )
        if len(band_regions) != 2:
            continue
        regions.extend(band_regions)
        if len(regions) >= 4:
            break
    return regions[:4]


def figure_text_geometry_lines_from_word_rows(
    rows: Iterable[dict[str, Any]],
    region_bbox: tuple[float, float, float, float],
) -> list[ocr_page_analysis.TextGeometryLine]:
    words = [
        word
        for row_index, row in enumerate(rows)
        if (word := ocr_layout.ocr_layout_word(row, row_index=row_index)) is not None
        and figure_ocr_word_is_useful(word)
    ]
    if not words:
        return []
    geometry_lines = ocr_layout.ocr_words_to_lines(words)
    segments: list[ocr_page_analysis.TextGeometryLine] = []
    for line in geometry_lines:
        for segment in split_figure_word_line(line):
            text = ocr_layout.render_ocr_word_line(segment)
            if not figure_text_line_is_useful(text, figure_line_confidence(segment)):
                continue
            segments.append(
                ocr_page_analysis.text_geometry_line_from_bbox(
                    text,
                    figure_word_segment_bbox(segment),
                    figure_line_confidence(segment),
                    source="figure_word_rows",
                    kind="figure_text_line",
                )
            )
    return order_figure_text_segments(segments, region_bbox)


def figure_ocr_word_is_useful(word: ocr_layout.OcrLayoutWord) -> bool:
    text = word.text.strip()
    if not text or not any(ch.isalnum() for ch in text):
        return False
    confidence = word.confidence if word.confidence is not None else 50
    if confidence < 15 and not any(ch.isdigit() for ch in text):
        return False
    if len(text) == 1 and text.islower() and confidence < 65:
        return False
    return figure_text_noise_ratio(text) <= 0.50


def split_figure_word_line(
    words: list[ocr_layout.OcrLayoutWord],
) -> list[list[ocr_layout.OcrLayoutWord]]:
    visible = [
        word
        for word in sorted(words, key=lambda item: (item.x0, item.word_num, item.row_index))
        if figure_ocr_word_is_useful(word)
    ]
    if len(visible) <= 1:
        return [visible] if visible else []
    heights = [word.height for word in visible if word.height > 0]
    gap_threshold = max(8.0, (median(heights) if heights else 8.0) * 2.5)
    segments: list[list[ocr_layout.OcrLayoutWord]] = []
    current: list[ocr_layout.OcrLayoutWord] = [visible[0]]
    for previous, word in zip(visible, visible[1:], strict=False):
        gap = word.x0 - previous.x1
        if gap >= gap_threshold:
            segments.append(current)
            current = [word]
        else:
            current.append(word)
    segments.append(current)
    return segments


def figure_text_line_is_useful(text: str, confidence: int | None = None) -> bool:
    stripped = text.strip()
    if not stripped or not any(ch.isalnum() for ch in stripped):
        return False
    if figure_text_noise_ratio(stripped) > 0.50:
        return False
    confidence_value = confidence if confidence is not None else 50
    tokens = normalized_text_tokens(stripped)
    if not tokens:
        return False
    has_digit = any(ch.isdigit() for ch in stripped)
    has_assignment = "=" in stripped or ":" in stripped
    has_connector = "_" in stripped or "+" in stripped
    has_upper = any(ch.isupper() for ch in stripped)
    has_descriptive_token = any(len(token) >= 4 for token in tokens)
    if not (has_digit or has_connector) and figure_text_line_looks_repeated_artifact(tokens):
        return False
    if figure_text_line_looks_alpha_noise(stripped, tokens):
        return False
    if len(tokens) <= 2:
        return confidence_value >= 35 and (
            has_digit or has_assignment or has_connector or has_upper or len(stripped) >= 4
        )
    if not (has_digit or has_assignment or has_connector or has_descriptive_token):
        return False
    return not (confidence_value < 20 and not (has_digit or has_assignment))


def figure_text_line_looks_alpha_noise(text: str, tokens: list[str]) -> bool:
    if not tokens:
        return True
    has_digit_or_connector = any(ch.isdigit() or ch in "+_/=" for ch in text)
    if len(tokens) == 1:
        token = tokens[0]
        if token.isdigit() and len(token) == 1:
            return True
        if token.isalpha() and len(token) == 1:
            return True
        raw = "".join(ch for ch in text if ch.isalpha())
        return bool(token.isalpha() and len(token) < 4 and not raw.isupper())
    if has_digit_or_connector:
        return False
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if not alpha_chars:
        return False
    uppercase_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
    return uppercase_ratio >= 0.65


def figure_text_line_looks_repeated_artifact(tokens: list[str]) -> bool:
    if not tokens:
        return True
    if len(tokens) >= 2 and all(len(token) <= 2 for token in tokens):
        return True
    compact = "".join(tokens)
    if len(compact) < 4:
        return False
    counts = Counter(compact)
    return max(counts.values()) / len(compact) >= 0.68


def figure_text_noise_ratio(text: str) -> float:
    nonspace = [ch for ch in text if not ch.isspace()]
    if not nonspace:
        return 1.0
    allowed_punctuation = frozenset("+-._/=():,[]{}<>")
    noisy = 0
    for ch in nonspace:
        if ch.isalnum() or ch in allowed_punctuation:
            continue
        if ch in OCR_ARTIFACT_CHARS:
            noisy += 1
            continue
        noisy += 1
    return noisy / len(nonspace)


def figure_word_segment_bbox(
    words: list[ocr_layout.OcrLayoutWord],
) -> tuple[float, float, float, float]:
    return (
        min(word.x0 for word in words),
        min(word.y0 for word in words),
        max(word.x1 for word in words),
        max(word.y1 for word in words),
    )


def figure_line_confidence(words: list[ocr_layout.OcrLayoutWord]) -> int | None:
    confidences = [word.confidence for word in words if word.confidence is not None]
    if not confidences:
        return None
    return int(round(sum(confidences) / len(confidences)))


def figure_text_lines_confidence(
    lines: list[ocr_page_analysis.TextGeometryLine],
    fallback: int | None,
) -> int | None:
    confidences = [line.confidence for line in lines if line.confidence is not None]
    if not confidences:
        return fallback
    return int(round(sum(confidences) / len(confidences)))


def figure_text_geometry_line_row(
    line: ocr_page_analysis.TextGeometryLine,
) -> dict[str, Any]:
    if line.observation.bbox is None:
        raise ValueError("Figure text geometry lines require a page bbox")
    x0, y0, x1, y1 = line.observation.bbox
    return {
        "text": line.text,
        "conf": line.confidence,
        "left": int(round(x0)),
        "top": int(round(y0)),
        "width": max(1, int(round(x1 - x0))),
        "height": max(1, int(round(y1 - y0))),
        "page_bbox": line.observation.bbox,
    }


def order_figure_text_segments(
    segments: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
) -> list[ocr_page_analysis.TextGeometryLine]:
    if len(segments) <= 2:
        return sorted(segments, key=figure_text_line_reading_order_key)
    x0, y0, x1, y1 = region_bbox
    region_height = max(1.0, y1 - y0)
    legend_cutoff = y0 + region_height * 0.16
    main = [
        line
        for line in segments
        if page_geometry.observation_mid_y(line.observation) > legend_cutoff
    ]
    legend = [
        line
        for line in segments
        if page_geometry.observation_mid_y(line.observation) <= legend_cutoff
    ]
    ordered = order_figure_segments_by_columns(main, region_bbox)
    ordered.extend(sorted(legend, key=figure_text_line_reading_order_key))
    return ordered


def order_figure_segments_by_columns(
    segments: list[ocr_page_analysis.TextGeometryLine],
    region_bbox: tuple[float, float, float, float],
) -> list[ocr_page_analysis.TextGeometryLine]:
    if len(segments) <= 2:
        return sorted(segments, key=figure_text_line_reading_order_key)
    width = max(1.0, region_bbox[2] - region_bbox[0])
    tolerance = max(14.0, min(32.0, width * 0.10))
    columns = figure_x_position_clusters(segments, tolerance)
    ordered: list[ocr_page_analysis.TextGeometryLine] = []
    for column in columns:
        ordered.extend(sorted(column, key=figure_text_line_reading_order_key))
    return ordered


def figure_text_line_reading_order_key(
    line: ocr_page_analysis.TextGeometryLine,
) -> tuple[float, float]:
    return page_geometry.observation_reading_order_key(line.observation)


def figure_x_position_clusters(
    segments: list[ocr_page_analysis.TextGeometryLine],
    tolerance: float,
) -> list[list[ocr_page_analysis.TextGeometryLine]]:
    clusters: list[list[ocr_page_analysis.TextGeometryLine]] = []
    for segment in sorted(
        segments,
        key=lambda line: page_geometry.observation_mid_x(line.observation),
    ):
        center = page_geometry.observation_mid_x(segment.observation)
        if not clusters:
            clusters.append([segment])
            continue
        cluster_center = sum(
            page_geometry.observation_mid_x(line.observation) for line in clusters[-1]
        ) / len(clusters[-1])
        if abs(center - cluster_center) <= tolerance:
            clusters[-1].append(segment)
        else:
            clusters.append([segment])
    return clusters


def fused_figure_ocr_candidate(
    region: ocr_page_analysis.FigureOcrRegion,
    candidates: list[OcrCandidate],
) -> OcrCandidate | None:
    usable = usable_figure_ocr_candidates(candidates)
    if not usable:
        return None
    line_clusters = figure_line_alternative_clusters(usable)
    if not line_clusters:
        return max(usable, key=figure_ocr_candidate_score)
    selected_lines = [max(cluster, key=figure_text_alternative_score) for cluster in line_clusters]
    candidate_boxes = {
        candidate.name: candidate.page_bbox
        for candidate in usable
        if candidate.page_bbox is not None
    }
    candidate_lines = {
        candidate.name: figure_candidate_text_geometry_lines(candidate) for candidate in usable
    }
    selected_lines = figure_drop_duplicate_selected_lines(selected_lines)
    selected_lines = repair_figure_reference_numeral_lines(selected_lines)
    selected_lines = precision_filter_figure_reference_lines(selected_lines)
    selected_lines = compose_complementary_figure_lines(selected_lines)
    selected_lines = refine_fused_figure_selected_lines(selected_lines)
    selected_lines = figure_drop_metadata_selected_lines(selected_lines)
    selected_lines = figure_drop_suspicious_callout_lines(
        selected_lines,
        candidate_boxes,
        candidate_lines,
    )
    selected_lines = figure_cleanup_selected_callout_lines(selected_lines)
    selected_lines = supplement_consensus_callout_cluster_lines(
        selected_lines,
        candidate_lines,
    )
    selected_lines = order_figure_text_segments(selected_lines, region.bbox)
    if not selected_lines:
        return max(usable, key=figure_ocr_candidate_score)
    line_rows = tuple(figure_text_geometry_line_row(line) for line in selected_lines)
    text = "\n".join(line.text for line in selected_lines if line.text.strip())
    confidence = figure_text_lines_confidence(
        selected_lines,
        max(usable, key=figure_ocr_candidate_score).result.confidence,
    )
    text_result = OcrTextResult(
        text,
        confidence,
        line_rows=line_rows,
        word_rows=tuple(row for candidate in usable for row in candidate.result.word_rows),
        symbol_rows=tuple(row for candidate in usable for row in candidate.result.symbol_rows),
    )
    page_bbox = union_page_bboxes(
        candidate.page_bbox for candidate in usable if candidate.page_bbox is not None
    )
    return ocr_candidates.OcrCandidate(
        f"figure_region_{region.item_index}_fused",
        text_result,
        region_count=1,
        page_bbox=page_bbox,
    )


def usable_figure_ocr_candidates(candidates: list[OcrCandidate]) -> list[OcrCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.result.text.strip() and figure_ocr_candidate_score(candidate) > 0.0
    ]


def figure_line_alternative_clusters(
    candidates: list[OcrCandidate],
) -> list[list[ocr_page_analysis.TextGeometryLine]]:
    clusters: list[list[ocr_page_analysis.TextGeometryLine]] = []
    for candidate in sorted(candidates, key=lambda item: item.name):
        for line in figure_candidate_text_geometry_lines(candidate):
            target = figure_matching_line_cluster(clusters, line)
            if target is None:
                clusters.append([line])
            else:
                target.append(line)
    return clusters


def figure_candidate_text_geometry_lines(
    candidate: OcrCandidate,
) -> list[ocr_page_analysis.TextGeometryLine]:
    lines: list[ocr_page_analysis.TextGeometryLine] = []
    for row in candidate.result.line_rows:
        text = str(row.get("text", "")).strip()
        bbox = page_geometry.rect_box_tuple(row.get("page_bbox"))
        if not text or bbox is None:
            continue
        confidence = row.get("conf")
        try:
            confidence_value = (
                int(round(ocr_float_value(confidence))) if confidence is not None else None
            )
        except (TypeError, ValueError):
            confidence_value = None
        normalized_bbox = page_geometry.normalize_rect(bbox)
        if normalized_bbox is None:
            continue
        lines.append(
            ocr_page_analysis.text_geometry_line_from_bbox(
                text,
                normalized_bbox,
                confidence_value,
                source=candidate.name,
                kind="figure_text_line",
            )
        )
    return lines


def figure_candidate_token_evidence(
    candidate: OcrCandidate,
    *,
    rows: str = "line_rows",
    token_extractor: Callable[[str], str | None] | None = None,
) -> list[FigureTokenEvidence]:
    extractor = precision_label_single_token if token_extractor is None else token_extractor
    row_items = candidate.result.word_rows if rows == "word_rows" else candidate.result.line_rows
    evidence: list[FigureTokenEvidence] = []
    confidence = page_geometry.numeric_confidence(candidate.result.confidence)
    for row in row_items:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        bbox = dominant_image_row_page_bbox(row)
        if bbox is None:
            continue
        token = extractor(text)
        if token is None:
            continue
        evidence.append(
            FigureTokenEvidence(
                token=token,
                text=text,
                bbox=bbox,
                source=str(candidate.name),
                confidence=confidence,
            )
        )
    return evidence


def precision_label_single_token(text: str) -> str | None:
    tokens = precision_label_candidate_tokens(text)
    return tokens[0] if len(tokens) == 1 else None


def dominant_image_alpha_fragment_token(text: str) -> str | None:
    cleaned = "".join(ch for ch in text if ch.isalpha())
    if len(cleaned) < 2 or len(cleaned) > 8:
        return None
    if cleaned.casefold() == cleaned and len(cleaned) < 4:
        return None
    return cleaned.casefold()


def figure_drop_duplicate_selected_lines(
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> list[ocr_page_analysis.TextGeometryLine]:
    kept: list[ocr_page_analysis.TextGeometryLine] = []
    for line in sorted(lines, key=figure_text_alternative_score, reverse=True):
        if any(figure_selected_lines_are_duplicates(line, existing) for existing in kept):
            continue
        kept.append(line)
    return kept


def figure_selected_lines_are_duplicates(
    left: ocr_page_analysis.TextGeometryLine,
    right: ocr_page_analysis.TextGeometryLine,
) -> bool:
    left_tokens = normalized_text_tokens(left.text)
    right_tokens = normalized_text_tokens(right.text)
    if not left_tokens or left_tokens != right_tokens:
        return False
    return figure_text_lines_match(left, right)


def repair_figure_reference_numeral_lines(
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> list[ocr_page_analysis.TextGeometryLine]:
    if not figure_lines_use_short_reference_numerals(lines):
        return lines
    repaired: list[ocr_page_analysis.TextGeometryLine] = []
    for line in lines:
        repaired_text = repair_figure_reference_numeral_text(line.text)
        if repaired_text == line.text:
            repaired.append(line)
            continue
        repaired.append(
            replace(
                line,
                text=repaired_text,
                observation=replace(line.observation, text=repaired_text),
            )
        )
    return repaired


def figure_lines_use_short_reference_numerals(
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> bool:
    lengths: list[int] = []
    for line in lines:
        if not figure_reference_numeral_context_line_is_eligible(line.text):
            continue
        lengths.extend(len(token) for token in normalized_text_tokens(line.text) if token.isdigit())
    if len(lengths) < 8:
        return False
    short_count = sum(1 for length in lengths if length <= 2)
    return short_count / len(lengths) >= 0.82


def figure_reference_numeral_context_line_is_eligible(text: str) -> bool:
    lowered_tokens = set(normalized_text_tokens(text.casefold()))
    if lowered_tokens & {"patent", "sheet"}:
        return False
    if re.search(r"\d,\d", text):
        return False
    tokens = normalized_text_tokens(text)
    if any(token.isdigit() and len(token) >= 4 for token in tokens):
        return False
    return any(token.isdigit() for token in tokens) and any(
        token.isalpha() and len(token) >= 4 for token in tokens
    )


def repair_figure_reference_numeral_text(text: str) -> str:
    if not figure_reference_numeral_context_line_is_eligible(text):
        return text

    def repair_token(match: re.Match[str]) -> str:
        token = match.group(0)
        cleaned = token.strip("~`'\"|/\\")
        if len(cleaned) == 3 and cleaned.isdigit() and cleaned[1:] != "00":
            return cleaned[1:]
        return cleaned

    return re.sub(
        r"(?<![A-Za-z0-9])[~`'\"|/\\]*\d{1,3}[~`'\"|/\\]*(?![A-Za-z0-9])",
        repair_token,
        text,
    )


def precision_filter_figure_reference_lines(
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> list[ocr_page_analysis.TextGeometryLine]:
    if not figure_lines_use_short_reference_numerals(lines):
        return lines
    filtered: list[ocr_page_analysis.TextGeometryLine] = []
    for line in lines:
        cleaned_text = precision_clean_figure_reference_text(line)
        if not cleaned_text:
            continue
        if cleaned_text == line.text:
            filtered.append(line)
            continue
        filtered.append(
            replace(
                line,
                text=cleaned_text,
                observation=replace(line.observation, text=cleaned_text),
            )
        )
    return drop_figure_reference_numeric_outlier_lines(filtered)


def refine_fused_figure_selected_lines(
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> list[ocr_page_analysis.TextGeometryLine]:
    if len(lines) < 6:
        return lines
    descriptive_lines = sum(1 for line in lines if figure_fused_line_is_descriptive(line.text))
    if descriptive_lines < 4:
        return lines
    refined = [
        line for line in lines if not figure_fused_line_is_fragment(line.text, line.confidence)
    ]
    return refined or lines


def figure_drop_metadata_selected_lines(
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> list[ocr_page_analysis.TextGeometryLine]:
    filtered = [line for line in lines if not figure_selected_line_is_metadata(line.text)]
    return filtered or lines


def figure_selected_line_is_metadata(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if figure_reference_precision_line_is_metadata(stripped):
        return True
    tokens = normalized_text_tokens(stripped.casefold())
    if not tokens:
        return True
    if tokens == ["fig"]:
        return True
    if len(tokens) == 2 and tokens[0].isdigit() and tokens[1] == "of":
        return True
    if len(tokens) == 3 and tokens[0].isdigit() and tokens[1] == "of" and tokens[2].isdigit():
        return True
    return bool({"oct", "us", "patent"} & set(tokens))


def figure_cleanup_selected_callout_lines(
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> list[ocr_page_analysis.TextGeometryLine]:
    filtered = [line for line in lines if not figure_selected_callout_line_should_drop(line)]
    if not filtered:
        filtered = lines
    return figure_drop_redundant_selected_lines(filtered)


def supplement_consensus_callout_cluster_lines(
    selected_lines: list[ocr_page_analysis.TextGeometryLine],
    candidate_lines: dict[str, list[ocr_page_analysis.TextGeometryLine]],
) -> list[ocr_page_analysis.TextGeometryLine]:
    clusters: dict[str, list[ocr_page_analysis.TextGeometryLine]] = {}
    for source, lines in candidate_lines.items():
        cluster_key = callout_cluster_source_group_key(source)
        if cluster_key is None:
            continue
        clusters.setdefault(cluster_key, []).extend(lines)
    additions: list[ocr_page_analysis.TextGeometryLine] = []
    for cluster_lines in clusters.values():
        addition = consensus_callout_cluster_line(cluster_lines, selected_lines)
        if addition is not None:
            additions.append(addition)
    if not additions:
        return selected_lines
    supplemented = list(selected_lines)
    for line in sorted(additions, key=figure_selected_line_priority, reverse=True):
        if any(figure_selected_lines_are_redundant(line, existing) for existing in supplemented):
            continue
        supplemented.append(line)
    return supplemented


def callout_cluster_source_group_key(source: str) -> str | None:
    if "callout_cluster_" not in source:
        return None
    for suffix in ("_base_psm", "_otsu_psm", "_sauvola_psm"):
        marker = source.find(suffix)
        if marker != -1:
            return source[:marker]
    return source


def consensus_callout_cluster_line(
    cluster_lines: list[ocr_page_analysis.TextGeometryLine],
    selected_lines: list[ocr_page_analysis.TextGeometryLine],
) -> ocr_page_analysis.TextGeometryLine | None:
    candidates = [
        line
        for line in cluster_lines
        if descriptive_callout_cluster_line_is_consensus_eligible(line, selected_lines)
    ]
    if not candidates:
        return None
    best = max(candidates, key=figure_text_alternative_score)
    support = callout_cluster_line_consensus_support(best, cluster_lines)
    if support < 2:
        return None
    return best


def descriptive_callout_cluster_line_is_consensus_eligible(
    line: ocr_page_analysis.TextGeometryLine,
    selected_lines: list[ocr_page_analysis.TextGeometryLine],
) -> bool:
    if figure_selected_callout_line_should_drop(line):
        return False
    confidence = line.confidence if line.confidence is not None else 0
    if confidence < 70:
        return False
    tokens = normalized_text_tokens(line.text)
    readable_alpha = sum(1 for token in tokens if token.isalpha() and len(token) >= 4)
    numeric_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    if readable_alpha < 3 or numeric_tokens < 1:
        return False
    return not any(
        figure_selected_lines_are_redundant(line, existing) for existing in selected_lines
    )


def callout_cluster_line_consensus_support(
    target: ocr_page_analysis.TextGeometryLine,
    cluster_lines: list[ocr_page_analysis.TextGeometryLine],
) -> int:
    target_tokens = set(normalized_text_tokens(target.text.casefold()))
    target_alpha = {token for token in target_tokens if token.isalpha() and len(token) >= 4}
    if len(target_alpha) < 3:
        return 0
    support = 0
    for line in cluster_lines:
        if line is target:
            support += 1
            continue
        if not figure_text_lines_match(target, line):
            continue
        line_tokens = set(normalized_text_tokens(line.text.casefold()))
        line_alpha = {token for token in line_tokens if token.isalpha() and len(token) >= 4}
        if len(line_alpha) < 2:
            continue
        if len(target_alpha & line_alpha) < min(3, len(target_alpha), len(line_alpha)):
            continue
        support += 1
    return support


def figure_selected_callout_line_should_drop(
    line: ocr_page_analysis.TextGeometryLine,
) -> bool:
    source = line.observation.source
    if "callout_cluster_" not in source:
        return False
    text = line.text.strip()
    tokens = normalized_text_tokens(text)
    if not tokens:
        return True
    if len(tokens) == 1 and tokens[0].isalpha() and len(tokens[0]) <= 4:
        return True
    if ocr_text_analysis.alphabetic_gibberish_score(text) >= 0.22:
        return True
    numeric_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    alpha_tokens = sum(1 for token in tokens if token.isalpha() and len(token) >= 4)
    short_alpha_tokens = sum(1 for token in tokens if token.isalpha() and len(token) <= 3)
    if numeric_tokens >= 1 and alpha_tokens >= 1 and short_alpha_tokens >= 1 and len(tokens) >= 3:
        return True
    return bool(numeric_tokens >= 2 and alpha_tokens >= 2 and len(tokens) >= 4)


def figure_drop_redundant_selected_lines(
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> list[ocr_page_analysis.TextGeometryLine]:
    if len(lines) < 2:
        return lines
    kept: list[ocr_page_analysis.TextGeometryLine] = []
    for line in sorted(lines, key=figure_selected_line_priority, reverse=True):
        if any(figure_selected_lines_are_redundant(line, existing) for existing in kept):
            continue
        kept.append(line)
    return kept or lines


def figure_selected_line_priority(
    line: ocr_page_analysis.TextGeometryLine,
) -> float:
    score = figure_text_alternative_score(line)
    if figure_selected_callout_line_should_drop(line):
        score -= 20.0
    return score


def figure_selected_lines_are_redundant(
    left: ocr_page_analysis.TextGeometryLine,
    right: ocr_page_analysis.TextGeometryLine,
) -> bool:
    left_bbox = left.observation.bbox
    right_bbox = right.observation.bbox
    if left_bbox is None or right_bbox is None:
        return False
    overlap = page_geometry.rect_intersection_area(left_bbox, right_bbox)
    if overlap <= 0.0:
        return False
    left_area = page_geometry.rect_area(left_bbox)
    right_area = page_geometry.rect_area(right_bbox)
    if min(left_area, right_area) <= 0.0:
        return False
    overlap_ratio = overlap / min(left_area, right_area)
    if overlap_ratio < 0.45:
        return False
    left_tokens = set(normalized_text_tokens(left.text.casefold()))
    right_tokens = set(normalized_text_tokens(right.text.casefold()))
    if not left_tokens or not right_tokens:
        return False
    return (
        left_tokens <= right_tokens
        or right_tokens <= left_tokens
        or len(left_tokens & right_tokens) >= min(len(left_tokens), len(right_tokens))
    )


def figure_drop_suspicious_callout_lines(
    lines: list[ocr_page_analysis.TextGeometryLine],
    candidate_boxes: dict[str, tuple[float, float, float, float]],
    candidate_lines: dict[str, list[ocr_page_analysis.TextGeometryLine]],
) -> list[ocr_page_analysis.TextGeometryLine]:
    filtered = [
        line
        for line in lines
        if not figure_callout_line_should_drop(line, candidate_boxes, candidate_lines)
    ]
    return filtered or lines


def figure_callout_line_should_drop(
    line: ocr_page_analysis.TextGeometryLine,
    candidate_boxes: dict[str, tuple[float, float, float, float]],
    candidate_lines: dict[str, list[ocr_page_analysis.TextGeometryLine]],
) -> bool:
    source = line.observation.source
    if "callout_cluster_" not in source:
        return False
    source_lines = candidate_lines.get(source, [])
    if figure_callout_source_is_fragment_heavy(source_lines) and (
        figure_callout_line_has_single_label_token(line.text)
        or figure_fused_line_is_fragment(line.text, line.confidence)
    ):
        return True
    line_bbox = line.observation.bbox
    candidate_bbox = candidate_boxes.get(source)
    if line_bbox is None or candidate_bbox is None:
        return False
    if not figure_callout_line_has_single_label_token(line.text):
        return False
    line_height = max(1.0, page_geometry.observation_height(line.observation))
    left_gap = max(0.0, line_bbox[0] - candidate_bbox[0])
    right_gap = max(0.0, candidate_bbox[2] - line_bbox[2])
    edge_threshold = max(8.0, line_height * 0.9)
    return left_gap <= edge_threshold or right_gap <= edge_threshold


def figure_callout_line_has_single_label_token(text: str) -> bool:
    tokens = normalized_text_tokens(text)
    if len(tokens) > 2:
        return False
    readable_alpha_tokens = [token for token in tokens if token.isalpha() and len(token) >= 4]
    return len(readable_alpha_tokens) == 1 and any(token.isdigit() for token in tokens)


def figure_callout_source_is_fragment_heavy(
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> bool:
    if len(lines) < 3:
        return False
    fragment_count = sum(1 for line in lines if figure_callout_source_line_is_fragment(line))
    return fragment_count >= 2


def figure_callout_source_line_is_fragment(
    line: ocr_page_analysis.TextGeometryLine,
) -> bool:
    tokens = normalized_text_tokens(line.text)
    if not tokens:
        return True
    if figure_fused_line_is_fragment(line.text, line.confidence):
        return True
    if len(tokens) == 1:
        token = tokens[0]
        return token.isalpha() and len(token) <= 4
    return False


def compose_complementary_figure_lines(
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> list[ocr_page_analysis.TextGeometryLine]:
    if len(lines) < 2:
        return lines
    ordered = sorted(lines, key=figure_text_line_reading_order_key)
    composed: list[ocr_page_analysis.TextGeometryLine] = []
    index = 0
    while index < len(ordered):
        current = ordered[index]
        if index + 1 >= len(ordered):
            composed.append(current)
            break
        right = ordered[index + 1]
        merged = compose_complementary_figure_line_pair(current, right)
        if merged is not None:
            composed.append(merged)
            index += 2
            continue
        composed.append(current)
        index += 1
    return composed


def compose_complementary_figure_line_pair(
    left: ocr_page_analysis.TextGeometryLine,
    right: ocr_page_analysis.TextGeometryLine,
) -> ocr_page_analysis.TextGeometryLine | None:
    left_bbox = left.observation.bbox
    right_bbox = right.observation.bbox
    if left_bbox is None or right_bbox is None:
        return None
    if not figure_lines_are_complementary_neighbors(left, right):
        return None
    merged_text = figure_merge_complementary_line_text(left.text, right.text)
    if merged_text is None:
        return None
    merged_bbox = (
        min(left_bbox[0], right_bbox[0]),
        min(left_bbox[1], right_bbox[1]),
        max(left_bbox[2], right_bbox[2]),
        max(left_bbox[3], right_bbox[3]),
    )
    confidence = figure_text_lines_confidence(
        [left, right],
        left.confidence if left.confidence is not None else right.confidence,
    )
    return ocr_page_analysis.text_geometry_line_from_bbox(
        merged_text,
        merged_bbox,
        confidence,
        source=left.observation.source,
        kind="figure_text_line",
        provenance={
            "composed_from": (
                left.observation.source,
                right.observation.source,
            )
        },
    )


def figure_lines_are_complementary_neighbors(
    left: ocr_page_analysis.TextGeometryLine,
    right: ocr_page_analysis.TextGeometryLine,
) -> bool:
    left_bbox = left.observation.bbox
    right_bbox = right.observation.bbox
    if left_bbox is None or right_bbox is None:
        return False
    left_height = max(1.0, page_geometry.observation_height(left.observation))
    right_height = max(1.0, page_geometry.observation_height(right.observation))
    center_delta = abs(
        page_geometry.observation_mid_y(left.observation)
        - page_geometry.observation_mid_y(right.observation)
    )
    if center_delta > max(left_height, right_height) * 0.8:
        return False
    horizontal_gap = max(0.0, right_bbox[0] - left_bbox[2])
    if horizontal_gap > max(left_height, right_height) * 4.5:
        return False
    overlap = max(
        0.0,
        min(left_bbox[2], right_bbox[2]) - max(left_bbox[0], right_bbox[0]),
    )
    max_allowed_overlap = (
        min(
            page_geometry.observation_width(left.observation),
            page_geometry.observation_width(right.observation),
        )
        * 0.55
    )
    return not overlap > max_allowed_overlap


def figure_merge_complementary_line_text(left_text: str, right_text: str) -> str | None:
    left_tokens = normalized_text_tokens(left_text)
    right_tokens = normalized_text_tokens(right_text)
    if not left_tokens or not right_tokens:
        return None
    if figure_reference_precision_line_is_metadata(
        left_text
    ) or figure_reference_precision_line_is_metadata(right_text):
        return None
    left_readable = figure_complement_readable_token_count(left_tokens)
    left_numeric = all(token.isdigit() for token in left_tokens)
    right_numeric = all(token.isdigit() for token in right_tokens)
    if left_numeric and right_numeric:
        return None
    if any(token.isdigit() for token in left_tokens):
        return None
    if left_readable < 1:
        return None
    if not right_numeric or len(right_tokens) != 1 or len(right_tokens[0]) > 2:
        return None
    merged = f"{left_text.strip()} {right_tokens[0]}"
    merged = " ".join(merged.split())
    if not figure_text_line_is_useful(merged):
        return None
    if merged == left_text.strip() or merged == right_text.strip():
        return None
    return merged


def figure_complement_readable_token_count(tokens: list[str]) -> int:
    return sum(1 for token in tokens if token.isalpha() and len(token) >= 4)


def figure_fused_line_is_descriptive(text: str) -> bool:
    tokens = normalized_text_tokens(text)
    readable_alpha = sum(1 for token in tokens if token.isalpha() and len(token) >= 4)
    return readable_alpha >= 1 and not figure_reference_precision_line_is_metadata(text)


def figure_fused_line_is_fragment(
    text: str,
    confidence: int | None,
) -> bool:
    stripped = text.strip()
    if not stripped or figure_reference_precision_line_is_metadata(stripped):
        return False
    tokens = normalized_text_tokens(stripped)
    if not tokens:
        return True
    if all(token.isdigit() for token in tokens):
        return len(tokens) <= 2
    readable_alpha = sum(1 for token in tokens if token.isalpha() and len(token) >= 4)
    if len(tokens) == 1:
        token = tokens[0]
        if token.isalpha() and len(token) < 4:
            return True
        if token.isdigit():
            return True
        if token.isalpha() and confidence is not None and confidence < 65:
            return True
    alpha_lengths = [len(token) for token in tokens if token.isalpha()]
    if readable_alpha == 0 and alpha_lengths and max(alpha_lengths) < 4:
        return True
    if readable_alpha == 0:
        if len(tokens) <= 2:
            return True
        if figure_text_noise_ratio(stripped) >= 0.18:
            return True
        if ocr_text_analysis.alphabetic_gibberish_score(stripped) >= 0.28:
            return True
    return bool(confidence is not None and confidence < 65 and readable_alpha == 0)


def precision_clean_figure_reference_text(
    line: ocr_page_analysis.TextGeometryLine,
) -> str:
    text = line.text.strip()
    if not text:
        return ""
    if figure_reference_precision_line_is_metadata(text):
        return text
    confidence = line.confidence if line.confidence is not None else 50
    tokens = [
        token
        for raw_token in text.split()
        if (token := precision_clean_figure_reference_token(raw_token)) is not None
    ]
    tokens = drop_ambiguous_leading_figure_reference_number(tokens)
    if not tokens:
        return ""
    if len(tokens) == 1 and tokens[0].isalpha() and confidence < 75:
        return ""
    if confidence < 60 and not any(
        token.isalpha() and len(token) >= 4 and token.upper() == token for token in tokens
    ):
        return ""
    cleaned = " ".join(tokens)
    if not figure_text_line_is_useful(cleaned, confidence):
        return ""
    return cleaned


def figure_reference_precision_line_is_metadata(text: str) -> bool:
    tokens = set(normalized_text_tokens(text.casefold()))
    if tokens & {"fig", "patent", "sheet", "oct", "us"}:
        return True
    return bool(re.search(r"\d,\d", text))


def precision_clean_figure_reference_token(token: str) -> str | None:
    stripped = token.strip()
    if not stripped:
        return None
    if any(ch in stripped for ch in '[]{}<>"=°'):
        return None
    cleaned = stripped.strip("~`'|/\\.,;:!?()")
    if not cleaned:
        return None
    if cleaned.isdigit():
        return cleaned if len(cleaned) >= 2 else None
    if cleaned.isalpha():
        if len(cleaned) <= 2 and len(set(cleaned.casefold())) == 1:
            return None
        if len(cleaned) < 3 and cleaned.upper() != cleaned:
            return None
        if not (cleaned.upper() == cleaned or cleaned.istitle()):
            return None
        return cleaned
    if re.fullmatch(r"[A-Z]\d", cleaned):
        return cleaned
    return None


def drop_ambiguous_leading_figure_reference_number(tokens: list[str]) -> list[str]:
    if len(tokens) < 3:
        return tokens
    if not tokens[0].isdigit():
        return tokens
    numeric_count = sum(1 for token in tokens if token.isdigit())
    alpha_count = sum(1 for token in tokens if token.isalpha())
    if numeric_count == 1 and alpha_count >= 2:
        return tokens[1:]
    return tokens


def drop_figure_reference_numeric_outlier_lines(
    lines: list[ocr_page_analysis.TextGeometryLine],
) -> list[ocr_page_analysis.TextGeometryLine]:
    values: list[int] = []
    isolated_values: dict[int, list[ocr_page_analysis.TextGeometryLine]] = {}
    for line in lines:
        if figure_reference_precision_line_is_metadata(line.text):
            continue
        tokens = normalized_text_tokens(line.text)
        numeric_tokens = [token for token in tokens if token.isdigit()]
        for token in numeric_tokens:
            try:
                value = int(token)
            except ValueError:
                continue
            if value < 10 or value > 999:
                continue
            values.append(value)
        if tokens and tokens == numeric_tokens and len(numeric_tokens) == 1:
            value = int(numeric_tokens[0])
            isolated_values.setdefault(value, []).append(line)
    if len(values) < 12 or not isolated_values:
        return lines
    sorted_values = sorted(values)
    p90_index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * 0.90) - 1))
    reference_ceiling = max(80, int(round(sorted_values[p90_index] * 1.40)))
    outlier_values = {
        value
        for value in isolated_values
        if value >= reference_ceiling and value == max(sorted_values)
    }
    if not outlier_values:
        return lines
    outlier_line_ids = {
        id(outlier_line) for value in outlier_values for outlier_line in isolated_values[value]
    }
    return [line for line in lines if id(line) not in outlier_line_ids]


def figure_matching_line_cluster(
    clusters: list[list[ocr_page_analysis.TextGeometryLine]],
    line: ocr_page_analysis.TextGeometryLine,
) -> list[ocr_page_analysis.TextGeometryLine] | None:
    for cluster in clusters:
        if any(figure_text_lines_match(existing, line) for existing in cluster):
            return cluster
    return None


def figure_text_lines_match(
    left: ocr_page_analysis.TextGeometryLine,
    right: ocr_page_analysis.TextGeometryLine,
) -> bool:
    return (
        page_geometry.observation_geometry_match_score(
            left.observation,
            right.observation,
        )
        >= 0.62
    )


def figure_text_alternative_score(line: ocr_page_analysis.TextGeometryLine) -> float:
    text = line.text.strip()
    tokens = normalized_text_tokens(text)
    confidence = line.confidence if line.confidence is not None else 50
    score = confidence * 0.10 + len(tokens)
    if "=" in text:
        score += 4.0
    if "_" in text:
        score += 4.0
    if any(ch.isdigit() for ch in text):
        score += 3.0
    if "subregion_" in line.observation.source and any(
        token.isalpha() and len(token) >= 4 for token in tokens
    ):
        score += 4.0
    if "callout_cluster_" in line.observation.source and any(
        token.isalpha() and len(token) >= 4 for token in tokens
    ):
        score += 6.0
    if figure_text_noise_ratio(text) > 0.25:
        score -= 8.0
    score -= figure_text_noise_ratio(text) * 12.0
    score -= ocr_text_analysis.alphabetic_gibberish_score(text) * 18.0
    score -= max(0, len(tokens) - 5) * 1.4
    return score


def figure_ocr_candidate_score(candidate: OcrCandidate) -> float:
    lines = [line.strip() for line in candidate.result.text.splitlines() if line.strip()]
    useful_lines = sum(1 for line in lines if figure_text_line_is_useful(line))
    tokens = normalized_text_tokens(candidate.result.text)
    confidence = candidate.result.confidence if candidate.result.confidence is not None else 45
    quality = text_ocr_quality_score(candidate.result.text)
    noise = figure_text_noise_ratio(candidate.result.text)
    gibberish = ocr_text_analysis.alphabetic_gibberish_score(candidate.result.text)
    clean_token_count = len(tokens) * (1.0 - min(0.75, gibberish))
    purity_penalty = figure_ocr_candidate_purity_penalty(candidate, lines)
    return (
        useful_lines * 8.0
        + clean_token_count * 2.0
        + min(95, confidence) * 0.10
        - quality * 35.0
        - noise * 30.0
        - gibberish * max(140.0, len(tokens) * 5.0)
        - purity_penalty
    )


def figure_ocr_candidate_purity_penalty(
    candidate: OcrCandidate,
    lines: list[str],
) -> float:
    if "callout_cluster_" not in candidate.name:
        return 0.0
    if len(lines) <= 1:
        return 0.0
    fragment_lines = sum(1 for line in lines if figure_callout_candidate_line_is_fragment(line))
    metadata_lines = sum(1 for line in lines if figure_selected_line_is_metadata(line))
    descriptive_lines = sum(
        1 for line in lines if figure_callout_candidate_line_is_descriptive(line)
    )
    penalty = fragment_lines * 9.0 + metadata_lines * 12.0
    if descriptive_lines == 0:
        penalty += 12.0
    elif descriptive_lines == 1 and len(lines) >= 3:
        penalty += 8.0
    if fragment_lines >= descriptive_lines and len(lines) >= 3:
        penalty += 10.0
    return penalty


def figure_callout_candidate_line_is_fragment(text: str) -> bool:
    tokens = normalized_text_tokens(text)
    if not tokens:
        return True
    confidence = 60 if figure_text_line_is_useful(text) else 40
    if figure_fused_line_is_fragment(text, confidence):
        return True
    return bool(
        len(tokens) <= 2 and not any(token.isalpha() and len(token) >= 4 for token in tokens)
    )


def figure_callout_candidate_line_is_descriptive(text: str) -> bool:
    tokens = normalized_text_tokens(text)
    readable_alpha = sum(1 for token in tokens if token.isalpha() and len(token) >= 4)
    return readable_alpha >= 1 and not figure_selected_line_is_metadata(text)


def combined_figure_ocr_page_result(
    selected_candidates: list[OcrCandidate],
    candidates: list[OcrCandidate],
) -> OcrPageTextResult:
    selected_candidates = [
        candidate for candidate in selected_candidates if candidate.result.text.strip()
    ]
    if not selected_candidates:
        return ocr_candidates.OcrPageTextResult("", candidates=tuple(candidates))
    text = "\n".join(candidate.result.text.strip() for candidate in selected_candidates)
    confidences = [
        candidate.result.confidence
        for candidate in selected_candidates
        if candidate.result.confidence is not None
    ]
    confidence = int(round(sum(confidences) / len(confidences))) if confidences else None
    line_rows = tuple(
        row for candidate in selected_candidates for row in candidate.result.line_rows
    )
    word_rows = tuple(
        row for candidate in selected_candidates for row in candidate.result.word_rows
    )
    symbol_rows = tuple(
        row for candidate in selected_candidates for row in candidate.result.symbol_rows
    )
    page_bbox = union_page_bboxes(
        candidate.page_bbox for candidate in selected_candidates if candidate.page_bbox is not None
    )
    text_result = OcrTextResult(
        text,
        confidence,
        line_rows=line_rows,
        word_rows=word_rows,
        symbol_rows=symbol_rows,
    )
    candidate = ocr_candidates.OcrCandidate(
        "figure_ocr_regions",
        text_result,
        region_count=len(selected_candidates),
        page_bbox=page_bbox,
    )
    return ocr_candidates.OcrPageTextResult(text, candidate, tuple(candidates))


def union_page_bboxes(
    boxes: Iterable[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    boxes = tuple(boxes)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def extract_ocr_page_text(page: PageExtractionHost, *, vector_text: str = "") -> str:
    return extract_ocr_page_result(page, vector_text=vector_text).text


def extract_ocr_page_result(
    page: PageExtractionHost,
    *,
    vector_text: str = "",
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
) -> OcrPageTextResult:
    if ocr_session is None:
        with ocr_session_runtime.OcrPageSession() as owned_session:
            try:
                return extract_ocr_page_result(
                    page,
                    vector_text=vector_text,
                    ocr_session=owned_session,
                )
            finally:
                record_ocr_deskew_diagnostics(page, owned_session)
    text = ""
    candidates: list[OcrCandidate] = []
    candidate: OcrCandidate | None = None
    selected_output_lines: tuple[observation_resolver.ResolvedTextLine, ...] = ()
    preserve_raw_text = False
    if ocr_postprocess.ocr_is_enabled():
        timeout = ocr_rendering.ocr_timeout_seconds()
        candidates = collect_ocr_candidates(
            page,
            timeout,
            vector_text=vector_text,
            ocr_session=ocr_session,
        )
        verification_candidates = tuple(
            candidate for candidate in candidates if candidate.name.startswith("verification_")
        )
        candidates = [
            candidate for candidate in candidates if not candidate.name.startswith("verification_")
        ]
        candidate = ocr_selection.select_ocr_candidate(candidates, support_text=vector_text)
        record_ocr_candidate_diagnostics(
            page,
            candidates,
            selected_candidate=candidate,
            support_text=vector_text,
        )
        if candidate is not None:
            text = candidate.result.text
            table_fusion = fuse_table_ocr_candidates(
                text,
                candidates,
                selected_candidate=candidate,
            )
            fused_text = dense_table_categorical_token_supplement(
                table_fusion.text,
                candidate,
                candidates,
            )
            selected_output_lines = table_fusion.output_lines
            output_lines = repair_ocr_output_lines_with_alternate_candidates(
                selected_output_lines,
                candidates,
                selected_candidate=candidate,
            )
            split_band_lines = ocr_postprocess.candidate_multi_column_band_split_lines(candidate)
            if split_band_lines and len(split_band_lines) >= len(output_lines) + 8:
                output_lines = ocr_postprocess.resolved_text_lines_from_geometry_lines(
                    split_band_lines,
                    source=f"{candidate.name}:band_split",
                    kind="ocr_textline",
                )
                selected_output_lines = output_lines
            rendered_output_text = render_resolved_text_lines(output_lines).rstrip()
            if (
                schematic_layout_render_drops_material_text(
                    page, candidate, fused_text, rendered_output_text
                )
                or dense_sparse_layout_render_drops_material_text(
                    candidate,
                    fused_text,
                    rendered_output_text,
                )
                or clean_full_page_ocr_should_preserve_raw_text(page, candidate, fused_text)
            ):
                text = fused_text
                output_lines = ()
                selected_output_lines = ()
                preserve_raw_text = True
            else:
                text = rendered_output_text
        else:
            output_lines = ()
    else:
        output_lines = ()
        verification_candidates = ()
    result = ocr_candidates.OcrPageTextResult(
        text,
        candidate,
        tuple(candidates),
        output_lines,
        selected_output_lines,
        verification_candidates,
        preserve_raw_text,
    )
    record_ocr_deskew_diagnostics(page, ocr_session)
    return result


def record_ocr_candidate_diagnostics(
    page: PageExtractionHost,
    candidates: Iterable[OcrCandidate],
    *,
    selected_candidate: OcrCandidate | None,
    support_text: str = "",
) -> None:
    """Cache compact candidate diagnostics for difficult OCR pages."""
    cache = getattr(page, "extraction_cache", None)
    if cache is None:
        return
    records = []
    for candidate in candidates:
        result = candidate.result
        records.append(
            {
                "name": candidate.name,
                "selected": candidate is selected_candidate,
                "characters": len(result.text),
                "tokens": ocr_text_analysis.extracted_text_token_count(result.text),
                "confidence": result.confidence,
                "score": ocr_selection.ocr_candidate_score(candidate, support_text=support_text),
                "quality": ocr_text_analysis.text_ocr_quality_score(result.text),
                "artifact": ocr_text_analysis.scanned_ocr_artifact_score(result.text),
                "line_rows": len(result.line_rows),
                "word_rows": len(result.word_rows),
            }
        )
    cache["ocr_candidate_diagnostics"] = tuple(records)


def record_ocr_deskew_diagnostics(
    page: PageExtractionHost,
    ocr_session: ocr_session_runtime.OcrPageSession,
) -> None:
    diagnostics = ocr_session.deskew_diagnostics()
    cache = getattr(page, "extraction_cache", None)
    if diagnostics and cache is not None:
        cache["ocr_deskew"] = diagnostics


def append_ocr_candidate_with_layout_variants(
    candidates: list[OcrCandidate],
    candidate: OcrCandidate,
    *,
    candidate_images: dict[str, OcrImage] | None = None,
    image: OcrImage | None = None,
    timeout: float | None = None,
) -> None:
    candidates.append(candidate)
    if candidate_images is not None and image is not None:
        candidate_images[candidate.name] = image


def collect_ocr_candidates(
    page: PageExtractionHost,
    timeout: float | None,
    *,
    vector_text: str = "",
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
) -> list[OcrCandidate]:
    candidates: list[OcrCandidate] = []

    image = full_page_image_for_ocr(page)
    if image is None:
        try:
            image = ocr_rendering.render_page_for_ocr_at_dpi(
                page,
                dpi=OCR_FALLBACK_DPI,
                source=f"rendered_page_{OCR_FALLBACK_DPI}dpi",
            )
        except Exception:
            image = None
    if image is None:
        return candidates
    image = optimized_full_page_primary_ocr_image(image)

    result = (
        ocr_session.image_to_text_result(
            image,
            psm=ocr_execution.OCR_DEFAULT_PAGE_SEGMENTATION_MODE,
        )
        if ocr_session is not None
        else ocr_full_page.ocr_image_to_text_result_with_timeout(image, timeout)
    )
    candidates.append(
        ocr_candidate_generation.ocr_candidate_from_image(
            "full_page_simple" if image.source.startswith("full_page_") else image.source,
            result,
            image,
            token_type_classifier=ocr_schematic.classify_schematic_token_type,
        )
    )
    if image.source.startswith("full_page_") and ocr_full_page.should_try_alternate_ocr(result):
        auto_result = (
            ocr_session.image_to_text_result(image, psm=3)
            if ocr_session is not None
            else ocr_execution.ocr_image_to_text_result_with_psm_timeout(
                image,
                psm=3,
                timeout=timeout,
            )
        )
        append_nonempty_ocr_candidate(candidates, "full_page_auto_psm3", auto_result, image)
        if getattr(page.get_page_profile(), "recommended_strategy", None) == "text_table":
            table_result = (
                ocr_session.image_to_text_result(image, psm=11)
                if ocr_session is not None
                else ocr_execution.ocr_image_to_text_result_with_psm_timeout(
                    image,
                    psm=11,
                    timeout=timeout,
                )
            )
            append_nonempty_ocr_candidate(
                candidates,
                "full_page_auto_psm11",
                table_result,
                image,
            )
    if should_try_dense_image_sparse_ocr_candidate(page, image, candidates[0]):
        sparse_image = dense_image_sparse_ocr_image(image)
        if sparse_image is not None:
            sparse_result = (
                ocr_session.image_to_text_result(
                    sparse_image,
                    psm=ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE,
                )
                if ocr_session is not None
                else ocr_execution.ocr_image_to_text_result_with_psm_timeout(
                    sparse_image,
                    psm=ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE,
                    timeout=timeout,
                )
            )
            append_nonempty_ocr_candidate(
                candidates,
                "full_page_high_resolution_sparse",
                sparse_result,
                sparse_image,
            )
            row_candidate = dense_image_table_row_ocr_candidate(
                image,
                sparse_image,
                result,
                timeout,
                ocr_session=ocr_session,
            )
            if row_candidate is not None:
                candidates.append(row_candidate)
    if should_try_image_only_layout_ocr_candidate(page, image, candidates[0]):
        alternate_result = (
            ocr_session.image_to_text_result(
                image,
                psm=ocr_full_page.OCR_FALLBACK_ALTERNATE_PAGE_SEGMENTATION_MODE,
            )
            if ocr_session is not None
            else ocr_execution.ocr_image_to_text_result_with_psm_timeout(
                image,
                psm=ocr_full_page.OCR_FALLBACK_ALTERNATE_PAGE_SEGMENTATION_MODE,
                timeout=timeout,
            )
        )
        append_nonempty_ocr_candidate(
            candidates,
            "verification_full_page_simple_psm4",
            alternate_result,
            image,
        )
    expand_rendered = should_expand_weak_full_page_ocr_candidates(
        page,
        image,
        candidates[0],
        vector_text=vector_text,
    )
    if expand_rendered:
        append_rendered_full_page_ocr_candidates(
            page,
            candidates,
            timeout,
            base_image=image,
            ocr_session=ocr_session,
        )
    elif should_collect_cross_source_verification_candidate(
        page,
        image,
        candidates[0],
    ):
        verification_candidates: list[OcrCandidate] = []
        append_rendered_full_page_ocr_candidates(
            page,
            verification_candidates,
            timeout,
            base_image=image,
            ocr_session=ocr_session,
        )
        candidates.extend(
            replace(candidate, name=f"verification_{candidate.name}")
            for candidate in verification_candidates
        )
    return candidates


def should_try_image_only_layout_ocr_candidate(
    page: PageExtractionHost,
    image: OcrImage,
    candidate: OcrCandidate,
) -> bool:
    if not image.source.startswith("full_page_"):
        return False
    try:
        profile = page.get_page_profile()
    except Exception:
        return False
    if getattr(profile, "recommended_strategy", None) != "image_or_ocr" or bool(
        getattr(profile, "has_text_showing_ops", False)
    ):
        return False
    result = candidate.result
    tokens = extracted_text_token_count(result.text)
    if not (
        OCR_IMAGE_ONLY_LAYOUT_RETRY_MIN_TOKENS <= tokens <= OCR_IMAGE_ONLY_LAYOUT_RETRY_MAX_TOKENS
    ):
        return False
    confidence = result.confidence if result.confidence is not None else 0
    if confidence > OCR_IMAGE_ONLY_LAYOUT_RETRY_MAX_CONFIDENCE:
        return False
    if text_ocr_quality_score(result.text) < OCR_IMAGE_ONLY_LAYOUT_RETRY_MIN_QUALITY:
        return False
    return (
        len(result.line_rows) >= OCR_IMAGE_ONLY_LAYOUT_RETRY_MIN_LINES
        and len(result.word_rows) >= OCR_IMAGE_ONLY_LAYOUT_RETRY_MIN_WORDS
    )


def optimized_full_page_primary_ocr_image(image: OcrImage) -> OcrImage:
    return optimized_full_page_ocr_image(
        image,
        max_pixels=OCR_FULL_PAGE_PRIMARY_MAX_PIXELS,
        source_suffix="scaled_primary",
    )


def should_try_dense_image_sparse_ocr_candidate(
    page: PageExtractionHost,
    image: OcrImage,
    candidate: OcrCandidate,
) -> bool:
    """Retry dense raster tables at a scale where their small cells are legible."""
    if not image.source.startswith("full_page_"):
        return False
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return False
    result = candidate.result
    tokens = extracted_text_token_count(result.text)
    if not (OCR_DENSE_IMAGE_SPARSE_MIN_TOKENS <= tokens <= OCR_DENSE_IMAGE_SPARSE_MAX_TOKENS):
        return False
    if (
        len(result.line_rows) < OCR_DENSE_IMAGE_SPARSE_MIN_LINES
        or len(result.word_rows) < OCR_DENSE_IMAGE_SPARSE_MIN_WORDS
    ):
        return False
    if numeric_token_ratio(result.text) < 0.22:
        return False
    if not ocr_text_analysis.text_has_many_digit_lines(result.text):
        return False
    confidence = result.confidence if result.confidence is not None else 0
    if confidence >= 90 and text_ocr_quality_score(result.text) <= 0.12:
        return False
    try:
        return ocr_page_analysis.has_dominant_page_image(page)
    except Exception:
        return False


def dense_image_sparse_ocr_image(image: OcrImage) -> OcrImage | None:
    """Return a bounded high-resolution view without duplicating source pixels."""
    current_width = image.target_width or image.width
    current_height = image.target_height or image.height
    if current_width <= 0 or current_height <= 0:
        return None
    current_pixels = current_width * current_height
    max_side_scale = OCR_DENSE_IMAGE_SPARSE_MAX_SIDE / max(
        current_width,
        current_height,
    )
    max_pixel_scale = math.sqrt(OCR_DENSE_IMAGE_SPARSE_MAX_PIXELS / current_pixels)
    scale = min(max_side_scale, max_pixel_scale)
    if scale < 1.20:
        return None
    target_width = max(1, int(round(current_width * scale)))
    target_height = max(1, int(round(current_height * scale)))
    if not leptonica_pix_size_is_supported(target_width, target_height):
        return None
    resolution = image.resolution or OCR_FALLBACK_DPI
    return replace(
        image,
        source=f"{image.source}_high_resolution_sparse",
        target_width=target_width,
        target_height=target_height,
        resolution=max(1, int(round(resolution * scale))),
    )


def dense_image_table_row_ocr_candidate(
    source_image: OcrImage,
    sparse_image: OcrImage,
    primary_result: OcrTextResult,
    timeout: float | None,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
) -> OcrCandidate | None:
    rectangles = dense_image_table_row_rectangles(source_image, primary_result)
    if not rectangles:
        return None
    variables = {
        **ocr_table_regions.OCR_TESSERACT_TABLE_VARIABLES,
        "load_freq_dawg": "0",
        "load_system_dawg": "0",
    }
    requests = [
        ocr_execution.RectangleOcrRequest(rectangle, 7, variables) for rectangle in rectangles
    ]
    results = (
        ocr_session.image_regions_to_text_results(sparse_image, requests)
        if ocr_session is not None
        else ocr_execution.ocr_image_regions_to_text_results_with_timeout(
            sparse_image,
            requests,
            timeout,
        )
    )
    texts: list[str] = []
    confidences: list[int] = []
    for result in results:
        normalized = ocr_table_regions.normalize_table_region_ocr_text(result.text)
        if not normalized:
            continue
        texts.append(normalized)
        if result.confidence is not None:
            confidences.append(result.confidence)
    if not texts:
        return None
    confidence = int(round(sum(confidences) / len(confidences))) if confidences else None
    return ocr_candidates.OcrCandidate(
        "dense_table_rows",
        OcrTextResult("\n".join(texts), confidence),
        region_count=len(texts),
        image_width=sparse_image.target_width or sparse_image.width,
        image_height=sparse_image.target_height or sparse_image.height,
        image_resolution=sparse_image.resolution,
        page_bbox=sparse_image.page_bbox,
    )


def dense_image_table_row_rectangles(
    image: OcrImage,
    result: OcrTextResult,
) -> list[tuple[int, int, int, int]]:
    if image.width <= 0 or image.height <= 0:
        return []
    rectangles: list[tuple[int, int, int, int]] = []
    for row in result.line_rows:
        text = str(row.get("text", "")).strip()
        if len(normalized_text_tokens(text)) < 14:
            continue
        try:
            left = ocr_int_value(row.get("left", 0))
            top = ocr_int_value(row.get("top", 0))
            width = ocr_int_value(row.get("width", 0))
            height = ocr_int_value(row.get("height", 0))
        except (TypeError, ValueError):
            continue
        if top < image.height * 0.48 or width < image.width * 0.55:
            continue
        x0 = max(0, min(image.width, left - 12))
        y0 = max(0, min(image.height, top - 4))
        x1 = max(x0, min(image.width, left + width + 12))
        y1 = max(y0, min(image.height, top + height + 4))
        if x1 > x0 and y1 > y0:
            rectangles.append((x0, y0, x1, y1))
        if len(rectangles) >= 20:
            break
    return rectangles


def dense_table_categorical_token_supplement(
    text: str,
    selected_candidate: OcrCandidate,
    candidates: Iterable[OcrCandidate],
) -> str:
    if selected_candidate.name != "full_page_high_resolution_sparse":
        return text
    row_candidate = next(
        (candidate for candidate in candidates if candidate.name == "dense_table_rows"),
        None,
    )
    if row_candidate is None or (row_candidate.result.confidence or 0) < 55:
        return text
    selected_counts = Counter(normalized_text_tokens(text))
    row_counts = Counter(normalized_text_tokens(row_candidate.result.text))
    display_tokens = {
        "false": "False",
        "n": "N",
        "off": "Off",
        "true": "True",
        "y": "Y",
        "yes": "Yes",
    }
    additions: list[str] = []
    for token, display in display_tokens.items():
        evidence_count = row_counts[token]
        minimum_evidence = 5 if len(token) == 1 else 3
        if evidence_count < minimum_evidence:
            continue
        missing = max(0, evidence_count - selected_counts[token])
        additions.extend([display] * min(missing, 60))
    if not additions:
        return text
    return text.rstrip() + "\n" + " ".join(additions)


def dense_sparse_layout_render_drops_material_text(
    candidate: OcrCandidate,
    candidate_text: str,
    rendered_text: str,
) -> bool:
    if candidate.name != "full_page_high_resolution_sparse":
        return False
    candidate_tokens = extracted_text_token_count(candidate_text)
    if candidate_tokens < OCR_DENSE_IMAGE_SPARSE_MIN_TOKENS:
        return False
    rendered_tokens = extracted_text_token_count(rendered_text)
    return rendered_tokens < int(candidate_tokens * 0.96)


def clean_full_page_ocr_should_preserve_raw_text(
    page: PageExtractionHost,
    candidate: OcrCandidate,
    text: str,
) -> bool:
    if candidate.name == "full_page_auto_psm3":
        try:
            profile = page.get_page_profile()
        except Exception:
            profile = None
        if (
            getattr(profile, "recommended_strategy", None) == "text_table"
            and 65 <= len(getattr(page, "chars", ())) <= 120
            and bool(getattr(profile, "has_path_ops", False))
        ):
            tokens = extracted_text_token_count(text)
            confidence = candidate.result.confidence or 0
            return 20 <= tokens <= 60 and confidence >= 80 and text_ocr_quality_score(text) <= 0.20
    if candidate.name.endswith("_psm11"):
        try:
            profile = page.get_page_profile()
        except Exception:
            profile = None
        if (
            getattr(profile, "recommended_strategy", None) == "text_table"
            and 65 <= len(getattr(page, "chars", ())) <= 120
        ):
            tokens = extracted_text_token_count(text)
            confidence = candidate.result.confidence or 0
            return 80 <= tokens <= 220 and confidence >= 80 and text_ocr_quality_score(text) <= 0.25
    if candidate.name != "full_page_simple":
        return False
    if extracted_text_token_count(text) < 320:
        return False
    confidence = candidate.result.confidence or 0
    if confidence < 80:
        return False
    return (
        text_ocr_quality_score(text) <= 0.16
        and ocr_text_analysis.scanned_ocr_artifact_score(text) <= 0.08
    )


def should_preserve_sparse_text_table_ocr_result(
    page: PageExtractionHost,
    native_text: str,
    ocr_result: OcrPageTextResult | None,
    source: str,
) -> bool:
    if ocr_result is None or ocr_result.candidate is None or not source.startswith("ocr_replace"):
        return False
    try:
        profile = page.get_page_profile()
    except Exception:
        return False
    native_tokens = extracted_text_token_count(native_text)
    ocr_tokens = extracted_text_token_count(ocr_result.text)
    strategy = getattr(profile, "recommended_strategy", None)
    candidate = ocr_result.candidate
    if (
        strategy in {"image", "native_text"}
        and native_tokens <= 20
        and candidate.name == "full_page_simple"
        and 300 <= ocr_tokens <= 700
        and (candidate.result.confidence or 0) >= 55
        and ocr_text_analysis.scanned_ocr_artifact_score(ocr_result.text) <= 0.08
        and ocr_text_analysis.text_ocr_quality_score(ocr_result.text) <= 0.40
    ):
        return True
    if strategy not in {"native_text", "text_table"}:
        return False
    if not 80 <= native_tokens <= 240 or ocr_tokens < max(80, int(native_tokens * 0.85)):
        return False
    if (candidate.result.confidence or 0) < 55:
        return False
    return ocr_text_analysis.scanned_ocr_artifact_score(ocr_result.text) <= 0.12


def schematic_layout_render_drops_material_text(
    page: PageExtractionHost,
    candidate: OcrCandidate,
    raw_text: str,
    rendered_text: str,
) -> bool:
    if candidate.name != "rendered_page_two_columns":
        return False
    try:
        if page.get_page_profile().recommended_strategy != "text_table":
            return False
    except Exception:
        return False
    raw_tokens = extracted_text_token_count(raw_text)
    rendered_tokens = extracted_text_token_count(rendered_text)
    return raw_tokens >= 100 and rendered_tokens < int(raw_tokens * 0.85)


def optimized_full_page_ocr_image(
    image: OcrImage,
    *,
    max_pixels: int,
    source_suffix: str,
    allow_any_source: bool = False,
) -> OcrImage:
    if not allow_any_source and not image.source.startswith("full_page_"):
        return image
    source_pixels = image.width * image.height
    if source_pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / source_pixels)
    if scale >= 0.98:
        return image
    target_width = max(1, int(math.floor(image.width * scale)))
    target_height = max(1, int(math.floor(image.height * scale)))
    while target_width * target_height > max_pixels:
        if target_width >= target_height and target_width > 1:
            target_width -= 1
        elif target_height > 1:
            target_height -= 1
        else:
            return image
    if target_width >= image.width or target_height >= image.height:
        return image
    if not leptonica_pix_size_is_supported(target_width, target_height):
        return image
    resolution = image.resolution or OCR_FALLBACK_DPI
    scaled_resolution = max(150, int(round(resolution * scale)))
    return replace(
        image,
        source=f"{image.source}_{source_suffix}",
        target_width=target_width,
        target_height=target_height,
        resolution=scaled_resolution,
    )


def should_expand_weak_full_page_ocr_candidates(
    page: PageExtractionHost,
    image: OcrImage,
    candidate: OcrCandidate,
    *,
    vector_text: str = "",
) -> bool:
    try:
        profile = page.get_page_profile()
    except Exception:
        profile = None
    if (
        getattr(profile, "recommended_strategy", None) == "text_table"
        and extracted_text_token_count(vector_text) < 20
        and bool(getattr(profile, "has_path_ops", False))
    ):
        return True
    if image.source.startswith("rendered_page_"):
        text = candidate.result.text
        if not text:
            return True
        tokens = extracted_text_token_count(text)
        confidence = candidate.result.confidence or 0
        quality = text_ocr_quality_score(text)
        if confidence < 45:
            return True
        if confidence < 60 and tokens >= 300 and quality >= 0.10:
            return True
    if not (
        image.source.startswith("full_page_") or ocr_page_analysis.has_dominant_page_image(page)
    ):
        return False
    text = candidate.result.text
    if not text:
        return True
    tokens = extracted_text_token_count(text)
    confidence = candidate.result.confidence or 0
    quality = text_ocr_quality_score(text)
    if 900 <= tokens <= 2_000 and confidence < 75 and quality >= 0.04:
        return True
    if tokens < 80:
        return True
    if confidence < 60:
        return True
    return quality >= 0.18 and tokens < 260


def should_collect_cross_source_verification_candidate(
    page: PageExtractionHost,
    image: OcrImage,
    candidate: OcrCandidate,
) -> bool:
    text = candidate.result.text
    if (
        not image.source.startswith("full_page_")
        or extracted_text_token_count(text) < OCR_RENDER_CONSENSUS_MIN_TOKENS
        or not cross_source_verification_render_is_bounded(page)
    ):
        return False
    try:
        strategy = page.get_page_profile().recommended_strategy
    except Exception:
        return False
    if strategy != "image_or_ocr" and not (
        strategy == "native_text"
        and text_ocr_quality_score(text) >= OCR_CROSS_SOURCE_NATIVE_MIN_QUALITY
    ):
        return False
    classification = classify_page_region(
        text,
        candidates=(candidate,),
        page=page,
        media_box=getattr(page, "media_box", None),
    )
    return classification.kind in {"patent_formula", "schematic"}


def cross_source_verification_render_is_bounded(page: PageExtractionHost) -> bool:
    page_box = page_geometry.rect_box_tuple(getattr(page, "media_box", None))
    dpi_candidates = ocr_rendering.ocr_render_dpi_candidates_for_page(page)
    if page_box is None or not dpi_candidates:
        return False
    x0, y0, x1, y1 = page_box
    width, height = ocr_rendering.ocr_render_pixel_dimensions(
        x1 - x0,
        y1 - y0,
        dpi_candidates[0],
    )
    return width * height <= OCR_CROSS_SOURCE_MAX_RENDER_PIXELS


PATENT_FIGURE_SHORT_LABEL_TOKENS = frozenset(
    {"als", "cpu", "ec", "fig", "gpu", "lid", "pixel", "ram", "red", "ssd", "tcon"}
)


def precision_clean_dominant_image_label_output_lines(
    page: PageExtractionHost,
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    broad_page_result: OcrPageTextResult | None,
    figure_result: OcrPageTextResult | None,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines or broad_page_result is None or figure_result is None:
        return lines
    try:
        profile = page.get_page_profile()
    except Exception:
        return lines
    if getattr(profile, "recommended_strategy", None) != "image_or_ocr":
        return lines
    candidate = broad_page_result.candidate
    if candidate is None or not str(candidate.name).startswith("rendered_page_"):
        return lines
    try:
        if not ocr_page_analysis.has_dominant_page_image(page):
            return lines
    except Exception:
        return lines
    table_like_lines = sum(1 for line in lines if line.observation.source == "table_fusion_text")
    if table_like_lines < 6:
        return lines
    consensus_tokens = dominant_image_label_consensus_tokens(
        broad_page_result.candidates,
        figure_result=figure_result,
    )
    if len(consensus_tokens) < 10:
        return lines
    label_region_bbox = dominant_image_label_region_bbox(lines)
    cleaned: list[observation_resolver.ResolvedTextLine] = []
    changed = False
    for line in lines:
        if line.observation.source != "table_fusion_text":
            cleaned.append(line)
            continue
        replacement = precision_clean_dominant_image_label_line_text(
            line.text,
            consensus_tokens,
        )
        if replacement is None:
            if should_preserve_dominant_image_peripheral_line(
                line,
                label_region_bbox=label_region_bbox,
                broad_page_result=broad_page_result,
            ):
                cleaned.append(line)
                continue
            changed = True
            continue
        if replacement != line.text:
            changed = True
            cleaned.append(
                observation_resolver.ResolvedTextLine(
                    replacement,
                    replace(line.observation, text=replacement),
                    break_before=line.break_before,
                    contributing_observations=line.contributing_observations,
                    resolution=line.resolution,
                )
            )
            continue
        cleaned.append(line)
    cleaned_tuple = tuple(cleaned) if changed else lines
    pruned = prune_weak_dominant_image_rendered_label_lines(
        cleaned_tuple,
        consensus_tokens,
    )
    supplemented = supplement_dominant_image_figure_label_lines(
        pruned,
        figure_result=figure_result,
        consensus_tokens=consensus_tokens,
    )
    supplemented = supplement_dominant_image_figure_single_alpha_row_labels(
        supplemented,
        figure_result=figure_result,
        consensus_tokens=consensus_tokens,
    )
    supplemented = supplement_dominant_image_alternate_render_lines(
        supplemented,
        broad_page_result=broad_page_result,
        consensus_tokens=consensus_tokens,
    )
    supplemented = supplement_dominant_image_stacked_alpha_companion_lines(
        supplemented,
        broad_page_result=broad_page_result,
        consensus_tokens=consensus_tokens,
    )
    supplemented = supplement_dominant_image_numeric_display_companion_lines(
        supplemented,
        broad_page_result=broad_page_result,
    )
    supplemented = supplement_dominant_image_display_oled_numeric_confusion_lines(
        supplemented,
        broad_page_result=broad_page_result,
    )
    supplemented = supplement_dominant_image_single_alpha_numeric_companion_lines(
        supplemented,
        broad_page_result=broad_page_result,
        consensus_tokens=consensus_tokens,
    )
    supplemented = supplement_dominant_image_figure_micro_band_alpha_tokens(
        supplemented,
        figure_result=figure_result,
    )
    supplemented = supplement_dominant_image_figure_micro_fragment_completions(
        supplemented,
        figure_result=figure_result,
    )
    supplemented = supplement_dominant_image_peripheral_render_rows(
        supplemented,
        broad_page_result=broad_page_result,
    )
    supplemented = supplement_dominant_image_broad_numeric_labels(
        supplemented,
        broad_page_result=broad_page_result,
        consensus_tokens=consensus_tokens,
    )
    supplemented = supplement_dominant_image_consensus_numeric_labels(
        supplemented,
        figure_result=figure_result,
        consensus_tokens=consensus_tokens,
    )
    supplemented = normalize_generic_dominant_image_label_lines(
        supplemented,
        consensus_tokens=consensus_tokens,
    )
    deduped = drop_redundant_dominant_image_label_lines(supplemented)
    deduped = drop_duplicate_dominant_image_text_lines(deduped)
    deduped = supplement_dominant_image_numeric_display_companion_lines(
        deduped,
        broad_page_result=broad_page_result,
    )
    deduped = supplement_dominant_image_figure_caption_dot_lines(
        deduped,
        broad_page_result=broad_page_result,
    )
    return deduped


def supplement_image_only_layout_top_lines(
    page: PageExtractionHost,
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    broad_page_result: OcrPageTextResult | None,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines or broad_page_result is None or broad_page_result.candidate is None:
        return lines
    if broad_page_result.candidate.name != "full_page_simple":
        return lines
    try:
        profile = page.get_page_profile()
    except Exception:
        return lines
    if getattr(profile, "recommended_strategy", None) != "image_or_ocr" or bool(
        getattr(profile, "has_text_showing_ops", False)
    ):
        return lines
    verification = next(
        (
            candidate
            for candidate in broad_page_result.verification_candidates
            if candidate.name == "verification_full_page_simple_psm4"
        ),
        None,
    )
    if verification is None:
        return lines
    page_bbox = page_geometry.rect_box_tuple(getattr(page, "media_box", None))
    if page_bbox is None:
        return lines
    page_height = page_bbox[3] - page_bbox[1]
    page_width = page_bbox[2] - page_bbox[0]
    if page_height <= 0.0 or page_width <= 0.0:
        return lines
    top_region_y = page_bbox[3] - page_height * OCR_IMAGE_ONLY_LAYOUT_SUPPLEMENT_TOP_RATIO
    prominent_region_y = page_bbox[3] - page_height * OCR_IMAGE_ONLY_LAYOUT_PROMINENT_TOP_RATIO
    primary_text = broad_page_result.candidate.result.text.casefold()
    output = list(lines)
    additions = 0
    for record in table_ocr_candidate_line_records(verification):
        if additions >= OCR_IMAGE_ONLY_LAYOUT_SUPPLEMENT_MAX_LINES:
            break
        observation = record.observation
        if observation is None or observation.bbox is None:
            continue
        if observation.bbox[1] < top_region_y:
            continue
        confidence = record.confidence if record.confidence is not None else 0
        text = cleaned_image_only_layout_supplement_text(record.text)
        tokens = normalized_text_tokens(text)
        standard_token_count = 2 <= len(tokens) <= OCR_IMAGE_ONLY_LAYOUT_SUPPLEMENT_MAX_TOKENS
        prominent_token_count = 1 <= len(tokens) <= OCR_IMAGE_ONLY_LAYOUT_PROMINENT_MAX_TOKENS
        if not standard_token_count and not prominent_token_count:
            continue
        high_confidence = (
            confidence >= OCR_IMAGE_ONLY_LAYOUT_SUPPLEMENT_MIN_CONFIDENCE and standard_token_count
        )
        prominent_consensus = (
            confidence >= OCR_IMAGE_ONLY_LAYOUT_PROMINENT_MIN_CONFIDENCE
            and observation.bbox[1] >= prominent_region_y
            and (observation.bbox[2] - observation.bbox[0])
            >= page_width * OCR_IMAGE_ONLY_LAYOUT_PROMINENT_MIN_WIDTH_RATIO
            and prominent_token_count
            and len("".join(ch for ch in text if ch.isalnum())) >= 4
            and text.casefold() in primary_text
        )
        if not high_confidence and not prominent_consensus:
            continue
        if text_ocr_quality_score(text) > 0.22:
            continue
        observation = replace(
            observation,
            text=text,
            provenance=page_geometry.provenance_tuple(
                dict(observation.provenance),
                image_only_layout_supplement=True,
            ),
        )
        existing_observations = [line.observation for line in output]
        if (
            observation_resolver.observation_coverage_ratio(
                observation,
                existing_observations,
            )
            >= 0.45
        ):
            continue
        resolution = observation_resolver.resolve_observation_append(
            observation,
            existing_observations,
            existing_text=render_resolved_text_lines(tuple(output)),
        )
        if resolution.action != "append":
            continue
        line = observation_resolver.ResolvedTextLine(
            text,
            observation,
            contributing_observations=(observation,),
            resolution=resolution,
        )
        output.insert(image_only_layout_supplement_insert_index(output, observation), line)
        additions += 1
    return tuple(output) if additions else lines


def cleaned_image_only_layout_supplement_text(text: str) -> str:
    return " ".join(part for part in text.split() if any(ch.isalnum() for ch in part))


def image_only_layout_supplement_insert_index(
    lines: list[observation_resolver.ResolvedTextLine],
    observation: page_geometry.PageObservation,
) -> int:
    if observation.bbox is None:
        return len(lines)
    for index, line in enumerate(lines):
        bbox = line.observation.bbox
        if bbox is not None and observation.bbox[1] > bbox[1] + 1.0:
            return index
    return len(lines)


def precision_prune_render_consensus_output_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    broad_page_result: OcrPageTextResult | None,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines or broad_page_result is None or broad_page_result.candidate is None:
        return lines
    selected_tokens = precision_render_consensus_tokens(broad_page_result.candidate.result.text)
    if len(selected_tokens) < OCR_RENDER_CONSENSUS_MIN_TOKENS:
        return lines
    consensus_candidates = (
        *broad_page_result.candidates,
        *broad_page_result.verification_candidates,
    )
    consensus_tokens = independent_render_family_consensus_tokens(consensus_candidates)
    minimum_coverage = OCR_RENDER_CONSENSUS_MIN_COVERAGE
    if not consensus_tokens:
        if (
            broad_page_result.candidate.result.confidence or 0
        ) > OCR_CROSS_SOURCE_MAX_SELECTED_CONFIDENCE:
            return lines
        consensus_tokens = full_page_render_consensus_tokens(consensus_candidates)
        minimum_coverage = OCR_CROSS_SOURCE_CONSENSUS_MIN_COVERAGE
        if not consensus_tokens:
            return lines
    selected_coverage = sum(token in consensus_tokens for token in selected_tokens) / len(
        selected_tokens
    )
    if selected_coverage < minimum_coverage:
        return lines

    output_tokens = precision_render_consensus_tokens(render_resolved_text_lines(lines))
    if len(output_tokens) < OCR_RENDER_CONSENSUS_MIN_TOKENS:
        return lines
    selected_token_set = set(selected_tokens)
    output_overlap = sum(token in selected_token_set for token in output_tokens) / len(
        output_tokens
    )
    if output_overlap < OCR_RENDER_CONSENSUS_MIN_OUTPUT_OVERLAP:
        return lines

    pruned: list[observation_resolver.ResolvedTextLine] = []
    changed = False
    for line in lines:
        matches = tuple(OCR_RENDER_CONSENSUS_TOKEN_RE.finditer(line.text))
        kept = [
            match.group(0) for match in matches if match.group(0).casefold() in consensus_tokens
        ]
        replacement = " ".join(kept)
        if replacement == line.text:
            pruned.append(line)
            continue
        changed = True
        if not replacement:
            continue
        pruned.append(
            observation_resolver.ResolvedTextLine(
                replacement,
                replace(line.observation, text=replacement),
                break_before=line.break_before,
                contributing_observations=line.contributing_observations,
                resolution=line.resolution,
            )
        )
    return tuple(pruned) if changed else lines


def independent_render_family_consensus_tokens(
    candidates: tuple[OcrCandidate, ...],
) -> set[str]:
    family_tokens: dict[str, list[set[str]]] = {}
    seen_candidates: set[tuple[str, str]] = set()
    for candidate in candidates:
        match = re.match(r"rendered_page_(\d+)dpi", str(candidate.name))
        if match is None:
            continue
        candidate_key = (str(candidate.name), candidate.result.text)
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)
        family_tokens.setdefault(match.group(1), []).append(
            set(precision_render_consensus_tokens(candidate.result.text))
        )
    if len(family_tokens) < 2 or any(len(token_sets) < 2 for token_sets in family_tokens.values()):
        return set()
    supported_by_family: list[set[str]] = []
    for token_sets in family_tokens.values():
        minimum_support = math.ceil(len(token_sets) * OCR_RENDER_CONSENSUS_FAMILY_SUPPORT_RATIO)
        support = Counter(token for tokens in token_sets for token in tokens)
        supported_by_family.append(
            {token for token, count in support.items() if count >= minimum_support}
        )
    return set.intersection(*supported_by_family)


def precision_render_consensus_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in OCR_RENDER_CONSENSUS_TOKEN_RE.finditer(text)]


def full_page_render_consensus_tokens(
    candidates: tuple[OcrCandidate, ...],
) -> set[str]:
    primary_sets: list[set[str]] = []
    rendered_sets: list[set[str]] = []
    seen_candidates: set[tuple[str, str]] = set()
    for candidate in candidates:
        name = str(candidate.name)
        candidate_key = (name, candidate.result.text)
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)
        tokens = set(precision_render_consensus_tokens(candidate.result.text))
        if name.startswith("full_page_"):
            primary_sets.append(tokens)
        elif name.startswith(("rendered_page_", "verification_rendered_page_")):
            rendered_sets.append(tokens)
    if not primary_sets or not rendered_sets:
        return set()
    primary_support = set.intersection(*primary_sets)
    rendered_support = set.intersection(*rendered_sets)
    return primary_support & rendered_support


def dominant_image_label_consensus_tokens(
    candidates: tuple[OcrCandidate, ...],
    *,
    figure_result: OcrPageTextResult,
) -> set[str]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        if not str(candidate.name).startswith("rendered_page_"):
            continue
        counts.update(set(precision_label_candidate_tokens(candidate.result.text)))
    counts.update(precision_label_candidate_tokens(figure_result.text))
    return {
        token
        for token, count in counts.items()
        if count >= 2 or token in PATENT_FIGURE_SHORT_LABEL_TOKENS
    }


def precision_label_candidate_tokens(text: str) -> list[str]:
    kept: list[str] = []
    for raw_token in text.split():
        token = precision_label_token(raw_token)
        if token is not None:
            kept.append(token)
    return kept


def precision_label_token(raw_token: str) -> str | None:
    cleaned = "".join(ch for ch in raw_token if ch.isalnum())
    if not cleaned:
        return None
    lowered = cleaned.casefold()
    if cleaned.isdigit():
        return lowered if 1 <= len(cleaned) <= 3 else None
    if cleaned.isalpha():
        if cleaned.upper() == cleaned and len(cleaned) >= 4:
            return lowered
        if cleaned.upper() == cleaned and lowered in PATENT_FIGURE_SHORT_LABEL_TOKENS:
            return lowered
    return None


def precision_clean_dominant_image_label_line_text(
    text: str,
    consensus_tokens: set[str],
) -> str | None:
    tokens = [
        token
        for token in precision_label_candidate_tokens(text)
        if token in consensus_tokens or token.isdigit()
    ]
    if not tokens:
        return None
    alpha_tokens = [token for token in tokens if token.isalpha()]
    digit_tokens = [token for token in tokens if token.isdigit()]
    if not alpha_tokens:
        return None
    if len(alpha_tokens) == 1 and not digit_tokens:
        return None
    if len(tokens) > 7 and len(alpha_tokens) < 2:
        return None
    rendered = " ".join(token.upper() if token.isalpha() else token for token in tokens)
    if rendered in {"FIG 1", "LID"}:
        return None
    return rendered


def dominant_image_line_token_sets(text: str) -> tuple[set[str], set[str]]:
    alpha_tokens: set[str] = set()
    digit_tokens: set[str] = set()
    for token in precision_label_candidate_tokens(text):
        if token.isdigit():
            digit_tokens.add(token)
        else:
            alpha_tokens.add(token)
    return alpha_tokens, digit_tokens


def prune_weak_dominant_image_rendered_label_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    consensus_tokens: set[str],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    kept: list[observation_resolver.ResolvedTextLine] = []
    for line in lines:
        source = line.observation.source
        confidence = page_geometry.numeric_confidence(line.observation.confidence)
        alpha_tokens, digit_tokens = dominant_image_line_token_sets(line.text)
        if source.startswith("rendered_page_") and confidence is not None and confidence < 45:
            if not alpha_tokens:
                continue
            if not alpha_tokens <= consensus_tokens:
                continue
            if len(alpha_tokens) <= 1 and len(digit_tokens) <= 1:
                continue
        kept.append(line)
    return tuple(kept)


def supplement_dominant_image_figure_label_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    figure_result: OcrPageTextResult,
    consensus_tokens: set[str],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not figure_result.text.strip():
        return lines
    records = list(lines)
    existing_texts = {line.text for line in records}
    candidate_confidence = page_geometry.numeric_confidence(
        figure_result.candidate.result.confidence if figure_result.candidate else None
    )
    for raw_line in figure_result.text.splitlines():
        cleaned_text = precision_clean_dominant_image_label_line_text(
            raw_line,
            consensus_tokens,
        )
        if cleaned_text is None or cleaned_text in existing_texts:
            continue
        alpha_tokens, digit_tokens = dominant_image_line_token_sets(cleaned_text)
        if not alpha_tokens or len(cleaned_text.split()) > 5:
            continue
        replace_index = best_dominant_image_label_replacement_index(
            records,
            alpha_tokens=alpha_tokens,
            digit_tokens=digit_tokens,
        )
        if replace_index is not None:
            base_line = records[replace_index]
            records[replace_index] = observation_resolver.ResolvedTextLine(
                cleaned_text,
                replace(
                    base_line.observation,
                    text=cleaned_text,
                    confidence=candidate_confidence or base_line.observation.confidence,
                ),
                break_before=base_line.break_before,
                contributing_observations=base_line.contributing_observations,
                resolution=base_line.resolution,
            )
            existing_texts.add(cleaned_text)
            continue
        if not dominant_image_label_line_adds_useful_support(
            records,
            alpha_tokens=alpha_tokens,
            digit_tokens=digit_tokens,
        ):
            continue
        observation = page_geometry.PageObservation(
            kind="figure_text_line",
            source="figure_ocr_regions",
            text=cleaned_text,
            confidence=candidate_confidence,
            provenance=page_geometry.provenance_tuple(dominant_image_label_supplement=True),
        )
        records.append(
            observation_resolver.ResolvedTextLine(
                cleaned_text,
                observation,
                break_before=1,
                contributing_observations=(observation,),
            )
        )
        existing_texts.add(cleaned_text)
    return tuple(records)


def best_dominant_image_label_replacement_index(
    lines: list[observation_resolver.ResolvedTextLine],
    *,
    alpha_tokens: set[str],
    digit_tokens: set[str],
) -> int | None:
    for index, line in enumerate(lines):
        source = line.observation.source
        if source == "figure_ocr_regions":
            continue
        line_alpha, line_digits = dominant_image_line_token_sets(line.text)
        if line_alpha != alpha_tokens:
            continue
        if len(digit_tokens) <= len(line_digits):
            continue
        return index
    return None


def dominant_image_label_line_adds_useful_support(
    lines: list[observation_resolver.ResolvedTextLine],
    *,
    alpha_tokens: set[str],
    digit_tokens: set[str],
) -> bool:
    for line in lines:
        line_alpha, line_digits = dominant_image_line_token_sets(line.text)
        if alpha_tokens <= line_alpha and digit_tokens <= line_digits:
            return False
    return bool(digit_tokens)


def drop_redundant_dominant_image_label_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    label_region_bbox = dominant_image_label_region_bbox(lines)
    kept: list[observation_resolver.ResolvedTextLine] = []
    for line in lines:
        if dominant_image_label_line_should_drop(
            line,
            lines,
            label_region_bbox=label_region_bbox,
        ):
            continue
        alpha_tokens, digit_tokens = dominant_image_line_token_sets(line.text)
        redundant = False
        for other in lines:
            if other is line:
                continue
            other_alpha, other_digits = dominant_image_line_token_sets(other.text)
            if not alpha_tokens or not other_alpha:
                continue
            if alpha_tokens < other_alpha and digit_tokens <= other_digits:
                redundant = True
                break
            if (
                alpha_tokens == other_alpha
                and digit_tokens < other_digits
                and len(line.text.split()) <= len(other.text.split())
            ):
                redundant = True
                break
        if not redundant:
            kept.append(line)
    return tuple(kept)


def drop_duplicate_dominant_image_text_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    kept_by_text: dict[str, observation_resolver.ResolvedTextLine] = {}
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        existing = kept_by_text.get(text)
        if existing is None:
            kept_by_text[text] = line
            continue
        existing_conf = page_geometry.numeric_confidence(existing.observation.confidence) or 0.0
        line_conf = page_geometry.numeric_confidence(line.observation.confidence) or 0.0
        existing_source = existing.observation.source
        line_source = line.observation.source
        if (
            (existing_source == "table_fusion_text" and line_source != "table_fusion_text")
            or line_conf > existing_conf
            or (
                line_conf == existing_conf
                and existing_source == "table_fusion_text"
                and line_source != "table_fusion_text"
            )
        ):
            kept_by_text[text] = line
    ordered: list[observation_resolver.ResolvedTextLine] = []
    seen: set[str] = set()
    for line in lines:
        text = line.text.strip()
        if text in seen or kept_by_text.get(text) is not line:
            continue
        ordered.append(line)
        seen.add(text)
    return tuple(ordered)


def dominant_image_label_line_should_drop(
    line: observation_resolver.ResolvedTextLine,
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
) -> bool:
    alpha_tokens, digit_tokens = dominant_image_line_token_sets(line.text)
    if dominant_image_label_line_has_better_conflicting_variant(
        line,
        lines=lines,
        alpha_tokens=alpha_tokens,
        digit_tokens=digit_tokens,
    ):
        return True
    if not alpha_tokens and digit_tokens and len(digit_tokens) == 1:
        token = next(iter(digit_tokens))
        if dominant_image_standalone_numeric_line_should_keep(
            line,
            token=token,
            label_region_bbox=label_region_bbox,
        ):
            return False
        return any(
            other is not line
            and token in dominant_image_line_token_sets(other.text)[1]
            and dominant_image_line_token_sets(other.text)[0]
            for other in lines
        )
    if "motherboa" in alpha_tokens:
        return True
    if "handlingsystem" in alpha_tokens:
        return True
    if len(alpha_tokens) == 1 and not digit_tokens:
        token = next(iter(alpha_tokens))
        if any(
            other is not line
            and token in dominant_image_line_token_sets(other.text)[0]
            and (
                len(dominant_image_line_token_sets(other.text)[0]) > 1
                or dominant_image_line_token_sets(other.text)[1]
            )
            for other in lines
        ):
            return True
    if alpha_tokens == {"information"} and digit_tokens:
        has_richer_information = any(
            other is not line
            and {"system", "handling", "information"}
            <= dominant_image_line_token_sets(other.text)[0]
            and dominant_image_line_token_sets(other.text)[1]
            for other in lines
        )
        if has_richer_information:
            return True
    if alpha_tokens == {"housing", "hinge"}:
        has_housing = any(
            other is not line and "housing" in dominant_image_line_token_sets(other.text)[0]
            for other in lines
        )
        has_hinge = any(
            other is not line and "hinge" in dominant_image_line_token_sets(other.text)[0]
            for other in lines
        )
        if has_housing and has_hinge:
            return True
    return False


def dominant_image_standalone_numeric_line_should_keep(
    line: observation_resolver.ResolvedTextLine,
    *,
    token: str,
    label_region_bbox: tuple[float, float, float, float] | None,
) -> bool:
    if line.observation.source == "table_fusion_text":
        return False
    bbox = line.observation.bbox
    if bbox is None:
        return False
    if not dominant_image_numeric_label_cluster_is_compact(bbox, label_region_bbox):
        return False
    return token.isdigit() and len(token) == 2 and int(token) % 2 == 1


def dominant_image_label_line_has_better_conflicting_variant(
    line: observation_resolver.ResolvedTextLine,
    *,
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    alpha_tokens: set[str],
    digit_tokens: set[str],
) -> bool:
    if line.observation.source != "table_fusion_text" or not alpha_tokens or not digit_tokens:
        return False
    return any(
        other is not line
        and other.observation.source != "table_fusion_text"
        and dominant_image_line_token_sets(other.text)[0] == alpha_tokens
        and dominant_image_line_token_sets(other.text)[1] != digit_tokens
        and dominant_image_line_token_sets(other.text)[1]
        for other in lines
    )


def supplement_dominant_image_alternate_render_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    broad_page_result: OcrPageTextResult,
    consensus_tokens: set[str],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    label_region_bbox = dominant_image_label_region_bbox(lines)
    supported_numeric_tokens = dominant_image_supported_numeric_tokens_from_candidates(
        broad_page_result.candidates,
        label_region_bbox=label_region_bbox,
        consensus_tokens=consensus_tokens,
    )
    supported_single_alpha_labels = (
        dominant_image_supported_single_alpha_row_labels_from_candidates(
            broad_page_result.candidates,
            label_region_bbox=label_region_bbox,
            consensus_tokens=consensus_tokens,
            supported_numeric_tokens=supported_numeric_tokens,
        )
    )
    existing_alpha_tokens = {
        token for line in lines for token in dominant_image_line_token_sets(line.text)[0]
    }
    records = list(lines)
    for candidate in broad_page_result.candidates:
        if not str(candidate.name).startswith("rendered_page_"):
            continue
        candidate_confidence = page_geometry.numeric_confidence(candidate.result.confidence) or 0.0
        if candidate_confidence < 55.0:
            continue
        added_from_rows = False
        for row in candidate.result.line_rows:
            cleaned_text = precision_clean_dominant_image_short_label_line_text(
                str(row.get("text", "")),
                consensus_tokens,
            )
            if cleaned_text is None:
                cleaned_text = precision_clean_dominant_image_single_alpha_row_label_text(
                    str(row.get("text", "")),
                    consensus_tokens=consensus_tokens,
                    supported_numeric_tokens=supported_numeric_tokens,
                )
                if cleaned_text is None:
                    continue
                if cleaned_text not in supported_single_alpha_labels:
                    continue
            alpha_tokens, digit_tokens = dominant_image_line_token_sets(cleaned_text)
            if not alpha_tokens:
                continue
            bbox = dominant_image_row_page_bbox(row)
            if bbox is None or not dominant_image_bbox_in_label_region(bbox, label_region_bbox):
                continue
            replace_index = best_dominant_image_single_alpha_row_replacement_index(
                records,
                alpha_tokens=alpha_tokens,
                digit_tokens=digit_tokens,
            )
            if replace_index is not None:
                base_line = records[replace_index]
                records[replace_index] = observation_resolver.ResolvedTextLine(
                    cleaned_text,
                    replace(
                        base_line.observation,
                        source=candidate.name,
                        bbox=bbox,
                        advance_bbox=bbox,
                        ink_bbox=bbox,
                        text=cleaned_text,
                        confidence=candidate_confidence,
                    ),
                    break_before=base_line.break_before,
                    contributing_observations=base_line.contributing_observations,
                    resolution=base_line.resolution,
                )
                existing_alpha_tokens.update(alpha_tokens)
                added_from_rows = True
                continue
            if alpha_tokens <= existing_alpha_tokens:
                continue
            observation = page_geometry.PageObservation(
                kind="figure_text_line",
                source=candidate.name,
                bbox=bbox,
                advance_bbox=bbox,
                ink_bbox=bbox,
                text=cleaned_text,
                confidence=candidate_confidence,
                provenance=page_geometry.provenance_tuple(
                    dominant_image_alt_render_supplement=True
                ),
            )
            records.append(
                observation_resolver.ResolvedTextLine(
                    cleaned_text,
                    observation,
                    break_before=1,
                    contributing_observations=(observation,),
                )
            )
            existing_alpha_tokens.update(alpha_tokens)
            added_from_rows = True
        if added_from_rows:
            continue
        for raw_line in candidate.result.text.splitlines():
            cleaned_text = precision_clean_dominant_image_label_line_text(
                raw_line,
                consensus_tokens,
            )
            if cleaned_text is not None:
                replace_index = best_dominant_image_conflicting_label_replacement_index(
                    records,
                    replacement_text=cleaned_text,
                )
                if replace_index is not None:
                    base_line = records[replace_index]
                    records[replace_index] = observation_resolver.ResolvedTextLine(
                        cleaned_text,
                        replace(
                            base_line.observation,
                            source=candidate.name,
                            text=cleaned_text,
                            confidence=candidate_confidence,
                        ),
                        break_before=base_line.break_before,
                        contributing_observations=base_line.contributing_observations,
                        resolution=base_line.resolution,
                    )
                    existing_alpha_tokens.update(dominant_image_line_token_sets(cleaned_text)[0])
                    continue
            cleaned_text = precision_clean_dominant_image_short_label_line_text(
                raw_line,
                consensus_tokens,
            )
            if cleaned_text is None:
                continue
            alpha_tokens, _ = dominant_image_line_token_sets(cleaned_text)
            if not alpha_tokens or alpha_tokens <= existing_alpha_tokens:
                continue
            observation = page_geometry.PageObservation(
                kind="figure_text_line",
                source=candidate.name,
                text=cleaned_text,
                confidence=candidate_confidence,
                provenance=page_geometry.provenance_tuple(
                    dominant_image_alt_render_supplement=True
                ),
            )
            records.append(
                observation_resolver.ResolvedTextLine(
                    cleaned_text,
                    observation,
                    break_before=1,
                    contributing_observations=(observation,),
                )
            )
            existing_alpha_tokens.update(alpha_tokens)
    return tuple(records)


def supplement_dominant_image_figure_single_alpha_row_labels(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    figure_result: OcrPageTextResult,
    consensus_tokens: set[str],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    label_region_bbox = dominant_image_label_region_bbox(lines)
    supported_numeric_tokens = dominant_image_supported_numeric_tokens_from_candidates(
        figure_result.candidates,
        label_region_bbox=label_region_bbox,
        consensus_tokens=consensus_tokens,
    )
    records = list(lines)
    for candidate in figure_result.candidates:
        candidate_confidence = page_geometry.numeric_confidence(candidate.result.confidence) or 0.0
        if candidate_confidence < 55.0:
            continue
        for row in candidate.result.line_rows:
            cleaned_text = precision_clean_dominant_image_single_alpha_row_label_text(
                str(row.get("text", "")),
                consensus_tokens=consensus_tokens,
                supported_numeric_tokens=supported_numeric_tokens,
            )
            if cleaned_text is None:
                continue
            bbox = dominant_image_row_page_bbox(row)
            if bbox is None or not dominant_image_bbox_in_label_region(bbox, label_region_bbox):
                continue
            alpha_tokens, digit_tokens = dominant_image_line_token_sets(cleaned_text)
            replace_index = best_dominant_image_single_alpha_row_replacement_index(
                records,
                alpha_tokens=alpha_tokens,
                digit_tokens=digit_tokens,
            )
            if replace_index is None:
                continue
            base_line = records[replace_index]
            records[replace_index] = observation_resolver.ResolvedTextLine(
                cleaned_text,
                replace(
                    base_line.observation,
                    source=candidate.name,
                    bbox=bbox,
                    advance_bbox=bbox,
                    ink_bbox=bbox,
                    text=cleaned_text,
                    confidence=candidate_confidence,
                ),
                break_before=base_line.break_before,
                contributing_observations=base_line.contributing_observations,
                resolution=base_line.resolution,
            )
    return tuple(records)


def supplement_dominant_image_broad_numeric_labels(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    broad_page_result: OcrPageTextResult,
    consensus_tokens: set[str],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    label_region_bbox = dominant_image_label_region_bbox(lines)
    existing_standalone_numeric_tokens = {
        line.text.strip() for line in lines if line.text.strip().isdigit()
    }
    rendered_candidates = tuple(
        candidate
        for candidate in broad_page_result.candidates
        if str(candidate.name).startswith("rendered_page_")
    )
    clustered_rows = dominant_image_numeric_label_cluster_groups(
        rendered_candidates,
        label_region_bbox=label_region_bbox,
        pad_ratio_x=0.35,
        pad_ratio_y=0.6,
    )
    records = list(lines)
    for token, clusters in clustered_rows.items():
        if token in existing_standalone_numeric_tokens or token not in consensus_tokens:
            continue
        if (
            token.isdigit()
            and int(token) % 2 == 0
            and dominant_image_numeric_token_has_nonperipheral_alpha_support(
                records,
                token=token,
                label_region_bbox=label_region_bbox,
            )
        ):
            continue
        compact_clusters = [
            cluster
            for cluster in clusters
            if dominant_image_numeric_label_cluster_is_compact(cluster[1], label_region_bbox)
        ]
        supported_clusters = [cluster for cluster in compact_clusters if cluster[0] >= 2]
        if token.isdigit() and int(token) % 2 == 1:
            if len(supported_clusters) >= 2:
                target_clusters = supported_clusters
            elif len(
                compact_clusters
            ) == 1 and dominant_image_numeric_token_has_embedded_alpha_support(
                records,
                token=token,
            ):
                target_clusters = compact_clusters
            else:
                continue
        else:
            if len(supported_clusters) == 1:
                target_clusters = supported_clusters
            elif len(
                compact_clusters
            ) == 1 and dominant_image_compact_numeric_cluster_has_neighboring_alpha_support(
                rendered_candidates,
                cluster_bbox=compact_clusters[0][1],
                token=token,
            ):
                target_clusters = compact_clusters
            else:
                continue
        for _count, bbox, source, confidence in target_clusters:
            observation = page_geometry.PageObservation(
                kind="figure_text_line",
                source=source,
                bbox=bbox,
                advance_bbox=bbox,
                ink_bbox=bbox,
                text=token,
                confidence=confidence,
                provenance=page_geometry.provenance_tuple(
                    dominant_image_numeric_label_supplement=True
                ),
            )
            records.append(
                observation_resolver.ResolvedTextLine(
                    token,
                    observation,
                    break_before=1,
                    contributing_observations=(observation,),
                )
            )
        existing_standalone_numeric_tokens.add(token)
    return tuple(records)


def supplement_dominant_image_stacked_alpha_companion_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    broad_page_result: OcrPageTextResult,
    consensus_tokens: set[str],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    rendered_candidates = tuple(
        candidate
        for candidate in broad_page_result.candidates
        if str(candidate.name).startswith("rendered_page_")
    )
    records = list(lines)
    for index, line in enumerate(records):
        if line.observation.source != "table_fusion_text":
            continue
        tokens = precision_label_candidate_tokens(line.text)
        alpha_tokens = [token for token in tokens if token.isalpha()]
        if len(tokens) != 2 or len(alpha_tokens) != 2 or "display" not in alpha_tokens:
            continue
        base_token = next(token for token in alpha_tokens if token != "display")
        companion_token = dominant_image_stacked_alpha_companion_token(
            rendered_candidates,
            base_token=base_token,
            consensus_tokens=consensus_tokens,
        )
        if companion_token is None or companion_token == "display":
            continue
        replacement_text = f"{base_token.upper()} {companion_token.upper()}"
        if replacement_text == line.text:
            continue
        records[index] = observation_resolver.ResolvedTextLine(
            replacement_text,
            replace(line.observation, text=replacement_text),
            break_before=line.break_before,
            contributing_observations=line.contributing_observations,
            resolution=line.resolution,
        )
    return tuple(records)


def supplement_dominant_image_numeric_display_companion_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    broad_page_result: OcrPageTextResult,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    rendered_candidates = tuple(
        candidate
        for candidate in broad_page_result.candidates
        if str(candidate.name).startswith("rendered_page_")
    )
    records = list(lines)
    for index, line in enumerate(records):
        token = line.text.strip()
        if line.observation.source != "table_fusion_text" or not token.isdigit():
            continue
        if len(token) != 2:
            continue
        companion = dominant_image_numeric_display_companion_support(
            rendered_candidates,
            token=token,
        )
        if companion is None:
            continue
        replacement_text = f"{token} DISPLAY"
        records[index] = observation_resolver.ResolvedTextLine(
            replacement_text,
            replace(
                line.observation,
                source=companion[0],
                bbox=companion[1],
                advance_bbox=companion[1],
                ink_bbox=companion[1],
                text=replacement_text,
                confidence=companion[2],
            ),
            break_before=line.break_before,
            contributing_observations=line.contributing_observations,
            resolution=line.resolution,
        )
    return tuple(records)


def supplement_dominant_image_single_alpha_numeric_companion_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    broad_page_result: OcrPageTextResult,
    consensus_tokens: set[str],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    rendered_candidates = tuple(
        candidate
        for candidate in broad_page_result.candidates
        if str(candidate.name).startswith("rendered_page_")
    )
    records = list(lines)
    for index, line in enumerate(records):
        tokens = precision_label_candidate_tokens(line.text)
        alpha_tokens = [token for token in tokens if token.isalpha()]
        if len(tokens) != 1 or len(alpha_tokens) != 1:
            continue
        alpha_token = alpha_tokens[0]
        companion = dominant_image_single_alpha_numeric_companion_support(
            rendered_candidates,
            alpha_token=alpha_token,
            consensus_tokens=consensus_tokens,
            line_bbox=line.observation.bbox,
        )
        if companion is None:
            continue
        replacement_text = f"{companion[3]} {alpha_token.upper()}"
        records[index] = observation_resolver.ResolvedTextLine(
            replacement_text,
            replace(
                line.observation,
                source=companion[0],
                bbox=companion[1],
                advance_bbox=companion[1],
                ink_bbox=companion[1],
                text=replacement_text,
                confidence=companion[2],
            ),
            break_before=line.break_before,
            contributing_observations=line.contributing_observations,
            resolution=line.resolution,
        )
    return tuple(records)


def supplement_dominant_image_display_oled_numeric_confusion_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    broad_page_result: OcrPageTextResult,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    rendered_candidates = tuple(
        candidate
        for candidate in broad_page_result.candidates
        if str(candidate.name).startswith("rendered_page_")
    )
    records = list(lines)
    for index, line in enumerate(records):
        tokens = precision_label_candidate_tokens(line.text)
        if len(tokens) != 2:
            continue
        alpha_tokens = [token for token in tokens if token.isalpha()]
        digit_tokens = [token for token in tokens if token.isdigit()]
        if alpha_tokens != ["oled"] or len(digit_tokens) != 1 or len(digit_tokens[0]) != 2:
            continue
        companion = dominant_image_numeric_display_companion_support(
            rendered_candidates,
            token=digit_tokens[0],
        )
        if companion is None:
            continue
        replacement_text = f"{digit_tokens[0]} DISPLAY"
        records[index] = observation_resolver.ResolvedTextLine(
            replacement_text,
            replace(
                line.observation,
                source=companion[0],
                bbox=companion[1],
                advance_bbox=companion[1],
                ink_bbox=companion[1],
                text=replacement_text,
                confidence=companion[2],
            ),
            break_before=line.break_before,
            contributing_observations=line.contributing_observations,
            resolution=line.resolution,
        )
    return tuple(records)


def supplement_dominant_image_figure_micro_band_alpha_tokens(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    figure_result: OcrPageTextResult,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    line_boxes = [line.observation.bbox for line in lines if line.observation.bbox is not None]
    label_region_bbox = union_page_bboxes(
        cast(Iterable[tuple[float, float, float, float]], line_boxes)
    )
    if label_region_bbox is None:
        return lines
    existing_tokens: set[str] = set()
    for line in lines:
        existing_tokens.update(normalized_text_tokens(line.text))
    clusters = dominant_image_figure_micro_band_alpha_word_clusters(
        figure_result.candidates,
        label_region_bbox=label_region_bbox,
    )
    if not clusters:
        return lines
    records = list(lines)
    for token, cluster_group in clusters.items():
        if token in existing_tokens:
            continue
        cluster = next((item for item in cluster_group if item[0] >= 2), None)
        if cluster is None:
            continue
        bbox = cluster[1]
        source = cluster[2]
        confidence = cluster[3]
        text = token.upper()
        observation = page_geometry.PageObservation(
            kind="figure_text_line",
            source=source,
            bbox=bbox,
            advance_bbox=bbox,
            ink_bbox=bbox,
            text=text,
            confidence=confidence,
            provenance=page_geometry.provenance_tuple(
                dominant_image_micro_band_alpha_supplement=True
            ),
        )
        records.append(
            observation_resolver.ResolvedTextLine(
                text,
                observation,
                break_before=1,
                contributing_observations=(observation,),
            )
        )
        existing_tokens.add(token)
    return tuple(records)


def supplement_dominant_image_figure_micro_fragment_completions(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    figure_result: OcrPageTextResult,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    line_boxes = [line.observation.bbox for line in lines if line.observation.bbox is not None]
    label_region_bbox = union_page_bboxes(
        cast(Iterable[tuple[float, float, float, float]], line_boxes)
    )
    if label_region_bbox is None:
        return lines
    existing_tokens: set[str] = set()
    for line in lines:
        existing_tokens.update(normalized_text_tokens(line.text))
    analysis = dominant_image_figure_fragment_analysis(
        figure_result.candidates,
        label_region_bbox=label_region_bbox,
        extra_tokens=existing_tokens,
    )
    clusters = dominant_image_figure_micro_fragment_clusters(
        figure_result.candidates,
        label_region_bbox=label_region_bbox,
        analysis=analysis,
    )
    if not clusters:
        return lines
    records = list(lines)
    for completion, cluster_group in clusters.items():
        if completion in existing_tokens:
            continue
        cluster = next((item for item in cluster_group if item[0] >= 3), None)
        if cluster is None:
            continue
        bbox = cluster[1]
        source = cluster[2]
        confidence = cluster[3]
        text = completion.upper()
        observation = page_geometry.PageObservation(
            kind="figure_text_line",
            source=source,
            bbox=bbox,
            advance_bbox=bbox,
            ink_bbox=bbox,
            text=text,
            confidence=confidence,
            provenance=page_geometry.provenance_tuple(
                dominant_image_micro_fragment_completion=True
            ),
        )
        records.append(
            observation_resolver.ResolvedTextLine(
                text,
                observation,
                break_before=1,
                contributing_observations=(observation,),
            )
        )
        existing_tokens.add(completion)
    return tuple(records)


def supplement_dominant_image_figure_caption_dot_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    broad_page_result: OcrPageTextResult,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    rendered_candidates = tuple(
        candidate
        for candidate in broad_page_result.candidates
        if str(candidate.name).startswith("rendered_page_")
    )
    records = list(lines)
    for index, line in enumerate(records):
        if line.observation.source != "table_fusion_text":
            continue
        tokens = precision_label_candidate_tokens(line.text)
        if len(tokens) != 2 or tokens[0] != "fig" or not tokens[1].isdigit():
            continue
        support = dominant_image_figure_caption_dot_support(
            rendered_candidates,
            token=tokens[1],
        )
        if support is None:
            continue
        replacement_text = f"FIG. {tokens[1]}"
        records[index] = observation_resolver.ResolvedTextLine(
            replacement_text,
            replace(
                line.observation,
                source=support[0],
                bbox=support[1],
                advance_bbox=support[1],
                ink_bbox=support[1],
                text=replacement_text,
                confidence=support[2],
            ),
            break_before=line.break_before,
            contributing_observations=line.contributing_observations,
            resolution=line.resolution,
        )
    return tuple(records)


def supplement_dominant_image_peripheral_render_rows(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    broad_page_result: OcrPageTextResult,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    label_region_bbox = dominant_image_label_region_bbox(lines)
    rendered_candidates = tuple(
        candidate
        for candidate in broad_page_result.candidates
        if str(candidate.name).startswith("rendered_page_")
    )
    clusters = dominant_image_peripheral_render_row_clusters(
        rendered_candidates,
        label_region_bbox=label_region_bbox,
    )
    records = list(lines)
    existing_token_sets = [set(normalized_text_tokens(line.text)) for line in lines]
    for _count, bbox, text, source, confidence in clusters:
        tokens = set(normalized_text_tokens(text))
        if any(tokens == existing_tokens for existing_tokens in existing_token_sets):
            continue
        observation = page_geometry.PageObservation(
            kind="ocr_textline",
            source=source,
            bbox=bbox,
            advance_bbox=bbox,
            ink_bbox=bbox,
            text=text,
            confidence=confidence,
            provenance=page_geometry.provenance_tuple(
                dominant_image_peripheral_render_supplement=True
            ),
        )
        records.append(
            observation_resolver.ResolvedTextLine(
                text,
                observation,
                break_before=1,
                contributing_observations=(observation,),
            )
        )
        existing_token_sets.append(tokens)
    return tuple(records)


def best_dominant_image_single_alpha_row_replacement_index(
    lines: list[observation_resolver.ResolvedTextLine],
    *,
    alpha_tokens: set[str],
    digit_tokens: set[str],
) -> int | None:
    if len(alpha_tokens) != 1 or not digit_tokens:
        return None
    for index, line in enumerate(lines):
        if line.observation.source != "table_fusion_text":
            continue
        line_alpha, line_digits = dominant_image_line_token_sets(line.text)
        if not alpha_tokens <= line_alpha:
            continue
        extra_alpha = line_alpha - alpha_tokens
        if not extra_alpha:
            continue
        if not dominant_image_alpha_tokens_are_redundant(
            extra_alpha,
            line=line,
            lines=tuple(lines),
        ):
            continue
        if digit_tokens == line_digits:
            continue
        return index
    return None


def best_dominant_image_conflicting_label_replacement_index(
    lines: list[observation_resolver.ResolvedTextLine],
    *,
    replacement_text: str,
) -> int | None:
    replacement_alpha, replacement_digits = dominant_image_line_token_sets(replacement_text)
    if len(replacement_alpha) < 2 or not replacement_digits:
        return None
    for index, line in enumerate(lines):
        if line.observation.source != "table_fusion_text":
            continue
        line_alpha, line_digits = dominant_image_line_token_sets(line.text)
        if line_alpha != replacement_alpha or not line_digits or line_digits == replacement_digits:
            continue
        return index
    return None


def precision_clean_dominant_image_short_label_line_text(
    text: str,
    consensus_tokens: set[str],
) -> str | None:
    tokens = precision_label_candidate_tokens(text)
    if len(tokens) != 1:
        return None
    token = tokens[0]
    if not token.isalpha():
        return None
    if token not in PATENT_FIGURE_SHORT_LABEL_TOKENS:
        return None
    if token not in consensus_tokens:
        return None
    return token.upper()


def precision_clean_dominant_image_single_alpha_row_label_text(
    text: str,
    *,
    consensus_tokens: set[str],
    supported_numeric_tokens: set[str],
) -> str | None:
    tokens = precision_label_candidate_tokens(text)
    if len(tokens) < 2:
        return None
    alpha_positions = [(index, token) for index, token in enumerate(tokens) if token.isalpha()]
    if len(alpha_positions) != 1:
        return None
    alpha_index, alpha_token = alpha_positions[0]
    if alpha_token not in consensus_tokens or alpha_token in PATENT_FIGURE_SHORT_LABEL_TOKENS:
        return None
    best_digit: str | None = None
    best_distance: int | None = None
    for index, token in enumerate(tokens):
        if not token.isdigit():
            continue
        if token not in supported_numeric_tokens and not plausible_dominant_image_row_digit(token):
            continue
        distance = abs(index - alpha_index)
        if best_distance is None or distance < best_distance:
            best_digit = token
            best_distance = distance
    if best_digit is None:
        return None
    if alpha_index == 0:
        return f"{alpha_token.upper()} {best_digit}"
    return f"{best_digit} {alpha_token.upper()}"


def plausible_dominant_image_row_digit(token: str) -> bool:
    if not token.isdigit() or len(token) != 2:
        return False
    value = int(token)
    return 10 <= value <= 60 and value % 2 == 0


def supplement_dominant_image_consensus_numeric_labels(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    figure_result: OcrPageTextResult,
    consensus_tokens: set[str],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    label_region_bbox = dominant_image_label_region_bbox(lines)
    existing_tokens = set()
    for line in lines:
        existing_tokens.update(precision_label_candidate_tokens(line.text))
    counts: Counter[str] = Counter()
    for candidate in figure_result.candidates:
        counts.update(set(precision_label_candidate_tokens(candidate.result.text)))
    clustered_rows = dominant_image_numeric_label_cluster_groups(
        figure_result.candidates,
        label_region_bbox=label_region_bbox,
    )
    records = list(lines)
    for token, count in counts.items():
        if token in existing_tokens:
            continue
        clusters = clustered_rows.get(token, ())
        cluster = next(
            (
                cluster
                for cluster in clusters
                if dominant_image_consensus_numeric_label_token(
                    token,
                    count=count,
                    clustered_count=cluster[0],
                )
            ),
            None,
        )
        if cluster is None:
            continue
        if token not in consensus_tokens:
            continue
        bbox = cluster[1]
        source = cluster[2]
        confidence = cluster[3]
        observation = page_geometry.PageObservation(
            kind="figure_text_line",
            source=source,
            bbox=bbox,
            advance_bbox=bbox,
            ink_bbox=bbox,
            text=token,
            confidence=confidence,
            provenance=page_geometry.provenance_tuple(dominant_image_numeric_label_supplement=True),
        )
        records.append(
            observation_resolver.ResolvedTextLine(
                token,
                observation,
                break_before=1,
                contributing_observations=(observation,),
            )
        )
        existing_tokens.add(token)
    return tuple(records)


def normalize_generic_dominant_image_label_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    consensus_tokens: set[str],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    label_region_bbox = dominant_image_label_region_bbox(lines)
    normalized: list[observation_resolver.ResolvedTextLine] = []
    for line in lines:
        replacement = normalized_generic_dominant_image_label_line(
            line,
            lines=lines,
            consensus_tokens=consensus_tokens,
            label_region_bbox=label_region_bbox,
        )
        if replacement is None:
            continue
        normalized.append(replacement)
    return tuple(normalized)


def normalized_generic_dominant_image_label_line(
    line: observation_resolver.ResolvedTextLine,
    *,
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    consensus_tokens: set[str],
    label_region_bbox: tuple[float, float, float, float] | None,
) -> observation_resolver.ResolvedTextLine | None:
    if line.observation.source != "table_fusion_text":
        return line
    bbox = line.observation.bbox
    if bbox is not None and dominant_image_bbox_is_peripheral_to_label_region(
        bbox,
        label_region_bbox,
    ):
        return line
    tokens = precision_label_candidate_tokens(line.text)
    if not tokens:
        return None
    alpha_tokens, digit_tokens = dominant_image_line_token_sets(line.text)
    if len(alpha_tokens) > 2:
        fig_replacement = normalized_dominant_image_figure_caption_noise_line(
            line,
            lines=lines,
        )
        if fig_replacement is not None:
            return fig_replacement
        return line
    if not alpha_tokens:
        return line
    if not dominant_image_alpha_tokens_are_redundant(
        alpha_tokens,
        line=line,
        lines=lines,
    ):
        return line
    kept_digits = [token for token in tokens if token.isdigit() and token in consensus_tokens]
    if kept_digits:
        replacement_text = " ".join(kept_digits)
        if replacement_text == line.text:
            return line
        observation = replace(line.observation, text=replacement_text)
        return replace(
            line,
            text=replacement_text,
            observation=observation,
            contributing_observations=(observation,),
        )
    return None


def normalized_dominant_image_figure_caption_noise_line(
    line: observation_resolver.ResolvedTextLine,
    *,
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> observation_resolver.ResolvedTextLine | None:
    alpha_tokens, digit_tokens = dominant_image_line_token_sets(line.text)
    if "fig" not in alpha_tokens or not digit_tokens:
        return None
    redundant_alpha_tokens = alpha_tokens - {"fig"}
    if not redundant_alpha_tokens:
        return None
    if not dominant_image_alpha_tokens_are_redundant(
        redundant_alpha_tokens,
        line=line,
        lines=lines,
    ):
        return None
    digits = sorted(digit_tokens, key=lambda token: (len(token), token))
    replacement_text = f"FIG {digits[0]}"
    if replacement_text == line.text:
        return line
    observation = replace(line.observation, text=replacement_text)
    return replace(
        line,
        text=replacement_text,
        observation=observation,
        contributing_observations=(observation,),
    )


def dominant_image_supported_numeric_tokens_from_candidates(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
    consensus_tokens: set[str],
) -> set[str]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        counts.update(set(precision_label_candidate_tokens(candidate.result.text)))
    clustered_rows = dominant_image_numeric_label_cluster_groups(
        candidates,
        label_region_bbox=label_region_bbox,
    )
    supported: set[str] = set()
    for token, count in counts.items():
        clusters = clustered_rows.get(token, ())
        if not any(
            dominant_image_consensus_numeric_label_token(
                token,
                count=count,
                clustered_count=cluster[0],
            )
            for cluster in clusters
        ):
            continue
        if token not in consensus_tokens and not clusters:
            continue
        supported.add(token)
    return supported


def dominant_image_supported_single_alpha_row_labels_from_candidates(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
    consensus_tokens: set[str],
    supported_numeric_tokens: set[str],
) -> set[str]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        if not str(candidate.name).startswith("rendered_page_"):
            continue
        for row in candidate.result.line_rows:
            bbox = dominant_image_row_page_bbox(row)
            if bbox is None or not dominant_image_bbox_in_label_region(bbox, label_region_bbox):
                continue
            cleaned_text = precision_clean_dominant_image_single_alpha_row_label_text(
                str(row.get("text", "")),
                consensus_tokens=consensus_tokens,
                supported_numeric_tokens=supported_numeric_tokens,
            )
            if cleaned_text is None:
                continue
            counts[cleaned_text] += 1
    return {text for text, count in counts.items() if count >= 2}


def dominant_image_alpha_tokens_are_redundant(
    alpha_tokens: set[str],
    *,
    line: observation_resolver.ResolvedTextLine,
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> bool:
    for token in alpha_tokens:
        if not any(
            other is not line
            and token in dominant_image_line_token_sets(other.text)[0]
            and (
                len(dominant_image_line_token_sets(other.text)[0]) > 1
                or dominant_image_line_token_sets(other.text)[1]
            )
            for other in lines
        ):
            return False
    return True


def dominant_image_consensus_numeric_label_token(
    token: str,
    *,
    count: int,
    clustered_count: int = 0,
) -> bool:
    if not token.isdigit() or len(token) != 2:
        return False
    value = int(token)
    if not 10 <= value <= 60:
        return False
    if value % 2 != 0:
        return clustered_count >= 2
    return count >= 6 or clustered_count >= 2


def dominant_image_label_region_bbox(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[float, float, float, float] | None:
    boxes = [
        line.observation.bbox
        for line in lines
        if line.observation.bbox is not None and dominant_image_line_token_sets(line.text)[0]
    ]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def dominant_image_row_page_bbox(
    row: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    bbox = page_geometry.normalize_rect(row.get("page_bbox"))
    return bbox if page_geometry.valid_rect(bbox) else None


def dominant_image_bbox_in_label_region(
    bbox: tuple[float, float, float, float],
    region_bbox: tuple[float, float, float, float] | None,
    *,
    pad_ratio_x: float = 0.08,
    pad_ratio_y: float = 0.08,
) -> bool:
    if region_bbox is None:
        return True
    center_x = (bbox[0] + bbox[2]) * 0.5
    center_y = (bbox[1] + bbox[3]) * 0.5
    width = max(1.0, region_bbox[2] - region_bbox[0])
    height = max(1.0, region_bbox[3] - region_bbox[1])
    pad_x = width * pad_ratio_x
    pad_y = height * pad_ratio_y
    return (
        region_bbox[0] - pad_x <= center_x <= region_bbox[2] + pad_x
        and region_bbox[1] - pad_y <= center_y <= region_bbox[3] + pad_y
    )


def dominant_image_numeric_label_cluster_groups(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
    pad_ratio_x: float = 0.08,
    pad_ratio_y: float = 0.08,
) -> dict[str, tuple[tuple[int, tuple[float, float, float, float], str, float | None], ...]]:
    grouped: dict[
        str,
        list[tuple[int, tuple[float, float, float, float], str, float | None]],
    ] = {}
    for candidate in candidates:
        candidate_confidence = page_geometry.numeric_confidence(candidate.result.confidence)
        for observation in dominant_image_label_observations(candidate):
            token = observation.numeric_token
            if token is None:
                continue
            bbox = observation.bbox
            if bbox is None or not dominant_image_bbox_in_label_region(
                bbox,
                label_region_bbox,
                pad_ratio_x=pad_ratio_x,
                pad_ratio_y=pad_ratio_y,
            ):
                continue
            clusters = grouped.setdefault(token, [])
            matched_index = next(
                (
                    index
                    for index, (
                        _count,
                        cluster_bbox,
                        _source,
                        _confidence,
                    ) in enumerate(clusters)
                    if dominant_image_numeric_label_boxes_match(bbox, cluster_bbox)
                ),
                None,
            )
            if matched_index is None:
                clusters.append((1, bbox, observation.source, observation.confidence))
                continue
            count, cluster_bbox, source, confidence = clusters[matched_index]
            clusters[matched_index] = (
                count + 1,
                dominant_image_union_bbox(cluster_bbox, bbox),
                source if (confidence or 0.0) >= (candidate_confidence or 0.0) else candidate.name,
                max(confidence or 0.0, candidate_confidence or 0.0),
            )
    cluster_groups: dict[
        str,
        tuple[tuple[int, tuple[float, float, float, float], str, float | None], ...],
    ] = {}
    for token, clusters in grouped.items():
        cluster_groups[token] = tuple(
            sorted(
                clusters,
                key=lambda cluster: (
                    -(cluster[0]),
                    -page_geometry.rect_area(cluster[1]),
                    -(cluster[3] or 0.0),
                    cluster[1][1],
                    cluster[1][0],
                ),
            ),
        )
    return cluster_groups


def dominant_image_label_observations(
    candidate: OcrCandidate,
) -> tuple[DominantImageLabelObservation, ...]:
    observations: list[DominantImageLabelObservation] = []
    confidence = page_geometry.numeric_confidence(candidate.result.confidence)
    for row in candidate.result.line_rows:
        text = str(row.get("text", "")).strip()
        bbox = dominant_image_row_page_bbox(row)
        if not text or bbox is None:
            continue
        observations.append(
            DominantImageLabelObservation(
                text=text,
                numeric_token=dominant_image_numeric_label_row_token(text),
                bbox=bbox,
                source=str(candidate.name),
                confidence=confidence,
            )
        )
    return tuple(observations)


def dominant_image_numeric_label_row_token(text: str) -> str | None:
    tokens = precision_label_candidate_tokens(text)
    if len(tokens) != 1:
        return None
    token = tokens[0]
    if not token.isdigit() or len(token) != 2:
        return None
    value = int(token)
    return token if 10 <= value <= 60 else None


def dominant_image_label_boxes_match(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    left_area = page_geometry.rect_area(left)
    right_area = page_geometry.rect_area(right)
    if left_area <= 0 or right_area <= 0:
        return False
    overlap = page_geometry.rect_intersection_area(left, right)
    if overlap / min(left_area, right_area) >= 0.25:
        return True
    left_center = ((left[0] + left[2]) * 0.5, (left[1] + left[3]) * 0.5)
    right_center = ((right[0] + right[2]) * 0.5, (right[1] + right[3]) * 0.5)
    left_width = max(1.0, left[2] - left[0])
    left_height = max(1.0, left[3] - left[1])
    right_width = max(1.0, right[2] - right[0])
    right_height = max(1.0, right[3] - right[1])
    return (
        abs(left_center[0] - right_center[0]) <= max(left_width, right_width) * 0.75
        and abs(left_center[1] - right_center[1]) <= max(left_height, right_height) * 0.75
    )


def dominant_image_numeric_label_boxes_match(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    left_area = page_geometry.rect_area(left)
    right_area = page_geometry.rect_area(right)
    if left_area <= 0 or right_area <= 0:
        return False
    left_width = max(1.0, left[2] - left[0])
    left_height = max(1.0, left[3] - left[1])
    right_width = max(1.0, right[2] - right[0])
    right_height = max(1.0, right[3] - right[1])
    area_ratio = max(left_area, right_area) / min(left_area, right_area)
    width_ratio = max(left_width, right_width) / min(left_width, right_width)
    height_ratio = max(left_height, right_height) / min(left_height, right_height)
    if area_ratio > 8.0 and (width_ratio > 4.0 or height_ratio > 4.0):
        return False
    overlap = page_geometry.rect_intersection_area(left, right)
    if overlap / min(left_area, right_area) >= 0.25:
        return True
    left_center = ((left[0] + left[2]) * 0.5, (left[1] + left[3]) * 0.5)
    right_center = ((right[0] + right[2]) * 0.5, (right[1] + right[3]) * 0.5)
    return (
        abs(left_center[0] - right_center[0]) <= min(left_width, right_width) * 2.0
        and abs(left_center[1] - right_center[1]) <= max(left_height, right_height) * 0.9
    )


def dominant_image_numeric_label_cluster_is_compact(
    bbox: tuple[float, float, float, float],
    label_region_bbox: tuple[float, float, float, float] | None,
) -> bool:
    if label_region_bbox is None:
        return True
    region_width = max(1.0, label_region_bbox[2] - label_region_bbox[0])
    region_height = max(1.0, label_region_bbox[3] - label_region_bbox[1])
    width = max(1.0, bbox[2] - bbox[0])
    height = max(1.0, bbox[3] - bbox[1])
    area_ratio = page_geometry.rect_area(bbox) / max(
        1.0, page_geometry.rect_area(label_region_bbox)
    )
    return width <= region_width * 0.12 and height <= region_height * 0.08 and area_ratio <= 0.008


def dominant_image_numeric_token_has_embedded_alpha_support(
    lines: list[observation_resolver.ResolvedTextLine],
    *,
    token: str,
) -> bool:
    for line in lines:
        alpha_tokens, digit_tokens = dominant_image_line_token_sets(line.text)
        if token not in digit_tokens:
            continue
        if len(alpha_tokens) >= 2:
            return True
    return False


def dominant_image_compact_numeric_cluster_has_neighboring_alpha_support(
    candidates: tuple[OcrCandidate, ...],
    *,
    cluster_bbox: tuple[float, float, float, float],
    token: str,
) -> bool:
    supporting_candidates: set[str] = set()
    for candidate in candidates:
        for row in candidate.result.line_rows:
            row_bbox = dominant_image_row_page_bbox(row)
            if row_bbox is None:
                continue
            row_text = str(row.get("text", ""))
            row_tokens = precision_label_candidate_tokens(row_text)
            alpha_tokens = [row_token for row_token in row_tokens if row_token.isalpha()]
            if not alpha_tokens or token in row_tokens:
                continue
            if not dominant_image_neighboring_row_supports_compact_cluster(
                cluster_bbox,
                row_bbox,
            ):
                continue
            supporting_candidates.add(str(candidate.name))
            break
    return len(supporting_candidates) >= 2


def dominant_image_numeric_token_has_nonperipheral_alpha_support(
    lines: list[observation_resolver.ResolvedTextLine],
    *,
    token: str,
    label_region_bbox: tuple[float, float, float, float] | None,
) -> bool:
    for line in lines:
        alpha_tokens, digit_tokens = dominant_image_line_token_sets(line.text)
        if token not in digit_tokens or not alpha_tokens:
            continue
        bbox = line.observation.bbox
        if bbox is not None and dominant_image_bbox_is_peripheral_to_label_region(
            bbox,
            label_region_bbox,
        ):
            continue
        return True
    return False


def dominant_image_stacked_alpha_companion_token(
    candidates: tuple[OcrCandidate, ...],
    *,
    base_token: str,
    consensus_tokens: set[str],
) -> str | None:
    support: Counter[str] = Counter()
    for candidate in candidates:
        rows = tuple(candidate.result.line_rows)
        for row in rows:
            row_bbox = dominant_image_row_page_bbox(row)
            row_tokens = precision_label_candidate_tokens(str(row.get("text", "")))
            if row_bbox is None or row_tokens != [base_token]:
                continue
            for other in rows:
                other_bbox = dominant_image_row_page_bbox(other)
                other_tokens = precision_label_candidate_tokens(str(other.get("text", "")))
                if (
                    other_bbox is None
                    or len(other_tokens) != 1
                    or other_tokens[0] not in consensus_tokens
                    or other_tokens[0] == base_token
                    or other_tokens[0] == "display"
                ):
                    continue
                if not dominant_image_neighboring_row_supports_compact_cluster(
                    row_bbox,
                    other_bbox,
                ):
                    continue
                if not dominant_image_display_row_exists_to_right(
                    rows,
                    anchor_bbox=row_bbox,
                    companion_bbox=other_bbox,
                ):
                    continue
                support[other_tokens[0]] += 1
    if not support:
        return None
    token, count = max(support.items(), key=lambda item: (item[1], item[0] == "oled"))
    return token if count >= 2 else None


def dominant_image_numeric_display_companion_support(
    candidates: tuple[OcrCandidate, ...],
    *,
    token: str,
) -> tuple[str, tuple[float, float, float, float], float | None] | None:
    supporting_candidates: set[str] = set()
    best: tuple[str, tuple[float, float, float, float], float | None] | None = None
    for candidate in candidates:
        rows = tuple(candidate.result.line_rows)
        numeric_boxes: list[tuple[float, float, float, float]] = []
        display_boxes: list[tuple[float, float, float, float]] = []
        for row in rows:
            row_tokens = precision_label_candidate_tokens(str(row.get("text", "")))
            bbox = dominant_image_row_page_bbox(row)
            if bbox is None:
                continue
            if row_tokens == [token]:
                numeric_boxes.append(bbox)
            elif row_tokens == ["display"]:
                display_boxes.append(bbox)
        match = next(
            (
                dominant_image_union_bbox(numeric_bbox, display_bbox)
                for numeric_bbox in numeric_boxes
                for display_bbox in display_boxes
                if dominant_image_numeric_display_boxes_match(numeric_bbox, display_bbox)
            ),
            None,
        )
        if match is None:
            continue
        supporting_candidates.add(str(candidate.name))
        confidence = page_geometry.numeric_confidence(candidate.result.confidence)
        if best is None or (confidence or 0.0) > (best[2] or 0.0):
            best = (str(candidate.name), match, confidence)
    return best if len(supporting_candidates) >= 2 else None


def dominant_image_single_alpha_numeric_companion_support(
    candidates: tuple[OcrCandidate, ...],
    *,
    alpha_token: str,
    consensus_tokens: set[str],
    line_bbox: tuple[float, float, float, float] | None,
) -> tuple[str, tuple[float, float, float, float], float | None, str] | None:
    supporting: Counter[str] = Counter()
    best: tuple[str, tuple[float, float, float, float], float | None, str] | None = None
    for candidate in candidates:
        confidence = page_geometry.numeric_confidence(candidate.result.confidence)
        rows = tuple(candidate.result.line_rows)
        alpha_boxes: list[tuple[float, float, float, float]] = []
        for row in rows:
            if precision_label_candidate_tokens(str(row.get("text", ""))) != [alpha_token]:
                continue
            bbox = dominant_image_row_page_bbox(row)
            if bbox is not None:
                alpha_boxes.append(bbox)
        if not alpha_boxes:
            continue
        for token, clusters in dominant_image_numeric_label_cluster_groups(
            (candidate,),
            label_region_bbox=None,
            pad_ratio_x=0.35,
            pad_ratio_y=0.6,
        ).items():
            if token not in consensus_tokens:
                continue
            for cluster_count, cluster_bbox, _source, _cluster_conf in clusters:
                if cluster_count < 1:
                    continue
                if not any(
                    dominant_image_neighboring_row_supports_compact_cluster(
                        cluster_bbox,
                        alpha_bbox,
                    )
                    for alpha_bbox in alpha_boxes
                ):
                    continue
                if (
                    line_bbox is not None
                    and not dominant_image_neighboring_row_supports_compact_cluster(
                        cluster_bbox,
                        line_bbox,
                    )
                ):
                    continue
                supporting[token] += 1
                if best is None or (token == alpha_token or (confidence or 0.0) > (best[2] or 0.0)):
                    best = (
                        str(candidate.name),
                        dominant_image_union_bbox(cluster_bbox, alpha_boxes[0]),
                        confidence,
                        token,
                    )
    if not supporting:
        return None
    token, count = max(supporting.items(), key=lambda item: (item[1], -abs(int(item[0]) - 44)))
    if count < 2:
        return None
    if best is None or best[3] != token:
        return None
    return best


def dominant_image_figure_micro_band_alpha_word_clusters(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
) -> dict[str, tuple[tuple[int, tuple[float, float, float, float], str, float | None], ...]]:
    grouped: dict[
        str,
        list[tuple[int, tuple[float, float, float, float], str, float | None]],
    ] = {}
    for candidate in candidates:
        if "_micro_band_" not in str(candidate.name):
            continue
        confidence = page_geometry.numeric_confidence(candidate.result.confidence)
        numeric_boxes = [
            bbox
            for row in candidate.result.word_rows
            if (bbox := dominant_image_row_page_bbox(row)) is not None
            and dominant_image_numeric_label_row_token(str(row.get("text", ""))) is not None
        ]
        if not numeric_boxes:
            continue
        for row in candidate.result.word_rows:
            token = dominant_image_alpha_word_row_token(str(row.get("text", "")))
            if token is None:
                continue
            bbox = dominant_image_row_page_bbox(row)
            if bbox is None or not dominant_image_bbox_in_label_region(
                bbox,
                label_region_bbox,
                pad_ratio_x=0.12,
                pad_ratio_y=0.12,
            ):
                continue
            if not any(
                dominant_image_micro_band_alpha_has_numeric_support(
                    bbox,
                    numeric_bbox,
                )
                for numeric_bbox in numeric_boxes
            ):
                continue
            clusters = grouped.setdefault(token, [])
            matched_index = next(
                (
                    index
                    for index, (
                        _count,
                        cluster_bbox,
                        _source,
                        _confidence,
                    ) in enumerate(clusters)
                    if dominant_image_label_boxes_match(bbox, cluster_bbox)
                ),
                None,
            )
            if matched_index is None:
                clusters.append((1, bbox, str(candidate.name), confidence))
                continue
            count, cluster_bbox, source, cluster_confidence = clusters[matched_index]
            clusters[matched_index] = (
                count + 1,
                dominant_image_union_bbox(cluster_bbox, bbox),
                source
                if (cluster_confidence or 0.0) >= (confidence or 0.0)
                else str(candidate.name),
                max(cluster_confidence or 0.0, confidence or 0.0),
            )
    cluster_groups: dict[
        str,
        tuple[tuple[int, tuple[float, float, float, float], str, float | None], ...],
    ] = {}
    for token, clusters in grouped.items():
        cluster_groups[token] = tuple(
            sorted(
                clusters,
                key=lambda cluster: (
                    -cluster[0],
                    -page_geometry.rect_area(cluster[1]),
                    -(cluster[3] or 0.0),
                    cluster[1][1],
                    cluster[1][0],
                ),
            )
        )
    return cluster_groups


def dominant_image_figure_micro_fragment_clusters(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
    fragment_clusters: dict[str, tuple[FigureFragmentCluster, ...]] | None = None,
    local_vocabulary: set[str] | None = None,
    fusion_support: dict[str, tuple[FigureFragmentFusionSupport, ...]] | None = None,
    analysis: FigureFragmentAnalysis | None = None,
) -> dict[str, tuple[tuple[int, tuple[float, float, float, float], str, float | None], ...]]:
    if fragment_clusters is not None:
        raw_clusters = fragment_clusters
    elif analysis is not None:
        raw_clusters = analysis.raw_clusters
    else:
        raw_clusters = dominant_image_figure_band_local_fragment_clusters(
            candidates,
            label_region_bbox=label_region_bbox,
        )
    if local_vocabulary is not None:
        vocabulary = local_vocabulary
    elif analysis is not None:
        vocabulary = analysis.vocabulary
    else:
        vocabulary = dominant_image_page_local_fragment_vocabulary(candidates)
    if fusion_support is not None:
        fragment_support = fusion_support
    elif analysis is not None:
        fragment_support = analysis.fusion_support
    else:
        fragment_support = dominant_image_figure_band_local_fragment_fusion_support(
            candidates,
            label_region_bbox=label_region_bbox,
            fragment_clusters=raw_clusters,
            slot_plan=None,
        )
    grouped: dict[
        str,
        list[tuple[int, tuple[float, float, float, float], str, float | None]],
    ] = {}
    for candidate in candidates:
        if "_micro_fragment_" not in str(candidate.name):
            continue
        confidence = page_geometry.numeric_confidence(candidate.result.confidence)
        for item in figure_candidate_token_evidence(
            candidate,
            rows="word_rows",
            token_extractor=dominant_image_alpha_fragment_token,
        ):
            bbox = item.bbox
            if not dominant_image_bbox_in_label_region(
                bbox,
                label_region_bbox,
                pad_ratio_x=0.16,
                pad_ratio_y=0.16,
            ):
                continue
            fragment = dominant_image_alpha_fragment_token(item.text)
            if fragment is None:
                continue
            raw_group = raw_clusters.get(fragment)
            if raw_group is None or not any(cluster.count >= 3 for cluster in raw_group):
                continue
            support_group = fragment_support.get(fragment)
            if support_group is None or not any(
                support.score >= 8.0 and support.alpha_tokens and support.numeric_tokens
                for support in support_group
            ):
                continue
            completion = dominant_image_page_local_fragment_completion(
                fragment,
                vocabulary=vocabulary,
            )
            if completion is None:
                continue
            clusters = grouped.setdefault(completion, [])
            matched_index = next(
                (
                    index
                    for index, (
                        _count,
                        cluster_bbox,
                        _source,
                        _confidence,
                    ) in enumerate(clusters)
                    if dominant_image_label_boxes_match(bbox, cluster_bbox)
                ),
                None,
            )
            if matched_index is None:
                clusters.append((1, bbox, str(candidate.name), confidence))
                continue
            count, cluster_bbox, source, cluster_confidence = clusters[matched_index]
            clusters[matched_index] = (
                count + 1,
                dominant_image_union_bbox(cluster_bbox, bbox),
                source
                if (cluster_confidence or 0.0) >= (confidence or 0.0)
                else str(candidate.name),
                max(cluster_confidence or 0.0, confidence or 0.0),
            )
    cluster_groups: dict[
        str,
        tuple[tuple[int, tuple[float, float, float, float], str, float | None], ...],
    ] = {}
    for token, clusters in grouped.items():
        cluster_groups[token] = tuple(
            sorted(
                clusters,
                key=lambda cluster: (
                    -cluster[0],
                    -page_geometry.rect_area(cluster[1]),
                    -(cluster[3] or 0.0),
                    cluster[1][1],
                    cluster[1][0],
                ),
            )
        )
    return cluster_groups


def dominant_image_page_local_fragment_vocabulary(
    candidates: tuple[OcrCandidate, ...],
    *,
    extra_tokens: Iterable[str] = (),
) -> set[str]:
    counts: Counter[str] = Counter()
    for token in extra_tokens:
        if token.isalpha() and len(token) >= 4:
            counts[token.casefold()] += 2
    for candidate in candidates:
        candidate_name = str(candidate.name)
        if "_micro_fragment_" in candidate_name:
            continue
        for token in precision_label_candidate_tokens(candidate.result.text):
            if token.isalpha() and len(token) >= 4:
                counts[token] += 1
    return {token for token, count in counts.items() if count >= 2}


def dominant_image_page_local_fragment_completion(
    fragment: str,
    *,
    vocabulary: set[str],
) -> str | None:
    if not fragment.isalpha() or len(fragment) < 2:
        return None
    normalized = fragment.casefold()
    best_word: str | None = None
    best_key: tuple[int, int, str] | None = None
    for word in vocabulary:
        if not word.startswith(normalized):
            continue
        if len(word) - len(normalized) > 4:
            continue
        if len(normalized) / len(word) < 0.45:
            continue
        key = (len(word) - len(normalized), len(word), word)
        if best_key is None or key < best_key:
            best_word = word
            best_key = key
    return best_word


def dominant_image_figure_fragment_analysis(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
    extra_tokens: Iterable[str] = (),
) -> FigureFragmentAnalysis:
    vocabulary = dominant_image_page_local_fragment_vocabulary(
        candidates,
        extra_tokens=extra_tokens,
    )
    raw_clusters = dominant_image_figure_band_local_fragment_clusters(
        candidates,
        label_region_bbox=label_region_bbox,
    )
    slot_plan = dominant_image_figure_band_slot_plan(
        candidates,
        label_region_bbox=label_region_bbox,
    )
    fusion_support = dominant_image_figure_band_local_fragment_fusion_support(
        candidates,
        label_region_bbox=label_region_bbox,
        fragment_clusters=raw_clusters,
        slot_plan=slot_plan,
    )
    slot_evidence = dominant_image_figure_band_slot_evidence(
        candidates,
        label_region_bbox=label_region_bbox,
        slot_plan=slot_plan,
    )
    return FigureFragmentAnalysis(
        vocabulary=vocabulary,
        raw_clusters=raw_clusters,
        slot_plan=slot_plan,
        fusion_support=fusion_support,
        slot_evidence=slot_evidence,
    )


def dominant_image_figure_band_local_fragment_clusters(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
) -> dict[str, tuple[FigureFragmentCluster, ...]]:
    grouped: dict[str, list[FigureFragmentCluster]] = {}
    for candidate in candidates:
        candidate_name = str(candidate.name)
        if "_micro_fragment_" not in candidate_name and "_micro_band_" not in candidate_name:
            continue
        confidence = page_geometry.numeric_confidence(candidate.result.confidence)
        rows = "word_rows"
        for item in figure_candidate_token_evidence(
            candidate,
            rows=rows,
            token_extractor=dominant_image_alpha_fragment_token,
        ):
            if not 2 <= len(item.token) <= 5:
                continue
            if not dominant_image_bbox_in_label_region(
                item.bbox,
                label_region_bbox,
                pad_ratio_x=0.16,
                pad_ratio_y=0.16,
            ):
                continue
            if item.token == "pixel":
                continue
            clusters = grouped.setdefault(item.token, [])
            matched_index = next(
                (
                    index
                    for index, cluster in enumerate(clusters)
                    if dominant_image_label_boxes_match(item.bbox, cluster.bbox)
                ),
                None,
            )
            if matched_index is None:
                clusters.append(
                    FigureFragmentCluster(
                        fragment=item.token,
                        count=1,
                        bbox=item.bbox,
                        source=candidate_name,
                        confidence=confidence,
                    )
                )
                continue
            cluster = clusters[matched_index]
            clusters[matched_index] = FigureFragmentCluster(
                fragment=cluster.fragment,
                count=cluster.count + 1,
                bbox=dominant_image_union_bbox(cluster.bbox, item.bbox),
                source=cluster.source
                if (cluster.confidence or 0.0) >= (confidence or 0.0)
                else candidate_name,
                confidence=max(cluster.confidence or 0.0, confidence or 0.0),
            )
    cluster_groups: dict[str, tuple[FigureFragmentCluster, ...]] = {}
    for fragment, clusters in grouped.items():
        cluster_groups[fragment] = tuple(
            sorted(
                clusters,
                key=lambda cluster: (
                    -cluster.count,
                    -page_geometry.rect_area(cluster.bbox),
                    -(cluster.confidence or 0.0),
                    cluster.bbox[1],
                    cluster.bbox[0],
                ),
            )
        )
    return cluster_groups


def dominant_image_figure_band_local_fragment_fusion_support(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
    fragment_clusters: dict[str, tuple[FigureFragmentCluster, ...]] | None = None,
    slot_plan: FigureBandSlotPlan | None = None,
) -> dict[str, tuple[FigureFragmentFusionSupport, ...]]:
    raw_clusters = (
        dominant_image_figure_band_local_fragment_clusters(
            candidates,
            label_region_bbox=label_region_bbox,
        )
        if fragment_clusters is None
        else fragment_clusters
    )
    resolved_slot_plan = (
        dominant_image_figure_band_slot_plan(
            candidates,
            label_region_bbox=label_region_bbox,
        )
        if slot_plan is None
        else slot_plan
    )
    alpha_clusters = resolved_slot_plan.alpha_clusters
    numeric_clusters = resolved_slot_plan.numeric_clusters
    band_slots = resolved_slot_plan.slots
    support_by_fragment: dict[str, tuple[FigureFragmentFusionSupport, ...]] = {}
    for fragment, clusters in raw_clusters.items():
        supports: list[FigureFragmentFusionSupport] = []
        for cluster in clusters:
            slot_supports = tuple(
                slot
                for slot in band_slots
                if dominant_image_band_local_fragment_supports_gap(
                    cluster.bbox,
                    slot.gap_bbox,
                )
            )
            alpha_tokens = tuple(
                sorted(
                    token
                    for token, token_clusters in alpha_clusters.items()
                    if any(
                        dominant_image_band_local_fragment_supports_neighbor(
                            cluster.bbox,
                            token_cluster[1],
                        )
                        for token_cluster in token_clusters
                    )
                )
            )
            numeric_tokens = tuple(
                sorted(
                    token
                    for token, token_clusters in numeric_clusters.items()
                    if any(
                        dominant_image_band_local_fragment_supports_neighbor(
                            cluster.bbox,
                            token_cluster[1],
                        )
                        for token_cluster in token_clusters
                    )
                )
            )
            score = (
                cluster.count * 2.0
                + len(alpha_tokens) * 2.0
                + len(numeric_tokens) * 1.5
                + len(slot_supports) * 1.5
            )
            supports.append(
                FigureFragmentFusionSupport(
                    fragment=fragment,
                    count=cluster.count,
                    bbox=cluster.bbox,
                    source=cluster.source,
                    confidence=cluster.confidence,
                    alpha_tokens=alpha_tokens,
                    numeric_tokens=numeric_tokens,
                    score=score,
                )
            )
        support_by_fragment[fragment] = tuple(
            sorted(
                supports,
                key=lambda support: (
                    -support.score,
                    -support.count,
                    -(support.confidence or 0.0),
                    support.bbox[1],
                    support.bbox[0],
                ),
            )
        )
    return support_by_fragment


def dominant_image_figure_band_slots(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
    alpha_clusters: dict[
        str,
        tuple[tuple[int, tuple[float, float, float, float], str, float | None], ...],
    ]
    | None = None,
    numeric_clusters: dict[
        str,
        tuple[tuple[int, tuple[float, float, float, float], str, float | None], ...],
    ]
    | None = None,
) -> tuple[FigureBandSlot, ...]:
    return dominant_image_figure_band_slot_plan(
        candidates,
        label_region_bbox=label_region_bbox,
        alpha_clusters=alpha_clusters,
        numeric_clusters=numeric_clusters,
    ).slots


def dominant_image_figure_band_slot_evidence(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
    slots: tuple[FigureBandSlot, ...] | None = None,
    slot_plan: FigureBandSlotPlan | None = None,
) -> tuple[FigureBandSlotEvidence, ...]:
    if slots is not None:
        band_slots = slots
    elif slot_plan is not None:
        band_slots = slot_plan.slots
    else:
        band_slots = dominant_image_figure_band_slots(
            candidates,
            label_region_bbox=label_region_bbox,
        )
    if not band_slots:
        return ()
    slot_evidence: list[FigureBandSlotEvidence] = []
    for slot in band_slots:
        fragment_counts: Counter[str] = Counter()
        word_counts: Counter[str] = Counter()
        source_names: set[str] = set()
        fragment_observations: list[tuple[float, str]] = []
        for candidate in candidates:
            candidate_name = str(candidate.name)
            if (
                "_micro_band_" not in candidate_name
                and "_micro_fragment_" not in candidate_name
                and "_band_slot_" not in candidate_name
            ):
                continue
            for row in candidate.result.word_rows:
                bbox = dominant_image_row_page_bbox(row)
                if bbox is None or not dominant_image_band_local_fragment_supports_gap(
                    bbox,
                    slot.gap_bbox,
                ):
                    continue
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                source_names.add(candidate_name)
                fragment = dominant_image_alpha_fragment_token(text)
                if fragment is not None and fragment != slot.anchor_token:
                    fragment_counts[fragment] += 1
                    bbox_center_x = (bbox[0] + bbox[2]) * 0.5
                    fragment_observations.append((bbox_center_x, fragment))
                for token in precision_label_candidate_tokens(text):
                    if token.isalpha() and token != slot.anchor_token:
                        word_counts[token] += 1
        fragments = tuple(token for token, _count in fragment_counts.most_common())
        ordered_fragments = dominant_image_ordered_slot_fragments(fragment_observations)
        words = tuple(token for token, _count in word_counts.most_common())
        score = (
            slot.score
            + sum(fragment_counts.values()) * 1.5
            + sum(word_counts.values()) * 1.0
            + len(source_names) * 1.0
        )
        slot_evidence.append(
            FigureBandSlotEvidence(
                slot=slot,
                fragments=fragments,
                ordered_fragments=ordered_fragments,
                words=words,
                sources=tuple(sorted(source_names)),
                score=score,
            )
        )
    return tuple(
        sorted(
            slot_evidence,
            key=lambda evidence: (
                -evidence.score,
                evidence.slot.gap_bbox[1],
                evidence.slot.gap_bbox[0],
            ),
        )
    )


def dominant_image_figure_band_slot_hypotheses(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
    vocabulary: set[str] | None = None,
    evidence: tuple[FigureBandSlotEvidence, ...] | None = None,
    analysis: FigureFragmentAnalysis | None = None,
) -> dict[str, tuple[FigureBandSlotHypothesis, ...]]:
    if evidence is not None:
        slot_evidence = evidence
    elif analysis is not None:
        slot_evidence = analysis.slot_evidence
    else:
        slot_evidence = dominant_image_figure_band_slot_evidence(
            candidates,
            label_region_bbox=label_region_bbox,
            slot_plan=None,
        )
    repeated_vocabulary = (
        analysis.vocabulary
        if analysis is not None and vocabulary is None
        else dominant_image_page_local_fragment_vocabulary(candidates)
    )
    hypothesis_vocabulary = (
        dominant_image_page_local_technical_lexicon(
            candidates,
            slot_evidence=slot_evidence,
        )
        if vocabulary is None
        else vocabulary
    )
    grouped: dict[str, tuple[FigureBandSlotHypothesis, ...]] = {}
    for item in slot_evidence:
        hypotheses: list[FigureBandSlotHypothesis] = []
        for token in hypothesis_vocabulary:
            if token == item.slot.anchor_token:
                continue
            fragment_matches = tuple(
                fragment
                for fragment in item.fragments
                if dominant_image_slot_hypothesis_matches_fragment(token, fragment)
            )
            word_matches = tuple(word for word in item.words if word == token)
            if not fragment_matches and not word_matches:
                continue
            score = (
                len(fragment_matches) * 2.5
                + len(word_matches) * 3.0
                + min(len(token), 8) * 0.1
                + (2.0 if token in repeated_vocabulary else 0.0)
            )
            hypotheses.append(
                FigureBandSlotHypothesis(
                    token=token,
                    score=score,
                    fragment_matches=fragment_matches,
                    word_matches=word_matches,
                )
            )
        grouped[item.slot.anchor_token] = tuple(
            sorted(
                hypotheses,
                key=lambda hypothesis: (
                    -hypothesis.score,
                    -len(hypothesis.fragment_matches),
                    -len(hypothesis.word_matches),
                    hypothesis.token,
                ),
            )
        )
    return grouped


def dominant_image_page_local_technical_lexicon(
    candidates: tuple[OcrCandidate, ...],
    *,
    slot_evidence: tuple[FigureBandSlotEvidence, ...] = (),
) -> set[str]:
    lexicon = set(dominant_image_page_local_fragment_vocabulary(candidates))
    for evidence in slot_evidence:
        ordered_fragments = getattr(evidence, "ordered_fragments", ())
        fragments = getattr(evidence, "fragments", ())
        lexicon.update(
            dominant_image_slot_fragment_synthetic_candidates(ordered_fragments or fragments)
        )
    return lexicon


def dominant_image_slot_fragment_synthetic_candidates(
    fragments: tuple[str, ...],
) -> set[str]:
    synthetic: set[str] = set()
    normalized = dominant_image_compact_slot_fragment_sequence(fragments)
    for left_index, left in enumerate(normalized):
        for right in normalized[left_index + 1 :]:
            candidate = dominant_image_merge_fragment_pair(left, right)
            if candidate is None:
                continue
            synthetic.add(candidate)
    return synthetic


def dominant_image_compact_slot_fragment_sequence(
    fragments: tuple[str, ...],
) -> list[str]:
    normalized = [
        fragment.casefold()
        for fragment in fragments
        if fragment.isalpha() and 2 <= len(fragment) <= 5
    ]
    if not normalized:
        return []
    compact: list[str] = []
    for fragment in normalized:
        if any(fragment != other and fragment in other for other in normalized):
            continue
        if compact and compact[-1] == fragment:
            continue
        compact.append(fragment)
    return compact


def dominant_image_ordered_slot_fragments(
    fragment_observations: list[tuple[float, str]],
) -> tuple[str, ...]:
    ordered = sorted(fragment_observations, key=lambda item: (item[0], -len(item[1]), item[1]))
    compact: list[str] = []
    for _center_x, fragment in ordered:
        if compact and compact[-1] == fragment:
            continue
        compact.append(fragment)
    return tuple(compact)


def dominant_image_merge_fragment_pair(left: str, right: str) -> str | None:
    if len(left) < 2 or len(right) < 2:
        return None
    merged = left + right
    max_overlap = min(len(left), len(right), 4)
    for overlap in range(max_overlap, 1, -1):
        if left[-overlap:] == right[:overlap]:
            merged = left + right[overlap:]
            break
    if len(merged) < 4 or len(merged) > 10:
        return None
    if not merged.isalpha():
        return None
    return merged


def dominant_image_slot_hypothesis_matches_fragment(token: str, fragment: str) -> bool:
    normalized_token = token.casefold()
    normalized_fragment = fragment.casefold()
    if len(normalized_fragment) < 2:
        return False
    if normalized_token.startswith(normalized_fragment):
        return True
    return bool(normalized_fragment in normalized_token and len(normalized_fragment) >= 3)


def dominant_image_band_slot_numeric_supports_anchor(
    anchor_bbox: tuple[float, float, float, float],
    numeric_bbox: tuple[float, float, float, float],
) -> bool:
    anchor_center_y = (anchor_bbox[1] + anchor_bbox[3]) * 0.5
    numeric_center_y = (numeric_bbox[1] + numeric_bbox[3]) * 0.5
    anchor_height = max(1.0, anchor_bbox[3] - anchor_bbox[1])
    numeric_height = max(1.0, numeric_bbox[3] - numeric_bbox[1])
    if abs(anchor_center_y - numeric_center_y) > max(anchor_height, numeric_height) * 5.0:
        return False
    return numeric_bbox[0] > anchor_bbox[2]


def dominant_image_band_local_fragment_supports_gap(
    fragment_bbox: tuple[float, float, float, float],
    gap_bbox: tuple[float, float, float, float],
) -> bool:
    overlap_width = max(
        0.0, min(fragment_bbox[2], gap_bbox[2]) - max(fragment_bbox[0], gap_bbox[0])
    )
    overlap_height = max(
        0.0, min(fragment_bbox[3], gap_bbox[3]) - max(fragment_bbox[1], gap_bbox[1])
    )
    fragment_area = page_geometry.rect_area(fragment_bbox)
    if fragment_area <= 0.0:
        return False
    return (overlap_width * overlap_height) / fragment_area >= 0.28


def dominant_image_band_local_fragment_supports_neighbor(
    fragment_bbox: tuple[float, float, float, float],
    neighbor_bbox: tuple[float, float, float, float],
) -> bool:
    fragment_center_y = (fragment_bbox[1] + fragment_bbox[3]) * 0.5
    neighbor_center_y = (neighbor_bbox[1] + neighbor_bbox[3]) * 0.5
    fragment_height = max(1.0, fragment_bbox[3] - fragment_bbox[1])
    neighbor_height = max(1.0, neighbor_bbox[3] - neighbor_bbox[1])
    if abs(fragment_center_y - neighbor_center_y) > max(fragment_height, neighbor_height) * 5.0:
        return False
    horizontal_gap = max(
        0.0,
        max(fragment_bbox[0], neighbor_bbox[0]) - min(fragment_bbox[2], neighbor_bbox[2]),
    )
    return (
        horizontal_gap
        <= max(
            fragment_bbox[2] - fragment_bbox[0],
            neighbor_bbox[2] - neighbor_bbox[0],
        )
        * 9.0
    )


def dominant_image_figure_caption_dot_support(
    candidates: tuple[OcrCandidate, ...],
    *,
    token: str,
) -> tuple[str, tuple[float, float, float, float], float | None] | None:
    supporting_candidates: set[str] = set()
    best: tuple[str, tuple[float, float, float, float], float | None] | None = None
    for candidate in candidates:
        confidence = page_geometry.numeric_confidence(candidate.result.confidence)
        for row in candidate.result.line_rows:
            text = str(row.get("text", "")).strip()
            if "." not in text:
                continue
            if precision_label_candidate_tokens(text) != ["fig", token]:
                continue
            bbox = dominant_image_row_page_bbox(row)
            if bbox is None:
                continue
            supporting_candidates.add(str(candidate.name))
            if best is None or (confidence or 0.0) > (best[2] or 0.0):
                best = (str(candidate.name), bbox, confidence)
            break
    return best if len(supporting_candidates) >= 2 else None


def dominant_image_alpha_word_row_token(text: str) -> str | None:
    cleaned = "".join(ch for ch in text if ch.isalpha())
    if len(cleaned) < 4:
        return None
    return cleaned.casefold()


def dominant_image_micro_fragment_completion(text: str) -> str | None:
    token = dominant_image_alpha_fragment_token(text)
    if token is None or len(token) < 3:
        return None
    completion = dominant_image_common_prefix_completion(token)
    if completion is None or completion == token.upper():
        return None
    return completion.casefold()


def dominant_image_micro_band_alpha_has_numeric_support(
    alpha_bbox: tuple[float, float, float, float],
    numeric_bbox: tuple[float, float, float, float],
) -> bool:
    alpha_center_y = (alpha_bbox[1] + alpha_bbox[3]) * 0.5
    numeric_center_y = (numeric_bbox[1] + numeric_bbox[3]) * 0.5
    alpha_height = max(1.0, alpha_bbox[3] - alpha_bbox[1])
    numeric_height = max(1.0, numeric_bbox[3] - numeric_bbox[1])
    if abs(alpha_center_y - numeric_center_y) > max(alpha_height, numeric_height) * 4.5:
        return False
    horizontal_gap = max(
        0.0,
        max(alpha_bbox[0], numeric_bbox[0]) - min(alpha_bbox[2], numeric_bbox[2]),
    )
    return (
        horizontal_gap
        <= max(
            alpha_bbox[2] - alpha_bbox[0],
            numeric_bbox[2] - numeric_bbox[0],
        )
        * 8.0
    )


def dominant_image_neighboring_row_supports_compact_cluster(
    cluster_bbox: tuple[float, float, float, float],
    row_bbox: tuple[float, float, float, float],
) -> bool:
    overlap_width = max(0.0, min(cluster_bbox[2], row_bbox[2]) - max(cluster_bbox[0], row_bbox[0]))
    min_width = max(
        1.0,
        min(cluster_bbox[2] - cluster_bbox[0], row_bbox[2] - row_bbox[0]),
    )
    if overlap_width / min_width < 0.45:
        return False
    cluster_center_y = (cluster_bbox[1] + cluster_bbox[3]) * 0.5
    row_center_y = (row_bbox[1] + row_bbox[3]) * 0.5
    cluster_height = max(1.0, cluster_bbox[3] - cluster_bbox[1])
    row_height = max(1.0, row_bbox[3] - row_bbox[1])
    return abs(cluster_center_y - row_center_y) <= max(cluster_height, row_height) * 3.5


def dominant_image_display_row_exists_to_right(
    rows: tuple[dict[str, Any], ...],
    *,
    anchor_bbox: tuple[float, float, float, float],
    companion_bbox: tuple[float, float, float, float],
) -> bool:
    right_edge = max(anchor_bbox[2], companion_bbox[2])
    anchor_center_y = (anchor_bbox[1] + anchor_bbox[3]) * 0.5
    anchor_height = max(1.0, anchor_bbox[3] - anchor_bbox[1])
    companion_height = max(1.0, companion_bbox[3] - companion_bbox[1])
    for row in rows:
        row_bbox = dominant_image_row_page_bbox(row)
        row_tokens = precision_label_candidate_tokens(str(row.get("text", "")))
        if row_bbox is None or row_tokens != ["display"]:
            continue
        if row_bbox[0] <= right_edge + 40.0:
            continue
        row_center_y = (row_bbox[1] + row_bbox[3]) * 0.5
        row_height = max(1.0, row_bbox[3] - row_bbox[1])
        if (
            abs(row_center_y - anchor_center_y)
            > max(
                anchor_height,
                companion_height,
                row_height,
            )
            * 1.4
        ):
            continue
        return True
    return False


def dominant_image_numeric_display_boxes_match(
    numeric_bbox: tuple[float, float, float, float],
    display_bbox: tuple[float, float, float, float],
) -> bool:
    overlap_width = max(
        0.0,
        min(numeric_bbox[2], display_bbox[2]) - max(numeric_bbox[0], display_bbox[0]),
    )
    min_width = max(
        1.0,
        min(numeric_bbox[2] - numeric_bbox[0], display_bbox[2] - display_bbox[0]),
    )
    if overlap_width / min_width < 0.4:
        return False
    numeric_center_y = (numeric_bbox[1] + numeric_bbox[3]) * 0.5
    display_center_y = (display_bbox[1] + display_bbox[3]) * 0.5
    numeric_height = max(1.0, numeric_bbox[3] - numeric_bbox[1])
    display_height = max(1.0, display_bbox[3] - display_bbox[1])
    return (
        abs(numeric_center_y - display_center_y)
        <= max(
            numeric_height,
            display_height,
        )
        * 1.8
    )


def dominant_image_peripheral_render_row_clusters(
    candidates: tuple[OcrCandidate, ...],
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
) -> tuple[tuple[int, tuple[float, float, float, float], str, str, float | None], ...]:
    clusters: list[
        tuple[
            set[str],
            set[str],
            tuple[float, float, float, float],
            str,
            str,
            float | None,
        ]
    ] = []
    for candidate in candidates:
        candidate_confidence = page_geometry.numeric_confidence(candidate.result.confidence)
        for row in candidate.result.line_rows:
            bbox = dominant_image_row_page_bbox(row)
            text = str(row.get("text", "")).strip()
            tokens = set(normalized_text_tokens(text))
            if (
                bbox is None
                or not dominant_image_bbox_is_peripheral_to_label_region(
                    bbox,
                    label_region_bbox,
                )
                or not dominant_image_peripheral_render_row_text_candidate(tokens, text)
            ):
                continue
            matched_index = next(
                (
                    index
                    for index, (
                        cluster_candidates,
                        cluster_tokens,
                        cluster_bbox,
                        _text,
                        _source,
                        _confidence,
                    ) in enumerate(clusters)
                    if dominant_image_label_boxes_match(bbox, cluster_bbox)
                    and len(tokens & cluster_tokens) >= 2
                ),
                None,
            )
            if matched_index is None:
                clusters.append(
                    (
                        {str(candidate.name)},
                        tokens,
                        bbox,
                        text,
                        str(candidate.name),
                        candidate_confidence,
                    )
                )
                continue
            (
                cluster_candidates,
                cluster_tokens,
                cluster_bbox,
                cluster_text,
                cluster_source,
                cluster_confidence,
            ) = clusters[matched_index]
            candidate_names = set(cluster_candidates)
            candidate_names.add(str(candidate.name))
            best_text = cluster_text
            best_source = cluster_source
            best_confidence = cluster_confidence
            if dominant_image_peripheral_render_row_text_is_better(
                text,
                cluster_text,
                candidate_confidence=candidate_confidence,
                current_confidence=cluster_confidence,
            ):
                best_text = text
                best_source = str(candidate.name)
                best_confidence = candidate_confidence
            clusters[matched_index] = (
                candidate_names,
                cluster_tokens | tokens,
                dominant_image_union_bbox(cluster_bbox, bbox),
                best_text,
                best_source,
                best_confidence,
            )
    supported = [
        (len(candidate_names), bbox, text, source, confidence)
        for candidate_names, _tokens, bbox, text, source, confidence in clusters
        if len(candidate_names) >= 2
    ]
    return tuple(
        sorted(
            supported,
            key=lambda cluster: (cluster[1][1], cluster[1][0], -cluster[0]),
        )
    )


def dominant_image_peripheral_render_row_text_candidate(
    tokens: set[str],
    text: str,
) -> bool:
    if not 2 <= len(tokens) <= 10:
        return False
    alpha_tokens = [token for token in tokens if token.isalpha()]
    if not alpha_tokens:
        return False
    if any(len(token) >= 6 for token in alpha_tokens):
        return True
    return text_ocr_quality_score(text) <= 0.28


def dominant_image_peripheral_render_row_text_is_better(
    candidate_text: str,
    current_text: str,
    *,
    candidate_confidence: float | None,
    current_confidence: float | None,
) -> bool:
    candidate_punct = sum(1 for ch in candidate_text if not ch.isalnum() and not ch.isspace())
    current_punct = sum(1 for ch in current_text if not ch.isalnum() and not ch.isspace())
    if candidate_punct != current_punct:
        return candidate_punct > current_punct
    candidate_quality = text_ocr_quality_score(candidate_text)
    current_quality = text_ocr_quality_score(current_text)
    if candidate_quality != current_quality:
        return candidate_quality < current_quality
    return (candidate_confidence or 0.0) > (current_confidence or 0.0)


def should_preserve_dominant_image_peripheral_line(
    line: observation_resolver.ResolvedTextLine,
    *,
    label_region_bbox: tuple[float, float, float, float] | None,
    broad_page_result: OcrPageTextResult,
) -> bool:
    bbox = line.observation.bbox
    if bbox is None:
        return False
    if not dominant_image_bbox_is_peripheral_to_label_region(
        bbox,
        label_region_bbox,
    ):
        return False
    tokens = normalized_text_tokens(line.text)
    if not 2 <= len(tokens) <= 10:
        return False
    alpha_tokens = [token for token in tokens if token.isalpha()]
    if not alpha_tokens:
        return False
    if text_ocr_quality_score(line.text) > 0.45 and max(len(token) for token in alpha_tokens) < 6:
        return False
    return dominant_image_peripheral_line_has_render_support(
        line,
        broad_page_result=broad_page_result,
    )


def dominant_image_bbox_is_peripheral_to_label_region(
    bbox: tuple[float, float, float, float],
    label_region_bbox: tuple[float, float, float, float] | None,
) -> bool:
    if label_region_bbox is None:
        return False
    region_width = max(1.0, label_region_bbox[2] - label_region_bbox[0])
    region_height = max(1.0, label_region_bbox[3] - label_region_bbox[1])
    center_x = (bbox[0] + bbox[2]) * 0.5
    center_y = (bbox[1] + bbox[3]) * 0.5
    above = (
        center_y <= label_region_bbox[1] - region_height * 0.03
        or bbox[3] <= label_region_bbox[1] + region_height * 0.01
    )
    below = (
        center_y >= label_region_bbox[3] + region_height * 0.03
        or bbox[1] >= label_region_bbox[3] - region_height * 0.01
    )
    left = (
        center_x <= label_region_bbox[0] - region_width * 0.08
        or bbox[2] <= label_region_bbox[0] + region_width * 0.02
    )
    right = (
        center_x >= label_region_bbox[2] + region_width * 0.08
        or bbox[0] >= label_region_bbox[2] - region_width * 0.02
    )
    return above or below or left or right


def dominant_image_peripheral_line_has_render_support(
    line: observation_resolver.ResolvedTextLine,
    *,
    broad_page_result: OcrPageTextResult,
) -> bool:
    bbox = line.observation.bbox
    if bbox is None:
        return False
    line_tokens = set(normalized_text_tokens(line.text))
    if len(line_tokens) < 2:
        return False
    supporting_candidates: set[str] = set()
    for candidate in broad_page_result.candidates:
        if not str(candidate.name).startswith("rendered_page_"):
            continue
        for row in candidate.result.line_rows:
            row_bbox = dominant_image_row_page_bbox(row)
            if row_bbox is None or not dominant_image_label_boxes_match(bbox, row_bbox):
                continue
            row_tokens = set(normalized_text_tokens(str(row.get("text", ""))))
            if len(line_tokens & row_tokens) < 2:
                continue
            supporting_candidates.add(str(candidate.name))
            break
    return len(supporting_candidates) >= 2


def dominant_image_union_bbox(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


DEGRADATION_CHART_SIGNAL_TOKENS = frozenset(
    {
        "calib",
        "calibration",
        "degradation",
        "phase",
        "mode",
        "life",
        "eco",
    }
)


def precision_clean_degradation_chart_output_lines(
    page: PageExtractionHost,
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines:
        return lines
    try:
        profile = page.get_page_profile()
    except Exception:
        return lines
    if getattr(profile, "recommended_strategy", None) != "image_or_ocr":
        return lines
    signal_lines = sum(
        1
        for line in lines
        if DEGRADATION_CHART_SIGNAL_TOKENS & set(normalized_text_tokens(line.text))
    )
    if signal_lines < 4:
        return lines
    cleaned: list[observation_resolver.ResolvedTextLine] = []
    changed = False
    for line in lines:
        replacement_text = precision_clean_degradation_chart_line_text(line.text)
        if replacement_text is None:
            changed = True
            continue
        if replacement_text != line.text:
            changed = True
            observation = replace(line.observation, text=replacement_text)
            cleaned.append(
                replace(
                    line,
                    text=replacement_text,
                    observation=observation,
                    contributing_observations=(observation,),
                )
            )
            continue
        cleaned.append(line)
    return tuple(cleaned) if changed else lines


def precision_clean_degradation_chart_line_text(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    tokens = stripped.split()
    tokens = cleaned_degradation_chart_tokens(tokens)
    if not tokens:
        return None
    stripped = " ".join(tokens)
    normalized = normalized_text_tokens(stripped)
    if len(tokens) == 1:
        token = tokens[0].strip()
        if len(token) <= 2 and not any(ch.isdigit() for ch in token):
            return None
        token_digits = "".join(ch for ch in token if ch.isdigit())
        token_alpha = "".join(ch for ch in token if ch.isalpha())
        if token_digits and len(token_digits) <= 2 and len(token_alpha) <= 1:
            return None
    if len(tokens) <= 2 and all(len(token) <= 3 for token in normalized):
        return None
    informative_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if degradation_chart_token_is_informative(token)
        ),
        None,
    )
    if informative_index is None:
        return stripped
    if informative_index == 0:
        return stripped
    if all(degradation_chart_token_is_leading_junk(token) for token in tokens[:informative_index]):
        return " ".join(tokens[informative_index:])
    return stripped


def cleaned_degradation_chart_tokens(tokens: list[str]) -> list[str]:
    cleaned = [token for token in tokens if token not in {"+", "=", "¢"}]
    if len(cleaned) >= 3 and cleaned[0].isdigit() and "YEAR-PHASE" in cleaned:
        trailing = cleaned[-1]
        if trailing.isdigit() and len(trailing) == 2:
            cleaned = cleaned[:-1]
    return cleaned


def degradation_chart_token_is_informative(token: str) -> bool:
    cleaned = "".join(ch for ch in token if ch.isalnum())
    if not cleaned:
        return False
    lowered = cleaned.casefold()
    if lowered in DEGRADATION_CHART_SIGNAL_TOKENS:
        return True
    return cleaned.isalpha() and cleaned.upper() == cleaned and len(cleaned) >= 4


def degradation_chart_token_is_leading_junk(token: str) -> bool:
    cleaned = "".join(ch for ch in token if ch.isalnum())
    if not cleaned:
        return True
    if cleaned.isdigit():
        return False
    return cleaned.casefold() == cleaned and len(cleaned) <= 3


def append_rendered_full_page_ocr_candidates(
    page: PageExtractionHost,
    candidates: list[OcrCandidate],
    timeout: float | None,
    *,
    base_image: OcrImage,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
) -> None:
    backend = TesseractCtypesBackend.from_system()
    dpi_candidates = ocr_rendering.ocr_render_dpi_candidates_for_page(page)
    max_render_dpi = max(dpi_candidates) if dpi_candidates else None
    try:
        profile = page.get_page_profile()
    except Exception:
        profile = None
    prioritize_sparse_layout = getattr(profile, "recommended_strategy", None) == "native_text"
    vector_diagram_sparse = (
        getattr(profile, "recommended_strategy", None) in {"vector_or_table", "text_table"}
        and bool(getattr(profile, "has_path_ops", False))
        and (
            not bool(getattr(profile, "has_text_showing_ops", True))
            or (
                getattr(profile, "recommended_strategy", None) == "text_table"
                and len(getattr(page, "chars", ())) <= 20
                and any(
                    stream.decoded_bytes >= 150_000
                    for stream in getattr(profile, "content_streams", ())
                )
            )
        )
    )
    for dpi in dpi_candidates:
        source = f"rendered_page_{dpi}dpi"
        rendered_image: OcrImage | None
        if base_image.source == source:
            rendered_image = base_image
        elif base_image.source.startswith("rendered_page_") and dpi < (
            base_image.resolution or dpi
        ):
            rendered_image = ocr_rendering.derived_ocr_image_at_dpi(
                base_image,
                dpi=dpi,
                source=source,
            )
        else:
            rendered_image = ocr_rendering.render_page_for_ocr_at_dpi(
                page,
                dpi=dpi,
                source=source,
            )
        if rendered_image is None:
            continue
        primary_result = (
            ocr_session.image_to_text_result(
                rendered_image,
                psm=ocr_execution.OCR_DEFAULT_PAGE_SEGMENTATION_MODE,
            )
            if ocr_session is not None
            else ocr_full_page.ocr_image_to_text_result_with_timeout(
                rendered_image,
                timeout,
            )
        )
        append_nonempty_ocr_candidate(
            candidates,
            source,
            primary_result,
            rendered_image,
        )
        if (
            getattr(profile, "recommended_strategy", None) == "text_table"
            and 65 <= len(getattr(page, "chars", ())) <= 120
            and bool(getattr(profile, "has_path_ops", False))
            and rendered_image.width * rendered_image.height <= 12_000_000
            and ocr_full_page.should_try_tesseract_table_profile_ocr(primary_result)
        ):
            candidates.extend(
                ocr_candidate_generation.line_art_text_mask_ocr_candidates(
                    ocr_candidates.OcrCandidate(source, primary_result),
                    rendered_image,
                    timeout,
                    ocr_image_to_text_result_with_psm=cast(
                        ocr_candidate_generation.OcrPsmTextResultFunction,
                        lambda image, *, psm, timeout, variables: (
                            ocr_session.image_to_text_result(image, psm=psm, variables=variables)
                            if ocr_session is not None
                            else ocr_execution.ocr_image_to_text_result_with_psm_timeout(
                                image,
                                psm=psm,
                                variables=variables,
                                timeout=timeout,
                            )
                        ),
                    ),
                    token_type_classifier=ocr_schematic.classify_schematic_token_type,
                )
            )
            candidates.extend(
                ocr_table_regions.collect_table_rectangle_ocr_candidates(
                    cast(ocr_table_regions.TableOcrPage, page),
                    rendered_image,
                    timeout,
                )
            )
        if dpi == max_render_dpi and vector_diagram_sparse:
            tiled_candidate = ocr_tiling.tiled_ocr_candidate_for_dpi(
                page,
                dpi,
                timeout,
                max_side_pixels=OCR_VECTOR_DIAGRAM_TILE_MAX_SIDE_PIXELS,
                token_type_classifier=ocr_schematic.classify_schematic_token_type,
            )
            if tiled_candidate is not None:
                candidates.append(tiled_candidate)
        if dpi == max_render_dpi and should_try_suspect_native_table_tiling(page):
            tiled_candidate = ocr_tiling.tiled_ocr_candidate_for_dpi(
                page,
                dpi,
                timeout,
                token_type_classifier=ocr_schematic.classify_schematic_token_type,
            )
            if tiled_candidate is not None:
                candidates.append(tiled_candidate)
        # Dense scanned tables are poorly served by the default sparse-page
        # segmentation.  Keep a dedicated block-mode candidate so the normal
        # line reconciliation can choose cells/rows with stable geometry.
        if ocr_full_page.should_try_tesseract_table_profile_ocr(primary_result):
            table_result = (
                ocr_session.image_to_text_result(
                    rendered_image,
                    psm=6,
                    variables={"preserve_interword_spaces": "1"},
                )
                if ocr_session is not None
                else ocr_execution.ocr_image_to_text_result_with_psm_timeout(
                    rendered_image,
                    psm=6,
                    variables={"preserve_interword_spaces": "1"},
                    timeout=timeout,
                )
            )
            append_nonempty_ocr_candidate(
                candidates,
                f"{source}_table_profile_psm6",
                table_result,
                rendered_image,
            )
        append_rendered_band_column_ocr_candidates(
            candidates,
            rendered_image,
            primary_result,
            timeout,
            ocr_session=ocr_session,
        )
        if ocr_selection.rendered_sparse_ocr_candidate_is_usable_without_region_retry(
            ocr_candidates.OcrCandidate(name=source, result=primary_result)
        ):
            break
        if not should_expand_weak_full_page_ocr_candidates(
            page,
            rendered_image,
            ocr_candidates.OcrCandidate(name=source, result=primary_result),
        ):
            continue
        if prioritize_sparse_layout and backend is not None:
            sparse_layout_result = ocr_full_page.rendered_sparse_layout_result(
                backend,
                rendered_image,
                primary_result,
            )
            sparse_layout_candidate = append_nonempty_ocr_candidate(
                candidates,
                f"{source}_sparse_layout",
                sparse_layout_result,
                rendered_image,
            )
            if (
                sparse_layout_candidate is not None
                and rendered_sparse_ocr_candidate_is_usable_for_page(
                    page,
                    sparse_layout_candidate,
                )
            ):
                return
        if max_render_dpi is None or dpi == max_render_dpi:
            append_rendered_two_column_ocr_candidates(
                candidates,
                rendered_image,
                timeout,
                result=primary_result,
                ocr_session=ocr_session,
                cache=page.extraction_cache,
            )
        if ocr_full_page.should_try_alternate_ocr(primary_result):
            alternate_result = (
                ocr_session.image_to_text_result(
                    rendered_image,
                    psm=ocr_full_page.OCR_FALLBACK_ALTERNATE_PAGE_SEGMENTATION_MODE,
                )
                if ocr_session is not None
                else ocr_execution.ocr_image_to_text_result_with_psm_timeout(
                    rendered_image,
                    psm=ocr_full_page.OCR_FALLBACK_ALTERNATE_PAGE_SEGMENTATION_MODE,
                    timeout=timeout,
                )
            )
            append_nonempty_ocr_candidate(
                candidates,
                f"{source}_psm{ocr_full_page.OCR_FALLBACK_ALTERNATE_PAGE_SEGMENTATION_MODE}",
                alternate_result,
                rendered_image,
            )
            if (
                getattr(profile, "recommended_strategy", None) == "text_table"
                and len(getattr(page, "chars", ())) <= 64
            ):
                auto_result = (
                    ocr_session.image_to_text_result(rendered_image, psm=3)
                    if ocr_session is not None
                    else ocr_execution.ocr_image_to_text_result_with_psm_timeout(
                        rendered_image,
                        psm=3,
                        timeout=timeout,
                    )
                )
                append_nonempty_ocr_candidate(
                    candidates,
                    "full_page_auto_psm3",
                    auto_result,
                    rendered_image,
                )
        if ocr_full_page.should_try_sparse_ocr(primary_result):
            sparse_result = (
                ocr_session.image_to_text_result(
                    rendered_image,
                    psm=ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE,
                )
                if ocr_session is not None
                else ocr_execution.ocr_image_to_text_result_with_psm_timeout(
                    rendered_image,
                    psm=ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE,
                    timeout=timeout,
                )
            )
            sparse_candidate = append_nonempty_ocr_candidate(
                candidates,
                f"{source}_psm{ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE}",
                sparse_result,
                rendered_image,
            )
            if (
                sparse_candidate is not None
                and dpi != max_render_dpi
                and ocr_selection.rendered_sparse_ocr_candidate_is_usable_without_resolution_retry(
                    sparse_candidate
                )
            ):
                return
        if vector_diagram_sparse and not ocr_full_page.should_try_sparse_ocr(primary_result):
            sparse_result = (
                ocr_session.image_to_text_result(
                    rendered_image,
                    psm=ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE,
                )
                if ocr_session is not None
                else ocr_execution.ocr_image_to_text_result_with_psm_timeout(
                    rendered_image,
                    psm=ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE,
                    timeout=timeout,
                )
            )
            append_nonempty_ocr_candidate(
                candidates,
                f"{source}_vector_sparse_psm{ocr_full_page.OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE}",
                sparse_result,
                rendered_image,
            )
        if (
            backend is not None
            and not prioritize_sparse_layout
            and ocr_full_page.should_try_rendered_sparse_layout_ocr(
                rendered_image,
                primary_result,
            )
        ):
            sparse_layout_result = ocr_full_page.rendered_sparse_layout_result(
                backend,
                rendered_image,
                primary_result,
            )
            sparse_layout_candidate = append_nonempty_ocr_candidate(
                candidates,
                f"{source}_sparse_layout",
                sparse_layout_result,
                rendered_image,
            )
            if (
                sparse_layout_candidate is not None
                and ocr_selection.rendered_sparse_ocr_candidate_is_usable_without_region_retry(
                    sparse_layout_candidate
                )
            ):
                return
            if (
                sparse_layout_candidate is not None
                and rendered_sparse_ocr_candidate_is_usable_for_page(
                    page,
                    sparse_layout_candidate,
                )
            ):
                return
        if not should_try_rotated_ocr_supplement(
            primary_result,
            rendered_image,
        ):
            continue
        for variant_index, variant in enumerate(
            ocr_image_orientation_variants(
                rendered_image,
                cache=page.extraction_cache,
            )[1:],
            start=1,
        ):
            variant_source = f"{source}_orientation_{variant_index}"
            variant_result = (
                ocr_session.image_to_text_result(
                    variant,
                    psm=ocr_execution.OCR_DEFAULT_PAGE_SEGMENTATION_MODE,
                )
                if ocr_session is not None
                else ocr_full_page.ocr_image_to_text_result_with_timeout(
                    variant,
                    timeout,
                )
            )
            append_nonempty_ocr_candidate(
                candidates,
                variant_source,
                variant_result,
                variant,
            )
            append_rendered_band_column_ocr_candidates(
                candidates,
                variant,
                variant_result,
                timeout,
                ocr_session=ocr_session,
            )


def should_try_suspect_native_table_tiling(page: PageExtractionHost) -> bool:
    cache = getattr(page, "extraction_cache", None)
    if not isinstance(cache, dict):
        return False
    assessment = cache.get("native_text_assessment")
    if not isinstance(assessment, dict) or assessment.get("status") != "suspect":
        return False
    try:
        profile = page.get_page_profile()
    except Exception:
        return False
    return getattr(profile, "recommended_strategy", None) == "text_table" and bool(
        getattr(profile, "has_path_ops", False)
    )


def rendered_sparse_ocr_candidate_is_usable_for_page(
    page: PageExtractionHost,
    candidate: OcrCandidate,
) -> bool:
    if not candidate.name.endswith("_sparse_layout"):
        return False
    try:
        profile = page.get_page_profile()
    except Exception:
        return False
    if getattr(profile, "recommended_strategy", None) != "native_text":
        return False
    text = candidate.result.text
    tokens = extracted_text_token_count(text)
    if tokens < 8 or tokens > 240:
        return False
    if ocr_text_analysis.text_ocr_quality_score(text) > 0.12:
        return False
    if ocr_text_analysis.sparse_text_looks_noisy(text):
        return False
    return ocr_selection.ocr_candidate_score(candidate) >= 34.0


def append_nonempty_ocr_candidate(
    candidates: list[OcrCandidate],
    name: str,
    result: OcrTextResult,
    image: OcrImage,
) -> OcrCandidate | None:
    if not result.text:
        return None
    candidate = ocr_candidate_generation.ocr_candidate_from_image(
        name,
        result,
        image,
        token_type_classifier=ocr_schematic.classify_schematic_token_type,
    )
    candidates.append(candidate)
    return candidate


def append_rendered_band_column_ocr_candidates(
    candidates: list[OcrCandidate],
    image: OcrImage,
    result: OcrTextResult,
    timeout: float | None,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
) -> None:
    for name, regions in multi_column_band_ocr_region_variants(image, result):
        candidate = two_column_ocr_candidate_from_regions(
            image,
            regions,
            timeout,
            name,
            ocr_session=ocr_session,
        )
        if candidate is not None:
            candidates.append(candidate)


def append_rendered_two_column_ocr_candidates(
    candidates: list[OcrCandidate],
    image: OcrImage,
    timeout: float | None,
    *,
    result: OcrTextResult | None = None,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
    cache: ExtractionCacheMapping | None = None,
) -> None:
    for name, regions in two_column_ocr_region_variants(
        image,
        result=result,
        timeout=timeout,
        cache=cache,
    ):
        candidate = two_column_ocr_candidate_from_regions(
            image,
            regions,
            timeout,
            name,
            ocr_session=ocr_session,
        )
        if candidate is not None:
            candidates.append(candidate)
    return None


def best_ocr_candidate_image(
    candidates: list[OcrCandidate], candidate_images: dict[str, OcrImage]
) -> OcrImage | None:
    candidate = ocr_selection.select_ocr_candidate(candidates)
    if candidate is None:
        return None
    return candidate_images.get(candidate.name)


def best_table_ocr_candidate_image(
    candidates: list[OcrCandidate], candidate_images: dict[str, OcrImage]
) -> OcrImage | None:
    rendered_candidates = [
        candidate
        for candidate in candidates
        if candidate.name.startswith("rendered_page_")
        and not candidate.name.endswith("_tiled")
        and candidate.name in candidate_images
    ]
    candidate = ocr_selection.select_ocr_candidate(rendered_candidates)
    if candidate is None:
        return None
    return candidate_images.get(candidate.name)


def best_column_ocr_candidate_image(
    candidates: list[OcrCandidate],
    candidate_images: dict[str, OcrImage],
    *,
    vector_text: str,
) -> OcrImage | None:
    if extracted_text_token_count(vector_text) > 0:
        return None
    rendered_candidates = [
        candidate
        for candidate in candidates
        if should_try_two_column_ocr_candidate(candidate) and candidate.name in candidate_images
    ]
    candidate = ocr_selection.select_ocr_candidate(rendered_candidates)
    if candidate is None:
        return None
    return candidate_images.get(candidate.name)


def should_limit_rendered_ocr_after_large_full_page_image(
    image: OcrImage | None,
) -> bool:
    if image is None:
        return False
    if image.width * image.height <= OCR_ORIENTATION_ENSEMBLE_MAX_PIXELS:
        return False
    return image.source.startswith("full_page_") and "_rotated_" not in image.source


def rendered_dpis_limited_for_large_full_page_image(
    candidates: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(dpi for dpi in candidates if dpi <= OCR_LARGE_FULL_PAGE_IMAGE_MAX_RENDER_DPI)


def collect_rotated_ocr_supplement_candidates(
    image: OcrImage, candidates: list[OcrCandidate]
) -> list[OcrCandidate]:
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return []
    base = ocr_selection.select_ocr_candidate(candidates)
    if base is None or not should_try_rotated_ocr_supplement(base, image):
        return []
    backend = TesseractCtypesBackend.from_system()
    if backend is None:
        return []
    best = OcrTextResult("", None)
    best_score = float("-inf")
    for clockwise in (False, True):
        rotated = ocr_execution.rotate_ocr_image_right_angle(
            image,
            clockwise=clockwise,
        )
        result = backend.image_to_text_result(
            rotated,
            psm=ocr_full_page.OCR_FALLBACK_ALTERNATE_PAGE_SEGMENTATION_MODE,
            resolution=image.resolution or OCR_FALLBACK_DPI,
        )
        score = rotated_ocr_supplement_score(base.result.text, result)
        if score > best_score:
            best = result
            best_score = score
    if not should_accept_rotated_ocr_supplement(base.result.text, best):
        return []
    confidence = max(
        base.result.confidence if base.result.confidence is not None else 50,
        best.confidence if best.confidence is not None else 50,
    )
    text = base.result.text.rstrip() + "\n" + " ".join(best.text.split())
    return [
        ocr_candidates.OcrCandidate(
            "rotated_supplement",
            OcrTextResult(text, confidence),
            bbox=base.bbox,
            region_count=base.region_count,
        )
    ]


def collect_early_rotated_ocr_supplement_candidates(
    image: OcrImage | None,
    base_candidate: OcrCandidate | None,
    candidates: list[OcrCandidate],
) -> list[OcrCandidate]:
    if image is None or base_candidate is None:
        return []
    if not image.source.startswith("full_page_") or "_rotated_" in image.source:
        return []
    supplements = collect_rotated_ocr_supplement_candidates(image, candidates)
    if not supplements:
        return []
    supplement = supplements[0]
    if not should_stop_after_early_rotated_ocr_supplement(
        base_candidate,
        supplement,
        candidates,
    ):
        return []
    return supplements


def should_stop_after_early_rotated_ocr_supplement(
    base_candidate: OcrCandidate,
    supplement: OcrCandidate,
    candidates: list[OcrCandidate],
) -> bool:
    if supplement.name != "rotated_supplement":
        return False
    selected = ocr_selection.select_ocr_candidate([*candidates, supplement])
    if selected is not supplement:
        return False
    supplement_score = ocr_selection.ocr_candidate_score(supplement)
    base_score = ocr_selection.ocr_candidate_score(base_candidate)
    if supplement_score < base_score + 10.0:
        return False
    confidence = supplement.result.confidence or 0
    if confidence < 90:
        return False
    tokens = extracted_text_token_count(supplement.result.text)
    if not (80 <= tokens <= 220):
        return False
    return text_ocr_quality_score(supplement.result.text) <= 0.24


def should_try_rotated_ocr_supplement(
    candidate: OcrCandidate | OcrTextResult,
    image: OcrImage | None = None,
) -> bool:
    result = candidate.result if isinstance(candidate, ocr_candidates.OcrCandidate) else candidate
    if not result.text:
        return False
    tokens = extracted_text_token_count(result.text)
    confidence = result.confidence if result.confidence is not None else 50
    quality = text_ocr_quality_score(result.text)
    if 25 <= tokens <= 90 and confidence >= 78 and quality <= 0.25:
        return True
    if image is None:
        return False
    if image.source.startswith("rendered_page_"):
        if isinstance(candidate, ocr_candidates.OcrCandidate) and not candidate.name.startswith(
            "rendered_page_"
        ):
            return False
        if tokens <= 4 and confidence < 60:
            return True
        return bool(tokens <= 20 and confidence < 60 and quality <= 0.3)
    if not isinstance(candidate, ocr_candidates.OcrCandidate):
        return False
    if not image.source.startswith("full_page_"):
        return False
    if candidate.name != "full_page_image":
        return False
    if not (80 <= tokens <= 180):
        return False
    if confidence < 55:
        return False
    return quality <= 0.20


def rotated_ocr_supplement_score(base_text: str, rotated: OcrTextResult) -> float:
    if not rotated.text:
        return float("-inf")
    tokens = normalized_text_tokens(rotated.text)
    if not tokens:
        return float("-inf")
    base_tokens = set(normalized_text_tokens(base_text))
    new_tokens = sum(1 for token in tokens if token not in base_tokens)
    confidence = rotated.confidence if rotated.confidence is not None else 50
    return (
        confidence
        + min(len(tokens), 24) * 2.0
        + (new_tokens / max(1, len(tokens))) * 20.0
        - text_ocr_quality_score(rotated.text) * 60.0
    )


def should_accept_rotated_ocr_supplement(base_text: str, rotated: OcrTextResult) -> bool:
    tokens = normalized_text_tokens(rotated.text)
    if not (3 <= len(tokens) <= 30):
        return False
    confidence = rotated.confidence if rotated.confidence is not None else 50
    if confidence < 78:
        return False
    if text_ocr_quality_score(rotated.text) > 0.35:
        return False
    base_tokens = set(normalized_text_tokens(base_text))
    new_tokens = sum(1 for token in tokens if token not in base_tokens)
    return new_tokens >= max(3, int(len(tokens) * 0.55))


def ocr_image_orientation_variants(
    image: OcrImage,
    *,
    cache: dict[Any, Any] | None = None,
) -> list[OcrImage]:
    if cache is not None:
        cache_key = (
            "ocr_image_orientation_variants",
            id(image),
            image.source,
            image.width,
            image.height,
            image.bytes_per_pixel,
            image.bytes_per_line,
            image.resolution,
            image.clockwise_quarter_turns,
            image.page_clockwise_quarter_turns,
        )
        cached = cache.get(cache_key)
        if isinstance(cached, tuple):
            return list(cast(tuple[OcrImage, ...], cached))
    if image.encoded is None and image.bytes_per_pixel not in {1, 3, 4}:
        return [image]
    if image.encoded is None and not image.data:
        return [image]
    if image.width <= 0 or image.height <= 0:
        return [image]
    if not should_use_full_orientation_ensemble(image):
        return [image]
    variants = [
        image,
        ocr_execution.rotate_ocr_image_right_angle(image, clockwise=True),
        ocr_execution.rotate_ocr_image_half_turn(image),
        ocr_execution.rotate_ocr_image_right_angle(image, clockwise=False),
    ]
    unique: list[OcrImage] = []
    seen: set[tuple[int, int, str]] = set()
    for variant in variants:
        key = (variant.width, variant.height, variant.source)
        if key in seen:
            continue
        seen.add(key)
        unique.append(variant)
    if cache is not None:
        cache[cache_key] = tuple(unique)
    return unique


def should_use_full_orientation_ensemble(image: OcrImage) -> bool:
    pixels = image.width * image.height
    if pixels <= OCR_ORIENTATION_ENSEMBLE_MAX_PIXELS:
        return True
    return "_rotated_" in image.source


def collect_region_ocr_candidates(
    image: OcrImage,
    timeout: float | None,
    *,
    result: OcrTextResult | None = None,
) -> list[OcrCandidate]:
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return []
    regions = native_ocr_regions_from_result(result, image, max_regions=4)
    if not regions:
        regions = native_ocr_regions_for_image(image, timeout, max_regions=4)
    if not regions:
        return []
    page_area = max(1, image.width * image.height)
    if len(regions) == 1 and regions[0].area >= page_area * 0.75:
        return []
    return collect_rectangle_region_ocr_candidates(image, regions, timeout)


def two_column_ocr_candidates(
    image: OcrImage,
    timeout: float | None,
    *,
    result: OcrTextResult | None = None,
    cache: ExtractionCacheMapping | None = None,
) -> list[OcrCandidate]:
    variants = two_column_ocr_region_variants(
        image,
        result=result,
        timeout=timeout,
        cache=cache,
    )
    candidates: list[OcrCandidate] = []
    for name, regions in variants:
        candidate = two_column_ocr_candidate_from_regions(image, regions, timeout, name)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def two_column_ocr_candidate(
    image: OcrImage,
    timeout: float | None,
    *,
    result: OcrTextResult | None = None,
    cache: ExtractionCacheMapping | None = None,
) -> OcrCandidate | None:
    variants = two_column_ocr_region_variants(
        image,
        result=result,
        timeout=timeout,
        cache=cache,
    )
    if not variants:
        return None
    name, regions = variants[0]
    return two_column_ocr_candidate_from_regions(image, regions, timeout, name)


def two_column_ocr_candidate_from_regions(
    image: OcrImage,
    regions: list[NativeOcrRegion],
    timeout: float | None,
    name: str,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
) -> OcrCandidate | None:
    requests = [
        ocr_execution.RectangleOcrRequest(
            (region.x0, region.y0, region.x1, region.y1),
            OCR_FALLBACK_PAGE_SEGMENTATION_MODE,
            ocr_table_regions.OCR_TESSERACT_TABLE_VARIABLES,
        )
        for region in regions
    ]
    results = (
        ocr_session.image_regions_to_text_results(image, requests)
        if ocr_session is not None
        else ocr_execution.ocr_image_regions_to_text_results_with_timeout(
            image,
            requests,
            timeout,
        )
    )
    texts: list[str] = []
    confidences: list[int] = []
    for result in results:
        text = result.text.strip()
        if not text:
            continue
        texts.append(text)
        if result.confidence is not None:
            confidences.append(result.confidence)
    if len(texts) < 2:
        return None
    confidence = int(round(sum(confidences) / len(confidences))) if confidences else None
    return ocr_candidates.OcrCandidate(
        name,
        OcrTextResult("\n".join(texts), confidence),
        region_count=len(texts),
        image_width=image.width,
        image_height=image.height,
        image_resolution=image.resolution,
    )


def multi_column_band_ocr_region_variants(
    image: OcrImage,
    result: OcrTextResult,
) -> list[tuple[str, list[NativeOcrRegion]]]:
    if not should_try_multi_column_band_ocr_candidate(image, result):
        return []
    rects = ocr_line_rects_from_result(result, image.width, image.height)
    if len(rects) < OCR_MULTI_COLUMN_BAND_MIN_LINES * 2:
        return []
    regions = multi_column_band_regions_from_rects(rects, image.width, image.height)
    if len(regions) < 2:
        return []
    return [(f"{image.source}_band_columns", regions)]


def should_try_multi_column_band_ocr_candidate(
    image: OcrImage,
    result: OcrTextResult,
) -> bool:
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return False
    if not image.source.startswith("rendered_page_") or "_tile_" in image.source:
        return False
    if image.width < 900 or image.height < 900:
        return False
    tokens = extracted_text_token_count(result.text)
    confidence = result.confidence if result.confidence is not None else 50
    quality = text_ocr_quality_score(result.text)
    token_limit = OCR_MULTI_COLUMN_BAND_MAX_TOKENS
    if confidence < 35 or quality < 0.08:
        token_limit = max(token_limit, 1200)
    if not (OCR_MULTI_COLUMN_BAND_MIN_TEXTLINES <= tokens <= token_limit):
        return False
    if confidence >= 82 and quality <= 0.14 and tokens >= 420:
        return False
    line_rows = len(result.line_rows)
    word_rows = len(result.word_rows)
    if line_rows and line_rows > OCR_MULTI_COLUMN_BAND_MAX_TEXTLINES:
        return False
    if line_rows and line_rows < OCR_MULTI_COLUMN_BAND_MIN_TEXTLINES:
        if word_rows < OCR_MULTI_COLUMN_BAND_MIN_LINES * 2:
            return False
    elif not line_rows and word_rows < OCR_MULTI_COLUMN_BAND_MIN_LINES * 2:
        return False
    return quality >= 0.10 or confidence < 76


def ocr_line_rects_from_result(
    result: OcrTextResult,
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    if result.line_rows:
        line_rects = two_column_text_rects_from_boxes(
            ocr_component_boxes_from_rows(
                result.line_rows,
                level=TESSERACT_RIL_TEXTLINE,
            ),
            image_width,
            image_height,
        )
        if len(line_rects) >= OCR_MULTI_COLUMN_BAND_MIN_LINES * 2:
            return line_rects
    boxes = tuple(box for box in result.component_boxes if box.level == TESSERACT_RIL_TEXTLINE)
    if boxes:
        textline_rects = two_column_text_rects_from_boxes(
            list(boxes),
            image_width,
            image_height,
        )
        if len(textline_rects) >= OCR_MULTI_COLUMN_BAND_MIN_LINES * 2:
            return textline_rects
    if result.word_rows:
        return two_column_text_rects_from_boxes(
            ocr_component_boxes_from_rows(
                result.word_rows,
                level=TESSERACT_RIL_WORD,
            ),
            image_width,
            image_height,
        )
    return []


def multi_column_band_regions_from_rects(
    rects: list[tuple[int, int, int, int]],
    image_width: int,
    image_height: int,
) -> list[NativeOcrRegion]:
    if image_width < 900 or image_height < 900:
        return []
    grouped = vertical_rect_groups(rects, image_height)
    regions: list[NativeOcrRegion] = []
    for group in grouped:
        band_regions = multi_column_band_regions_from_group(
            group,
            image_width=image_width,
            image_height=image_height,
        )
        regions.extend(band_regions)
        if len(regions) >= OCR_MULTI_COLUMN_BAND_MAX_REGION_COUNT:
            break
    return regions[:OCR_MULTI_COLUMN_BAND_MAX_REGION_COUNT]


def vertical_rect_groups(
    rects: list[tuple[int, int, int, int]],
    image_height: int,
) -> list[list[tuple[int, int, int, int]]]:
    if not rects:
        return []
    ordered = sorted(rects, key=lambda rect: ((rect[1] + rect[3]) * 0.5, rect[0], rect[2]))
    heights = [rect[3] - rect[1] for rect in ordered if rect[3] > rect[1]]
    median_height = median(heights) if heights else 12.0
    gap_threshold = max(20.0, median_height * 2.8, image_height * 0.014)
    groups: list[list[tuple[int, int, int, int]]] = []
    current: list[tuple[int, int, int, int]] = []
    previous_center_y: float | None = None
    for rect in ordered:
        center_y = (rect[1] + rect[3]) * 0.5
        if previous_center_y is not None and center_y - previous_center_y > gap_threshold:
            if current:
                groups.append(current)
            current = [rect]
        else:
            current.append(rect)
        previous_center_y = center_y
    if current:
        groups.append(current)
    return groups


def multi_column_band_regions_from_group(
    rects: list[tuple[int, int, int, int]],
    *,
    image_width: int,
    image_height: int,
) -> list[NativeOcrRegion]:
    if len(rects) < OCR_MULTI_COLUMN_BAND_MIN_LINES:
        return []
    band_bbox = rect_union(rects)
    if band_bbox is None:
        return []
    band_width = band_bbox[2] - band_bbox[0]
    band_height = band_bbox[3] - band_bbox[1]
    page_area = max(1.0, float(image_width * image_height))
    band_area_ratio = (band_width * band_height) / page_area
    band_height_ratio = band_height / max(1.0, float(image_height))
    if band_width < image_width * 0.45:
        return []
    if band_height < max(120.0, image_height * 0.06):
        return []
    if band_height_ratio > OCR_MULTI_COLUMN_BAND_MAX_HEIGHT_RATIO:
        return []
    if band_area_ratio > OCR_MULTI_COLUMN_BAND_MAX_PAGE_AREA_RATIO:
        return []
    translated = [
        (x0 - band_bbox[0], y0 - band_bbox[1], x1 - band_bbox[0], y1 - band_bbox[1])
        for x0, y0, x1, y1 in rects
    ]
    if not rects_support_two_column_gutter(translated, band_width):
        return []
    split = two_column_split_from_rects(translated, band_width)
    if split is None:
        return []
    left_count = 0
    right_count = 0
    crossing_count = 0
    gutter_half = max(18, int(band_width * 0.05))
    for x0, ignored_y0, x1, ignored_y1 in translated:
        if x1 <= split - gutter_half:
            left_count += 1
        elif x0 >= split + gutter_half:
            right_count += 1
        else:
            crossing_count += 1
    if left_count < 3 or right_count < 3:
        return []
    if crossing_count > max(2, int(len(translated) * 0.20)):
        return []
    regions = band_ocr_regions_from_split(
        band_bbox,
        split=band_bbox[0] + split,
        image_width=image_width,
        image_height=image_height,
    )
    return regions if len(regions) == 2 else []


def rect_union(
    rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    if not rects:
        return None
    return (
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    )


def band_ocr_regions_from_split(
    band_bbox: tuple[int, int, int, int],
    *,
    split: int,
    image_width: int,
    image_height: int,
) -> list[NativeOcrRegion]:
    x0, y0, x1, y1 = band_bbox
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return []
    split = max(x0 + int(width * 0.28), min(x1 - int(width * 0.28), split))
    gutter = max(24, int(width * 0.06))
    top = max(0, min(image_height, y0 - max(10, int(height * 0.04))))
    bottom = max(top, min(image_height, y1 + max(10, int(height * 0.04))))
    left = NativeOcrRegion(
        max(0, x0),
        top,
        max(0, min(image_width, split + gutter // 2)),
        bottom,
    )
    right = NativeOcrRegion(
        max(0, min(image_width, split - gutter // 2)),
        top,
        max(0, min(image_width, x1)),
        bottom,
    )
    if left.width < 80 or right.width < 80:
        return []
    if left.height < 80 or right.height < 80:
        return []
    return [left, right]


def two_column_ocr_region_variants(
    image: OcrImage,
    result: OcrTextResult | None = None,
    timeout: float | None = None,
    *,
    cache: ExtractionCacheMapping | None = None,
) -> list[tuple[str, list[NativeOcrRegion]]]:
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return []
    if not image.source.startswith("rendered_page_") or "_tile_" in image.source:
        return []
    if image.width < 900 or image.height < 900:
        return []
    if result is not None:
        split = two_column_split_from_result(result, image.width, image.height)
        if split is not None:
            midpoint = image.width // 2
            if abs(split - midpoint) > max(48, int(image.width * 0.08)):
                return []
            if abs(split - midpoint) < max(18, int(image.width * 0.010)):
                split = midpoint
            inferred_regions = two_column_ocr_regions_from_split(image, split)
            if inferred_regions:
                return [("rendered_page_two_columns", inferred_regions)]
    split = two_column_split_from_component_boxes(image, timeout, cache=cache)
    if split is None:
        return []
    midpoint = image.width // 2
    if abs(split - midpoint) > max(48, int(image.width * 0.08)):
        return []
    if abs(split - midpoint) < max(18, int(image.width * 0.010)):
        split = midpoint
    inferred_regions = two_column_ocr_regions_from_split(image, split)
    if not inferred_regions:
        return []
    return [("rendered_page_two_columns", inferred_regions)]


def two_column_ocr_regions(
    image: OcrImage,
    timeout: float | None = None,
    *,
    result: OcrTextResult | None = None,
    cache: ExtractionCacheMapping | None = None,
) -> list[NativeOcrRegion]:
    variants = two_column_ocr_region_variants(
        image,
        result=result,
        timeout=timeout,
        cache=cache,
    )
    return variants[0][1] if variants else []


def two_column_split_from_result(
    result: OcrTextResult,
    image_width: int,
    image_height: int,
) -> int | None:
    for rows, level in (
        (result.line_rows, TESSERACT_RIL_TEXTLINE),
        (result.word_rows, TESSERACT_RIL_WORD),
    ):
        if not rows:
            continue
        split = two_column_split_from_boxes(
            ocr_component_boxes_from_rows(rows, level=level),
            image_width,
            image_height,
        )
        if split is not None:
            return split
    boxes = tuple(box for box in result.component_boxes if box.level == TESSERACT_RIL_TEXTLINE)
    if boxes:
        return two_column_split_from_boxes(list(boxes), image_width, image_height)
    return None


def two_column_ocr_regions_from_split(
    image: OcrImage,
    split: int,
) -> list[NativeOcrRegion]:
    split = max(int(image.width * 0.28), min(int(image.width * 0.72), split))
    gutter = max(24, int(image.width * 0.035))
    top = max(0, int(image.height * 0.035))
    bottom = min(image.height, int(image.height * 0.97))
    if bottom - top < 300:
        return []
    return [
        NativeOcrRegion(0, top, max(0, split + gutter // 2), bottom),
        NativeOcrRegion(
            min(image.width, split - gutter // 2),
            top,
            image.width,
            bottom,
        ),
    ]


def two_column_split_from_component_boxes(
    image: OcrImage,
    timeout: float | None,
    *,
    cache: ExtractionCacheMapping | None = None,
) -> int | None:
    boxes = ocr_execution.ocr_component_boxes_with_timeout(
        image,
        TESSERACT_RIL_WORD,
        timeout,
        variables=ocr_table_regions.OCR_TESSERACT_TABLE_VARIABLES,
        cache=cache,
    )
    split = two_column_split_from_boxes(boxes, image.width, image.height)
    if split is not None:
        return split
    boxes = ocr_execution.ocr_component_boxes_with_timeout(
        image,
        TESSERACT_RIL_TEXTLINE,
        timeout,
        variables=ocr_table_regions.OCR_TESSERACT_TABLE_VARIABLES,
        cache=cache,
    )
    return two_column_split_from_boxes(boxes, image.width, image.height)


def two_column_split_from_boxes(
    boxes: list[OcrComponentBox],
    image_width: int,
    image_height: int,
) -> int | None:
    if image_width < 900 or image_height < 900 or len(boxes) < 12:
        return None
    rects = two_column_text_rects_from_boxes(boxes, image_width, image_height)
    if len(rects) < 12:
        return None
    return two_column_split_from_rects(rects, image_width)


def two_column_split_from_rects(
    rects: list[tuple[int, int, int, int]],
    region_width: int,
) -> int | None:
    if region_width < 300 or len(rects) < 12:
        return None
    page_midpoint = region_width * 0.5
    left_band = region_width * 0.28
    right_band = region_width * 0.72
    evidence: list[tuple[float, float]] = []
    rects_by_line = sorted(rects, key=lambda rect: (rect[1], rect[0], rect[2]))
    for left_rect, right_rect in zip(rects_by_line, rects_by_line[1:], strict=False):
        if not two_column_rects_share_line(left_rect, right_rect):
            continue
        gap = right_rect[0] - left_rect[2]
        if gap < max(20.0, region_width * 0.025):
            continue
        boundary = left_rect[2] + gap * 0.5
        if not (left_band <= boundary <= right_band):
            continue
        distance = abs(boundary - page_midpoint) / max(1.0, region_width)
        if distance > 0.22:
            continue
        evidence.append((boundary, gap))
    if evidence:
        candidate_boundary = two_column_boundary_from_evidence(evidence, region_width)
        if candidate_boundary is not None:
            return candidate_boundary
    return two_column_split_from_coverage_profile(rects, region_width)


def two_column_text_rects_from_boxes(
    boxes: list[OcrComponentBox],
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    min_width = max(3, int(image_width * 0.002))
    min_height = max(3, int(image_height * 0.0015))
    max_height = max(40, int(image_height * 0.045))
    for box in boxes:
        left = max(0, min(image_width, int(box.left)))
        top = max(0, min(image_height, int(box.top)))
        right = max(left, min(image_width, int(box.left) + int(box.width)))
        bottom = max(top, min(image_height, int(box.top) + int(box.height)))
        width = right - left
        height = bottom - top
        if width < min_width or height < min_height:
            continue
        if height > max_height:
            continue
        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        if not (image_width * 0.03 <= center_x <= image_width * 0.97):
            continue
        if not (image_height * 0.03 <= center_y <= image_height * 0.97):
            continue
        rects.append((left, top, right, bottom))
    return rects


def two_column_rects_share_line(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    if right[0] <= left[2]:
        return False
    overlap = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    min_height = max(1, min(left[3] - left[1], right[3] - right[1]))
    center_delta = abs((left[1] + left[3]) * 0.5 - (right[1] + right[3]) * 0.5)
    return overlap / min_height >= 0.35 or center_delta <= min_height * 0.75


def two_column_boundary_from_evidence(
    evidence: list[tuple[float, float]],
    image_width: int,
) -> int | None:
    tolerance = max(8.0, image_width * 0.012)
    groups: list[list[tuple[float, float]]] = []
    for item in sorted(evidence, key=lambda value: value[0]):
        if groups and abs(item[0] - two_column_boundary_group_center(groups[-1])) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])
    if not groups:
        return None
    best = max(
        groups,
        key=lambda group: (
            len(group),
            sum(gap for ignored_boundary, gap in group) / len(group),
            -abs(two_column_boundary_group_center(group) - image_width * 0.5),
        ),
    )
    if len(best) < 3:
        return None
    return int(round(two_column_boundary_group_center(best)))


def two_column_boundary_group_center(group: list[tuple[float, float]]) -> float:
    return sum(boundary for boundary, ignored_gap in group) / len(group)


def two_column_split_from_coverage_profile(
    rects: list[tuple[int, int, int, int]],
    image_width: int,
) -> int | None:
    buckets = 80
    coverage = [0] * buckets
    for left, ignored_top, right, ignored_bottom in rects:
        start = max(0, min(buckets - 1, int(left / image_width * buckets)))
        stop = max(start, min(buckets - 1, int((right - 1) / image_width * buckets)))
        for index in range(start, stop + 1):
            coverage[index] += 1
    left = int(buckets * 0.32)
    right = int(buckets * 0.68)
    if right <= left:
        return None
    min_value = min(coverage[left:right])
    candidates = [index for index in range(left, right) if coverage[index] <= min_value + 1]
    if not candidates:
        return None
    midpoint_bucket = buckets * 0.5
    bucket = min(candidates, key=lambda index: abs(index + 0.5 - midpoint_bucket))
    neighboring = coverage[max(0, bucket - 8) : min(buckets, bucket + 9)]
    if not neighboring:
        return None
    if max(neighboring) < max(4, min_value + 4):
        return None
    return int(round((bucket + 0.5) / buckets * image_width))


def should_try_two_column_ocr_candidate(candidate: OcrCandidate) -> bool:
    if not candidate.name.startswith("rendered_page_"):
        return False
    if "_tile_" in candidate.name or candidate.name.endswith("_sparse"):
        return False
    text = candidate.result.text
    tokens = extracted_text_token_count(text)
    if not (280 <= tokens <= 1_500):
        return False
    if numeric_token_ratio(text) >= 0.40:
        return False
    confidence = candidate.result.confidence if candidate.result.confidence is not None else 50
    quality = text_ocr_quality_score(text)
    if confidence >= 92 and quality <= 0.08 and tokens >= 900:
        return False
    if tokens <= 340 and compact_label_heavy_candidate_text(text):
        return False
    if not candidate_rows_support_two_column_retry(candidate):
        return False
    return confidence < 86 or quality >= 0.10 or tokens < 900


def compact_label_heavy_candidate_text(text: str) -> bool:
    return (
        extracted_text_token_count(text) <= 340
        and ocr_candidate_generation.compact_uppercase_label_count(text) >= 10
    )


def candidate_rows_support_two_column_retry(candidate: OcrCandidate) -> bool:
    image_width = candidate.image_width
    image_height = candidate.image_height
    if image_width is None or image_height is None:
        return True
    result = candidate.result
    line_rows = result.line_rows
    word_rows = result.word_rows
    if len(line_rows) < 18 and len(word_rows) < 120:
        return False
    rows = line_rows if len(line_rows) >= 12 else word_rows
    if not rows:
        return True
    level = TESSERACT_RIL_TEXTLINE if rows is line_rows else TESSERACT_RIL_WORD
    boxes = ocr_component_boxes_from_rows(rows, level=level)
    rects = two_column_text_rects_from_boxes(boxes, image_width, image_height)
    if len(rects) < 12:
        return False
    return rects_support_two_column_gutter(rects, image_width)


def rects_support_two_column_center_gutter(
    rects: list[tuple[int, int, int, int]],
    image_width: int,
) -> bool:
    return rects_support_two_column_gutter(rects, image_width)


def rects_support_two_column_gutter(
    rects: list[tuple[int, int, int, int]],
    region_width: int,
) -> bool:
    if region_width <= 0 or len(rects) < 12:
        return False
    midpoint = region_width * 0.5
    gutter_width = max(32.0, region_width * 0.08)
    left_limit = midpoint - gutter_width * 0.5
    right_limit = midpoint + gutter_width * 0.5
    left = 0
    right = 0
    crossing = 0
    for x0, ignored_y0, x1, ignored_y1 in rects:
        if x1 <= left_limit:
            left += 1
        elif x0 >= right_limit:
            right += 1
        else:
            crossing += 1
    if left < 6 or right < 6:
        return False
    return crossing <= max(3, int(len(rects) * 0.18))


def native_ocr_regions_for_image(
    image: OcrImage, timeout: float | None, *, max_regions: int
) -> list[NativeOcrRegion]:
    boxes = ocr_execution.ocr_component_boxes_with_timeout(
        image,
        TESSERACT_RIL_BLOCK,
        timeout,
        variables=ocr_table_regions.OCR_TESSERACT_TABLE_VARIABLES,
    )
    regions = native_ocr_regions_from_component_boxes(
        boxes,
        image.width,
        image.height,
        max_regions=max_regions,
    )
    if regions:
        return regions
    boxes = ocr_execution.ocr_component_boxes_with_timeout(
        image,
        TESSERACT_RIL_TEXTLINE,
        timeout,
        variables=ocr_table_regions.OCR_TESSERACT_TABLE_VARIABLES,
    )
    return native_ocr_regions_from_component_boxes(
        boxes,
        image.width,
        image.height,
        max_regions=max_regions,
    )


def native_ocr_regions_from_result(
    result: OcrTextResult | None,
    image: OcrImage,
    *,
    max_regions: int,
) -> list[NativeOcrRegion]:
    if result is None:
        return []
    for rows, level in (
        (result.line_rows, TESSERACT_RIL_TEXTLINE),
        (result.word_rows, TESSERACT_RIL_WORD),
    ):
        if not rows:
            continue
        boxes = ocr_component_boxes_from_rows(rows, level=level)
        regions = native_ocr_regions_from_component_boxes(
            boxes,
            image.width,
            image.height,
            max_regions=max_regions,
        )
        if regions:
            return regions
    return []


def ocr_component_boxes_from_rows(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    level: int,
) -> list[OcrComponentBox]:
    boxes: list[OcrComponentBox] = []
    for index, row in enumerate(rows):
        try:
            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        boxes.append(
            OcrComponentBox(
                level=level,
                index=index,
                left=left,
                top=top,
                width=width,
                height=height,
            )
        )
    return boxes


def native_ocr_regions_from_component_boxes(
    boxes: list[OcrComponentBox],
    image_width: int,
    image_height: int,
    *,
    max_regions: int,
) -> list[NativeOcrRegion]:
    if image_width <= 0 or image_height <= 0 or max_regions <= 0:
        return []
    page_area = max(1, image_width * image_height)
    min_area = max(64, page_area // 12_000)
    regions: list[NativeOcrRegion] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        x0 = max(0, min(image_width, int(box.left)))
        y0 = max(0, min(image_height, int(box.top)))
        x1 = max(x0, min(image_width, int(box.left) + int(box.width)))
        y1 = max(y0, min(image_height, int(box.top) + int(box.height)))
        if x1 <= x0 or y1 <= y0:
            continue
        region = NativeOcrRegion(x0, y0, x1, y1)
        if region.area < min_area:
            continue
        key = (region.x0, region.y0, region.x1, region.y1)
        if key in seen:
            continue
        seen.add(key)
        regions.append(region)
    regions = merge_native_ocr_regions(regions, image_width, image_height)
    if len(regions) > max_regions:
        regions = sorted(regions, key=lambda region: region.area, reverse=True)[:max_regions]
    regions.sort(key=lambda region: (region.y0, region.x0))
    return regions


def merge_native_ocr_regions(
    regions: list[NativeOcrRegion],
    image_width: int,
    image_height: int,
) -> list[NativeOcrRegion]:
    if len(regions) < 2:
        return regions
    widths = sorted(region.width for region in regions)
    heights = sorted(region.height for region in regions)
    median_width = widths[len(widths) // 2]
    median_height = heights[len(heights) // 2]
    pad_x = max(24, int(round(median_width * 1.25)), int(round(image_width * 0.02)))
    pad_y = max(16, int(round(median_height * 1.5)), int(round(image_height * 0.015)))

    parents = list(range(len(regions)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(regions):
        left_x0 = max(0, left.x0 - pad_x)
        left_y0 = max(0, left.y0 - pad_y)
        left_x1 = min(image_width, left.x1 + pad_x)
        left_y1 = min(image_height, left.y1 + pad_y)
        for right_index in range(left_index + 1, len(regions)):
            right = regions[right_index]
            right_x0 = max(0, right.x0 - pad_x)
            right_y0 = max(0, right.y0 - pad_y)
            right_x1 = min(image_width, right.x1 + pad_x)
            right_y1 = min(image_height, right.y1 + pad_y)
            if left_x1 < right_x0 or right_x1 < left_x0 or left_y1 < right_y0 or right_y1 < left_y0:
                continue
            union(left_index, right_index)

    merged: dict[int, NativeOcrRegion] = {}
    for index, region in enumerate(regions):
        root = find(index)
        current = merged.get(root)
        if current is None:
            merged[root] = region
        else:
            merged[root] = NativeOcrRegion(
                min(current.x0, region.x0),
                min(current.y0, region.y0),
                max(current.x1, region.x1),
                max(current.y1, region.y1),
            )
    return list(merged.values())


def collect_rectangle_region_ocr_candidates(
    image: OcrImage,
    regions: list[Any],
    timeout: float | None,
    *,
    ocr_session: ocr_session_runtime.OcrPageSession | None = None,
) -> list[OcrCandidate]:
    candidates: list[OcrCandidate] = []
    texts: list[str] = []
    confidences: list[int] = []
    for region in regions:
        rectangle = (region.x0, region.y0, region.x1, region.y1)
        psm = rectangle_region_page_segmentation_mode(region)
        result = (
            ocr_session.image_region_to_text_result(
                image,
                rectangle,
                psm=psm,
                variables=ocr_table_regions.OCR_TESSERACT_TABLE_VARIABLES,
            )
            if ocr_session is not None
            else ocr_execution.ocr_image_region_to_text_result_with_timeout(
                image,
                rectangle,
                psm=psm,
                variables=ocr_table_regions.OCR_TESSERACT_TABLE_VARIABLES,
                timeout=timeout,
            )
        )
        if not result.text:
            continue
        texts.append(result.text)
        if result.confidence is not None:
            confidences.append(result.confidence)
        candidates.append(
            ocr_candidates.OcrCandidate(
                "rectangle_region",
                result,
                bbox=rectangle,
                region_count=1,
                image_width=image.width,
                image_height=image.height,
                image_resolution=image.resolution,
            )
        )
    if len(texts) >= 2:
        confidence = int(round(sum(confidences) / len(confidences))) if confidences else None
        candidates.append(
            ocr_candidates.OcrCandidate(
                "rectangle_regions",
                OcrTextResult("\n".join(texts), confidence),
                region_count=len(texts),
            )
        )
    return candidates


def rectangle_region_page_segmentation_mode(region: Any) -> int:
    width = max(1, int(region.x1) - int(region.x0))
    height = max(1, int(region.y1) - int(region.y0))
    if height <= 96 and width >= height * 4:
        return 7
    return OCR_FALLBACK_PAGE_SEGMENTATION_MODE


def should_try_region_ocr_candidate(candidate: OcrCandidate | None) -> bool:
    if candidate is None or not candidate.result.text:
        return True
    result = candidate.result
    tokens = extracted_text_token_count(result.text)
    confidence = result.confidence
    if ocr_selection.rendered_sparse_ocr_candidate_is_usable_without_region_retry(candidate):
        return False
    if ocr_selection.high_density_full_page_ocr_candidate_is_usable_without_region_retry(candidate):
        return False
    if confidence is not None and confidence < 52 and tokens < 900:
        return True
    if tokens < 80:
        return True
    return tokens < 180 and sparse_text_looks_noisy(result.text)


def full_page_images_for_ocr(page: PageExtractionHost) -> list[OcrImage]:
    image = full_page_image_for_ocr(page)
    if image is None:
        return []
    return [image]


def full_page_image_for_ocr(page: PageExtractionHost) -> OcrImage | None:
    cache = getattr(page, "extraction_cache", None)
    cache_key = "full_page_image_for_ocr"
    if isinstance(cache, dict) and cache_key in cache:
        return cache[cache_key]
    rendered = ocr_page_analysis.rendered_page_for_ocr_analysis(page)
    page_area = max(1.0, float(rendered.width) * float(rendered.height))
    best_item: Any | None = None
    best_item_index = -1
    best_area = 0.0
    best_pixels = 0
    best_box: tuple[float, float, float, float] | None = None
    for item_index, item in enumerate(rendered.display_list.items):
        if item.kind not in {"image", "inline-image"}:
            continue
        metadata = item.data.get("image_metadata")
        pixels = int(metadata.get("pixels") or 0) if isinstance(metadata, dict) else 0
        box = page_geometry.rect_box_tuple(item.data.get("bbox"))
        if box is None:
            continue
        x0, y0, x1, y1 = box
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area > best_area or (area == best_area and pixels > best_pixels):
            best_area = area
            best_pixels = pixels
            best_item = item
            best_item_index = item_index
            best_box = box
    if (
        best_item is None
        or best_pixels < 100_000
        or best_area < page_area * OCR_FALLBACK_IMAGE_AREA_RATIO
    ):
        if isinstance(cache, dict):
            cache[cache_key] = None
        return None
    if best_area < page_area * 0.80 and rendered_page_has_substantial_text_overlay(rendered):
        if isinstance(cache, dict):
            cache[cache_key] = None
        return None

    image: OcrImage | None = None
    if best_box is not None and best_area >= page_area * 0.92:
        # Portrait scanned pages often carry a single dominant image that can
        # be OCR'd directly without paying for an additional full-page render.
        if rendered.height > rendered.width:
            image = ocr_image_from_rendered_image_item(
                best_item,
                encoded_source="full_page_encoded_image",
                rgb_source="full_page_rgb_image",
                cache=cache,
                cache_key=("rendered_page_item", best_item_index),
            )
        if image is None:
            image = rendered_full_page_image_item_crop_for_ocr(
                page,
                best_box,
            )
    if image is None:
        image = ocr_image_from_rendered_image_item(
            best_item,
            encoded_source="full_page_encoded_image",
            rgb_source="full_page_rgb_image",
            cache=cache,
            cache_key=("rendered_page_item", best_item_index),
        )
    if isinstance(cache, dict):
        cache[cache_key] = image
    return image


def rendered_full_page_image_item_crop_for_ocr(
    page: PageExtractionHost,
    page_bbox: tuple[float, float, float, float],
) -> OcrImage | None:
    rendered_page_image = ocr_rendering.render_page_for_ocr_at_dpi(
        page,
        dpi=OCR_FALLBACK_DPI,
        source=f"rendered_page_{OCR_FALLBACK_DPI}dpi",
    )
    if rendered_page_image is None:
        return None
    crop = figure_crop_page_bbox_from_image(
        rendered_page_image,
        page_bbox,
        cache=page.extraction_cache,
        cache_key=("rendered_full_page_image_item_crop_for_ocr", page_bbox),
    )
    if crop is None:
        return None
    return replace(
        crop,
        source="full_page_rendered_crop",
        page_bbox=page_geometry.normalize_rect(page_bbox),
    )


def ocr_image_from_rendered_image_item(
    item: Any,
    *,
    encoded_source: str,
    rgb_source: str,
    prefer_decoded: bool = False,
    cache: dict[Any, Any] | None = None,
    cache_key: Any | None = None,
) -> OcrImage | None:
    image_cache_key = (
        ("ocr_image_from_rendered_image_item", cache_key, prefer_decoded)
        if cache_key is not None
        else None
    )
    if cache is not None and image_cache_key is not None:
        cached = cache.get(image_cache_key)
        if cached is None and image_cache_key in cache:
            return None
        if isinstance(cached, OcrImage):
            return replace(
                cached,
                source=encoded_source if cached.encoded is not None else rgb_source,
                cache_key=cache_key,
            )
    data = item.data
    dictionary = data.get("dictionary")
    raw = data.get("raw_data")
    if not isinstance(dictionary, dict) or not isinstance(raw, (bytes, bytearray, memoryview)):
        if cache is not None and image_cache_key is not None:
            cache[image_cache_key] = None
        return None
    width = pdf_int(lookup_dict_key(dictionary, "Width"), 0)
    height = pdf_int(lookup_dict_key(dictionary, "Height"), 0)
    if width <= 0 or height <= 0:
        if cache is not None and image_cache_key is not None:
            cache[image_cache_key] = None
        return None
    box = page_geometry.rect_box_tuple(data.get("bbox"))
    orientation_matches = image_orientation_matches_box(width, height, box)
    rotate_clockwise = image_display_rotation_is_clockwise(data.get("items"))
    if not orientation_matches and rotate_clockwise is None:
        if cache is not None and image_cache_key is not None:
            cache[image_cache_key] = None
        return None
    page_bbox = page_geometry.normalize_rect(box) if box is not None else None

    raw_bytes = raw if isinstance(raw, bytes) else bytes(raw)
    filter_names = [
        name for name in image_filter_names(lookup_dict_key(dictionary, "Filter")) if name
    ]
    filters = set(filter_names)
    converted_cache_key = (
        "ocr_rendered_image_converted",
        id(dictionary),
        id(raw),
        width,
        height,
        prefer_decoded,
    )
    if cache is not None:
        cached_converted = cache.get(converted_cache_key)
        if isinstance(cached_converted, bytes):
            converted = cached_converted
            image = OcrImage(
                data=converted,
                width=width,
                height=height,
                bytes_per_pixel=3,
                bytes_per_line=width * 3,
                source=rgb_source,
                cache_key=cache_key,
                page_bbox=page_bbox,
            )
            if not orientation_matches and rotate_clockwise is not None:
                image = ocr_execution.rotate_ocr_image_right_angle(
                    image,
                    clockwise=rotate_clockwise,
                )
            if image_cache_key is not None:
                cache[image_cache_key] = image
            return image
        if cached_converted is None and converted_cache_key in cache:
            if image_cache_key is not None:
                cache[image_cache_key] = None
            return None
    if (
        not prefer_decoded
        and orientation_matches
        and len(filter_names) == 1
        and filters & {"DCTDecode", "DCT"}
    ):
        image = OcrImage(
            data=b"",
            width=width,
            height=height,
            bytes_per_pixel=0,
            bytes_per_line=0,
            encoded=raw_bytes,
            source=encoded_source,
            cache_key=cache_key,
            page_bbox=page_bbox,
        )
        if cache is not None and image_cache_key is not None:
            cache[image_cache_key] = image
        return image
    if orientation_matches and len(filter_names) == 1 and filters & {"DCTDecode", "DCT"}:
        filters = {"DCTDecode"}
    if orientation_matches and filter_names in (
        ["FlateDecode", "DCTDecode"],
        ["FlateDecode", "DCT"],
    ):
        try:
            encoded = apply_flate(raw_bytes, None)
        except Exception:
            encoded = b""
        if encoded is None:
            encoded = b""
        if encoded.startswith(b"\xff\xd8") and not prefer_decoded:
            return OcrImage(
                data=b"",
                width=width,
                height=height,
                bytes_per_pixel=0,
                bytes_per_line=0,
                encoded=encoded,
                source=encoded_source,
                cache_key=cache_key,
                page_bbox=page_bbox,
            )
        if encoded.startswith(b"\xff\xd8"):
            raw_bytes = encoded
            filters = {"DCTDecode"}
    if (
        not prefer_decoded
        and orientation_matches
        and len(filter_names) == 1
        and filters & {"JPXDecode"}
    ):
        image = OcrImage(
            data=b"",
            width=width,
            height=height,
            bytes_per_pixel=0,
            bytes_per_line=0,
            encoded=raw_bytes,
            source=encoded_source,
            cache_key=cache_key,
            page_bbox=page_bbox,
        )
        if cache is not None and image_cache_key is not None:
            cache[image_cache_key] = image
        return image

    decoded_image = decode_image_for_ocr(raw_bytes, dictionary, width, height, filters)
    if decoded_image is None:
        if cache is not None and image_cache_key is not None:
            cache[image_cache_key] = None
        return None
    if cache is not None:
        cache[converted_cache_key] = decoded_image
    image = OcrImage(
        data=decoded_image,
        width=width,
        height=height,
        bytes_per_pixel=3,
        bytes_per_line=width * 3,
        source=rgb_source,
        cache_key=cache_key,
        page_bbox=page_bbox,
    )
    if not orientation_matches and rotate_clockwise is not None:
        image = ocr_execution.rotate_ocr_image_right_angle(
            image,
            clockwise=rotate_clockwise,
        )
    if cache is not None and image_cache_key is not None:
        cache[image_cache_key] = image
    return image


def rendered_page_has_substantial_text_overlay(rendered: Any) -> bool:
    count = 0
    for item in getattr(rendered.display_list, "items", ()):
        if item.kind not in {"text", "glyph"}:
            continue
        count += 1
        if count >= 40:
            return True
    return False


def image_orientation_matches_box(
    width: int,
    height: int,
    box: tuple[float, float, float, float] | None,
) -> bool:
    if box is None or width <= 0 or height <= 0:
        return True
    x0, y0, x1, y1 = box
    box_width = abs(x1 - x0)
    box_height = abs(y1 - y0)
    if box_width <= 0.0 or box_height <= 0.0:
        return True
    image_aspect = width / height
    box_aspect = box_width / box_height
    ratio = max(image_aspect / box_aspect, box_aspect / image_aspect)
    return ratio <= 1.35


def image_display_rotation_is_clockwise(items: Any) -> bool | None:
    items_type = type(items)
    if items_type is not list and items_type is not tuple:
        return None
    quad: Any | None = None
    for kind, value in items:
        if kind == "quad":
            quad = value
            break
    quad_type = type(quad)
    if quad_type is not list and quad_type is not tuple:
        return None
    quad = cast(list[Any] | tuple[Any, ...], quad)
    if len(quad) < 3:
        return None
    try:
        p00 = (float(quad[0][0]), float(quad[0][1]))
        p10 = (float(quad[1][0]), float(quad[1][1]))
        p01 = (float(quad[2][0]), float(quad[2][1]))
    except TypeError:
        return None
    width_dx = p10[0] - p00[0]
    width_dy = p10[1] - p00[1]
    height_dx = p01[0] - p00[0]
    height_dy = p01[1] - p00[1]
    if abs(width_dy) <= abs(width_dx) or abs(height_dx) <= abs(height_dy):
        return None
    if width_dy < 0.0 and height_dx > 0.0:
        return True
    if width_dy > 0.0 and height_dx < 0.0:
        return False
    return None


def decode_image_for_ocr(
    raw: bytes,
    dictionary: dict[Any, Any],
    width: int,
    height: int,
    filters: set[str],
) -> bytes | None:
    expected_gray = width * height
    expected_rgb = expected_gray * 3
    if len(raw) in {expected_gray, expected_rgb}:
        decoded = raw
    else:
        try:
            decoded = decode_stream_data(raw, dictionary)
        except Exception:
            return None
    if len(decoded) < expected_gray:
        bpc = pdf_int(lookup_dict_key(dictionary, "BitsPerComponent"), 8)
        row_bytes = (width * bpc + 7) // 8
        if bpc not in {1, 2, 4} or len(decoded) < row_bytes * height:
            return None
    try:
        converted = ImageColorManager.convert_image_data(
            decoded,
            dictionary,
            prefer_embedded_icc=False,
        )
    except Exception:
        return None
    if converted is None or len(converted) < expected_rgb:
        return None
    return converted


def page_is_overlay_ocr(page: PageExtractionHost) -> bool:
    try:
        rendered = ocr_page_analysis.rendered_page_for_ocr_analysis(page)
    except Exception:
        return False
    return float(rendered.width) > float(
        rendered.height
    ) and ocr_page_analysis.page_has_many_non_image_drawings(page)


def remove_dense_table_ocr_artifact_tokens(text: str) -> str:
    """Drop standalone scan speckles that Tesseract promotes to table tokens."""
    if not text:
        return text
    lines = [
        " ".join(
            cleaned_token for token in line.split() if (cleaned_token := token.strip("'\"•!?~|*"))
        )
        for line in text.splitlines()
    ]
    return "\n".join(line for line in lines if line)


def supplement_dense_table_native_symbols(
    text: str,
    native_text: str,
    region_classification: Any | None,
) -> str:
    if (
        not text
        or not native_text
        or region_classification is None
        or region_classification.kind != "dense_table"
        or not bool(region_classification.signals.get("form_signal"))
    ):
        return text
    additions: list[str] = []
    for symbol, limit in (('"', 35), (".", 28)):
        native_count = native_text.count(symbol)
        output_count = text.count(symbol)
        additions.extend([symbol] * min(limit, max(0, native_count - output_count)))
    return text if not additions else text.rstrip() + "\n" + " ".join(additions)


def extract_page_text(page: PageExtractionHost) -> str:
    cache = page.extraction_cache
    if cache is None:
        page.extraction_cache = cache = create_extraction_cache()
    snapshot = cache.get_as("page_extraction_snapshot", PageExtractionSnapshot)
    if snapshot is not None:
        return snapshot.text
    profile = page.get_page_profile()
    ocr_enabled = ocr_postprocess.ocr_is_enabled()
    decision = page_extraction_decision(
        profile,
        ocr_enabled=ocr_enabled,
    )
    if decision.skip_text:
        cache["native_layout_geometry_summary"] = layout_geometry_summary_record(
            page_layout_geometry_summary([])
        )
        return cache_page_extraction_snapshot(cache, "", ())
    if decision.route == "native_text_fast":
        fast_text = try_extract_native_text_fast(page, profile, cache)
        if fast_text is not None:
            return cache_page_extraction_snapshot(cache, fast_text, ())
    capture_state = page.get_text_and_graphics_state()
    chars = native_text_runs_for_extraction(page.chars)
    chars = native_text_runs_inside_page_bounds(chars, page.media_box, rotate=page.rotation)
    chars = native_text_runs_inside_visible_row_bands(chars, page.media_box, page)
    rendered = None
    native_output_lines = render_page_observation_lines(
        chars,
        rotate=page.rotation,
        media_box=page.media_box,
        layout=True,
    )
    text = render_resolved_text_lines(native_output_lines)
    selected_text = select_native_text_layout(
        chars,
        text,
        rotate=page.rotation,
        media_box=page.media_box,
    )
    if selected_text != text:
        native_output_lines = best_effort_resolved_text_lines(
            selected_text,
            native_output_lines,
        )
    text = selected_text
    final_output_lines = native_output_lines
    if should_try_rendered_glyph_repair(chars, text):
        chars, text, native_output_lines = apply_rendered_glyph_repair_to_native_text(
            page,
            chars,
            text,
            native_output_lines,
        )
        final_output_lines = native_output_lines
    native_geometry_summary = (
        native_layout_geometry_summary_for_runs(chars) if ocr_enabled else None
    )
    if cache is not None and native_geometry_summary is not None:
        cache["native_layout_geometry_summary"] = layout_geometry_summary_record(
            native_geometry_summary
        )
    if ocr_schematic.vector_table_symbol_marks_from_drawings(
        capture_state.drawings
    ) and ocr_schematic.should_try_vector_table_symbol_supplement(
        text,
        chars,
        page.media_box,
    ):
        try:
            if rendered is None:
                rendered = ocr_rendering.rendered_page_for_ocr_render(
                    page,
                    source="vector_table_symbols",
                )
            supplemented_text = ocr_schematic.append_vector_table_symbol_supplement(
                text,
                chars,
                rendered,
                page.media_box,
            )
            if supplemented_text != text:
                text = supplemented_text
                final_output_lines = best_effort_resolved_text_lines(
                    text,
                    final_output_lines,
                )
        except Exception:
            pass
    vector_text = ""
    vector_result = VectorStrokeOcrResult("", None)
    broad_ocr_result: OcrPageTextResult | None = None
    figure_ocr_result: OcrPageTextResult | None = None
    embedded_image_text_result: OcrPageTextResult | None = None
    pre_reconciliation_text_source = "native"
    replaced_fragmented_invisible_text_layer = False
    preserve_complete_page_ocr_text = False
    preserved_raw_ocr_text = False
    preserved_substantial_native_text_table = False
    preserve_substantial_native_lines = False
    pre_ocr_native_text = text
    pre_ocr_native_output_lines = final_output_lines
    schematic_ocr_supplement_candidate: OcrCandidate | None = None
    schematic_ocr_supplement_candidates: tuple[OcrCandidate, ...] = ()
    fragmented_invisible_text_layer = bool(
        native_geometry_summary is not None
        and native_invisible_text_layer_has_fragmented_geometry(
            chars,
            text,
            native_geometry_summary,
        )
    )
    trusted_invisible_text_layer = bool(
        native_geometry_summary is not None
        and native_invisible_text_layer_is_trustworthy(
            chars,
            text,
            native_geometry_summary,
        )
    )
    dominant_image_requires_ocr_verification = (
        ocr_enabled and ocr_page_analysis.dominant_image_requires_ocr_verification(page)
    )
    omit_native_text_from_ocr_render = (
        ocr_enabled and ocr_page_analysis.native_text_should_be_omitted_from_ocr_render(page, text)
    )
    if cache is not None:
        cache["ocr_render_exclude_native_text"] = omit_native_text_from_ocr_render
        native_assessment = ocr_page_analysis.assess_native_text(text)
        cache["native_text_assessment"] = {
            "status": native_assessment.status,
            "reason": native_assessment.reason,
            "token_count": native_assessment.token_count,
            "uninterpretable_count": native_assessment.uninterpretable_count,
        }
    if cache is not None and dominant_image_requires_ocr_verification:
        cache["ocr_verification_reason"] = "dominant_image_sparse_visible_text"
    if (
        cache is not None
        and trusted_invisible_text_layer
        and not dominant_image_requires_ocr_verification
    ):
        cache["ocr_skipped_for_trusted_invisible_text_layer"] = True
    if ocr_enabled and (
        not trusted_invisible_text_layer
        or dominant_image_requires_ocr_verification
        or omit_native_text_from_ocr_render
    ):
        assert native_geometry_summary is not None
        ocr_session = ocr_session_runtime.OcrPageSession()
        try:
            trusted_vector_stroke_text = False
            replaced_with_figure_ocr = False
            if ocr_postprocess.should_try_vector_stroke_ocr(page, text):
                vector_result = extract_vector_stroke_page_result(page)
                vector_text = vector_result.text
                trusted_vector_stroke_text = (
                    ocr_postprocess.should_trust_vector_stroke_text_without_full_ocr(vector_result)
                )
                if (
                    trusted_vector_stroke_text
                    or ocr_postprocess.should_replace_text_with_vector_stroke_ocr(
                        text,
                        vector_text,
                        vector_result.confidence,
                    )
                ):
                    text = vector_text
                    final_output_lines = vector_stroke_result_output_lines(
                        vector_result,
                        text,
                    )
                else:
                    supplement_lines = (
                        ocr_postprocess.vector_stroke_page_result_supplemental_resolved_lines(
                            page, text, vector_result
                        )
                    )
                    text, final_output_lines = append_resolved_supplement_lines(
                        text,
                        final_output_lines,
                        supplement_lines,
                    )
            if not trusted_vector_stroke_text and (
                fragmented_invisible_text_layer
                or omit_native_text_from_ocr_render
                or ocr_postprocess.should_ocr_fallback(page, text)
                or ocr_postprocess.should_try_full_ocr_after_vector_stroke(vector_result)
            ):
                ocr_result = extract_ocr_page_result(
                    page,
                    vector_text=vector_text,
                    ocr_session=ocr_session,
                )
                ocr_text = ocr_result.text
                if ocr_postprocess.should_reject_full_page_ocr_result(
                    text,
                    ocr_text,
                ):
                    if cache is not None:
                        cache["ocr_page_result_rejected"] = "garbled_full_page_ocr"
                else:
                    broad_ocr_result = ocr_result
                    if cache is not None:
                        cache["ocr_page_result_summary"] = {
                            "candidate": (
                                ocr_result.candidate.name
                                if ocr_result.candidate is not None
                                else None
                            ),
                            "tokens": ocr_text_analysis.extracted_text_token_count(ocr_text),
                            "characters": len(ocr_text),
                        }
                    preserved_substantial_native_text_table = (
                        should_preserve_substantial_text_table_native_text(
                            page,
                            pre_ocr_native_text,
                            ocr_text,
                        )
                    )
                    merged_ocr_text = ocr_postprocess.merge_ocr_with_vector_stroke_geometry(
                        page,
                        ocr_result,
                        vector_result,
                    )
                    (
                        reconciled_ocr_text,
                        reconciled_ocr_lines,
                    ) = ocr_postprocess.reconcile_native_ocr_lines_by_geometry(
                        page,
                        text,
                        ocr_result,
                    )
                    if preserved_substantial_native_text_table:
                        pass
                    elif ocr_postprocess.should_use_merged_vector_stroke_ocr(
                        text,
                        ocr_text,
                        merged_ocr_text,
                    ):
                        text = merged_ocr_text
                        pre_reconciliation_text_source = "merged_vector_ocr"
                        final_output_lines = best_effort_resolved_text_lines(
                            text,
                            ocr_result.output_lines,
                            ocr_result.selected_output_lines,
                            vector_stroke_result_output_lines(vector_result, vector_text),
                            final_output_lines,
                        )
                        schematic_ocr_supplement_candidate = ocr_result.candidate
                        schematic_ocr_supplement_candidates = ocr_result.candidates
                    elif ocr_postprocess.should_replace_vector_stroke_text_with_ocr(
                        text,
                        ocr_text,
                        vector_text,
                    ):
                        text = ocr_text
                        pre_reconciliation_text_source = "ocr_replace_vector"
                        final_output_lines = ocr_result_output_lines(
                            page,
                            ocr_result,
                            text,
                        )
                        schematic_ocr_supplement_candidate = ocr_result.candidate
                        schematic_ocr_supplement_candidates = ocr_result.candidates
                    elif should_replace_symbol_encoded_text_with_ocr(
                        page,
                        text,
                        ocr_text,
                    ):
                        text = ocr_text
                        pre_reconciliation_text_source = "ocr_replace_symbol_encoded"
                        final_output_lines = ocr_result_output_lines(
                            page,
                            ocr_result,
                            text,
                        )
                        schematic_ocr_supplement_candidate = ocr_result.candidate
                        schematic_ocr_supplement_candidates = ocr_result.candidates
                    elif (
                        omit_native_text_from_ocr_render
                        and ocr_text_analysis.extracted_text_token_count(ocr_text) >= 20
                    ):
                        text = ocr_text
                        pre_reconciliation_text_source = "ocr_replace_suspect_native"
                        final_output_lines = ocr_result_output_lines(
                            page,
                            ocr_result,
                            text,
                        )
                        schematic_ocr_supplement_candidate = ocr_result.candidate
                        schematic_ocr_supplement_candidates = ocr_result.candidates
                    elif should_replace_noisy_native_text_with_compact_ocr(
                        page,
                        text,
                        ocr_text,
                    ):
                        text = ocr_text
                        pre_reconciliation_text_source = "ocr_replace_noisy_native"
                        final_output_lines = ocr_result_output_lines(
                            page,
                            ocr_result,
                            text,
                        )
                        schematic_ocr_supplement_candidate = ocr_result.candidate
                        schematic_ocr_supplement_candidates = ocr_result.candidates
                    elif should_replace_dominant_image_native_text_with_ocr(
                        page,
                        text,
                        ocr_text,
                    ):
                        text = ocr_text
                        pre_reconciliation_text_source = "ocr_replace_dominant_image"
                        final_output_lines = ocr_result_output_lines(
                            page,
                            ocr_result,
                            text,
                        )
                        schematic_ocr_supplement_candidate = ocr_result.candidate
                        schematic_ocr_supplement_candidates = ocr_result.candidates
                    elif fragmented_invisible_text_layer_should_yield_to_ocr(
                        text,
                        ocr_text,
                        (
                            ocr_result.candidate.result.confidence
                            if ocr_result.candidate is not None
                            else None
                        ),
                        native_layer_is_fragmented=fragmented_invisible_text_layer,
                    ):
                        text = ocr_text
                        replaced_fragmented_invisible_text_layer = True
                        pre_reconciliation_text_source = "ocr_replace_fragmented_invisible"
                        final_output_lines = ocr_result_output_lines(
                            page,
                            ocr_result,
                            text,
                        )
                        schematic_ocr_supplement_candidate = ocr_result.candidate
                        schematic_ocr_supplement_candidates = ocr_result.candidates
                    elif ocr_postprocess.should_use_reconciled_native_ocr_text(
                        text,
                        ocr_text,
                        reconciled_ocr_text,
                    ):
                        text = reconciled_ocr_text
                        pre_reconciliation_text_source = "reconciled_native_ocr"
                        final_output_lines = (
                            reconciled_ocr_lines
                            or best_effort_resolved_text_lines(
                                text,
                                final_output_lines,
                                native_output_lines,
                                ocr_result.output_lines,
                                ocr_result.selected_output_lines,
                            )
                        )
                        schematic_ocr_supplement_candidate = ocr_result.candidate
                        schematic_ocr_supplement_candidates = ocr_result.candidates
                    elif should_replace_text_with_ocr(
                        page,
                        text,
                        ocr_result,
                        native_runs=chars,
                        native_geometry=native_geometry_summary,
                        vector_text=vector_text,
                    ):
                        text = ocr_text
                        pre_reconciliation_text_source = "ocr_replace_general"
                        final_output_lines = ocr_result_output_lines(
                            page,
                            ocr_result,
                            text,
                        )
                        schematic_ocr_supplement_candidate = ocr_result.candidate
                        schematic_ocr_supplement_candidates = ocr_result.candidates
            if (
                broad_ocr_result is None
                and ocr_postprocess.should_try_figure_ocr_supplement(page, text)
                and ocr_postprocess.should_try_ocr_supplement(page, text)
                and page_has_image_only_full_page_figure_region(page)
            ):
                ocr_result = extract_ocr_page_result(
                    page,
                    ocr_session=ocr_session,
                )
                ocr_text = ocr_result.text
                if ocr_postprocess.should_reject_full_page_ocr_result(
                    text,
                    ocr_text,
                ):
                    if cache is not None:
                        cache["ocr_page_result_rejected"] = "garbled_full_page_ocr"
                else:
                    broad_ocr_result = ocr_result
                    if should_replace_text_with_ocr(
                        page,
                        text,
                        ocr_result,
                        native_runs=chars,
                        native_geometry=native_geometry_summary,
                        vector_text=vector_text,
                    ):
                        text = ocr_text
                        pre_reconciliation_text_source = "ocr_replace_general"
                        final_output_lines = ocr_result_output_lines(
                            page,
                            ocr_result,
                            text,
                        )
                        schematic_ocr_supplement_candidate = ocr_result.candidate
                        schematic_ocr_supplement_candidates = ocr_result.candidates
            if ocr_postprocess.should_try_figure_ocr_supplement(page, text):
                ocr_result = extract_figure_ocr_page_result(
                    page,
                    ocr_session=ocr_session,
                    broad_candidate=(
                        broad_ocr_result.candidate if broad_ocr_result is not None else None
                    ),
                )
                figure_ocr_result = ocr_result
                figure_should_replace = ocr_postprocess.should_replace_text_with_figure_ocr(
                    page,
                    text,
                    ocr_result,
                )
                if figure_should_replace and broad_page_ocr_should_win_over_figure_ocr(
                    broad_ocr_result,
                    ocr_result,
                ):
                    preserve_complete_page_ocr_text = True
                    final_output_lines = ()
                elif figure_should_replace:
                    text = ocr_result.text
                    pre_reconciliation_text_source = "figure_ocr_replace"
                    final_output_lines = ocr_result_output_lines(
                        page,
                        ocr_result,
                        text,
                    )
                    replaced_with_figure_ocr = True
            if ocr_postprocess.should_try_embedded_image_text_ocr_supplement(
                page,
                text,
            ):
                embedded_image_text_result = extract_embedded_image_text_ocr_page_result(
                    page,
                    ocr_session=ocr_session,
                )
            if (
                broad_ocr_result is None
                and not replaced_with_figure_ocr
                and ocr_postprocess.should_try_ocr_supplement(page, text)
            ):
                ocr_result = extract_ocr_page_result(
                    page,
                    ocr_session=ocr_session,
                )
                ocr_text = ocr_result.text
                if ocr_postprocess.should_reject_full_page_ocr_result(
                    text,
                    ocr_text,
                ):
                    if cache is not None:
                        cache["ocr_page_result_rejected"] = "garbled_full_page_ocr"
                else:
                    broad_ocr_result = ocr_result
                    if should_replace_text_with_ocr(
                        page,
                        text,
                        ocr_result,
                        native_runs=chars,
                        native_geometry=native_geometry_summary,
                    ):
                        text = ocr_text
                        pre_reconciliation_text_source = "ocr_supplement_replace"
                        final_output_lines = ocr_result_output_lines(
                            page,
                            ocr_result,
                            text,
                        )
            preserved_raw_ocr_text = bool(
                preserve_complete_page_ocr_text
                or should_preserve_sparse_text_table_ocr_result(
                    page,
                    pre_ocr_native_text,
                    broad_ocr_result,
                    pre_reconciliation_text_source,
                )
                or (
                    broad_ocr_result is not None
                    and broad_ocr_result.preserve_raw_text
                    and text == broad_ocr_result.text
                )
                or (
                    omit_native_text_from_ocr_render
                    and broad_ocr_result is not None
                    and pre_reconciliation_text_source != "native"
                )
            )
            if (
                preserved_raw_ocr_text
                and broad_ocr_result is not None
                and broad_ocr_result.candidate is not None
            ):
                text = broad_ocr_result.candidate.result.text
                final_output_lines = ()
            has_reconciliation_source = bool(
                (broad_ocr_result is not None and broad_ocr_result.text)
                or (figure_ocr_result is not None and figure_ocr_result.text)
                or (embedded_image_text_result is not None and embedded_image_text_result.text)
                or vector_result.text
            )
            native_text_tokens = extracted_text_token_count(text)
            preserve_substantial_native_lines = bool(
                pre_reconciliation_text_source == "native"
                and native_text_tokens >= 700
                and ocr_page_analysis.native_text_layer_has_substantial_page_coverage(
                    page,
                    native_text_tokens,
                )
            )
            if has_reconciliation_source and not (
                replaced_fragmented_invisible_text_layer
                or preserved_raw_ocr_text
                or preserve_substantial_native_lines
            ):
                reconciliation = ocr_line_reconciliation.reconcile_page_text_lines(
                    text,
                    final_output_lines,
                    ocr_line_reconciliation.OcrLineReconciliationSources(
                        broad_page_result=broad_ocr_result,
                        figure_result=(None if replaced_with_figure_ocr else figure_ocr_result),
                        embedded_image_result=embedded_image_text_result,
                        vector_result=vector_result,
                    ),
                )
                if reconciliation.text_lines:
                    reconciled_text = render_resolved_text_lines(reconciliation.text_lines)
                    if reconciled_text and reconciled_text != text:
                        text = reconciled_text
                        final_output_lines = reconciliation.text_lines
                    elif reconciled_text == text:
                        final_output_lines = reconciliation.text_lines
            if cache is not None:
                cache["ocr_line_reconciliation_input"] = {
                    "text_source": pre_reconciliation_text_source,
                    "text_lines": len(text.splitlines()),
                    "resolved_lines": len(final_output_lines),
                    "preserved_raw_text": preserved_raw_ocr_text,
                }
                cache["ocr_pre_reconciliation_source"] = pre_reconciliation_text_source
            pruned_output_lines = precision_clean_dominant_image_label_output_lines(
                page,
                final_output_lines,
                broad_page_result=broad_ocr_result,
                figure_result=figure_ocr_result,
            )
            pruned_output_lines = precision_prune_render_consensus_output_lines(
                pruned_output_lines,
                broad_page_result=broad_ocr_result,
            )
            pruned_output_lines = ocr_postprocess.prune_weak_ocr_artifact_output_lines(
                pruned_output_lines
            )
            pruned_output_lines = ocr_postprocess.prune_embedded_image_band_noise_output_lines(
                pruned_output_lines
            )
            pruned_output_lines = ocr_postprocess.prune_malformed_edge_url_output_lines(
                pruned_output_lines
            )
            pruning_ocr_result = broad_ocr_result
            if pruning_ocr_result is not None and pruning_ocr_result.candidate is not None:
                pruned_output_lines = ocr_postprocess.repair_word_geometry_noise_output_lines(
                    pruned_output_lines,
                    pruning_ocr_result.candidate,
                )
            pruned_output_lines = ocr_postprocess.prune_edge_noise_output_lines(pruned_output_lines)
            pruned_output_lines = precision_clean_degradation_chart_output_lines(
                page,
                pruned_output_lines,
            )
            pruned_output_lines = supplement_image_only_layout_top_lines(
                page,
                pruned_output_lines,
                broad_page_result=broad_ocr_result,
            )
            if (
                replaced_fragmented_invisible_text_layer
                or preserved_raw_ocr_text
                or preserve_substantial_native_lines
            ):
                # The native layer is known to be character-fragmented and
                # the replacement candidate has already passed strict gates,
                # or rendering the selected dense OCR geometry was proven to
                # discard material text. Reconciliation and generic pruning
                # must not repeat the same loss in either narrow recovery path.
                pruned_output_lines = final_output_lines
            if pruned_output_lines != final_output_lines:
                final_output_lines = pruned_output_lines
                text = render_resolved_text_lines(final_output_lines)
            if preserved_substantial_native_text_table or preserve_substantial_native_lines:
                text = pre_ocr_native_text
                final_output_lines = pre_ocr_native_output_lines
                broad_ocr_result = None
                schematic_ocr_supplement_candidate = None
                schematic_ocr_supplement_candidates = ()
        finally:
            record_ocr_deskew_diagnostics(page, ocr_session)
            ocr_session.close()
    if preserve_substantial_native_lines:
        text = ocr_text_analysis.repair_formula_control_delimiters(pre_ocr_native_text)
        return cache_page_extraction_snapshot(cache, text, pre_ocr_native_output_lines)
    schematic_consensus_candidates = schematic_ocr_supplement_candidates or (
        (schematic_ocr_supplement_candidate,)
        if schematic_ocr_supplement_candidate is not None
        else ()
    )
    region_classification = (
        classify_page_region(
            text,
            vector_text=vector_text,
            candidates=schematic_consensus_candidates,
            page=page,
            native_runs=chars,
            media_box=page.media_box,
            include_dominant_image=ocr_enabled,
        )
        if ocr_enabled or vector_text or schematic_consensus_candidates
        else None
    )
    if (
        preserved_raw_ocr_text
        and region_classification is not None
        and region_classification.kind == "schematic"
    ):
        text = ocr_schematic.remove_schematic_ocr_artifact_tokens(text)
    if region_classification is not None and region_classification.kind == "dense_table":
        text = remove_dense_table_ocr_artifact_tokens(text)
    if cache is not None and region_classification is not None:
        cache["page_region_classification"] = region_classification
    if region_classification is not None and (
        ocr_schematic.region_classification_supports_schematic_consensus(region_classification)
    ):
        previous_text = text
        schematic_text = ocr_schematic.repair_schematic_ocr_text_with_support(
            text,
            vector_text,
        )
        if schematic_text != previous_text:
            final_output_lines = ()
        text = ocr_schematic.schematic_ocr_text_candidates_supplement(
            schematic_text,
            schematic_consensus_candidates,
            vector_text or text,
            coverage_lines=ocr_schematic.schematic_supplement_coverage_lines(
                page,
                vector_result,
            ),
            allow_rendered_candidates=not vector_text,
        )
        if text != schematic_text:
            final_output_lines = ()
    cached_ocr_result = broad_ocr_result
    cached_candidates = tuple(
        candidate
        for candidate in (cached_ocr_result.candidates if cached_ocr_result is not None else ())
        if isinstance(candidate, ocr_candidates.OcrCandidate)
    )
    if cached_candidates:
        supplemented_text = ocr_postprocess.append_line_art_ocr_candidate_supplement(
            text,
            tuple(cached_candidates),
        )
        if supplemented_text != text:
            text = supplemented_text
            final_output_lines = ()
    text = ocr_text_analysis.repair_formula_control_delimiters(text)
    token_repair_support_texts: list[str] = [
        text,
        render_resolved_text_lines(native_output_lines),
        vector_text,
    ]
    token_repair_support_texts.extend(
        str(candidate.result.text) for candidate in cached_candidates if candidate.result.text
    )
    repaired_output_lines = ocr_postprocess.repair_document_local_identifier_output_lines(
        final_output_lines,
        support_texts=token_repair_support_texts,
    )
    if (
        not (replaced_fragmented_invisible_text_layer or preserved_raw_ocr_text)
        and repaired_output_lines != final_output_lines
    ):
        final_output_lines = repaired_output_lines
        text = render_resolved_text_lines(final_output_lines)
    cached_ocr_result = broad_ocr_result
    if cached_ocr_result is not None and cached_ocr_result.candidate is not None:
        geometry_repaired_output_lines = ocr_postprocess.repair_word_geometry_noise_output_lines(
            final_output_lines,
            cached_ocr_result.candidate,
        )
        if (
            not (replaced_fragmented_invisible_text_layer or preserved_raw_ocr_text)
            and geometry_repaired_output_lines != final_output_lines
        ):
            final_output_lines = geometry_repaired_output_lines
            text = render_resolved_text_lines(final_output_lines)
    shadow_pruned_output_lines = ocr_postprocess.prune_shadowed_selected_output_lines(
        final_output_lines
    )
    if (
        not (replaced_fragmented_invisible_text_layer or preserved_raw_ocr_text)
        and shadow_pruned_output_lines != final_output_lines
    ):
        final_output_lines = shadow_pruned_output_lines
        text = render_resolved_text_lines(final_output_lines)
    fully_covered_fusion_lines = ocr_postprocess.prune_fully_covered_fusion_lines(
        final_output_lines
    )
    if fully_covered_fusion_lines != final_output_lines:
        final_output_lines = fully_covered_fusion_lines
        text = render_resolved_text_lines(final_output_lines)
    suffix_pruned_output_lines = ocr_postprocess.prune_shadowed_band_split_suffix_output_lines(
        final_output_lines
    )
    if (
        not (replaced_fragmented_invisible_text_layer or preserved_raw_ocr_text)
        and suffix_pruned_output_lines != final_output_lines
    ):
        final_output_lines = suffix_pruned_output_lines
        text = render_resolved_text_lines(final_output_lines)
    repaired_text = ocr_postprocess.repair_document_local_identifier_text(
        text,
        support_texts=token_repair_support_texts,
        normalize_ocr_noise=pre_reconciliation_text_source != "native",
    )
    if repaired_text != text:
        text = repaired_text
        final_output_lines = best_effort_resolved_text_lines(
            text,
            final_output_lines,
        )
    direct_hyphen_repaired_text = ocr_postprocess.repair_direct_hyphenated_line_continuations_text(
        text
    )
    if direct_hyphen_repaired_text != text:
        text = direct_hyphen_repaired_text
        final_output_lines = best_effort_resolved_text_lines(
            text,
            final_output_lines,
        )
    compacted_footnote_url_text = ocr_postprocess.compact_footnote_url_markers_text(text)
    if compacted_footnote_url_text != text:
        text = compacted_footnote_url_text
        final_output_lines = best_effort_resolved_text_lines(
            text,
            final_output_lines,
        )
    if region_classification is not None and region_classification.kind == "dense_table":
        text = remove_dense_table_ocr_artifact_tokens(text)
        text = supplement_dense_table_native_symbols(
            text,
            pre_ocr_native_text,
            region_classification,
        )
    final_lines_text = (
        ocr_text_analysis.repair_formula_control_delimiters(
            render_resolved_text_lines(final_output_lines)
        )
        if final_output_lines
        else ""
    )
    if final_output_lines and text == final_lines_text:
        resolved_output_lines = final_output_lines
    else:
        resolved_output_lines = ()
    return cache_page_extraction_snapshot(cache, text, resolved_output_lines)

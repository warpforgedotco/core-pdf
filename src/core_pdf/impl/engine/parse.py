# SPDX-License-Identifier: AGPL-3.0-only
"""Capture, recognize, and emit PDF content."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum
from functools import cache
from html.parser import HTMLParser
from importlib import import_module
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import numpy

from core_pdf.impl.engine.array_views import contiguous_bytes, resample_nearest, uint8_image_view
from core_pdf.impl.engine.execution import RUNTIME, TaskScope, WorkStage
from core_pdf.impl.engine.image_cache import ImageCacheKey
from core_pdf.impl.engine.layout.geometry import rect_tuple
from core_pdf.impl.engine.layout.glyphs import GlyphObservation, GlyphUnicodeSemantics
from core_pdf.impl.engine.layout.models import TextRun
from core_pdf.impl.engine.layout.spatial import (
    SpatialIndex,
    bbox_intersection_area,
)
from core_pdf.impl.engine.layout.spatial import (
    bbox_overlap_ratio as spatial_bbox_overlap_ratio,
)
from core_pdf.impl.engine.layout.text_lines import (
    reconstruct_layout_line_text,
)
from core_pdf.impl.engine.newstroke import NewstrokeDecode, decode_newstroke_drawings
from core_pdf.impl.engine.rendering import (
    DisplayList,
    PathPaintItem,
    PathPaintKind,
    RasterImage,
    RenderedPage,
    RenderOptions,
    compose_page,
    rasterize_packed_stroked_paths,
)
from core_pdf.impl.engine.spec.s_07_content.capture import CapturedDrawing, CapturedInlineImage
from core_pdf.impl.engine.spec.s_07_content.operations import (
    ContentOperatorCounts,
)
from core_pdf.impl.engine.spec.s_07_content.page_program import PageProgram
from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_08_graphics.image_decode import decode_pdf_image
from core_pdf.impl.engine.spec.s_08_graphics.image_metadata import (
    image_filter_names,
    pdf_positive_int,
)
from core_pdf.impl.engine.stroked_text import (
    GlyphSignature,
    StrokedTextDecode,
    StrokedTextObservation,
    StrokedTextProfile,
    StrokedTextRun,
    StrokedTextSeed,
    decode_stroked_text_profile,
    decode_stroked_text_profile_with_alphabet,
    decode_stroked_text_profile_with_supplemental_seeds,
    profile_stroked_text,
)
from core_pdf.impl.engine.structured import (
    Block,
    BlockKind,
    Diagnostic,
    Document,
    Figure,
    Page,
    Table,
    TableCell,
    TextLine,
    TextSpan,
)
from core_pdf.impl.models import LineRecord
from core_pdf.impl.objects import PdfStream

# ===== model =====


FloatArray = numpy.ndarray[Any, numpy.dtype[numpy.float32]]
IntArray = numpy.ndarray[Any, numpy.dtype[numpy.int64]]
ByteArray = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]
BoolArray = numpy.ndarray[Any, numpy.dtype[numpy.bool_]]

PRIMARY_OCR_PIXELS = 6_000_000
OCR_PREFLIGHT_PIXELS = 1_000_000
HIDDEN_TEXT_VERIFY_PIXELS = 2_000_000
MAX_OCR_PIXELS = 16_000_000
MAX_OCR_RASTER_BYTES = MAX_OCR_PIXELS * 4
OCR_RESCUE_MIN_WEAK_INK_RATIO = 0.03
OCR_RESCUE_SATURATED_MEAN_INK = 0.85
OCR_RESCUE_MIN_CONFIDENCE = 95.0
OCR_RESCUE_LARGE_TEXT_HEIGHT = 32.0
OCR_PARALLEL_TILE_MIN_VECTOR_COMPLEXITY = 100_000
HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE = 80.0
HIDDEN_TEXT_VERIFY_MIN_MATCHED_TOKENS = 24
HIDDEN_TEXT_VERIFY_MIN_TOKEN_OVERLAP = 0.64
HIDDEN_TEXT_VERIFY_MIN_SPATIAL_OVERLAP = 0.55


class ObservationSource(IntEnum):
    NATIVE = 0
    OCR = 1
    STRUCTURE = 2
    FORM = 3


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


class PagePreflightClass(StrEnum):
    NATIVE_TEXT = "native-text"
    IMAGE_ONLY = "image-only"
    VECTOR_DIAGRAM = "vector-diagram"
    MIXED = "mixed"
    LIKELY_MALFORMED = "likely-malformed"


def internal_readonly(array: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Columnar text observations optimized for vectorized geometry operations."""

    text: tuple[str, ...]
    bbox: FloatArray
    polygon: FloatArray
    source: ByteArray
    confidence: FloatArray
    sequence: IntArray
    visible: BoolArray
    rotation: IntArray
    font_size: FloatArray
    line_break_before: BoolArray
    references: tuple[Any | None, ...]

    def __post_init__(self) -> None:
        size = len(self.text)
        if self.bbox.shape != (size, 4):
            raise ValueError("observation bboxes must have shape (n, 4)")
        if self.polygon.shape != (size, 8):
            raise ValueError("observation polygons must have shape (n, 8)")
        if len(self.references) != size:
            raise ValueError("observation references must match the text column")
        for column in (
            self.bbox,
            self.polygon,
            self.source,
            self.confidence,
            self.sequence,
            self.visible,
            self.rotation,
            self.font_size,
            self.line_break_before,
        ):
            if len(column) != size:
                raise ValueError("observation columns must have equal length")
            internal_readonly(column)

    def __len__(self) -> int:
        return len(self.text)

    @classmethod
    def empty(cls) -> ObservationBatch:
        return cls(
            (),
            numpy.empty((0, 4), dtype=numpy.float32),
            numpy.empty((0, 8), dtype=numpy.float32),
            numpy.empty(0, dtype=numpy.uint8),
            numpy.empty(0, dtype=numpy.float32),
            numpy.empty(0, dtype=numpy.int64),
            numpy.empty(0, dtype=numpy.bool_),
            numpy.empty(0, dtype=numpy.int64),
            numpy.empty(0, dtype=numpy.float32),
            numpy.empty(0, dtype=numpy.bool_),
            (),
        )

    @classmethod
    def from_columns(
        cls,
        text: Iterable[str],
        bbox: Iterable[tuple[float, float, float, float]],
        *,
        polygon: Iterable[tuple[float, ...]] | None = None,
        source: ObservationSource,
        confidence: Iterable[float] | None = None,
        sequence: Iterable[int] | None = None,
        visible: Iterable[bool] | None = None,
        rotation: Iterable[int] | None = None,
        font_size: Iterable[float] | None = None,
        line_break_before: Iterable[bool] | None = None,
        references: Iterable[Any | None] | None = None,
    ) -> ObservationBatch:
        texts = cast(tuple[str, ...], text if isinstance(text, tuple) else tuple(text))
        size = len(texts)
        if size == 0:
            return cls.empty()
        boxes = numpy.asarray(
            bbox if isinstance(bbox, (list, tuple)) else tuple(bbox), dtype=numpy.float32
        ).reshape((size, 4))
        polygons = (
            numpy.asarray(
                polygon if isinstance(polygon, (list, tuple)) else tuple(polygon),
                dtype=numpy.float32,
            ).reshape((size, 8))
            if polygon is not None
            else numpy.full((size, 8), numpy.nan, dtype=numpy.float32)
        )
        conf_arr = (
            numpy.asarray(
                confidence if isinstance(confidence, (list, tuple)) else tuple(confidence),
                dtype=numpy.float32,
            )
            if confidence is not None
            else numpy.full(size, numpy.nan, dtype=numpy.float32)
        )
        seq_arr = (
            numpy.asarray(
                sequence if isinstance(sequence, (list, tuple, range)) else tuple(sequence),
                dtype=numpy.int64,
            )
            if sequence is not None
            else numpy.arange(size, dtype=numpy.int64)
        )
        vis_arr = (
            numpy.asarray(
                visible if isinstance(visible, (list, tuple)) else tuple(visible),
                dtype=numpy.bool_,
            )
            if visible is not None
            else numpy.ones(size, dtype=numpy.bool_)
        )
        rot_arr = (
            numpy.asarray(
                rotation if isinstance(rotation, (list, tuple)) else tuple(rotation),
                dtype=numpy.int64,
            )
            if rotation is not None
            else numpy.zeros(size, dtype=numpy.int64)
        )
        font_arr = (
            numpy.asarray(
                font_size if isinstance(font_size, (list, tuple)) else tuple(font_size),
                dtype=numpy.float32,
            )
            if font_size is not None
            else numpy.full(size, numpy.nan, dtype=numpy.float32)
        )
        line_arr = (
            numpy.asarray(
                line_break_before
                if isinstance(line_break_before, (list, tuple))
                else tuple(line_break_before),
                dtype=numpy.bool_,
            )
            if line_break_before is not None
            else numpy.zeros(size, dtype=numpy.bool_)
        )
        ref_tuple = (
            (references if isinstance(references, tuple) else tuple(references))
            if references is not None
            else (None,) * size
        )
        return cls(
            texts,
            boxes,
            polygons,
            numpy.full(size, int(source), dtype=numpy.uint8),
            conf_arr,
            seq_arr,
            vis_arr,
            rot_arr,
            font_arr,
            line_arr,
            ref_tuple,
        )

    def take(self, indexes: Sequence[int] | IntArray) -> ObservationBatch:
        if (
            isinstance(indexes, (list, tuple))
            and len(indexes) == len(self)
            and all(int(cast(Any, index)) == position for position, index in enumerate(indexes))
        ):
            # A no-op selection is already immutable and does not need a
            # second allocation of every column.
            return self
        indexes = numpy.asarray(indexes, dtype=numpy.int64)
        return ObservationBatch(
            tuple(self.text[int(index)] for index in indexes),
            self.bbox[indexes],
            self.polygon[indexes],
            self.source[indexes],
            self.confidence[indexes],
            self.sequence[indexes],
            self.visible[indexes],
            self.rotation[indexes],
            self.font_size[indexes],
            self.line_break_before[indexes],
            tuple(self.references[int(index)] for index in indexes),
        )

    def select(self, mask: BoolArray) -> ObservationBatch:
        return self.take(numpy.flatnonzero(mask))

    def view(self, mask: BoolArray | None = None) -> ObservationView:
        indexes = (
            numpy.arange(len(self), dtype=numpy.int64)
            if mask is None
            else numpy.flatnonzero(mask).astype(numpy.int64, copy=False)
        )
        return ObservationView(self, indexes)

    @classmethod
    def concatenate(cls, *batches: ObservationBatch) -> ObservationBatch:
        batches = tuple(batch for batch in batches if len(batch))
        if not batches:
            return cls.empty()
        if len(batches) == 1:
            return batches[0]
        return cls(
            tuple(text for batch in batches for text in batch.text),
            numpy.concatenate(tuple(batch.bbox for batch in batches)),
            numpy.concatenate(tuple(batch.polygon for batch in batches)),
            numpy.concatenate(tuple(batch.source for batch in batches)),
            numpy.concatenate(tuple(batch.confidence for batch in batches)),
            numpy.concatenate(tuple(batch.sequence for batch in batches)),
            numpy.concatenate(tuple(batch.visible for batch in batches)),
            numpy.concatenate(tuple(batch.rotation for batch in batches)),
            numpy.concatenate(tuple(batch.font_size for batch in batches)),
            numpy.concatenate(tuple(batch.line_break_before for batch in batches)),
            tuple(reference for batch in batches for reference in batch.references),
        )

    @classmethod
    def concatenate_selected(
        cls,
        primary: ObservationBatch,
        secondary: ObservationBatch,
        secondary_mask: BoolArray,
    ) -> ObservationBatch:
        """Combine one complete batch and one selection with one numeric allocation.

        NumPy boolean indexing materializes a copy. Building a selected batch and then
        concatenating it would therefore copy every selected numeric column twice.
        """
        if not len(primary) and bool(numpy.all(secondary_mask)):
            return secondary
        indexes = numpy.flatnonzero(secondary_mask)
        if not len(primary):
            return secondary.take(indexes)
        if not len(indexes):
            return primary
        size = len(primary) + len(indexes)

        def combine(column: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
            shape = (size, *column.shape[1:])
            result = numpy.empty(shape, dtype=column.dtype)
            split = len(primary)
            primary_column = getattr(primary, column_name)
            result[:split] = primary_column
            result[split:] = column[indexes]
            return result

        columns: dict[str, numpy.ndarray[Any, Any]] = {}
        for column_name in (
            "bbox",
            "polygon",
            "source",
            "confidence",
            "sequence",
            "visible",
            "rotation",
            "font_size",
            "line_break_before",
        ):
            columns[column_name] = combine(getattr(secondary, column_name))
        return cls(
            (*primary.text, *(secondary.text[int(index)] for index in indexes)),
            columns["bbox"],
            columns["polygon"],
            columns["source"],
            columns["confidence"],
            columns["sequence"],
            columns["visible"],
            columns["rotation"],
            columns["font_size"],
            columns["line_break_before"],
            (*primary.references, *(secondary.references[int(index)] for index in indexes)),
        )


@dataclass(frozen=True, slots=True)
class ObservationView:
    """An immutable zero-copy selection over an observation batch.

    The index vector is tiny compared with copying every geometry column. Consumers
    materialize only when they must hand ownership to another stage.
    """

    batch: ObservationBatch
    indexes: IntArray

    def __post_init__(self) -> None:
        if self.indexes.ndim != 1:
            raise ValueError("observation view indexes must be one-dimensional")
        internal_readonly(self.indexes)

    def __len__(self) -> int:
        return len(self.indexes)

    def materialize(self) -> ObservationBatch:
        return self.batch.take(self.indexes)


@dataclass(frozen=True, slots=True)
class TextQualityStats:
    """Shape-only text quality signals used for routing and diagnostics."""

    token_count: int = 0
    wordlike_ratio: float = 0.0
    short_token_ratio: float = 0.0
    symbol_ratio: float = 0.0
    non_ascii_ratio: float = 0.0
    digit_token_ratio: float = 0.0

    @property
    def noise_score(self) -> float:
        return max(
            0.0,
            min(
                1.0,
                self.short_token_ratio * 0.35
                + self.symbol_ratio * 0.30
                + self.non_ascii_ratio * 0.20
                + self.digit_token_ratio * 0.15
                - self.wordlike_ratio * 0.25,
            ),
        )

    def as_cache_dict(self) -> dict[str, float | int]:
        return {
            "token_count": self.token_count,
            "wordlike_ratio": self.wordlike_ratio,
            "short_token_ratio": self.short_token_ratio,
            "symbol_ratio": self.symbol_ratio,
            "non_ascii_ratio": self.non_ascii_ratio,
            "digit_token_ratio": self.digit_token_ratio,
            "noise_score": self.noise_score,
        }


def internal_text_quality_stats(text: str) -> TextQualityStats:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return TextQualityStats()
    wordlike = 0
    short_tokens = 0
    digit_tokens = 0
    nonspace = 0
    symbols = 0
    non_ascii = 0
    for token in tokens:
        compact = [character for character in token if not character.isspace()]
        if len(compact) <= 2:
            short_tokens += 1
        if any(character.isdigit() for character in compact):
            digit_tokens += 1
        letters = [character for character in compact if character.isalpha()]
        if len(letters) >= 3 and any(character.casefold() in "aeiou" for character in letters):
            wordlike += 1
        for character in compact:
            nonspace += 1
            if not character.isalnum():
                symbols += 1
            if ord(character) > 127:
                non_ascii += 1
    if not nonspace:
        return TextQualityStats(token_count=len(tokens))
    return TextQualityStats(
        token_count=len(tokens),
        wordlike_ratio=wordlike / len(tokens),
        short_token_ratio=short_tokens / len(tokens),
        symbol_ratio=symbols / nonspace,
        non_ascii_ratio=non_ascii / nonspace,
        digit_token_ratio=digit_tokens / len(tokens),
    )


@dataclass(frozen=True, slots=True)
class GlyphEvidence:
    """Page-level evidence about whether glyph identifiers carry real Unicode."""

    glyph_count: int = 0
    visible_glyphs: int = 0
    semantic_characters: int = 0
    authoritative_glyphs: int = 0
    heuristic_glyphs: int = 0
    unknown_glyphs: int = 0
    unsupported_glyphs: int = 0
    low_confidence_glyphs: int = 0
    actual_text_characters: int = 0
    source_counts: tuple[tuple[str, int], ...] = ()

    @property
    def mapped_glyphs(self) -> int:
        return self.authoritative_glyphs + self.heuristic_glyphs

    @property
    def mapped_ratio(self) -> float:
        return self.mapped_glyphs / max(1, self.glyph_count)

    @property
    def authoritative_ratio(self) -> float:
        return self.authoritative_glyphs / max(1, self.glyph_count)

    @property
    def unknown_ratio(self) -> float:
        return self.unknown_glyphs / max(1, self.glyph_count)

    @property
    def low_confidence_ratio(self) -> float:
        return self.low_confidence_glyphs / max(1, self.glyph_count)

    @property
    def unsupported_ratio(self) -> float:
        return self.unsupported_glyphs / max(1, self.glyph_count)

    def inflation(self, characters: int) -> float:
        return self.glyph_count / max(1, characters)

    def as_cache_dict(self) -> dict[str, object]:
        return {
            "glyph_count": self.glyph_count,
            "visible_glyphs": self.visible_glyphs,
            "semantic_characters": self.semantic_characters,
            "authoritative_glyphs": self.authoritative_glyphs,
            "heuristic_glyphs": self.heuristic_glyphs,
            "unknown_glyphs": self.unknown_glyphs,
            "unsupported_glyphs": self.unsupported_glyphs,
            "low_confidence_glyphs": self.low_confidence_glyphs,
            "actual_text_characters": self.actual_text_characters,
            "mapped_ratio": self.mapped_ratio,
            "authoritative_ratio": self.authoritative_ratio,
            "unknown_ratio": self.unknown_ratio,
            "low_confidence_ratio": self.low_confidence_ratio,
            "unsupported_ratio": self.unsupported_ratio,
            "source_counts": dict(self.source_counts),
        }


@dataclass(frozen=True, slots=True)
class StrokedVectorTextEvidence:
    """Compact path families that are likely to be flattened single-line text."""

    trusted: bool = False
    drawing_indexes: tuple[int, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    dominant_compact_paths: int = 0
    candidate_paths: int = 0
    style_count: int = 0


@dataclass(frozen=True, slots=True)
class PageEvidence:
    """Reusable, capture-time evidence for routing and progressive extraction."""

    page_area: float
    native_characters: int
    visible_native_characters: int
    suspicious_characters: int
    image_count: int
    image_area_ratio: float
    vector_complexity: int
    image_boxes: tuple[tuple[float, float, float, float], ...] = ()
    text_coverage: float = 0.0
    full_page_image: bool = False
    uncovered_vector_area: float | None = None
    annotation_text: bool = False
    layout_reasons: tuple[str, ...] = ()
    text_quality: TextQualityStats = field(default_factory=TextQualityStats)
    all_text_quality: TextQualityStats = field(default_factory=TextQualityStats)
    glyphs: GlyphEvidence = field(default_factory=GlyphEvidence)
    painted_native_characters: int | None = None
    painted_text_coverage: float | None = None
    trusted_hidden_text: bool = False
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
    def suspicious_ratio(self) -> float:
        return self.suspicious_characters / max(1, self.native_characters)

    @property
    def visible_text_density(self) -> float:
        return self.visible_native_characters / max(1.0, self.page_area)

    @property
    def hidden_text_layer(self) -> bool:
        painted = (
            self.visible_native_characters
            if self.painted_native_characters is None
            else self.painted_native_characters
        )
        return self.native_characters >= 100 and painted < self.native_characters * 0.20

    @property
    def vector_text_segment_coverage(self) -> float:
        return self.vector_text_matched_segments / max(1, self.vector_text_candidate_segments)


@dataclass(frozen=True, slots=True)
class CapturedPage:
    page: Any
    program: PageProgram
    observations: ObservationBatch
    runs: tuple[TextRun, ...]
    drawings: tuple[CapturedDrawing, ...]
    grid_lines: Any
    inline_images: tuple[CapturedInlineImage, ...]
    evidence: PageEvidence


@dataclass(frozen=True, slots=True)
class PagePreflightFeatures:
    """Cheap page signals gathered before full content interpretation."""

    page_area: float
    stream_count: int
    decoded_stream_bytes: int
    raw_stream_bytes: int
    stream_filters: tuple[str, ...]
    has_fonts: bool
    font_count: int
    image_xobject_count: int
    form_xobject_count: int
    image_pixels: int
    image_raw_bytes: int
    image_filters: tuple[str, ...]
    operator_counts: ContentOperatorCounts


@dataclass(frozen=True, slots=True)
class PagePreflightRecommendation:
    page_class: PagePreflightClass
    capture: str
    ocr: str
    reason: str


@dataclass(frozen=True, slots=True)
class PagePreflight:
    features: PagePreflightFeatures
    recommendation: PagePreflightRecommendation

    def as_cache_dict(self) -> dict[str, object]:
        counts = self.features.operator_counts
        return {
            "class": self.recommendation.page_class.value,
            "capture": self.recommendation.capture,
            "ocr": self.recommendation.ocr,
            "reason": self.recommendation.reason,
            "features": {
                "page_area": self.features.page_area,
                "stream_count": self.features.stream_count,
                "decoded_stream_bytes": self.features.decoded_stream_bytes,
                "raw_stream_bytes": self.features.raw_stream_bytes,
                "stream_filters": self.features.stream_filters,
                "has_fonts": self.features.has_fonts,
                "font_count": self.features.font_count,
                "image_xobject_count": self.features.image_xobject_count,
                "form_xobject_count": self.features.form_xobject_count,
                "image_pixels": self.features.image_pixels,
                "image_raw_bytes": self.features.image_raw_bytes,
                "image_filters": self.features.image_filters,
                "operators": {
                    "text": counts.text,
                    "image": counts.image,
                    "vector_path": counts.vector_path,
                    "vector_paint": counts.vector_paint,
                    "graphics_state": counts.graphics_state,
                    "unknown": counts.unknown,
                    "malformed": counts.malformed,
                },
            },
        }


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
    reason: str = ""
    ocr_passes: tuple[OcrPass, ...] = ()
    verify_hidden_text: bool = False

    @property
    def image_regions_only(self) -> bool:
        return bool(self.ocr_passes) and all(
            ocr_pass.scope is OcrPassScope.IMAGE_REGIONS for ocr_pass in self.ocr_passes
        )


@dataclass(frozen=True, slots=True)
class ParsedLine:
    text: str
    bbox: tuple[float, float, float, float]
    source: str
    confidence: float | None = None
    sequence: int = 0
    rotation: int = 0
    font_size: float | None = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    mark: bool = False
    superscript: bool = False
    subscript: bool = False
    spans: tuple[TextSpan, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    lines: tuple[ParsedLine, ...]
    bbox: tuple[float, float, float, float]
    column_index: int | None = None
    kind: str = "paragraph"
    level: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedPage:
    page_number: int
    width: float
    height: float
    rotation: int
    route: PageRoute
    blocks: tuple[ParsedBlock, ...]
    tables: tuple[Any, ...] = ()
    figures: tuple[Any, ...] = ()
    diagnostics: tuple[str, ...] = ()
    metrics: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def lines(self) -> tuple[ParsedLine, ...]:
        return tuple(line for block in self.blocks for line in block.lines)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


# ===== capture =====


WORD_TOKEN_RE = re.compile(r"\w+")
VECTOR_PAINT_KINDS = frozenset({"fill", "fillstroke", "shading", "stroke"})
VECTOR_PAINT_OPERATION_WEIGHT = 3
PREFLIGHT_CACHE_KEY = "page_preflight_v1"


def internal_suspicious_character_count(text: str) -> int:
    return sum(
        character == "\ufffd"
        or 0xE000 <= ord(character) <= 0xF8FF
        or (not character.isprintable() and not character.isspace())
        for character in text
    )


def internal_normalized_tokens(runs: tuple[Any, ...] | list[Any]) -> tuple[str, ...]:
    return tuple(
        token.casefold()
        for run in runs
        for token in WORD_TOKEN_RE.findall(str(getattr(run, "text", "")))
    )


def internal_token_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    matched = sum(min(count, right_counts.get(token, 0)) for token, count in left_counts.items())
    return matched / min(len(left), len(right))


def internal_clip_bbox(run: Any) -> tuple[float, float, float, float] | None:
    for key, value in reversed(tuple(getattr(run, "provenance", ()))):
        if key != "clip_bbox" or not isinstance(value, (list, tuple)) or len(value) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(part) for part in value)
        except (TypeError, ValueError):
            return None
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
    return None


def internal_discard_duplicate_nested_layers(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    by_depth: dict[int, list[TextRun]] = {}
    for run in runs:
        by_depth.setdefault(run.xobject_depth, []).append(run)
    page_runs = by_depth.get(0, [])
    if not page_runs or len(by_depth) == 1:
        return runs
    tokens_by_depth = {
        depth: internal_normalized_tokens(nested) for depth, nested in by_depth.items()
    }
    page_tokens = tokens_by_depth[0]
    duplicate_depths = {
        depth
        for depth, nested in by_depth.items()
        if depth > 0
        and len(tokens_by_depth[depth]) >= 24
        and internal_token_overlap(tokens_by_depth[depth], page_tokens) >= 0.60
    }
    return tuple(run for run in runs if run.xobject_depth not in duplicate_depths)


def internal_discard_duplicate_clipped_layers(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    groups: dict[tuple[float, float, float, float], list[TextRun]] = {}
    run_boxes: list[tuple[TextRun, tuple[float, float, float, float] | None]] = []
    for run in runs:
        box = internal_clip_bbox(run)
        run_boxes.append((run, box))
        if box is not None:
            groups.setdefault(box, []).append(run)
    if len(groups) < 2:
        return runs
    primary_box, primary_runs = max(
        groups.items(), key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1])
    )
    tokens_by_box = {box: internal_normalized_tokens(group) for box, group in groups.items()}
    primary_tokens = tokens_by_box[primary_box]
    if len(primary_tokens) < 24:
        return runs
    duplicate_boxes = {
        box
        for box, group in groups.items()
        if box != primary_box
        and len(tokens_by_box[box]) >= 24
        and internal_token_overlap(tokens_by_box[box], primary_tokens) >= 0.50
    }
    return tuple(run for run, box in run_boxes if box not in duplicate_boxes)


def internal_extractable_runs(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    active = tuple(run for run in runs if run.text and run.inside_active_clip)
    return internal_discard_duplicate_clipped_layers(
        internal_discard_duplicate_nested_layers(active)
    )


def internal_run_uses_actual_text(run: TextRun) -> bool:
    return any(
        key == "unicode_source" and value in {"actual_text", "structure_actual_text"}
        for key, value in run.provenance
    )


def internal_run_mcid(run: TextRun) -> int | None:
    for key, value in reversed(run.provenance):
        if key == "mcid" and type(value) is int:
            return value
    return None


def internal_apply_structure_actual_text(
    page: Any,
    runs: tuple[TextRun, ...],
) -> tuple[TextRun, ...]:
    if not any(internal_run_mcid(run) is not None for run in runs):
        return runs
    try:
        structure = page.structure
    except (IndexError, TypeError, ValueError):
        return runs
    replaced_mcids: set[int] = set()
    output: list[TextRun] = []
    for run in runs:
        mcid = internal_run_mcid(run)
        if mcid is None:
            output.append(run)
            continue
        if mcid in replaced_mcids:
            continue
        try:
            element = structure[mcid] if 0 <= mcid < len(structure) else None
            actual_text = getattr(element, "actual_text", None)
        except (IndexError, TypeError, ValueError):
            actual_text = None
        if not isinstance(actual_text, str) or not actual_text:
            output.append(run)
            continue
        replaced_mcids.add(mcid)
        output.append(
            internal_copy_run(
                run,
                text=actual_text,
                provenance=(("unicode_source", "structure_actual_text"),),
            )
        )
    return tuple(output)


def internal_glyph_evidence(
    glyphs: tuple[GlyphObservation, ...],
    runs: tuple[TextRun, ...],
) -> GlyphEvidence:
    authoritative = 0
    heuristic = 0
    unknown = 0
    unsupported = 0
    low_confidence = 0
    visible = 0
    semantic_characters = 0
    sources: Counter[str] = Counter()
    glyph_count = 0
    for glyph in glyphs:
        if not glyph.text or glyph.text.isspace():
            continue
        glyph_count += 1
        visible += int(glyph.visible)
        learned_text = internal_learned_glyph_text(glyph)
        source = "learned_ocr" if learned_text is not None else glyph.unicode_source
        text = learned_text or glyph.text
        sources[source or "unspecified"] += 1
        semantics = (
            GlyphUnicodeSemantics.HEURISTIC if learned_text is not None else glyph.unicode_semantics
        )
        if semantics is GlyphUnicodeSemantics.AUTHORITATIVE:
            authoritative += 1
            semantic_characters += sum(not character.isspace() for character in text)
        elif semantics is GlyphUnicodeSemantics.HEURISTIC:
            heuristic += 1
            semantic_characters += sum(not character.isspace() for character in text)
        elif semantics is GlyphUnicodeSemantics.UNSUPPORTED:
            unsupported += 1
        else:
            unknown += 1
        if learned_text is None and (glyph.confidence is None or glyph.confidence < 0.50):
            low_confidence += 1
    actual_text_characters = sum(
        sum(not character.isspace() for character in run.text)
        for run in runs
        if internal_run_uses_actual_text(run)
    )
    return GlyphEvidence(
        glyph_count=glyph_count,
        visible_glyphs=visible,
        semantic_characters=semantic_characters,
        authoritative_glyphs=authoritative,
        heuristic_glyphs=heuristic,
        unknown_glyphs=unknown,
        unsupported_glyphs=unsupported,
        low_confidence_glyphs=low_confidence,
        actual_text_characters=actual_text_characters,
        source_counts=tuple(sorted(sources.items())),
    )


def internal_hidden_text_is_trusted(
    *,
    native_characters: int,
    painted_characters: int,
    suspicious_characters: int,
    quality: TextQualityStats,
    glyphs: GlyphEvidence,
) -> bool:
    if native_characters < 100 or painted_characters >= native_characters * 0.20:
        return False
    if suspicious_characters / max(1, native_characters) > 0.01:
        return False
    if glyphs.actual_text_characters >= max(32, int(native_characters * 0.80)):
        return True
    if not glyphs.glyph_count:
        return False
    clean_mapping = glyphs.low_confidence_ratio <= 0.01 and glyphs.unsupported_ratio <= 0.01
    if not clean_mapping:
        return False
    if glyphs.authoritative_ratio >= 0.90:
        return True
    return (
        glyphs.mapped_ratio >= 0.99
        and glyphs.unknown_ratio <= 0.01
        and quality.wordlike_ratio >= 0.65
        and quality.noise_score <= 0.05
    )


def internal_hidden_text_needs_verification(evidence: PageEvidence) -> bool:
    """Select dense numeric scan layers for a cheap raster-to-text consistency check.

    Clean prose layers can be trusted directly. Numeric tables and indexes do not
    satisfy that language-shaped rule even when their embedded OCR text is accurate.
    Restricting the probe to clean, mapped, image-backed layers avoids spending another
    OCR pass on obviously corrupt encodings and on prose pages where native text can
    omit material visible content.
    """
    quality = evidence.all_text_quality
    glyphs = evidence.glyphs
    return (
        evidence.hidden_text_layer
        and evidence.full_page_image
        and evidence.suspicious_ratio <= 0.01
        and glyphs.glyph_count >= 100
        and glyphs.mapped_ratio >= 0.99
        and glyphs.unknown_ratio <= 0.01
        and glyphs.unsupported_ratio <= 0.01
        and glyphs.low_confidence_ratio <= 0.01
        and quality.token_count >= 100
        and quality.digit_token_ratio >= 0.18
        and quality.symbol_ratio <= 0.30
        and quality.noise_score <= 0.15
    )


def internal_copy_run(
    run: TextRun,
    *,
    text: str | None = None,
    visible: bool | None = None,
    provenance: tuple[tuple[str, object], ...] = (),
) -> TextRun:
    return TextRun(
        text=run.text if text is None else text,
        x0=run.x0,
        y0=run.y0,
        x1=run.x1,
        y1=run.y1,
        tx=run.tx,
        ty=run.ty,
        font_size=run.font_size,
        space_width=run.space_width,
        order=run.order,
        stream_order=run.stream_order,
        xobject_depth=run.xobject_depth,
        font_name=run.font_name,
        is_vertical=run.is_vertical,
        rotation_angle=run.rotation_angle,
        visible=run.visible if visible is None else visible,
        inside_active_clip=run.inside_active_clip,
        line_break_before=run.line_break_before,
        seqno=run.seqno,
        fill_color=run.fill_color,
        advance_bbox=run.advance_bbox,
        ink_bbox=run.ink_bbox,
        baseline=run.baseline,
        provenance=(*run.provenance, *provenance),
        confidence=run.confidence,
        glyph_clusters=run.glyph_clusters,
    )


def internal_promote_hidden_run(run: TextRun) -> TextRun:
    """Create an extraction-only view without changing PDF paint visibility."""
    return internal_copy_run(
        run,
        visible=True,
        provenance=(("extraction_visibility", "trusted-hidden-layer"),),
    )


def internal_observations_from_runs(runs: tuple[TextRun, ...]) -> ObservationBatch:
    """Build the native observation columns shared by capture and hidden-layer promotion."""
    if not runs:
        return ObservationBatch.empty()
    n = len(runs)
    texts = [run.text for run in runs]
    boxes = numpy.empty((n, 4), dtype=numpy.float32)
    polygons = numpy.full((n, 8), numpy.nan, dtype=numpy.float32)
    source = numpy.full(n, int(ObservationSource.NATIVE), dtype=numpy.uint8)
    confidence = numpy.empty(n, dtype=numpy.float32)
    sequence = numpy.empty(n, dtype=numpy.int64)
    visible = numpy.empty(n, dtype=numpy.bool_)
    rotation = numpy.empty(n, dtype=numpy.int64)
    font_size = numpy.empty(n, dtype=numpy.float32)
    line_break_before = numpy.empty(n, dtype=numpy.bool_)

    for i, run in enumerate(runs):
        c = run.coords
        boxes[i, 0] = c[0]
        boxes[i, 1] = c[1]
        boxes[i, 2] = c[2]
        boxes[i, 3] = c[3]
        conf = run.confidence
        confidence[i] = conf if conf is not None else math.nan
        seq = run.seqno
        sequence[i] = seq if seq >= 0 else i
        visible[i] = run.visible
        rotation[i] = run.rotation_angle
        font_size[i] = c[6]
        line_break_before[i] = run.line_break_before

    return ObservationBatch(
        text=tuple(texts),
        bbox=boxes,
        polygon=polygons,
        source=source,
        confidence=confidence,
        sequence=sequence,
        visible=visible,
        rotation=rotation,
        font_size=font_size,
        line_break_before=line_break_before,
        references=runs,
    )


def internal_promoted_hidden_observations(capture: CapturedPage) -> ObservationBatch:
    """Expose a verified hidden layer while preserving its original geometry and ordering."""
    runs = tuple(
        internal_promote_hidden_run(run) if not run.visible else run for run in capture.runs
    )
    return internal_observations_from_runs(runs)


def internal_learned_glyph_text(glyph: GlyphObservation) -> str | None:
    learned = getattr(glyph.font_decoder, "learned_unicode", None)
    if not isinstance(learned, dict):
        return None
    text = learned.get(glyph.code_bytes)
    return text if isinstance(text, str) and len(text) == 1 else None


def internal_apply_learned_unicode_to_run(run: TextRun) -> TextRun:
    if not run.glyph_clusters:
        return run
    source = run.text
    cursor = 0
    output: list[str] = []
    changed = False
    for cluster in run.glyph_clusters:
        for glyph in cluster.glyphs:
            replacement = internal_learned_glyph_text(glyph)
            original = glyph.text
            if replacement is None or not original:
                continue
            position = source.find(original, cursor)
            if position < 0:
                continue
            output.append(source[cursor:position])
            output.append(replacement)
            cursor = position + len(original)
            changed = changed or replacement != original
    if not changed:
        return run
    output.append(source[cursor:])
    return internal_copy_run(
        run,
        text="".join(output),
        provenance=(("unicode_source", "learned_ocr"),),
    )


def internal_vector_complexity(drawings: tuple[Any, ...], grid_lines: Any) -> int:
    """Estimate vector workload without depending on graphics-state bookkeeping.

    Every derived segment contributes geometric work. Paint operations carry a larger
    fixed dispatch and raster cost, while clips, groups, and state markers are control
    records rather than visible vector content.
    """
    paint_operations = sum(
        getattr(drawing, "kind", None) in VECTOR_PAINT_KINDS for drawing in drawings
    )
    return len(grid_lines) + paint_operations * VECTOR_PAINT_OPERATION_WEIGHT


def internal_resource_dict(page: Any) -> dict[Any, Any]:
    try:
        resources = getattr(page, "cached_resources", {})
    except Exception:
        return {}
    return resources if isinstance(resources, dict) else {}


def internal_resolve_pdf_object(page: Any, value: object) -> object:
    resolver = getattr(getattr(page, "document", None), "resolver", None)
    resolve = getattr(resolver, "resolve", None)
    if callable(resolve):
        try:
            return resolve(value)
        except Exception:
            return None
    return value


def internal_resolve_pdf_stream(page: Any, value: object) -> PdfStream | None:
    resolved = internal_resolve_pdf_object(page, value)
    return resolved if isinstance(resolved, PdfStream) else None


def internal_preflight_resource_features(
    page: Any,
) -> tuple[int, int, int, int, int, tuple[str, ...]]:
    resources = internal_resource_dict(page)
    fonts = lookup_dict_key(resources, "Font")
    font_count = len(fonts) if isinstance(fonts, dict) else 0
    xobjects = lookup_dict_key(resources, "XObject")
    if not isinstance(xobjects, dict):
        return font_count, 0, 0, 0, 0, ()

    image_count = 0
    form_count = 0
    image_pixels = 0
    image_raw_bytes = 0
    filters: list[str] = []
    for value in xobjects.values():
        stream = internal_resolve_pdf_stream(page, value)
        if stream is None:
            continue
        dictionary = stream.dictionary
        subtype = normalize_pdf_name(lookup_dict_key(dictionary, "Subtype"))
        if subtype == "Image":
            image_count += 1
            width = pdf_positive_int(lookup_dict_key(dictionary, "Width"))
            height = pdf_positive_int(lookup_dict_key(dictionary, "Height"))
            image_pixels += width * height
            image_raw_bytes += len(stream.raw_data)
            filters.extend(image_filter_names(lookup_dict_key(dictionary, "Filter")))
        elif subtype == "Form":
            form_count += 1
    return font_count, image_count, form_count, image_pixels, image_raw_bytes, tuple(filters)


def internal_preflight_stream_features(
    page: Any,
) -> tuple[int, int, int, tuple[str, ...]]:
    stream_count = 0
    decoded_bytes = 0
    raw_bytes = 0
    filters: list[str] = []
    try:
        content_streams = tuple(getattr(page, "content_streams", ()))
    except Exception:
        return 0, 0, 0, ()
    for stream in content_streams:
        if not isinstance(stream, PdfStream):
            continue
        stream_count += 1
        raw_bytes += len(stream.raw_data)
        filters.extend(image_filter_names(lookup_dict_key(stream.dictionary, "Filter")))
        try:
            view = stream.data_view
            decoded_bytes += len(view)
        except Exception:
            continue
    return stream_count, decoded_bytes, raw_bytes, tuple(filters)


def internal_program_operator_counts(capture: CapturedPage) -> ContentOperatorCounts:
    """Derive routing counts from the already interpreted page program.

    These are deliberately semantic counts rather than raw token counts.  They
    avoid a second lexical pass and better describe the work downstream
    extraction and rendering will actually perform.
    """
    drawings = capture.drawings
    products = getattr(getattr(capture, "program", None), "products", None)
    program_runs = getattr(products, "runs", None)
    operator_runs = capture.runs if program_runs is None else program_runs
    return ContentOperatorCounts(
        # Capture can synthesize deterministic text from vector paths. Keep
        # preflight about the actual PDF operators rather than derived runs.
        text=len(operator_runs),
        image=len(capture.inline_images)
        + sum(getattr(drawing, "kind", None) == "image" for drawing in drawings),
        vector_path=len(capture.grid_lines),
        vector_paint=sum(
            getattr(drawing, "kind", None) in VECTOR_PAINT_KINDS for drawing in drawings
        ),
        graphics_state=sum(
            getattr(drawing, "kind", None)
            in {"state-push", "state-pop", "clip", "group-begin", "group-end"}
            for drawing in drawings
        ),
    )


def internal_classify_preflight(features: PagePreflightFeatures) -> PagePreflightRecommendation:
    counts = features.operator_counts
    has_images = features.image_xobject_count > 0 or counts.image > 0
    has_vectors = counts.vector >= 12 or counts.vector_paint >= 3
    has_text = counts.text > 0 or features.has_fonts
    if counts.malformed or (
        features.stream_count == 0 and not features.has_fonts and features.image_xobject_count == 0
    ):
        return PagePreflightRecommendation(
            PagePreflightClass.LIKELY_MALFORMED,
            "text-only",
            "fallback-page",
            "stream-scan-malformed-or-empty",
        )
    if (
        has_text
        and not has_images
        and not has_vectors
        and counts.unknown <= max(5, counts.total // 4)
    ):
        return PagePreflightRecommendation(
            PagePreflightClass.NATIVE_TEXT,
            "text-only",
            "none",
            "text-operators-with-font-resources",
        )
    if has_images and not has_text and not has_vectors:
        return PagePreflightRecommendation(
            PagePreflightClass.IMAGE_ONLY,
            "images-only",
            "page-or-image-regions",
            "image-resources-without-text-or-vector-paint",
        )
    if has_vectors and not has_images and counts.text <= 4:
        return PagePreflightRecommendation(
            PagePreflightClass.VECTOR_DIAGRAM,
            "graphics",
            "schematic-regions",
            "vector-paint-dominates-stream",
        )
    return PagePreflightRecommendation(
        PagePreflightClass.MIXED,
        "render",
        "current-plan",
        "mixed-or-ambiguous-preflight-signals",
    )


def preflight_page(page: Any, capture: CapturedPage | None = None) -> PagePreflight:
    cache = getattr(page, "extraction_cache", None)
    if cache is not None:
        cached = cache.get(PREFLIGHT_CACHE_KEY)
        if isinstance(cached, PagePreflight):
            return cached

    capture = capture if capture is not None else capture_page(page)
    stream_count, decoded_bytes, raw_bytes, stream_filters = internal_preflight_stream_features(
        page
    )
    counts = internal_program_operator_counts(capture)
    font_count, image_count, form_count, image_pixels, image_raw_bytes, image_filters = (
        internal_preflight_resource_features(page)
    )
    page_width = float(getattr(page, "width", 0.0))
    page_height = float(getattr(page, "height", 0.0))
    features = PagePreflightFeatures(
        page_area=max(1.0, page_width * page_height),
        stream_count=stream_count,
        decoded_stream_bytes=decoded_bytes,
        raw_stream_bytes=raw_bytes,
        stream_filters=stream_filters,
        has_fonts=font_count > 0,
        font_count=font_count,
        image_xobject_count=image_count,
        form_xobject_count=form_count,
        image_pixels=image_pixels,
        image_raw_bytes=image_raw_bytes,
        image_filters=image_filters,
        operator_counts=counts,
    )
    preflight = PagePreflight(features, internal_classify_preflight(features))
    if cache is not None:
        cache[PREFLIGHT_CACHE_KEY] = preflight
        cache["preflight"] = preflight.as_cache_dict()
    return preflight


STROKED_VECTOR_COMPACT_DIMENSION = 4.0
STROKED_VECTOR_RENDER_DIMENSION = 5.0
STROKED_VECTOR_MIN_DOMINANT_PATHS = 300
STROKED_VECTOR_MIN_STYLE_PATHS = 8
STROKED_VECTOR_MIN_COMPACT_RATIO = 0.60
STROKED_VECTOR_MIN_AXIS_COVERAGE = 0.35


def internal_stroked_vector_style(drawing: Any) -> tuple[object, ...] | None:
    """Return a stable paint-style key for an opaque, solid stroked path."""
    if (
        getattr(drawing, "kind", None) not in {"stroke", "fillstroke"}
        or getattr(drawing, "path", None) is None
        or getattr(drawing, "stroke_pattern", None) is not None
    ):
        return None
    raw_color = getattr(drawing, "stroke_color", None)
    if not isinstance(raw_color, (list, tuple)) or not raw_color:
        return None
    try:
        color = tuple(float(component) for component in raw_color)
        raw_opacity = getattr(drawing, "stroke_opacity", None)
        opacity = 1.0 if raw_opacity is None else float(raw_opacity)
        line_width = float(getattr(drawing, "line_width", 0.0))
    except (TypeError, ValueError):
        return None
    if opacity <= 0.0 or not (0.0 < line_width <= 1.5):
        return None
    raw_dash = getattr(drawing, "dash_pattern", None)
    if raw_dash:
        try:
            dash = (tuple(float(value) for value in raw_dash[0]), float(raw_dash[1]))
        except (IndexError, TypeError, ValueError):
            return None
        if dash[0]:
            return None
    else:
        dash = None
    return (
        getattr(drawing, "kind", None),
        color,
        opacity,
        line_width,
        int(getattr(drawing, "line_cap", 0) or 0),
        int(getattr(drawing, "line_join", 0) or 0),
        dash,
        getattr(drawing, "blend_mode", None),
        getattr(drawing, "soft_mask_alpha", None),
    )


def internal_stroked_vector_text_evidence(
    drawings: tuple[Any, ...],
    *,
    page_width: float,
    page_height: float,
    rotation: int = 0,
) -> StrokedVectorTextEvidence:
    """Detect distributed single-line fonts from repeated compact path styles.

    Flattened CAD exports typically paint every glyph stroke as a tiny path with
    one of a few repeated styles.  Wires, frames, and component outlines are much
    larger in at least one axis.  Retaining the qualifying drawing indexes lets
    OCR rasterize only the likely text layer instead of recomposing the page.
    """
    if len(drawings) < 180 or rotation % 360 or page_width <= 0.0 or page_height <= 0.0:
        return StrokedVectorTextEvidence()

    # Values are total paths, compact paths, and compact-path bounds.
    styles: dict[tuple[object, ...], list[float]] = {}
    indexed: list[tuple[int, tuple[object, ...], tuple[float, float, float, float], float]] = []
    for index, drawing in enumerate(drawings):
        style = internal_stroked_vector_style(drawing)
        if style is None:
            continue
        box = rect_tuple(getattr(drawing, "rect", None))
        if box is None:
            continue
        maximum_dimension = max(box[2] - box[0], box[3] - box[1])
        if maximum_dimension <= 0.0:
            continue
        stats = styles.setdefault(
            style,
            [0.0, 0.0, math.inf, -math.inf, math.inf, -math.inf],
        )
        stats[0] += 1.0
        if maximum_dimension <= STROKED_VECTOR_COMPACT_DIMENSION:
            stats[1] += 1.0
            stats[2] = min(stats[2], box[0])
            stats[3] = max(stats[3], box[2])
            stats[4] = min(stats[4], box[1])
            stats[5] = max(stats[5], box[3])
        indexed.append((index, style, box, maximum_dimension))
    if not styles:
        return StrokedVectorTextEvidence()

    dominant_style, dominant = max(styles.items(), key=lambda item: item[1][1])
    del dominant_style
    dominant_paths = int(dominant[1])
    dominant_ratio = dominant_paths / max(1.0, dominant[0])
    width_coverage = (dominant[3] - dominant[2]) / page_width
    height_coverage = (dominant[5] - dominant[4]) / page_height
    if (
        dominant_paths < STROKED_VECTOR_MIN_DOMINANT_PATHS
        or dominant_ratio < STROKED_VECTOR_MIN_COMPACT_RATIO
        or width_coverage < STROKED_VECTOR_MIN_AXIS_COVERAGE
        or height_coverage < STROKED_VECTOR_MIN_AXIS_COVERAGE
    ):
        return StrokedVectorTextEvidence(dominant_compact_paths=dominant_paths)

    selected_styles = {
        style
        for style, stats in styles.items()
        if stats[1] >= STROKED_VECTOR_MIN_STYLE_PATHS
        and stats[1] / max(1.0, stats[0]) >= STROKED_VECTOR_MIN_COMPACT_RATIO
    }
    selected = tuple(
        (index, box)
        for index, style, box, maximum_dimension in indexed
        if style in selected_styles and maximum_dimension <= STROKED_VECTOR_RENDER_DIMENSION
    )
    if len(selected) < STROKED_VECTOR_MIN_DOMINANT_PATHS:
        return StrokedVectorTextEvidence(dominant_compact_paths=dominant_paths)
    bbox = (
        min(box[0] for _, box in selected),
        min(box[1] for _, box in selected),
        max(box[2] for _, box in selected),
        max(box[3] for _, box in selected),
    )
    return StrokedVectorTextEvidence(
        trusted=True,
        drawing_indexes=tuple(index for index, _ in selected),
        bbox=bbox,
        dominant_compact_paths=dominant_paths,
        candidate_paths=len(selected),
        style_count=len(selected_styles),
    )


def internal_uncovered_vector_area(
    drawings: tuple[Any, ...],
    observations: ObservationBatch,
    *,
    page_area: float | None = None,
) -> float | None:
    """Estimate filled vector area not represented by native text.

    This expensive signal is only used on text-bearing, vector-heavy pages. It
    is intentionally conservative: overlap with any native text subtracts from
    the filled path, so false OCR escalation is preferred over missing vector
    text.
    """
    if not drawings or not len(observations):
        return None
    if len(drawings) < 180:
        return None
    native = observations.bbox
    rectangles: list[tuple[float, float, float, float, float]] = []
    for drawing in drawings:
        if getattr(drawing, "kind", None) not in {"fill", "fillstroke"}:
            continue
        rect = rect_tuple(getattr(drawing, "rect", None))
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        # Page backgrounds and exporter-generated white canvases are not text.
        # Charging their bounding boxes as filled glyph area forces otherwise
        # sparse vector documents into a full-page OCR route.
        if page_area is not None and area >= max(1.0, page_area) * 0.80:
            continue
        if area > 0.0:
            rectangles.append((x0, y0, x1, y1, area))
    if not rectangles:
        return 0.0
    if len(rectangles) * len(native) > 65_536:
        native_index = SpatialIndex.from_boxes(native)
        uncovered = 0.0
        for x0, y0, x1, y1, area in rectangles:
            box = (x0, y0, x1, y1)
            covered = sum(
                bbox_intersection_area(box, hit.bbox) for hit in native_index.intersecting_hits(box)
            )
            uncovered += max(0.0, area - min(area, covered))
        return uncovered

    # Evaluate a bounded batch at a time. This keeps the overlap calculation
    # vectorized without allocating a drawings-by-observations matrix for the
    # entire page.
    uncovered = 0.0
    for offset in range(0, len(rectangles), 64):
        batch = numpy.asarray(rectangles[offset : offset + 64], dtype=numpy.float32)
        batch_x0 = batch[:, 0, None]
        batch_y0 = batch[:, 1, None]
        batch_x1 = batch[:, 2, None]
        batch_y1 = batch[:, 3, None]
        overlap_x = numpy.maximum(
            0.0,
            numpy.minimum(native[None, :, 2], batch_x1)
            - numpy.maximum(native[None, :, 0], batch_x0),
        )
        overlap_y = numpy.maximum(
            0.0,
            numpy.minimum(native[None, :, 3], batch_y1)
            - numpy.maximum(native[None, :, 1], batch_y0),
        )
        batch_covered = numpy.sum(overlap_x * overlap_y, axis=1, dtype=numpy.float64)
        areas = batch[:, 4]
        uncovered += float(
            numpy.sum(numpy.maximum(0.0, areas - numpy.minimum(areas, batch_covered)))
        )
    return uncovered


def internal_capture_with_newstroke_text(
    capture: CapturedPage,
    decoded: NewstrokeDecode,
) -> CapturedPage:
    """Promote a page-level, template-verified vector font into native observations."""
    runs = decoded.runs
    observations = internal_observations_from_runs(runs)
    text = "".join(run.text for run in runs)
    characters = sum(not character.isspace() for character in text)
    boxes = observations.bbox
    widths = numpy.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = numpy.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    text_coverage = min(
        1.0,
        float(numpy.sum(widths * heights, dtype=numpy.float64)) / capture.evidence.page_area,
    )
    text_quality = internal_text_quality_stats(text)
    evidence = replace(
        capture.evidence,
        native_characters=characters,
        visible_native_characters=characters,
        suspicious_characters=internal_suspicious_character_count(text),
        text_coverage=text_coverage,
        uncovered_vector_area=internal_uncovered_vector_area(
            capture.drawings,
            observations,
            page_area=capture.evidence.page_area,
        ),
        text_quality=text_quality,
        all_text_quality=text_quality,
        painted_native_characters=characters,
        painted_text_coverage=text_coverage,
        vector_text_characters=characters,
        vector_text_candidate_segments=decoded.candidate_segments,
        vector_text_matched_segments=decoded.matched_segments,
        vector_text_sequences=decoded.sequences,
        vector_text_maximum_error=decoded.maximum_error,
        vector_text_trusted=True,
    )
    return replace(capture, observations=observations, runs=runs, evidence=evidence)


def internal_capture_from_program(
    page: Any,
    program: PageProgram,
) -> CapturedPage:
    cache = getattr(page, "extraction_cache", None)
    cache_key = "captured_page_program_v3"
    if cache is not None:
        cached = cache.get(cache_key)
        if isinstance(cached, CapturedPage) and cached.program is program:
            return cached
    products = program.products
    program_runs = tuple(products.runs)
    glyphs_by_seqno: dict[int, list[str]] = defaultdict(list)
    for glyph in products.glyphs:
        if glyph.font_name:
            glyphs_by_seqno[int(glyph.seqno)].append(glyph.font_name)
    glyph_seqnos = tuple(sorted(glyphs_by_seqno))
    enriched_runs: list[TextRun] = []
    for index, run in enumerate(program_runs):
        next_seqno = (
            program_runs[index + 1].seqno if index + 1 < len(program_runs) else float("inf")
        )
        font_counts = Counter(
            font_name
            for seqno in glyph_seqnos
            if run.seqno <= seqno < next_seqno
            for font_name in glyphs_by_seqno[seqno]
        )
        if font_counts:
            enriched = internal_copy_run(run)
            enriched.font_name = font_counts.most_common(1)[0][0]
            enriched_runs.append(enriched)
        else:
            enriched_runs.append(run)
    structured_runs = internal_apply_structure_actual_text(page, tuple(enriched_runs))
    raw_runs = tuple(
        internal_apply_learned_unicode_to_run(run)
        for run in internal_extractable_runs(structured_runs)
    )
    raw_text = "".join(run.text for run in raw_runs)
    painted_text = "".join(run.text for run in raw_runs if run.visible)
    native_characters = sum(not character.isspace() for character in raw_text)
    painted_native_characters = sum(not character.isspace() for character in painted_text)
    suspicious_characters = internal_suspicious_character_count(raw_text)
    all_text_quality = internal_text_quality_stats(raw_text)
    glyph_evidence = internal_glyph_evidence(tuple(products.glyphs), raw_runs)
    trusted_hidden_text = internal_hidden_text_is_trusted(
        native_characters=native_characters,
        painted_characters=painted_native_characters,
        suspicious_characters=suspicious_characters,
        quality=all_text_quality,
        glyphs=glyph_evidence,
    )
    runs = (
        tuple(internal_promote_hidden_run(run) if not run.visible else run for run in raw_runs)
        if trusted_hidden_text
        else raw_runs
    )
    observations = internal_observations_from_runs(runs)
    visible_text = "".join(run.text for run in runs if run.visible)
    drawings = tuple(products.drawings)
    inline_images = tuple(products.inline_images)
    page_width = float(page.width)
    page_height = float(page.height)
    page_area = max(1.0, page_width * page_height)
    visible = observations.visible
    boxes = observations.bbox
    visible_widths = numpy.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    visible_heights = numpy.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    text_coverage = min(
        1.0,
        float(numpy.sum(visible_widths * visible_heights * visible, dtype=numpy.float64))
        / page_area,
    )
    painted_mask = numpy.fromiter(
        (run.visible for run in raw_runs),
        dtype=numpy.bool_,
        count=len(raw_runs),
    )
    painted_text_coverage = min(
        1.0,
        float(numpy.sum(visible_widths * visible_heights * painted_mask, dtype=numpy.float64))
        / page_area,
    )
    visible_image_areas: list[float] = []
    visible_image_boxes: list[tuple[float, float, float, float]] = []
    for drawing in drawings:
        if getattr(drawing, "kind", None) != "image":
            continue
        box = rect_tuple(getattr(drawing, "rect", None))
        if box is None:
            continue
        width = max(0.0, min(page_width, box[2]) - max(0.0, box[0]))
        height = max(0.0, min(page_height, box[3]) - max(0.0, box[1]))
        if width > 0.0 and height > 0.0:
            visible_image_areas.append(width * height)
            visible_image_boxes.append(
                (
                    max(0.0, box[0]),
                    max(0.0, box[1]),
                    min(page_width, box[2]),
                    min(page_height, box[3]),
                )
            )
    image_count = len(inline_images) + sum(
        area >= page_area * 0.001 for area in visible_image_areas
    )
    grid_lines = products.lines
    full_page_image = any(
        width >= page_width * 0.90 and height >= page_height * 0.90
        for width, height in (
            (
                max(0.0, box[2] - box[0]),
                max(0.0, box[3] - box[1]),
            )
            for box in visible_image_boxes
        )
    )
    uncovered_vector_area = internal_uncovered_vector_area(
        drawings,
        observations,
        page_area=page_area,
    )
    captured = CapturedPage(
        page=page,
        program=program,
        observations=observations,
        runs=runs,
        drawings=drawings,
        grid_lines=grid_lines,
        inline_images=inline_images,
        evidence=PageEvidence(
            page_area=page_area,
            native_characters=native_characters,
            visible_native_characters=sum(not character.isspace() for character in visible_text),
            suspicious_characters=suspicious_characters,
            image_count=image_count,
            image_area_ratio=min(1.0, sum(visible_image_areas) / page_area),
            vector_complexity=internal_vector_complexity(drawings, grid_lines),
            image_boxes=tuple(
                box
                for box, area in zip(
                    visible_image_boxes,
                    visible_image_areas,
                    strict=True,
                )
                if area >= page_area * 0.001
            ),
            text_coverage=text_coverage,
            full_page_image=full_page_image,
            uncovered_vector_area=uncovered_vector_area,
            text_quality=internal_text_quality_stats(visible_text),
            all_text_quality=all_text_quality,
            glyphs=glyph_evidence,
            painted_native_characters=painted_native_characters,
            painted_text_coverage=painted_text_coverage,
            trusted_hidden_text=trusted_hidden_text,
        ),
    )
    newstroke_decode: NewstrokeDecode | None = None
    newstroke_seconds = 0.0
    if not captured.runs and internal_requires_high_resolution_vector_ocr(captured):
        newstroke_started = time.perf_counter()
        newstroke_decode = decode_newstroke_drawings(captured.drawings)
        newstroke_seconds = time.perf_counter() - newstroke_started
        if newstroke_decode.trusted:
            captured = internal_capture_with_newstroke_text(captured, newstroke_decode)
    if not captured.evidence.vector_text_trusted:
        stroked_vector_text = internal_stroked_vector_text_evidence(
            drawings,
            page_width=page_width,
            page_height=page_height,
            rotation=int(getattr(page, "rotation", 0) or 0),
        )
        captured = replace(
            captured,
            evidence=replace(captured.evidence, stroked_vector_text=stroked_vector_text),
        )
    if cache is not None:
        diagnostics = cache.setdefault("capture_diagnostics", {})
        diagnostics["text_quality"] = captured.evidence.text_quality.as_cache_dict()
        diagnostics["all_text_quality"] = captured.evidence.all_text_quality.as_cache_dict()
        diagnostics["glyph_evidence"] = captured.evidence.glyphs.as_cache_dict()
        diagnostics["trusted_hidden_text"] = captured.evidence.trusted_hidden_text
        diagnostics["painted_native_characters"] = captured.evidence.painted_native_characters or 0
        diagnostics["painted_text_coverage"] = captured.evidence.painted_text_coverage or 0.0
        diagnostics["stroked_vector_text"] = {
            "accepted": captured.evidence.stroked_vector_text.trusted,
            "dominant_compact_paths": (
                captured.evidence.stroked_vector_text.dominant_compact_paths
            ),
            "candidate_paths": captured.evidence.stroked_vector_text.candidate_paths,
            "style_count": captured.evidence.stroked_vector_text.style_count,
            "bbox": captured.evidence.stroked_vector_text.bbox,
        }
        if newstroke_decode is not None:
            diagnostics["newstroke"] = {
                "accepted": newstroke_decode.trusted,
                "candidate_segments": newstroke_decode.candidate_segments,
                "matched_segments": newstroke_decode.matched_segments,
                "matched_coverage": newstroke_decode.matched_coverage,
                "glyphs": newstroke_decode.glyphs,
                "characters": newstroke_decode.characters,
                "sequences": newstroke_decode.sequences,
                "maximum_error": newstroke_decode.maximum_error,
                "seconds": newstroke_seconds,
            }
    if cache is not None:
        cache[cache_key] = captured
    return captured


def capture_page(page: Any) -> CapturedPage:
    """Build the canonical page products once and derive routing evidence from them."""
    return internal_capture_from_program(page, page.get_page_program())


# ===== route =====


PSM_AUTO = 3
PSM_SPARSE_TEXT = 11
PSM_SPARSE_TEXT_OSD = 12

# Precision-first extraction thresholds.  Raster text below these confidence
# levels is more likely to be a layout artifact than a useful observation on
# the document classes routed through these passes.
HIDDEN_TEXT_MIN_CONFIDENCE = 90.0
RASTER_TEXT_MIN_CONFIDENCE = 90.0
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


def internal_requires_high_resolution_vector_ocr(capture: CapturedPage) -> bool:
    """Identify pure-vector diagrams whose tiny stroked labels need the maximum raster."""
    evidence = capture.evidence
    if not (
        evidence.image_count == 0
        and evidence.vector_complexity >= 100_000
        and evidence.text_coverage < 0.05
    ):
        return False
    paint_count = 0
    stroke_count = 0
    compact_stroke_count = 0
    for drawing in getattr(capture, "drawings", ()):
        kind = getattr(drawing, "kind", None)
        if kind not in VECTOR_PAINT_KINDS:
            continue
        paint_count += 1
        if kind != "stroke":
            continue
        stroke_count += 1
        box = rect_tuple(getattr(drawing, "rect", None))
        if box is not None and max(box[2] - box[0], box[3] - box[1]) <= 6.0:
            compact_stroke_count += 1
    return (
        stroke_count >= 10_000
        and stroke_count >= paint_count * 0.95
        and compact_stroke_count >= stroke_count * 0.95
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
                    run_if_characters_below=300,
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
    mode = PSM_AUTO if schematic and vector_complexity >= 150_000 else PSM_SPARSE_TEXT
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
                        else NATIVE_UNAVAILABLE_MIN_CONFIDENCE
                    ),
                    adaptive_scale=not high_resolution_vector,
                    character_confidence_threshold=(
                        55.0 if schematic and not high_resolution_vector else None
                    ),
                    region_first=True,
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
                    minimum_confidence=NATIVE_UNAVAILABLE_MIN_CONFIDENCE,
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
                image_modes if schematic else (PSM_SPARSE_TEXT,),
                minimum_confidence=RASTER_TEXT_MIN_CONFIDENCE,
                adaptive_scale=True,
                region_first=True,
                pixel_budget=PRIMARY_OCR_PIXELS,
                include_native_text=True,
            ),
            OcrPass(
                "fallback-regions" if schematic else "fallback-page",
                OcrPassScope.WEAK_REGIONS if schematic else OcrPassScope.PAGE,
                scale,
                (PSM_SPARSE_TEXT,),
                tiles=4 if schematic else 1,
                minimum_confidence=RASTER_TEXT_MIN_CONFIDENCE,
                run_if_characters_below=300 if schematic else 32,
                region_first=False,
                include_native_text=True,
            ),
        ),
    )


def plan_page(capture: CapturedPage) -> WorkPlan:
    return internal_base_plan_page(capture)


# ===== fusion =====


FUSION_GEOMETRY_CHUNK = 256

# Upper bound on elements materialized per vectorized overlap chunk.  The
# chunked path broadcasts FUSION_GEOMETRY_CHUNK candidates against the full
# native set, so only the native box count bounds per-chunk memory.
FUSION_VECTORIZED_ELEMENTS = 1_000_000


def internal_compact_text(text: str) -> str:
    return "".join(character.casefold() for character in text if character.isalnum())


def internal_text_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (internal_compact_text(part) for part in text.casefold().split())
        if len(token) >= 2
    )


def internal_duplicate_of_native_text(
    native_compact: str,
    native_tokens: frozenset[str],
    ocr_text: str,
) -> bool:
    """Detect raster OCR that repeats the page's native text.

    Vector pages can have different coordinate systems after rasterization, so
    geometry alone cannot identify duplicate OCR.  A compact text containment
    check is deliberately limited to reasonably long observations to avoid
    discarding short schematic labels such as ``R1`` or ``+5V``.
    """
    compact = internal_compact_text(ocr_text)
    if len(compact) < 8:
        tokens = internal_text_tokens(ocr_text)
        return bool(tokens) and all(token in native_tokens for token in tokens)
    if compact in native_compact:
        return True
    tokens = internal_text_tokens(ocr_text)
    return bool(tokens) and all(token in native_tokens for token in tokens)


def maximum_candidate_coverage(
    candidate_boxes: numpy.ndarray,
    native_boxes: numpy.ndarray,
) -> numpy.ndarray:
    """Return each candidate's maximum covered-area ratio in bounded chunks."""
    if not len(candidate_boxes) or not len(native_boxes):
        return numpy.zeros(len(candidate_boxes), dtype=numpy.float32)
    if len(native_boxes) * FUSION_GEOMETRY_CHUNK > FUSION_VECTORIZED_ELEMENTS:
        native_index = SpatialIndex.from_boxes(native_boxes)
        output = numpy.zeros(len(candidate_boxes), dtype=numpy.float32)
        for index, box in enumerate(candidate_boxes):
            area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
            maximum = 0.0
            for hit in native_index.intersecting_hits(box):
                maximum = max(maximum, bbox_intersection_area(box, hit.bbox))
            output[index] = maximum / area
        return output
    output = numpy.zeros(len(candidate_boxes), dtype=numpy.float32)
    native_x0 = native_boxes[:, 0][None, :]
    native_y0 = native_boxes[:, 1][None, :]
    native_x1 = native_boxes[:, 2][None, :]
    native_y1 = native_boxes[:, 3][None, :]
    for start in range(0, len(candidate_boxes), FUSION_GEOMETRY_CHUNK):
        stop = min(len(candidate_boxes), start + FUSION_GEOMETRY_CHUNK)
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
        output[start:stop] = numpy.max(widths * heights, axis=1) / areas
    return output


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
        plan.reason == "native-text-needs-augmentation"
        and len(native) < 16
        and len(ocr) >= len(native) * 4
        and sum(character.isalnum() for text in native.text for character in text) <= 4
    ):
        return ocr
    if plan.reason == "noisy-native-text":
        native_candidate = internal_candidate(PSM_SPARSE_TEXT, native)
        ocr_candidate = internal_candidate(PSM_SPARSE_TEXT, ocr)
        if (
            len(ocr) >= 4
            and ocr_candidate.metrics.mean_confidence >= RASTER_TEXT_MIN_CONFIDENCE
            and ocr_candidate.metrics.utility >= native_candidate.metrics.utility * 1.05
            and ocr_candidate.metrics.alphanumeric_characters
            >= native_candidate.metrics.alphanumeric_characters * 0.80
        ):
            return ocr

    minimum_confidence = (
        75.0
        if plan.image_regions_only
        else RASTER_TEXT_MIN_CONFIDENCE
        if plan.reason == "noisy-native-text"
        else 30.0
        if plan.reason == "uncovered-vector-text"
        else 45.0
    )
    confidence_mask = ocr.confidence >= minimum_confidence
    if plan.image_regions_only or plan.reason == "uncovered-vector-text":
        alphanumeric_mask = numpy.fromiter(
            (sum(character.isalnum() for character in text) >= 1 for text in ocr.text),
            dtype=numpy.bool_,
            count=len(ocr),
        )
    else:
        alphanumeric_mask = numpy.ones(len(ocr), dtype=numpy.bool_)
    if len(native):
        native_compact = "".join(internal_compact_text(text) for text in native.text)
        native_tokens = frozenset(
            token for text in native.text for token in internal_text_tokens(text)
        )
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


# ===== ocr =====


# OCR already has an explicit worker limit. Prevent Tesseract's OpenMP kernels
# from creating another layer of workers on top of it.
os.environ["OMP_THREAD_LIMIT"] = "1"


def internal_import_tesserocr() -> Any:
    """Initialize cysignals from the main thread before OCR workers use it."""
    if "tesserocr" not in sys.modules and threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "core_pdf must initialize OCR on the main thread; import PdfDocument or call "
            "prewarm_runtime() during application startup"
        )
    return import_module("tesserocr")


internal_TESSEROCR = internal_import_tesserocr()
internal_OCR_LOCAL = threading.local()
OCR_TIMEOUT_MILLISECONDS = 12_000
MIN_DIRECT_OCR_RESOLUTION = 240
OCR_BATCH_MAX_TASKS = 8
OCR_BATCH_MAX_PIXELS = 8_000_000


@dataclass(frozen=True, slots=True)
class internal_CandidateMetrics:
    characters: int
    alphanumeric_characters: int
    tokens: int
    line_count: int
    mean_confidence: float
    symbol_ratio: float
    utility: float
    median_text_height: float = 0.0

    def as_record(self) -> dict[str, float | int]:
        return {
            "characters": self.characters,
            "alphanumeric_characters": self.alphanumeric_characters,
            "tokens": self.tokens,
            "line_count": self.line_count,
            "mean_confidence": self.mean_confidence,
            "symbol_ratio": self.symbol_ratio,
            "utility": self.utility,
            "median_text_height": self.median_text_height,
        }


@dataclass(frozen=True, slots=True)
class internal_Candidate:
    mode: int
    observations: ObservationBatch
    metrics: internal_CandidateMetrics
    symbols: ObservationBatch = field(default_factory=ObservationBatch.empty)
    api_seconds: float = 0.0
    setup_seconds: float = 0.0
    recognition_seconds: float = 0.0
    iterator_seconds: float = 0.0
    cleanup_seconds: float = 0.0
    candidate_seconds: float = 0.0
    recognition_status: str = "not-run"


internal_OCR_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
internal_OCR_TOKEN_TRANSLATION = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)


def internal_normalized_ocr_token_key(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(internal_OCR_TOKEN_TRANSLATION).casefold()


@dataclass(frozen=True, slots=True)
class internal_HiddenTextVerification:
    hidden_tokens: int
    preview_tokens: int
    matched_tokens: int
    spatially_matched_tokens: int
    token_overlap: float
    spatial_overlap: float
    accepted: bool
    reason: str

    def as_record(self) -> dict[str, int | float | bool | str]:
        return {
            "hidden_tokens": self.hidden_tokens,
            "preview_tokens": self.preview_tokens,
            "matched_tokens": self.matched_tokens,
            "spatially_matched_tokens": self.spatially_matched_tokens,
            "token_overlap": self.token_overlap,
            "spatial_overlap": self.spatial_overlap,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def internal_hidden_text_verification(
    hidden: ObservationBatch,
    preview: ObservationBatch,
) -> internal_HiddenTextVerification:
    """Compare a word-level raster preview with hidden text and its page geometry."""
    hidden_by_token: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for text, raw_box in zip(hidden.text, hidden.bbox, strict=True):
        box = cast(tuple[float, float, float, float], tuple(float(value) for value in raw_box))
        for token in internal_text_tokens(text):
            hidden_by_token[token].append(box)

    preview_entries = tuple(
        (token, cast(tuple[float, float, float, float], tuple(float(value) for value in raw_box)))
        for text, raw_box in zip(preview.text, preview.bbox, strict=True)
        for token in internal_text_tokens(text)
    )
    used: dict[str, set[int]] = defaultdict(set)
    matched = 0
    spatially_matched = 0
    for token, preview_box in preview_entries:
        candidates = hidden_by_token.get(token, ())
        available = (
            (index, box) for index, box in enumerate(candidates) if index not in used[token]
        )
        preview_center_x = (preview_box[0] + preview_box[2]) * 0.5
        preview_center_y = (preview_box[1] + preview_box[3]) * 0.5
        closest = min(
            available,
            key=lambda item: (
                ((item[1][0] + item[1][2]) * 0.5 - preview_center_x) ** 2
                + ((item[1][1] + item[1][3]) * 0.5 - preview_center_y) ** 2
            ),
            default=None,
        )
        if closest is None:
            continue
        index, hidden_box = closest
        used[token].add(index)
        matched += 1
        hidden_center_x = (hidden_box[0] + hidden_box[2]) * 0.5
        hidden_center_y = (hidden_box[1] + hidden_box[3]) * 0.5
        x_tolerance = max(
            12.0,
            (preview_box[2] - preview_box[0]) * 1.5,
            (hidden_box[2] - hidden_box[0]) * 1.5,
        )
        y_tolerance = max(
            6.0,
            (preview_box[3] - preview_box[1]) * 1.5,
            (hidden_box[3] - hidden_box[1]) * 1.5,
        )
        spatially_matched += int(
            abs(preview_center_x - hidden_center_x) <= x_tolerance
            and abs(preview_center_y - hidden_center_y) <= y_tolerance
        )

    preview_tokens = len(preview_entries)
    hidden_tokens = sum(len(boxes) for boxes in hidden_by_token.values())
    token_overlap = matched / max(1, preview_tokens)
    spatial_overlap = spatially_matched / max(1, preview_tokens)
    if matched < HIDDEN_TEXT_VERIFY_MIN_MATCHED_TOKENS:
        accepted = False
        reason = "insufficient-matched-tokens"
    elif token_overlap < HIDDEN_TEXT_VERIFY_MIN_TOKEN_OVERLAP:
        accepted = False
        reason = "low-token-overlap"
    elif spatial_overlap < HIDDEN_TEXT_VERIFY_MIN_SPATIAL_OVERLAP:
        accepted = False
        reason = "low-spatial-overlap"
    else:
        accepted = True
        reason = "semantic-and-spatial-match"
    return internal_HiddenTextVerification(
        hidden_tokens=hidden_tokens,
        preview_tokens=preview_tokens,
        matched_tokens=matched,
        spatially_matched_tokens=spatially_matched,
        token_overlap=token_overlap,
        spatial_overlap=spatial_overlap,
        accepted=accepted,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class internal_Raster:
    image: RasterImage
    resolution: int

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    @property
    def nbytes(self) -> int:
        return self.image.nbytes


@dataclass(frozen=True, slots=True)
class internal_RasterRegion:
    """A decoded raster coupled to the page-space area it actually represents."""

    raster: internal_Raster
    page_box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class internal_StrokedTextCell:
    """A translated vector-text run in a packed OCR raster."""

    source_box: tuple[float, float, float, float]
    packed_box: tuple[float, float, float, float]
    drawing_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class internal_PackedStrokedTextRaster:
    """One compact raster plus the piecewise map back into PDF page space."""

    raster: internal_Raster
    packed_box: tuple[float, float, float, float]
    cells: tuple[internal_StrokedTextCell, ...]


@dataclass(frozen=True, slots=True)
class internal_OcrRegion:
    """A ranked page-space region selected before compositor rasterization."""

    page_box: tuple[float, float, float, float]
    score: float
    reasons: tuple[str, ...]

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.page_box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass(frozen=True, slots=True)
class internal_OcrTask:
    mode: int
    image: RasterImage
    rectangle: tuple[int, int, int, int]
    page_box: tuple[float, float, float, float]
    resolution: int
    minimum_confidence: float = 20.0
    character_confidence_threshold: float | None = None
    recognize_words: bool = False
    collect_symbols: bool = False


def internal_valid_tessdata_path(path: str | os.PathLike[str]) -> Path | None:
    candidate = Path(path).expanduser()
    if (candidate / "eng.traineddata").is_file():
        return candidate.resolve()
    return None


@cache
def internal_tessdata_path() -> str:
    """Resolve English traineddata without relying on wheel build prefixes."""
    configured = os.environ.get("TESSDATA_PREFIX")
    if configured:
        resolved = internal_valid_tessdata_path(configured)
        if resolved is None:
            raise RuntimeError(
                "TESSDATA_PREFIX must name a tessdata directory containing eng.traineddata"
            )
        return str(resolved)

    try:
        default_path, languages = internal_TESSEROCR.get_languages()
    except RuntimeError:
        default_path, languages = "", ()
    if "eng" in languages:
        resolved = internal_valid_tessdata_path(default_path)
        if resolved is not None:
            return str(resolved)

    executable = shutil.which("tesseract")
    if executable is not None:
        try:
            completed = subprocess.run(
                [executable, "--list-langs"],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None:
            output = f"{completed.stdout}\n{completed.stderr}"
            match = re.search(r'List of available languages in "([^"]+)"', output)
            if match is not None:
                resolved = internal_valid_tessdata_path(match.group(1))
                if resolved is not None:
                    return str(resolved)

    raise RuntimeError(
        "English Tesseract data was not found; set TESSDATA_PREFIX to a tessdata directory "
        "containing eng.traineddata"
    )


def internal_api(mode: int) -> Any:
    api = getattr(internal_OCR_LOCAL, "api", None)
    if api is None:
        psm = mode
        api = internal_TESSEROCR.PyTessBaseAPI(
            path=internal_tessdata_path(),
            psm=psm,
            oem=internal_TESSEROCR.OEM.LSTM_ONLY,
        )
        api.SetVariable("preserve_interword_spaces", "0")
        api.SetVariable("textord_tablefind_recognize_tables", "0")
        api.SetVariable("textord_tabfind_find_tables", "0")
        internal_OCR_LOCAL.api = api
    api.SetPageSegMode(mode)
    return api


def internal_prepare_ocr() -> None:
    """Validate OCR startup and construct the caller thread's reusable API."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("prewarm_runtime() must be called on the main thread")
    internal_api(3)


def prewarm_runtime() -> None:
    """Start shared workers and validate OCR during main-thread startup."""
    RUNTIME.prewarm()
    internal_prepare_ocr()


class internal_HocrCharacterParser(HTMLParser):
    """Extract line text after dropping low-confidence hOCR characters."""

    def __init__(self, threshold: float) -> None:
        super().__init__(convert_charrefs=True)
        self.threshold = threshold
        self.lines: dict[tuple[int, int, int, int], str] = {}
        self.internal_line_box: tuple[int, int, int, int] | None = None
        self.internal_words: list[str] = []
        self.internal_chars: list[str] = []
        self.internal_char_confidence = threshold
        self.internal_in_char = False
        self.internal_in_word = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "span":
            return
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        title = attributes.get("title") or ""
        if "ocr_line" in classes:
            match = re.search(r"bbox (\d+) (\d+) (\d+) (\d+)", title)
            self.internal_line_box = None
            if match:
                left, top, right, bottom = (int(value) for value in match.groups())
                self.internal_line_box = (left, top, right, bottom)
            self.internal_words = []
        elif "ocrx_word" in classes:
            self.internal_in_word = True
            self.internal_chars = []
        elif "ocrx_cinfo" in classes and self.internal_in_word:
            match = re.search(r"(?:x_conf|x_wconf) (-?\d+(?:\.\d+)?)", title)
            self.internal_char_confidence = float(match.group(1)) if match else 0.0
            self.internal_in_char = True

    def handle_data(self, data: str) -> None:
        if self.internal_in_char and self.internal_char_confidence >= self.threshold:
            self.internal_chars.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "span":
            return
        if self.internal_in_char:
            self.internal_in_char = False
        elif self.internal_in_word:
            self.internal_words.append("".join(self.internal_chars))
            self.internal_chars = []
            self.internal_in_word = False
        elif self.internal_line_box is not None:
            text = " ".join(word for word in self.internal_words if word).strip()
            self.lines[self.internal_line_box] = text
            self.internal_line_box = None


def internal_hocr_filtered_lines(
    api: Any, threshold: float | None
) -> dict[tuple[int, int, int, int], str]:
    if threshold is None or not hasattr(api, "GetHOCRText"):
        return {}
    try:
        hocr = api.GetHOCRText(0)
    except (RuntimeError, TypeError):
        return {}
    if not hocr:
        return {}
    parser = internal_HocrCharacterParser(threshold)
    parser.feed(hocr.decode("utf-8", "replace") if isinstance(hocr, bytes) else hocr)
    return parser.lines


def internal_acceptable_text(
    text: str, confidence: float, minimum_confidence: float = 20.0
) -> bool:
    if confidence < minimum_confidence or not text:
        return False
    stripped = " ".join(text.split())
    if not stripped:
        return False
    length = len(stripped)
    printable_count = 0
    nonspace_count = 0
    alphanumeric_count = 0
    first_char = ""
    same_char_count = 0
    for ch in stripped:
        if ch.isprintable():
            printable_count += 1
        if not ch.isspace():
            nonspace_count += 1
            if ch.isalnum():
                alphanumeric_count += 1
        ch_lower = ch.casefold()
        if not first_char:
            first_char = ch_lower
            same_char_count = 1
        elif ch_lower == first_char:
            same_char_count += 1

    if printable_count / length < 0.95:
        return False
    if nonspace_count >= 4:
        symbol_ratio = 1.0 - alphanumeric_count / nonspace_count
        if symbol_ratio >= 0.65 and confidence < 85.0:
            return False
        if alphanumeric_count == 0:
            return False
    if confidence < 70.0 and length == 1 and not stripped.isalnum():
        return False
    return not (length >= 8 and same_char_count == length)


def internal_observation_utility(text: str, confidence: float) -> float:
    """Estimate useful recovered content without rewarding punctuation noise."""
    nonspace_characters = [character for character in text if not character.isspace()]
    if not nonspace_characters:
        return 0.0
    alphanumeric = sum(character.isalnum() for character in nonspace_characters)
    symbols = len(nonspace_characters) - alphanumeric
    # Symbols are useful in forms and schematics, but an unlimited symbol reward lets
    # noisy segmentation beat a smaller, readable pass. Cap their contribution relative
    # to actual text while preserving short labels such as "+5V" and "R/C".
    symbol_credit = min(symbols, max(2.0, alphanumeric * 0.5)) * 0.30
    confidence_factor = 0.25 + 0.75 * min(100.0, max(0.0, confidence)) / 100.0
    repetition_penalty = 1.0
    if len(nonspace_characters) >= 6:
        dominant_ratio = max(
            Counter(character.casefold() for character in nonspace_characters).values()
        ) / len(nonspace_characters)
        if dominant_ratio > 0.60:
            repetition_penalty = max(0.20, 1.0 - (dominant_ratio - 0.60) * 2.0)
    return (alphanumeric + symbol_credit) * confidence_factor * repetition_penalty


def internal_text_utility_stats(text: str, confidence: float) -> tuple[int, int, float]:
    """Return non-space count, alphanumeric count, and utility in one character scan."""
    counts: Counter[str] = Counter()
    nonspace = 0
    alphanumeric = 0
    for character in text:
        if character.isspace():
            continue
        nonspace += 1
        alphanumeric += character.isalnum()
        counts[character.casefold()] += 1
    if not nonspace:
        return 0, 0, 0.0
    symbols = nonspace - alphanumeric
    symbol_credit = min(symbols, max(2.0, alphanumeric * 0.5)) * 0.30
    confidence_factor = 0.25 + 0.75 * min(100.0, max(0.0, confidence)) / 100.0
    repetition_penalty = 1.0
    if nonspace >= 6:
        dominant_ratio = max(counts.values()) / nonspace
        if dominant_ratio > 0.60:
            repetition_penalty = max(0.20, 1.0 - (dominant_ratio - 0.60) * 2.0)
    utility = (alphanumeric + symbol_credit) * confidence_factor * repetition_penalty
    return nonspace, alphanumeric, utility


def internal_candidate(
    mode: int,
    observations: ObservationBatch,
    *,
    symbols: ObservationBatch | None = None,
    api_seconds: float = 0.0,
    setup_seconds: float = 0.0,
    recognition_seconds: float = 0.0,
    iterator_seconds: float = 0.0,
    cleanup_seconds: float = 0.0,
    candidate_seconds: float = 0.0,
    recognition_status: str = "not-run",
    median_text_height: float = 0.0,
) -> internal_Candidate:
    confidences = observations.confidence
    finite_confidences = confidences[numpy.isfinite(confidences)]
    mean_confidence = float(numpy.mean(finite_confidences)) if len(finite_confidences) else 0.0
    characters = max(0, len(observations) - 1)
    nonspace_characters = 0
    alphanumeric = 0
    tokens = 0
    utility = 0.0
    for text, confidence in zip(
        observations.text,
        observations.confidence,
        strict=True,
    ):
        characters += len(text)
        tokens += len(text.split())
        nonspace, text_alphanumeric, text_utility = internal_text_utility_stats(
            text,
            float(confidence),
        )
        nonspace_characters += nonspace
        alphanumeric += text_alphanumeric
        utility += text_utility
    symbol_characters = nonspace_characters - alphanumeric
    return internal_Candidate(
        mode,
        observations,
        internal_CandidateMetrics(
            characters=characters,
            alphanumeric_characters=alphanumeric,
            tokens=sum(len(text.split()) for text in observations.text),
            line_count=len(observations),
            mean_confidence=mean_confidence,
            symbol_ratio=symbol_characters / max(1, nonspace_characters),
            utility=utility,
            median_text_height=median_text_height,
        ),
        symbols=symbols if symbols is not None else ObservationBatch.empty(),
        api_seconds=api_seconds,
        setup_seconds=setup_seconds,
        recognition_seconds=recognition_seconds,
        iterator_seconds=iterator_seconds,
        cleanup_seconds=cleanup_seconds,
        candidate_seconds=candidate_seconds,
        recognition_status=recognition_status,
    )


def internal_select_character_filtered_candidate(
    raw: internal_Candidate,
    filtered: internal_Candidate,
) -> internal_Candidate:
    """Keep raw OCR unless filtering earns its recall cost.

    hOCR character confidence is useful for removing isolated noise, but treating
    every low-confidence character as false creates large recall losses on dense
    schematics and degraded scans.  The filtered candidate may give up only a small
    amount of local utility while retaining nearly all recovered content.
    """
    raw_metrics = raw.metrics
    filtered_metrics = filtered.metrics
    if not len(filtered.observations):
        return raw
    if filtered_metrics.line_count < raw_metrics.line_count * 0.98:
        return raw
    if filtered_metrics.alphanumeric_characters < raw_metrics.alphanumeric_characters:
        return raw
    if filtered_metrics.utility < raw_metrics.utility * 0.98:
        return raw
    return filtered


def internal_map_ocr_box(
    task: internal_OcrTask,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Map one Tesseract pixel box into the task's PDF coordinate space."""
    x0, y0, x1, y1 = bbox
    page_x0, page_y0, page_x1, page_y1 = task.page_box
    page_width = page_x1 - page_x0
    page_height = page_y1 - page_y0
    return (
        page_x0 + x0 * page_width / task.image.width,
        page_y1 - y1 * page_height / task.image.height,
        page_x0 + x1 * page_width / task.image.width,
        page_y1 - y0 * page_height / task.image.height,
    )


def internal_recognized_symbols(api: Any, task: internal_OcrTask) -> ObservationBatch:
    """Read character boxes from an existing recognition without another OCR pass."""
    iterator = api.GetIterator()
    if iterator is None:
        return ObservationBatch.empty()
    level = internal_TESSEROCR.RIL.SYMBOL
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    confidences: list[float] = []
    while True:
        try:
            text = (iterator.GetUTF8Text(level) or "").strip()
            confidence = float(iterator.Confidence(level))
            bbox = iterator.BoundingBox(level)
        except RuntimeError:
            text = ""
            confidence = 0.0
            bbox = None
        if bbox is not None and len(text) == 1 and text.isprintable() and math.isfinite(confidence):
            texts.append(text)
            boxes.append(internal_map_ocr_box(task, bbox))
            confidences.append(confidence)
        if not iterator.Next(level):
            break
    return ObservationBatch.from_columns(
        texts,
        boxes,
        source=ObservationSource.OCR,
        confidence=confidences,
        sequence=range(len(texts)),
    )


def internal_recognize(
    task: internal_OcrTask,
    *,
    api_override: Any | None = None,
    image_prepared: bool = False,
) -> internal_Candidate:
    tesserocr = internal_TESSEROCR
    api_started = time.perf_counter()
    api = api_override if api_override is not None else internal_api(task.mode)
    api_seconds = time.perf_counter() - api_started if api_override is None else 0.0
    setup_started = time.perf_counter()
    if not image_prepared:
        api.SetImageBytes(
            task.image.tesseract_bytes(),
            task.image.width,
            task.image.height,
            task.image.channels,
            task.image.stride,
        )
    x, y, rectangle_width, rectangle_height = task.rectangle
    x0 = max(0, min(task.image.width, int(x)))
    y0 = max(0, min(task.image.height, int(y)))
    x1 = max(x0, min(task.image.width, int(x + rectangle_width)))
    y1 = max(y0, min(task.image.height, int(y + rectangle_height)))
    if (
        x1 > x0
        and y1 > y0
        and (
            image_prepared
            or (x0, y0, x1 - x0, y1 - y0) != (0, 0, task.image.width, task.image.height)
        )
    ):
        api.SetRectangle(x0, y0, x1 - x0, y1 - y0)
    api.SetSourceResolution(task.resolution)
    setup_seconds = time.perf_counter() - setup_started
    recognition_started = time.perf_counter()
    recognized = api.Recognize(timeout=OCR_TIMEOUT_MILLISECONDS)
    recognition_seconds = time.perf_counter() - recognition_started
    if recognized:
        recognition_status = "ok"
    elif recognition_seconds >= OCR_TIMEOUT_MILLISECONDS / 1000.0 * 0.9:
        recognition_status = "timeout"
    else:
        recognition_status = "failed"
    iterator_started = time.perf_counter()
    iterator = api.GetIterator() if recognized else None
    level = tesserocr.RIL.WORD if task.recognize_words else tesserocr.RIL.TEXTLINE
    filtered_lines = (
        {}
        if task.recognize_words
        else internal_hocr_filtered_lines(api, task.character_confidence_threshold)
    )
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    polygons: list[tuple[float, ...]] = []
    confidences: list[float] = []
    filtered_texts: list[str] = []
    filtered_boxes: list[tuple[float, float, float, float]] = []
    filtered_polygons: list[tuple[float, ...]] = []
    filtered_confidences: list[float] = []
    text_heights: list[float] = []
    line_breaks: list[bool] = []
    filtered_line_breaks: list[bool] = []
    pending_line_break = True
    if iterator is not None:
        sequence = 0
        while True:
            if task.recognize_words:
                is_at_beginning = getattr(iterator, "IsAtBeginningOf", None)
                if callable(is_at_beginning):
                    pending_line_break |= bool(is_at_beginning(tesserocr.RIL.TEXTLINE))
            else:
                pending_line_break = True
            try:
                text = iterator.GetUTF8Text(level) or ""
                confidence = float(iterator.Confidence(level))
                bbox = iterator.BoundingBox(level)
            except RuntimeError:
                text = ""
                confidence = 0.0
                bbox = None
            text = " ".join(text.split())
            if bbox is not None and internal_acceptable_text(
                text,
                confidence,
                task.minimum_confidence,
            ):
                x0, y0, x1, y1 = bbox
                bbox_key = (int(x0), int(y0), int(x1), int(y1))
                filtered = filtered_lines.get(bbox_key)
                filtered_text = " ".join(filtered.split()) if filtered is not None else text
                texts.append(text)
                mapped_box = internal_map_ocr_box(task, (x0, y0, x1, y1))
                boxes.append(mapped_box)
                polygon = (
                    mapped_box[0],
                    mapped_box[1],
                    mapped_box[2],
                    mapped_box[1],
                    mapped_box[2],
                    mapped_box[3],
                    mapped_box[0],
                    mapped_box[3],
                )
                polygons.append(polygon)
                confidences.append(confidence)
                line_breaks.append(pending_line_break)
                pending_line_break = False
                if internal_acceptable_text(
                    filtered_text,
                    confidence,
                    task.minimum_confidence,
                ):
                    filtered_texts.append(filtered_text)
                    filtered_boxes.append(mapped_box)
                    filtered_polygons.append(polygon)
                    filtered_confidences.append(confidence)
                    filtered_line_breaks.append(line_breaks[-1])
                text_heights.append(float(y1 - y0))
                sequence += 1
            if not iterator.Next(level):
                break
    symbols = (
        internal_recognized_symbols(api, task)
        if recognized and task.collect_symbols
        else ObservationBatch.empty()
    )
    iterator_seconds = time.perf_counter() - iterator_started
    cleanup_started = time.perf_counter()
    clear_adaptive = getattr(api, "ClearAdaptiveClassifier", None)
    if callable(clear_adaptive):
        clear_adaptive()
    cleanup_seconds = time.perf_counter() - cleanup_started
    candidate_started = time.perf_counter()
    observations = ObservationBatch.from_columns(
        texts,
        boxes,
        source=ObservationSource.OCR,
        polygon=polygons,
        confidence=confidences,
        sequence=range(len(texts)),
        line_break_before=line_breaks,
    )
    candidate = internal_candidate(
        task.mode,
        observations,
        symbols=symbols,
        api_seconds=api_seconds,
        setup_seconds=setup_seconds,
        recognition_seconds=recognition_seconds,
        iterator_seconds=iterator_seconds,
        cleanup_seconds=cleanup_seconds,
        recognition_status=recognition_status,
        median_text_height=(float(numpy.median(text_heights)) if text_heights else 0.0),
    )
    candidate_seconds = time.perf_counter() - candidate_started
    candidate = replace(candidate, candidate_seconds=candidate_seconds)
    if task.character_confidence_threshold is None:
        return candidate
    filtered_observations = ObservationBatch.from_columns(
        filtered_texts,
        filtered_boxes,
        source=ObservationSource.OCR,
        polygon=filtered_polygons,
        confidence=filtered_confidences,
        sequence=range(len(filtered_texts)),
        line_break_before=filtered_line_breaks,
    )
    filtered_candidate = internal_candidate(
        task.mode,
        filtered_observations,
        symbols=symbols,
        api_seconds=api_seconds,
        setup_seconds=setup_seconds,
        recognition_seconds=recognition_seconds,
        iterator_seconds=iterator_seconds,
        cleanup_seconds=cleanup_seconds,
        candidate_seconds=candidate_seconds,
        recognition_status=recognition_status,
        median_text_height=(float(numpy.median(text_heights)) if text_heights else 0.0),
    )
    return internal_select_character_filtered_candidate(candidate, filtered_candidate)


def internal_recognize_group(tasks: tuple[internal_OcrTask, ...]) -> tuple[internal_Candidate, ...]:
    """Recognize same-raster tasks while reusing Tesseract image setup."""
    if not tasks:
        return ()
    if len(tasks) == 1:
        return (internal_recognize(tasks[0]),)
    first = tasks[0]
    api = internal_api(first.mode)
    candidates = [internal_recognize(first, api_override=api)]
    for task in tasks[1:]:
        candidates.append(internal_recognize(task, api_override=api, image_prepared=True))
    return tuple(candidates)


def internal_ocr_task_groups(
    tasks: tuple[internal_OcrTask, ...],
) -> tuple[tuple[internal_OcrTask, ...], ...]:
    """Create ordered same-raster/mode batches without duplicating image setup."""
    groups: list[tuple[internal_OcrTask, ...]] = []
    current: list[internal_OcrTask] = []
    current_pixels = 0
    for task in tasks:
        pixels = task.rectangle[2] * task.rectangle[3]
        if current and (
            task.image is not current[0].image
            or task.mode != current[0].mode
            or len(current) >= OCR_BATCH_MAX_TASKS
            or current_pixels + pixels > OCR_BATCH_MAX_PIXELS
        ):
            groups.append(tuple(current))
            current = []
            current_pixels = 0
        current.append(task)
        current_pixels += pixels
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def internal_tile_tasks(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    ocr_pass: OcrPass,
    *,
    compact_image: bool | str = False,
) -> tuple[internal_OcrTask, ...]:
    if ocr_pass.preprocess == "binary-clean":
        raster = internal_adaptive_ocr_raster(raster)
    image = (
        internal_compact_ocr_image(raster.image, grayscale=compact_image == "grayscale")
        if compact_image
        else raster.image
    )
    requested_tiles = ocr_pass.tiles if ocr_pass.scope is OcrPassScope.TILES else 1
    tiles = max(1, min(requested_tiles, raster.height))
    if tiles == 1:
        return tuple(
            internal_OcrTask(
                mode=mode,
                image=image,
                rectangle=(0, 0, raster.width, raster.height),
                page_box=page_box,
                resolution=raster.resolution,
                minimum_confidence=ocr_pass.minimum_confidence,
                character_confidence_threshold=ocr_pass.character_confidence_threshold,
                recognize_words=ocr_pass.recognize_words,
                collect_symbols=ocr_pass.collect_symbols,
            )
            for mode in ocr_pass.modes
        )
    overlap = max(24, int(round(raster.resolution * 0.35)))
    base_height = math.ceil(raster.height / tiles)
    tasks = []
    for mode in ocr_pass.modes:
        for tile_index in range(tiles):
            y0 = max(0, tile_index * base_height - overlap)
            y1 = min(raster.height, (tile_index + 1) * base_height + overlap)
            tasks.append(
                internal_OcrTask(
                    mode=mode,
                    image=image,
                    rectangle=(0, y0, raster.width, y1 - y0),
                    page_box=page_box,
                    resolution=raster.resolution,
                    minimum_confidence=ocr_pass.minimum_confidence,
                    character_confidence_threshold=ocr_pass.character_confidence_threshold,
                    recognize_words=ocr_pass.recognize_words,
                    collect_symbols=ocr_pass.collect_symbols,
                )
            )
    return tuple(tasks)


def internal_raster_ink_grid(
    raster: internal_Raster, rows: int, columns: int
) -> numpy.ndarray[Any, Any]:
    """Measure visual ink per coarse region from a bounded zero-copy raster sample."""
    if rows <= 0 or columns <= 0:
        return numpy.zeros(max(0, rows * columns), dtype=numpy.float32)
    pixels = raster.image.array()
    y_step = max(1, raster.height // 512)
    x_step = max(1, raster.width // 512)
    sampled = pixels[::y_step, ::x_step]
    if raster.image.channels == 1:
        intensity = sampled[:, :, 0]
    else:
        intensity = numpy.min(sampled[:, :, :3], axis=2)
    ink = intensity < 245
    integral = numpy.pad(
        ink.cumsum(axis=0, dtype=numpy.int32).cumsum(axis=1, dtype=numpy.int32),
        ((1, 0), (1, 0)),
    )
    y_bounds = numpy.arange(rows + 1, dtype=numpy.intp) * len(ink) // rows
    x_bounds = numpy.arange(columns + 1, dtype=numpy.intp) * ink.shape[1] // columns
    y0, y1 = y_bounds[:-1], y_bounds[1:]
    x0, x1 = x_bounds[:-1], x_bounds[1:]
    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    counts = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    grid_output = numpy.zeros((rows, columns), dtype=numpy.float32)
    numpy.divide(sums, counts, out=grid_output, where=counts != 0)
    return grid_output.reshape(-1)


def internal_estimated_text_height(raster: internal_Raster) -> float:
    """Estimate ordinary text-band height from a bounded raster preview.

    Horizontal projections are substantially cheaper than an exploratory OCR
    pass.  Sampling several vertical strips avoids letting table borders or a
    single illustration join otherwise independent text lines.
    """
    pixels = raster.image.array()
    sample_step = max(1, math.ceil(math.sqrt(raster.width * raster.height / 1_000_000)))
    sampled = pixels[::sample_step, ::sample_step]
    gray = sampled[:, :, 0] if raster.image.channels == 1 else numpy.min(sampled[:, :, :3], axis=2)
    background = float(numpy.percentile(gray, 90.0))
    threshold = max(80.0, min(225.0, background - 24.0))
    ink = gray < threshold
    if not numpy.any(ink):
        return 0.0
    strip_count = max(4, min(12, ink.shape[1] // 48))
    heights: list[int] = []
    for strip in numpy.array_split(ink, strip_count, axis=1):
        if strip.shape[1] < 4:
            continue
        required = max(2, int(math.ceil(strip.shape[1] * 0.01)))
        active = numpy.count_nonzero(strip, axis=1) >= required
        # Close a one-row break caused by ascenders, punctuation, or scan noise.
        if len(active) >= 3:
            active[1:-1] |= active[:-2] & active[2:]
        padded = numpy.pad(active.astype(numpy.int8), (1, 1))
        transitions = numpy.diff(padded)
        starts = numpy.flatnonzero(transitions == 1)
        ends = numpy.flatnonzero(transitions == -1)
        for height in ends - starts:
            if 2 <= height <= max(12, sampled.shape[0] // 12):
                heights.append(int(height))
    if len(heights) < 4:
        return 0.0
    values = numpy.asarray(heights, dtype=numpy.float32)
    lower = float(numpy.percentile(values, 25.0))
    upper = float(numpy.percentile(values, 85.0))
    typical = values[(values >= lower) & (values <= upper)]
    return float(numpy.median(typical if len(typical) else values)) * sample_step


def internal_observation_coverage_grid(
    observations: ObservationBatch,
    page_box: tuple[float, float, float, float],
    rows: int,
    columns: int,
) -> numpy.ndarray[Any, Any]:
    if not len(observations):
        return numpy.zeros(rows * columns, dtype=numpy.float32)
    x0, y0, x1, y1 = page_box
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    output = numpy.zeros((rows, columns), dtype=numpy.float32)
    for text, confidence, raw_box in zip(
        observations.text,
        observations.confidence,
        observations.bbox,
        strict=True,
    ):
        box_x0 = max(x0, float(raw_box[0]))
        box_y0 = max(y0, float(raw_box[1]))
        box_x1 = min(x1, float(raw_box[2]))
        box_y1 = min(y1, float(raw_box[3]))
        box_width = box_x1 - box_x0
        box_height = box_y1 - box_y0
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        utility = internal_observation_utility(text, float(confidence))
        if utility <= 0.0:
            continue
        column_start = max(0, min(columns - 1, int((box_x0 - x0) * columns / width)))
        column_end = max(
            column_start,
            min(columns - 1, math.ceil((box_x1 - x0) * columns / width) - 1),
        )
        row_start = max(0, min(rows - 1, int((y1 - box_y1) * rows / height)))
        row_end = max(
            row_start,
            min(rows - 1, math.ceil((y1 - box_y0) * rows / height) - 1),
        )
        box_area = box_width * box_height
        for row in range(row_start, row_end + 1):
            cell_y0 = y1 - (row + 1) * height / rows
            cell_y1 = y1 - row * height / rows
            overlap_y = max(0.0, min(box_y1, cell_y1) - max(box_y0, cell_y0))
            if overlap_y <= 0.0:
                continue
            for column in range(column_start, column_end + 1):
                cell_x0 = x0 + column * width / columns
                cell_x1 = x0 + (column + 1) * width / columns
                overlap_x = max(0.0, min(box_x1, cell_x1) - max(box_x0, cell_x0))
                if overlap_x > 0.0:
                    output[row, column] += utility * overlap_x * overlap_y / box_area
    return output.reshape(-1)


def internal_observation_utility_grid(
    observations: ObservationBatch,
    page_box: tuple[float, float, float, float],
    rows: int,
    columns: int,
) -> numpy.ndarray[Any, Any]:
    """Assign each observation to one cell for stable weak-region ranking."""
    if not len(observations):
        return numpy.zeros(rows * columns, dtype=numpy.float32)
    x0, y0, x1, y1 = page_box
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    centers_x = (observations.bbox[:, 0] + observations.bbox[:, 2]) * 0.5
    centers_y = (observations.bbox[:, 1] + observations.bbox[:, 3]) * 0.5
    inside = (centers_x >= x0) & (centers_x <= x1) & (centers_y >= y0) & (centers_y <= y1)
    if not numpy.any(inside):
        return numpy.zeros(rows * columns, dtype=numpy.float32)
    centers_x = centers_x[inside]
    centers_y = centers_y[inside]
    columns_by_observation = numpy.clip(
        ((centers_x - x0) * columns / width).astype(numpy.int64),
        0,
        columns - 1,
    )
    rows_by_observation = numpy.clip(
        ((y1 - centers_y) * rows / height).astype(numpy.int64),
        0,
        rows - 1,
    )
    utility = numpy.fromiter(
        (
            internal_observation_utility(text, float(confidence))
            for text, confidence in zip(
                (
                    text
                    for text, selected in zip(observations.text, inside, strict=True)
                    if selected
                ),
                observations.confidence[inside],
                strict=True,
            )
        ),
        dtype=numpy.float32,
        count=int(numpy.count_nonzero(inside)),
    )
    return numpy.bincount(
        rows_by_observation * columns + columns_by_observation,
        weights=utility,
        minlength=rows * columns,
    ).astype(numpy.float32, copy=False)


def internal_weak_region_grid_shape(
    raster: internal_Raster,
    ocr_pass: OcrPass,
    primary: ObservationBatch,
) -> tuple[int, int]:
    rows = max(1, min(ocr_pass.tiles, raster.height))
    columns = max(1, min(ocr_pass.region_columns, raster.width))
    if len(primary) >= 40:
        rows = min(rows, 6)
        columns = min(columns, 3)
    return rows, columns


def internal_weak_region_rectangles(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
) -> tuple[tuple[int, int, int, int], ...]:
    """Find visually occupied cells where the primary OCR recovered little text."""
    rows, columns = internal_weak_region_grid_shape(raster, ocr_pass, primary)
    ink = internal_raster_ink_grid(raster, rows, columns)
    utility = internal_observation_utility_grid(primary, page_box, rows, columns)
    expected_utility = float(numpy.sum(utility)) / max(1, rows * columns)
    utility_limit = max(4.0, expected_utility * 0.45)
    eligible = numpy.flatnonzero((ink >= 0.01) & (utility < utility_limit))
    if not len(eligible):
        return ()
    priority = ink[eligible] / (1.0 + utility[eligible] * 0.05)
    region_limit = ocr_pass.max_regions
    if len(primary) >= 40:
        region_limit = max(1, region_limit // 2)
        region_limit = min(region_limit, 8)
    ranked = eligible[numpy.argsort(priority)[::-1][:region_limit]]
    # Tesseract's sparse-text layout pass scans connected components that can cross
    # the requested rectangle.  A narrow horizontal margin can therefore make
    # Leptonica reject a component as being outside the active rectangle.  Keep a
    # generous, resolution-scaled margin so region boundaries do not bisect glyphs
    # or text lines.
    overlap_x = min(
        max(48, int(round(raster.resolution * 0.20))),
        max(0, (raster.width // columns - 1) // 2),
    )
    overlap_y = min(
        max(48, int(round(raster.resolution * 0.20))),
        max(0, (raster.height // rows - 1) // 2),
    )
    rectangles: list[tuple[int, int, int, int]] = []
    for raw_cell in ranked:
        cell = int(raw_cell)
        row, column = divmod(cell, columns)
        cell_x0 = column * raster.width // columns
        cell_x1 = (column + 1) * raster.width // columns
        cell_y0 = row * raster.height // rows
        cell_y1 = (row + 1) * raster.height // rows
        rectangle_x0 = max(0, cell_x0 - overlap_x)
        rectangle_x1 = min(raster.width, cell_x1 + overlap_x)
        rectangle_y0 = max(0, cell_y0 - overlap_y)
        rectangle_y1 = min(raster.height, cell_y1 + overlap_y)
        rectangles.append(
            (
                rectangle_x0,
                rectangle_y0,
                rectangle_x1 - rectangle_x0,
                rectangle_y1 - rectangle_y0,
            )
        )
    return tuple(rectangles)


def internal_weak_region_tasks(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
    *,
    compact_image: bool | str = False,
) -> tuple[internal_OcrTask, ...]:
    """Create OCR tasks for weak regions in an already materialized raster."""
    image = (
        internal_compact_ocr_image(raster.image, grayscale=compact_image == "grayscale")
        if compact_image
        else raster.image
    )
    return tuple(
        internal_OcrTask(
            mode=mode,
            image=image,
            rectangle=rectangle,
            page_box=page_box,
            resolution=raster.resolution,
            minimum_confidence=ocr_pass.minimum_confidence,
            character_confidence_threshold=ocr_pass.character_confidence_threshold,
            recognize_words=ocr_pass.recognize_words,
            collect_symbols=ocr_pass.collect_symbols,
        )
        for mode in ocr_pass.modes
        for rectangle in internal_weak_region_rectangles(raster, page_box, ocr_pass, primary)
    )


@dataclass(frozen=True, slots=True)
class internal_RescueCoverage:
    raster_count: int = 0
    cell_count: int = 0
    ink_cells: int = 0
    weak_cells: int = 0
    ink: float = 0.0
    weak_ink: float = 0.0

    @property
    def mean_ink(self) -> float:
        return self.ink / max(1, self.cell_count)

    @property
    def weak_ink_ratio(self) -> float:
        return self.weak_ink / max(1e-9, self.ink)

    def as_record(self) -> dict[str, int | float]:
        return {
            "raster_count": self.raster_count,
            "cell_count": self.cell_count,
            "ink_cells": self.ink_cells,
            "weak_cells": self.weak_cells,
            "mean_ink": self.mean_ink,
            "weak_ink_ratio": self.weak_ink_ratio,
        }


def internal_adaptive_rescue_coverage(
    source_tasks: tuple[internal_OcrTask, ...],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
) -> internal_RescueCoverage:
    """Measure ink not spatially explained by the primary OCR observations."""
    raster_count = 0
    cell_count = 0
    ink_cells = 0
    weak_cells = 0
    total_ink = 0.0
    weak_ink = 0.0
    seen: set[tuple[int, tuple[float, float, float, float], int]] = set()
    for task in source_tasks:
        key = (id(task.image), task.page_box, task.resolution)
        if key in seen:
            continue
        seen.add(key)
        raster = internal_Raster(task.image, task.resolution)
        rows, columns = internal_weak_region_grid_shape(raster, ocr_pass, primary)
        ink = internal_raster_ink_grid(raster, rows, columns)
        coverage = internal_observation_coverage_grid(primary, task.page_box, rows, columns)
        utility_limit = max(4.0, float(numpy.sum(coverage)) / (rows * columns) * 0.45)
        occupied = ink >= 0.01
        weak = occupied & (coverage < utility_limit)
        raster_count += 1
        cell_count += rows * columns
        ink_cells += int(numpy.count_nonzero(occupied))
        weak_cells += int(numpy.count_nonzero(weak))
        total_ink += float(numpy.sum(ink, dtype=numpy.float64))
        weak_ink += float(numpy.sum(ink[weak], dtype=numpy.float64))
    return internal_RescueCoverage(
        raster_count=raster_count,
        cell_count=cell_count,
        ink_cells=ink_cells,
        weak_cells=weak_cells,
        ink=total_ink,
        weak_ink=weak_ink,
    )


def internal_primary_text_is_sufficient(candidate: internal_Candidate) -> bool:
    """Return whether a sparse primary result is already large and trustworthy.

    Resolution escalation cannot add detail to text that is already comfortably
    sampled. Keep this decision shared by the adaptive rescue and subsequent
    full-page fallbacks so the latter cannot repeat work the former rejected.
    """
    metrics = candidate.metrics
    return (
        metrics.characters < 32
        and metrics.median_text_height >= OCR_RESCUE_LARGE_TEXT_HEIGHT
        and metrics.mean_confidence >= OCR_RESCUE_MIN_CONFIDENCE
    )


def internal_adaptive_rescue_decision(
    candidate: internal_Candidate,
    source_tasks: tuple[internal_OcrTask, ...],
    ocr_pass: OcrPass,
) -> tuple[bool, dict[str, object]]:
    """Decide whether another raster pass has enough unresolved visual evidence."""
    metrics = candidate.metrics
    coverage_pass = replace(
        ocr_pass,
        scope=OcrPassScope.WEAK_REGIONS,
        tiles=max(6, ocr_pass.tiles),
        region_columns=max(3, ocr_pass.region_columns),
        max_regions=max(8, ocr_pass.max_regions),
    )
    coverage = internal_adaptive_rescue_coverage(
        source_tasks,
        coverage_pass,
        candidate.observations,
    )
    reason = "unresolved-ink"
    run = True
    if internal_primary_text_is_sufficient(candidate):
        run = False
        reason = "primary-text-already-large"
    elif (
        metrics.characters >= 1_000
        and metrics.mean_confidence >= OCR_RESCUE_MIN_CONFIDENCE
        and coverage.mean_ink >= OCR_RESCUE_SATURATED_MEAN_INK
    ):
        # A nearly solid source gives the coarse ink grid no useful localization
        # signal.  Reprocessing arbitrary cells cannot target missing text.
        run = False
        reason = "ink-map-saturated"
    elif (
        metrics.characters >= 300
        and metrics.mean_confidence >= OCR_RESCUE_MIN_CONFIDENCE
        and coverage.raster_count
        and coverage.weak_ink_ratio < OCR_RESCUE_MIN_WEAK_INK_RATIO
        and (metrics.characters >= 600 or coverage.weak_ink_ratio == 0.0)
    ):
        run = False
        reason = "primary-covers-ink"
    return run, {
        "run": run,
        "reason": reason,
        "characters": metrics.characters,
        "mean_confidence": metrics.mean_confidence,
        "median_text_height": metrics.median_text_height,
        **coverage.as_record(),
    }


def internal_compact_ocr_image(image: RasterImage, *, grayscale: bool = False) -> RasterImage:
    """Drop redundant channels before sending a scan to Tesseract."""
    if image.channels == 3 and grayscale:
        if image.width * image.height < 5_000_000:
            return image
        samples = image.array()
        # Tesseract converts RGB to grayscale internally. Do it once here so its
        # segmentation pass processes one third as many bytes for scan-heavy pages.
        gray = (
            samples[:, :, 0].astype(numpy.uint16) * 77
            + samples[:, :, 1].astype(numpy.uint16) * 150
            + samples[:, :, 2].astype(numpy.uint16) * 29
            + 128
        ) >> 8
        gray = gray.astype(numpy.uint8)
        return RasterImage(contiguous_bytes(gray), image.width, image.height, 1)
    if image.channels not in {2, 4}:
        return image
    samples = image.array()
    alpha_index = image.channels - 1
    if not numpy.all(samples[:, :, alpha_index] == 255):
        if image.channels == 2:
            # Tesseract accepts gray, RGB, and RGBA byte layouts, but not the
            # gray-plus-alpha layout produced by PDF soft masks. Composite it
            # onto the same white background used by page rendering.
            distance_from_white = numpy.multiply(
                255 - samples[:, :, 0],
                samples[:, :, 1],
                dtype=numpy.uint16,
            )
            distance_from_white += 127
            distance_from_white //= 255
            gray_alpha = 255 - distance_from_white.astype(numpy.uint8)
            return RasterImage(contiguous_bytes(gray_alpha), image.width, image.height, 1)
        return image
    if image.channels == 2:
        return RasterImage(contiguous_bytes(samples[:, :, 0]), image.width, image.height, 1)
    if numpy.array_equal(samples[:, :, 0], samples[:, :, 1]) and numpy.array_equal(
        samples[:, :, 1], samples[:, :, 2]
    ):
        return RasterImage(contiguous_bytes(samples[:, :, 0]), image.width, image.height, 1)
    return RasterImage(contiguous_bytes(samples[:, :, :3]), image.width, image.height, 3)


OCR_IMAGE_TEXT_SAMPLE_PIXELS = 300_000
OCR_IMAGE_TEXT_EDGE_DELTA = 24
OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGES = 0.015
OCR_IMAGE_TEXT_PHOTO_MAX_WHITE = 0.20
OCR_IMAGE_TEXT_PHOTO_MIN_ENTROPY = 3.0
OCR_IMAGE_TEXT_STRONG_HORIZONTAL_EDGES = 0.09
OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGE_SHARE = 0.85


@dataclass(frozen=True, slots=True)
class internal_RasterTextSignal:
    likely_text: bool
    reason: str
    sampled_pixels: int
    white_ratio: float
    grayscale_entropy: float
    horizontal_edge_ratio: float
    vertical_edge_ratio: float

    def as_record(self) -> dict[str, bool | float | int | str]:
        return {
            "likely_text": self.likely_text,
            "reason": self.reason,
            "sampled_pixels": self.sampled_pixels,
            "white_ratio": self.white_ratio,
            "grayscale_entropy": self.grayscale_entropy,
            "horizontal_edge_ratio": self.horizontal_edge_ratio,
            "vertical_edge_ratio": self.vertical_edge_ratio,
        }


def internal_raster_text_signal(image: RasterImage) -> internal_RasterTextSignal:
    """Reject obvious non-text image supplements using a bounded pixel sample.

    This gate is intentionally limited to image supplements on pages that already
    have native text.  Full-page scan OCR and compositor fallbacks never use it.
    Text and line art have frequent horizontal intensity transitions; continuous-
    tone photographs may also have many edges, but those edges are less strongly
    horizontal and occur without a light document background.
    """
    pixels = image.array()
    sample_step = max(
        1,
        math.ceil(math.sqrt(image.width * image.height / OCR_IMAGE_TEXT_SAMPLE_PIXELS)),
    )
    sampled = pixels[::sample_step, ::sample_step]
    if image.channels == 1:
        gray = sampled[:, :, 0]
    elif image.channels == 2:
        source = sampled[:, :, 0].astype(numpy.uint16)
        alpha = sampled[:, :, 1].astype(numpy.uint16)
        gray = (255 - ((255 - source) * alpha + 127) // 255).astype(numpy.uint8)
    else:
        colour = sampled[:, :, :3]
        if image.channels == 4:
            alpha = sampled[:, :, 3:].astype(numpy.uint16)
            colour = (255 - ((255 - colour.astype(numpy.uint16)) * alpha + 127) // 255).astype(
                numpy.uint8
            )
        gray = numpy.min(colour, axis=2)

    gray_16 = gray.astype(numpy.int16)
    horizontal_edges = (
        float(numpy.mean(numpy.abs(numpy.diff(gray_16, axis=1)) >= OCR_IMAGE_TEXT_EDGE_DELTA))
        if gray.shape[1] > 1
        else 0.0
    )
    vertical_edges = (
        float(numpy.mean(numpy.abs(numpy.diff(gray_16, axis=0)) >= OCR_IMAGE_TEXT_EDGE_DELTA))
        if gray.shape[0] > 1
        else 0.0
    )
    white_ratio = float(numpy.mean(gray >= 245))
    histogram = numpy.bincount((gray // 8).reshape(-1), minlength=32).astype(numpy.float64)
    histogram /= max(1.0, float(numpy.sum(histogram)))
    occupied = histogram > 0.0
    entropy = float(-numpy.sum(histogram[occupied] * numpy.log2(histogram[occupied])))

    likely_text = True
    reason = "text-structure"
    if horizontal_edges < OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGES:
        likely_text = False
        reason = "low-edge-density"
    else:
        horizontal_edge_share = horizontal_edges / max(1e-9, vertical_edges)
        strongly_structured = bool(
            horizontal_edges >= OCR_IMAGE_TEXT_STRONG_HORIZONTAL_EDGES
            and horizontal_edge_share >= OCR_IMAGE_TEXT_MIN_HORIZONTAL_EDGE_SHARE
        )
        if (
            white_ratio < OCR_IMAGE_TEXT_PHOTO_MAX_WHITE
            and entropy >= OCR_IMAGE_TEXT_PHOTO_MIN_ENTROPY
            and not strongly_structured
        ):
            likely_text = False
            reason = "continuous-tone-image"
    return internal_RasterTextSignal(
        likely_text=likely_text,
        reason=reason,
        sampled_pixels=int(gray.size),
        white_ratio=white_ratio,
        grayscale_entropy=entropy,
        horizontal_edge_ratio=horizontal_edges,
        vertical_edge_ratio=vertical_edges,
    )


def internal_adaptive_ocr_raster(raster: internal_Raster) -> internal_Raster:
    """Binarize faded scans against their local background for a fallback pass."""
    pixels = raster.image.array()
    gray = (
        pixels[:, :, 0] if raster.image.channels == 1 else numpy.min(pixels[:, :, :3], axis=2)
    ).astype(numpy.float32)
    radius = max(8, min(24, min(raster.width, raster.height) // 80))
    integral = numpy.pad(gray, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y = numpy.arange(raster.height)
    x = numpy.arange(raster.width)
    y0 = numpy.maximum(0, y - radius)
    y1 = numpy.minimum(raster.height, y + radius + 1)
    x0 = numpy.maximum(0, x - radius)
    x1 = numpy.minimum(raster.width, x + radius + 1)
    local_sum = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    local_area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(numpy.float32)
    threshold = local_sum / local_area - 9.0
    binary = numpy.where(gray <= threshold, numpy.uint8(0), numpy.uint8(255))
    return internal_Raster(
        RasterImage(contiguous_bytes(binary), raster.width, raster.height, 1),
        raster.resolution,
    )


def internal_bbox_overlap_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    return spatial_bbox_overlap_ratio(left, right)


def internal_candidate_text_containment(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> bool:
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if not shorter or sum(len(token) for token in shorter) < 4:
        return False
    width = len(shorter)
    return any(longer[index : index + width] == shorter for index in range(len(longer) - width + 1))


def internal_merge_candidate_batches(
    candidates: tuple[internal_Candidate, ...],
) -> internal_Candidate:
    if not candidates:
        return internal_candidate(-1, ObservationBatch.empty())
    if len(candidates) == 1:
        return candidates[0]
    modes = {candidate.mode for candidate in candidates}
    merged_by_mode: list[internal_Candidate] = []
    for mode in sorted(modes):
        mode_candidates = tuple(candidate for candidate in candidates if candidate.mode == mode)
        combined = ObservationBatch.concatenate(
            *(candidate.observations for candidate in mode_candidates)
        )
        combined_symbols = ObservationBatch.concatenate(
            *(candidate.symbols for candidate in mode_candidates)
        )
        fuzzy_tile_deduplication = len(mode_candidates) > 1
        order = numpy.lexsort((combined.bbox[:, 0], -combined.bbox[:, 1]))
        normalized_text = tuple(" ".join(text.casefold().split()) for text in combined.text)
        normalized_tokens = tuple(
            tuple(
                internal_normalized_ocr_token_key(match.group(0))
                for match in internal_OCR_TOKEN.finditer(text)
                if internal_normalized_ocr_token_key(match.group(0))
            )
            for text in combined.text
        )
        observation_utility = numpy.fromiter(
            (
                internal_observation_utility(text, float(confidence))
                for text, confidence in zip(
                    combined.text,
                    combined.confidence,
                    strict=True,
                )
            ),
            dtype=numpy.float32,
            count=len(combined),
        )
        deduplicated: list[int] = []
        for raw_index in order:
            index = int(raw_index)
            duplicate_index = next(
                (
                    accepted_position
                    for accepted_position in range(
                        max(0, len(deduplicated) - 24), len(deduplicated)
                    )
                    if (
                        internal_bbox_overlap_ratio(
                            combined.bbox[index],
                            combined.bbox[deduplicated[accepted_position]],
                        )
                        >= (
                            0.35
                            if internal_candidate_text_containment(
                                normalized_tokens[deduplicated[accepted_position]],
                                normalized_tokens[index],
                            )
                            else (
                                0.45
                                if normalized_text[deduplicated[accepted_position]]
                                == normalized_text[index]
                                else (0.70 if fuzzy_tile_deduplication else math.inf)
                            )
                        )
                    )
                ),
                None,
            )
            if duplicate_index is None:
                deduplicated.append(index)
                continue
            accepted_index = deduplicated[duplicate_index]
            containment = internal_candidate_text_containment(
                normalized_tokens[accepted_index],
                normalized_tokens[index],
            )
            if containment and len(normalized_text[index]) != len(normalized_text[accepted_index]):
                if len(normalized_text[index]) > len(normalized_text[accepted_index]):
                    deduplicated[duplicate_index] = index
            elif observation_utility[index] > observation_utility[accepted_index]:
                deduplicated[duplicate_index] = index
        heights = tuple(
            candidate.metrics.median_text_height
            for candidate in mode_candidates
            if candidate.metrics.median_text_height > 0.0
        )
        merged_by_mode.append(
            internal_candidate(
                mode,
                combined.take(deduplicated),
                symbols=combined_symbols,
                median_text_height=float(numpy.median(heights)) if heights else 0.0,
            )
        )
    return max(merged_by_mode, key=lambda candidate: candidate.metrics.utility)


def internal_merge_candidates(candidates: tuple[internal_Candidate, ...]) -> ObservationBatch:
    return internal_merge_candidate_batches(candidates).observations


def internal_augment_candidate(
    primary: internal_Candidate,
    supplement: internal_Candidate,
    *,
    minimum_confidence: float,
) -> tuple[internal_Candidate, int]:
    """Add only high-quality supplement observations absent from the primary pass."""
    if not len(supplement.observations):
        return primary, 0
    observations = supplement.observations
    confidence = observations.confidence
    informative = numpy.fromiter(
        (sum(character.isalnum() for character in text) >= 2 for text in observations.text),
        dtype=numpy.bool_,
        count=len(observations),
    )
    useful = numpy.fromiter(
        (
            internal_observation_utility(text, float(value)) >= 2.0
            for text, value in zip(observations.text, confidence, strict=True)
        ),
        dtype=numpy.bool_,
        count=len(observations),
    )
    coverage = maximum_candidate_coverage(
        observations.bbox,
        primary.observations.bbox,
    )
    additions = (
        (confidence >= max(70.0, minimum_confidence)) & informative & useful & (coverage < 0.30)
    )
    added = int(numpy.count_nonzero(additions))
    if not added:
        return primary, 0
    combined = ObservationBatch.concatenate_selected(
        primary.observations,
        observations,
        additions,
    )
    return internal_candidate(primary.mode, combined, symbols=primary.symbols), added


def internal_record_candidates(
    capture: CapturedPage,
    candidates: tuple[tuple[str, internal_Candidate], ...],
    selected_name: str,
) -> None:
    cache = capture.page.extraction_cache
    cache["ocr_candidate_diagnostics"] = tuple(
        {
            "name": name,
            "mode": candidate.mode,
            "selected": name == selected_name,
            **candidate.metrics.as_record(),
        }
        for name, candidate in candidates
    )
    if os.environ.get("CORE_PDF_CANDIDATE_ANALYSIS", "").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        cache["ocr_candidate_analysis"] = tuple(
            {
                "name": name,
                "mode": candidate.mode,
                "selected": name == selected_name,
                "text": "\n".join(candidate.observations.text),
                **candidate.metrics.as_record(),
            }
            for name, candidate in candidates
        )


def internal_decoded_image_raster(
    image: Any,
    display_area: float,
    *,
    cache: Any | None = None,
    image_cache: Any | None = None,
    max_pixels: int = MAX_OCR_PIXELS,
) -> internal_Raster | None:
    source = getattr(image, "image_source", None)
    source_key = getattr(source, "cache_key", None)
    if not isinstance(source_key, tuple):
        source_key = ("image", id(image))
    shared_key = ImageCacheKey(
        "ocr-raster",
        tuple(source_key),
        (float(display_area), int(max_pixels)),
    )
    page_cache_key = ("decoded_ocr_image_v3", *source_key, float(display_area), max_pixels)
    if image_cache is not None:
        cached = image_cache.get(shared_key)
        if isinstance(cached, internal_Raster):
            return cached
    if cache is not None:
        cached = cache.get(page_cache_key)
        if isinstance(cached, internal_Raster):
            return cached
    shared = source.decode() if source is not None and hasattr(source, "decode") else None
    samples: numpy.ndarray[Any, Any] | None
    data: bytes | memoryview | None
    if shared is not None:
        samples = shared.array
        data = None
        decoded_width = shared.width
        decoded_height = shared.height
        decoded_channels = shared.channels
    else:
        raw = getattr(image, "raw_data", None)
        dictionary = getattr(image, "dictionary", None)
        if not isinstance(raw, (bytes, bytearray, memoryview)) or not isinstance(dictionary, dict):
            return None
        decoded = decode_pdf_image(raw, dictionary)
        if decoded is None:
            return None
        if isinstance(decoded.data, numpy.ndarray):
            array = cast(numpy.ndarray[Any, Any], decoded.data)
            samples = array.reshape((decoded.height, decoded.width, decoded.channels))
            data = None
        elif isinstance(decoded.data, (bytes, memoryview)):
            samples = None
            data = decoded.data
        else:
            samples = None
            data = memoryview(decoded.data).cast("B")
        decoded_width = decoded.width
        decoded_height = decoded.height
        decoded_channels = decoded.channels
    pixels_per_point = math.sqrt(decoded_width * decoded_height / max(1.0, display_area))
    resolution = max(70, min(600, int(round(72.0 * pixels_per_point))))
    width = decoded_width
    height = decoded_height
    channels = decoded_channels
    if width * height > max_pixels:
        reduction = math.sqrt(max_pixels / (width * height)) * 0.999
        target_width = max(1, int(width * reduction))
        target_height = max(1, int(height * reduction))
        if samples is None:
            assert data is not None
            samples = uint8_image_view(data, (height, width, channels))
        samples = resample_nearest(samples, target_height, target_width)
        data = None
        resolution = max(70, int(round(resolution * target_width / width)))
        width = target_width
        height = target_height
    scale = max(1, math.ceil(MIN_DIRECT_OCR_RESOLUTION / resolution))
    max_scale = max(1, int(math.sqrt(max_pixels / max(1, width * height))))
    scale = min(scale, max_scale)
    if scale > 1:
        if samples is None:
            assert data is not None
            samples = uint8_image_view(data, (height, width, channels))
        samples = resample_nearest(samples, height * scale, width * scale)
        data = None
        width *= scale
        height *= scale
        resolution *= scale
    if data is None:
        assert samples is not None
        data = contiguous_bytes(samples)
    raster = internal_Raster(RasterImage(data, width, height, channels), resolution)
    if image_cache is not None:
        image_cache.put(shared_key, raster)
    elif cache is not None:
        cache[page_cache_key] = raster
    return raster


class DirectImageOrientation(StrEnum):
    IDENTITY = "identity"
    FLIP_X = "flip-x"
    FLIP_Y = "flip-y"
    FLIP_XY = "flip-xy"
    TRANSPOSE = "transpose"
    TRANSPOSE_FLIP_X = "transpose-flip-x"
    TRANSPOSE_FLIP_Y = "transpose-flip-y"
    TRANSPOSE_FLIP_XY = "transpose-flip-xy"


internal_DIRECT_IMAGE_ORIENTATIONS: dict[DirectImageOrientation, tuple[int, int, int, int]] = {
    DirectImageOrientation.IDENTITY: (0, 1, 2, 3),
    DirectImageOrientation.FLIP_X: (1, 0, 3, 2),
    DirectImageOrientation.FLIP_Y: (2, 3, 0, 1),
    DirectImageOrientation.FLIP_XY: (3, 2, 1, 0),
    DirectImageOrientation.TRANSPOSE: (0, 2, 1, 3),
    DirectImageOrientation.TRANSPOSE_FLIP_X: (2, 0, 3, 1),
    DirectImageOrientation.TRANSPOSE_FLIP_Y: (1, 3, 0, 2),
    DirectImageOrientation.TRANSPOSE_FLIP_XY: (3, 1, 2, 0),
}


def internal_direct_image_orientation(
    image: Any,
    *,
    maximum_axis_deviation: float = 1e-5,
) -> DirectImageOrientation | None:
    items = getattr(image, "items", ())
    quad = next(
        (
            value
            for kind, value in items
            if kind == "quad" and isinstance(value, (list, tuple)) and len(value) == 4
        ),
        None,
    )
    if quad is None:
        return None
    try:
        points = tuple((float(point[0]), float(point[1])) for point in quad)
    except (IndexError, TypeError, ValueError):
        return None
    x0 = min(point[0] for point in points)
    y0 = min(point[1] for point in points)
    x1 = max(point[0] for point in points)
    y1 = max(point[1] for point in points)
    if x1 <= x0 or y1 <= y0:
        return None
    target_corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    tolerance = max(0.01, max(x1 - x0, y1 - y0) * maximum_axis_deviation)
    target_to_raw = [-1, -1, -1, -1]
    for raw_index, point in enumerate(points):
        target_index = min(
            range(4),
            key=lambda index: (
                abs(point[0] - target_corners[index][0]) + abs(point[1] - target_corners[index][1])
            ),
        )
        target = target_corners[target_index]
        if max(abs(point[0] - target[0]), abs(point[1] - target[1])) > tolerance:
            return None
        if target_to_raw[target_index] != -1:
            return None
        target_to_raw[target_index] = raw_index
    orientation_corners = tuple(target_to_raw)
    return next(
        (
            orientation
            for orientation, corners in internal_DIRECT_IMAGE_ORIENTATIONS.items()
            if corners == orientation_corners
        ),
        None,
    )


def internal_orient_direct_image_raster(
    image: Any,
    raster: internal_Raster,
    *,
    orientation: DirectImageOrientation | None = None,
) -> internal_Raster:
    orientation = orientation or internal_direct_image_orientation(image)
    if orientation in {None, DirectImageOrientation.IDENTITY}:
        return raster
    samples = raster.image.array()
    match orientation:
        case DirectImageOrientation.FLIP_X:
            oriented = samples[:, ::-1]
        case DirectImageOrientation.FLIP_Y:
            oriented = samples[::-1]
        case DirectImageOrientation.FLIP_XY:
            oriented = samples[::-1, ::-1]
        case DirectImageOrientation.TRANSPOSE:
            oriented = samples.transpose(1, 0, 2)
        case DirectImageOrientation.TRANSPOSE_FLIP_X:
            oriented = samples.transpose(1, 0, 2)[::-1]
        case DirectImageOrientation.TRANSPOSE_FLIP_Y:
            oriented = samples.transpose(1, 0, 2)[:, ::-1]
        case DirectImageOrientation.TRANSPOSE_FLIP_XY:
            oriented = samples.transpose(1, 0, 2)[::-1, ::-1]
        case _:
            return raster
    height, width, channels = oriented.shape
    return internal_Raster(
        RasterImage(contiguous_bytes(oriented), int(width), int(height), int(channels)),
        raster.resolution,
    )


def internal_page_image_regions(
    capture: CapturedPage,
    *,
    minimum_area_ratio: float,
    max_pixels: int = MAX_OCR_PIXELS,
    maximum_axis_deviation: float = 1e-5,
) -> tuple[internal_RasterRegion, ...]:
    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    page_area = max(1.0, page_width * page_height)
    regions: list[internal_RasterRegion] = []
    for image in getattr(capture, "drawings", ()):
        if getattr(image, "kind", None) != "image":
            continue
        orientation = internal_direct_image_orientation(
            image,
            maximum_axis_deviation=maximum_axis_deviation,
        )
        if orientation is None:
            continue
        box = rect_tuple(getattr(image, "rect", None))
        if box is None:
            continue
        clipped = (
            max(0.0, box[0]),
            max(0.0, box[1]),
            min(page_width, box[2]),
            min(page_height, box[3]),
        )
        # A decoded source raster represents the full image. If the image is clipped by
        # the page, mapping that full raster onto the clipped rectangle would compress
        # its OCR coordinates. Let the page compositor produce the correct crop instead.
        clip_tolerance = max(2.0, max(page_width, page_height) * 0.005)
        if any(
            abs(float(original) - clipped_value) > clip_tolerance
            for original, clipped_value in zip(box, clipped, strict=True)
        ):
            continue
        display_area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
        if display_area / page_area < minimum_area_ratio:
            continue
        raster = internal_decoded_image_raster(
            image,
            display_area,
            cache=getattr(capture.page, "extraction_cache", None),
            image_cache=getattr(getattr(capture.page, "document", None), "image_cache", None),
            max_pixels=max_pixels,
        )
        if raster is not None:
            oriented = raster
            if orientation is not DirectImageOrientation.IDENTITY:
                source = getattr(image, "image_source", None)
                source_key = getattr(source, "cache_key", None)
                if not isinstance(source_key, tuple):
                    source_key = ("image", id(image))
                oriented_key = ImageCacheKey(
                    "ocr-oriented-raster",
                    tuple(source_key),
                    (orientation.value, float(display_area), int(max_pixels)),
                )
                cache = getattr(getattr(capture.page, "document", None), "image_cache", None)
                cached_oriented = cache.get(oriented_key) if cache is not None else None
                if isinstance(cached_oriented, internal_Raster):
                    oriented = cached_oriented
                else:
                    oriented = internal_orient_direct_image_raster(
                        image,
                        raster,
                        orientation=orientation,
                    )
                    if cache is not None:
                        cache.put(oriented_key, oriented)
            regions.append(
                internal_RasterRegion(
                    oriented,
                    clipped,
                )
            )
    return tuple(regions)


def internal_dominant_image_region(
    capture: CapturedPage,
    *,
    max_pixels: int = MAX_OCR_PIXELS,
) -> internal_RasterRegion | None:
    def box_area(box: tuple[float, float, float, float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    regions = internal_page_image_regions(
        capture,
        minimum_area_ratio=0.65,
        max_pixels=max_pixels,
    )
    substantial = tuple(
        region for region in regions if region.raster.width * region.raster.height >= 4_096
    )
    if not substantial:
        return None
    if len(substantial) > 1:
        largest = max(substantial, key=lambda region: box_area(region.page_box))
        largest_area = max(1.0, box_area(largest.page_box))
        overlapping = sum(
            max(
                0.0,
                min(region.page_box[2], largest.page_box[2])
                - max(region.page_box[0], largest.page_box[0]),
            )
            * max(
                0.0,
                min(region.page_box[3], largest.page_box[3])
                - max(region.page_box[1], largest.page_box[1]),
            )
            / largest_area
            >= 0.90
            for region in substantial
            if region is not largest
        )
        if overlapping:
            return None
    return max(substantial, key=lambda region: region.raster.width * region.raster.height)


OCR_REGION_INITIAL_COUNT = 8
OCR_REGION_MAX_COUNT = 16
OCR_REGION_INITIAL_AREA_RATIO = 0.25
OCR_REGION_MAX_AREA_RATIO = 0.60
OCR_DIRECT_REGION_MIN_COVERAGE = 0.65
# Small affine placement noise is cheaper to absorb in OCR coordinates than to
# recompose and rasterize the entire page around an otherwise usable source image.
OCR_IMAGE_REGIONS_MAX_AXIS_DEVIATION = 0.01


def internal_ocr_region_box(
    box: tuple[float, float, float, float],
    *,
    page_width: float,
    page_height: float,
    padding: float,
) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = box
    clipped = (
        max(0.0, x0 - padding),
        max(0.0, y0 - padding),
        min(page_width, x1 + padding),
        min(page_height, y1 + padding),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def internal_ocr_region_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    smaller = min(
        max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1]),
        max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1]),
    )
    return intersection / smaller if smaller else 0.0


def internal_ocr_region_coverage(
    target: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
) -> float:
    """Return how much of a requested OCR target is covered by a candidate raster."""
    target_area = max(0.0, target[2] - target[0]) * max(0.0, target[3] - target[1])
    return bbox_intersection_area(target, candidate) / target_area if target_area else 0.0


def internal_merge_ocr_regions(regions: list[internal_OcrRegion]) -> tuple[internal_OcrRegion, ...]:
    merged: list[internal_OcrRegion] = []
    for region in sorted(regions, key=lambda item: (-item.score, item.page_box)):
        match = next(
            (
                index
                for index, existing in enumerate(merged)
                if internal_ocr_region_overlap(existing.page_box, region.page_box) >= 0.35
            ),
            None,
        )
        if match is None:
            merged.append(region)
            continue
        existing = merged[match]
        merged[match] = internal_OcrRegion(
            (
                min(existing.page_box[0], region.page_box[0]),
                min(existing.page_box[1], region.page_box[1]),
                max(existing.page_box[2], region.page_box[2]),
                max(existing.page_box[3], region.page_box[3]),
            ),
            max(existing.score, region.score) + min(existing.score, region.score) * 0.15,
            tuple(dict.fromkeys((*existing.reasons, *region.reasons))),
        )
    return tuple(sorted(merged, key=lambda item: (-item.score, item.page_box)))


def internal_candidate_ocr_regions(capture: CapturedPage) -> tuple[internal_OcrRegion, ...]:
    """Select likely OCR areas using capture-time geometry only.

    This deliberately does not render a preview image.  Native text, image bounds,
    captured paths, and grid lines are already available from the canonical page IR.
    """
    cache = getattr(capture.page, "extraction_cache", None)
    cache_key = "ocr_candidate_regions_v1"
    if cache is not None:
        cached = cache.get(cache_key)
        if isinstance(cached, tuple) and all(
            isinstance(item, internal_OcrRegion) for item in cached
        ):
            return cached

    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    page_area = max(1.0, page_width * page_height)
    padding = max(6.0, min(36.0, min(page_width, page_height) * 0.01))
    candidates: list[internal_OcrRegion] = []

    for box in getattr(capture.evidence, "image_boxes", ()):
        image_box = cast(
            tuple[float, float, float, float],
            tuple(float(value) for value in box),
        )
        padded = internal_ocr_region_box(
            image_box,
            page_width=page_width,
            page_height=page_height,
            padding=padding,
        )
        if padded is not None:
            candidates.append(internal_OcrRegion(padded, 5.0, ("image",)))

    native = getattr(capture, "observations", ObservationBatch.empty())
    native_boxes = tuple(tuple(float(value) for value in box) for box in native.bbox)
    native_index = SpatialIndex.from_boxes(native_boxes) if native_boxes else None

    def native_overlap(box: tuple[float, float, float, float]) -> float:
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        if native_index is not None:
            return min(
                1.0,
                sum(
                    bbox_intersection_area(box, hit.bbox)
                    for hit in native_index.intersecting_hits(box)
                )
                / area,
            )
        return min(
            1.0,
            sum(
                max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
                * max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
                for other in native_boxes
            )
            / area,
        )

    for drawing in getattr(capture, "drawings", ()):
        if getattr(drawing, "kind", None) not in {"fill", "fillstroke", "stroke"}:
            continue
        box = rect_tuple(getattr(drawing, "rect", None))
        if box is None:
            continue
        drawing_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if drawing_area <= 0.0 or drawing_area >= page_area * 0.80:
            continue
        uncovered = native_overlap(box) < 0.25
        if uncovered and getattr(drawing, "kind", None) in {"fill", "fillstroke"}:
            padded = internal_ocr_region_box(
                box,
                page_width=page_width,
                page_height=page_height,
                padding=padding,
            )
            if padded is not None:
                candidates.append(internal_OcrRegion(padded, 3.5, ("uncovered-vector",)))

    if hasattr(capture, "grid_lines"):
        horizontal, vertical = internal_axis_segments(capture)
    else:
        horizontal = numpy.empty((0, 3), dtype=numpy.float32)
        vertical = numpy.empty((0, 3), dtype=numpy.float32)
    for component_horizontal, component_vertical in internal_grid_components(horizontal, vertical):
        x0 = min(float(component_horizontal[:, 0].min()), float(component_vertical[:, 0].min()))
        y0 = min(float(component_horizontal[:, 2].min()), float(component_vertical[:, 1].min()))
        x1 = max(float(component_horizontal[:, 1].max()), float(component_vertical[:, 0].max()))
        y1 = max(float(component_horizontal[:, 2].max()), float(component_vertical[:, 2].max()))
        for split_horizontal, split_vertical in internal_split_grid_component(
            component_horizontal,
            component_vertical,
        ):
            split_box = (
                min(float(split_horizontal[:, 0].min()), float(split_vertical[:, 0].min())),
                min(float(split_horizontal[:, 2].min()), float(split_vertical[:, 1].min())),
                max(float(split_horizontal[:, 1].max()), float(split_vertical[:, 0].max())),
                max(float(split_horizontal[:, 2].max()), float(split_vertical[:, 2].max())),
            )
            padded = internal_ocr_region_box(
                split_box,
                page_width=page_width,
                page_height=page_height,
                padding=padding,
            )
            if padded is not None and (
                (padded[2] - padded[0]) * (padded[3] - padded[1]) < page_area * 0.45
            ):
                candidates.append(internal_OcrRegion(padded, 4.0, ("grid",)))
        if not component_horizontal.size or not component_vertical.size:
            continue
        component_box = (x0, y0, x1, y1)
        component_area = (x1 - x0) * (y1 - y0)
        if component_area < page_area * 0.45 and native_overlap(component_box) < 0.25:
            padded = internal_ocr_region_box(
                component_box,
                page_width=page_width,
                page_height=page_height,
                padding=padding,
            )
            if padded is not None:
                candidates.append(internal_OcrRegion(padded, 3.0, ("grid-labels",)))

    columns = 6
    rows = max(2, min(8, int(round(columns * page_height / max(1.0, page_width)))))
    vector_density = numpy.zeros(rows * columns, dtype=numpy.float32)
    for drawing in getattr(capture, "drawings", ()):
        if getattr(drawing, "kind", None) not in VECTOR_PAINT_KINDS:
            continue
        box = rect_tuple(getattr(drawing, "rect", None))
        if box is None:
            continue
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        column = min(columns - 1, max(0, int(center_x * columns / max(1.0, page_width))))
        row = min(rows - 1, max(0, int(center_y * rows / max(1.0, page_height))))
        vector_density[row * columns + column] += 1.0
    for line in getattr(capture, "grid_lines", ()):
        center_x = (float(line.x0) + float(line.x1)) * 0.5
        center_y = (float(line.y0) + float(line.y1)) * 0.5
        column = min(columns - 1, max(0, int(center_x * columns / max(1.0, page_width))))
        row = min(rows - 1, max(0, int(center_y * rows / max(1.0, page_height))))
        vector_density[row * columns + column] += 0.5

    native_counts = numpy.zeros(rows * columns, dtype=numpy.float32)
    for text, box in zip(native.text, native.bbox, strict=True):
        center_x = (float(box[0]) + float(box[2])) * 0.5
        center_y = (float(box[1]) + float(box[3])) * 0.5
        column = min(columns - 1, max(0, int(center_x * columns / max(1.0, page_width))))
        row = min(rows - 1, max(0, int(center_y * rows / max(1.0, page_height))))
        native_counts[row * columns + column] += sum(not char.isspace() for char in text)

    for cell, density in enumerate(vector_density):
        if density <= 0.0:
            continue
        row, column = divmod(cell, columns)
        cell_box = (
            column * page_width / columns,
            row * page_height / rows,
            (column + 1) * page_width / columns,
            (row + 1) * page_height / rows,
        )
        sparse = native_counts[cell] < 8.0
        header_band = row in {0, rows - 1} and native_counts[cell] < 24.0
        if not sparse and not header_band:
            continue
        padded = internal_ocr_region_box(
            cell_box,
            page_width=page_width,
            page_height=page_height,
            padding=padding,
        )
        if padded is not None:
            reasons = ["vector-density"]
            if sparse:
                reasons.append("sparse-label")
            if header_band:
                reasons.append("header-band")
            candidates.append(
                internal_OcrRegion(
                    padded,
                    1.5 + min(2.0, float(density) / 8.0),
                    tuple(reasons),
                )
            )

    if (
        capture.evidence.vector_complexity >= 180
        and capture.evidence.text_coverage < 0.05
        and (not native_boxes or len(native_boxes) >= 8)
    ):
        # Component labels are often isolated from the larger paths they
        # annotate. Use finer cells for these vector-only pages so the region
        # budget can select several label clusters instead of one broad artwork
        # box. The existing coarse density pass remains responsible for larger
        # diagram areas.
        label_columns = 12
        label_rows = max(
            4,
            min(12, int(round(label_columns * page_height / max(1.0, page_width)))),
        )
        label_density = numpy.zeros(label_rows * label_columns, dtype=numpy.float32)
        label_boxes: list[list[tuple[float, float, float, float]]] = [
            [] for _ in range(label_rows * label_columns)
        ]
        for drawing in getattr(capture, "drawings", ()):
            if getattr(drawing, "kind", None) not in VECTOR_PAINT_KINDS:
                continue
            box = rect_tuple(getattr(drawing, "rect", None))
            if box is None:
                continue
            center_x = (box[0] + box[2]) * 0.5
            center_y = (box[1] + box[3]) * 0.5
            column = min(
                label_columns - 1,
                max(0, int(center_x * label_columns / max(1.0, page_width))),
            )
            row = min(
                label_rows - 1,
                max(0, int(center_y * label_rows / max(1.0, page_height))),
            )
            label_density[row * label_columns + column] += 1.0
            label_boxes[row * label_columns + column].append(box)

        for cell, density in enumerate(label_density):
            if density <= 0.0:
                continue
            row, column = divmod(cell, label_columns)
            cell_box = (
                column * page_width / label_columns,
                row * page_height / label_rows,
                (column + 1) * page_width / label_columns,
                (row + 1) * page_height / label_rows,
            )
            component_boxes = label_boxes[cell]
            component_box = (
                min(box[0] for box in component_boxes),
                min(box[1] for box in component_boxes),
                max(box[2] for box in component_boxes),
                max(box[3] for box in component_boxes),
            )
            component_area = max(0.0, component_box[2] - component_box[0]) * max(
                0.0, component_box[3] - component_box[1]
            )
            label_padding = max(
                padding,
                min(72.0, min(page_width, page_height) * 0.03),
            )
            candidate_box = component_box if component_area <= page_area * 0.08 else cell_box
            padded = internal_ocr_region_box(
                candidate_box,
                page_width=page_width,
                page_height=page_height,
                padding=label_padding if candidate_box == component_box else padding,
            )
            if padded is not None:
                candidates.append(
                    internal_OcrRegion(
                        padded,
                        1.0 + min(3.0, float(density) / 8.0),
                        ("vector-label-density", "vector-label-neighborhood")
                        if candidate_box == component_box
                        else ("vector-label-density",),
                    )
                )

    regions = internal_merge_ocr_regions(candidates)
    if not regions:
        regions = (
            internal_OcrRegion(
                (0.0, 0.0, page_width, page_height),
                0.0,
                ("page-fallback",),
            ),
        )
    if cache is not None:
        cache[cache_key] = regions
    return regions


def internal_has_distributed_outline_text(capture: CapturedPage) -> bool:
    """Detect pages whose text was converted into many small filled vector paths."""
    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    max_width = max(24.0, page_width * 0.04)
    max_height = max(24.0, page_height * 0.04)
    boxes = tuple(
        box
        for drawing in getattr(capture, "drawings", ())
        if getattr(drawing, "kind", None) in {"fill", "fillstroke"}
        and (box := rect_tuple(getattr(drawing, "rect", None))) is not None
        and 0.0 < box[2] - box[0] <= max_width
        and 0.0 < box[3] - box[1] <= max_height
    )
    if len(boxes) < 200:
        return False
    bounds = (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
    width_ratio = (bounds[2] - bounds[0]) / max(1.0, page_width)
    height_ratio = (bounds[3] - bounds[1]) / max(1.0, page_height)
    return width_ratio >= 0.60 and height_ratio >= 0.60


def internal_rendered_page_raster(
    capture: CapturedPage,
    requested_scale: float,
    *,
    crop: tuple[float, float, float, float] | None = None,
    rendered: Any | None = None,
    cache: bool = True,
    max_pixels: int = MAX_OCR_PIXELS,
    include_native_text: bool = False,
) -> internal_Raster | None:
    page = capture.page
    compose_started = time.perf_counter()
    if rendered is None:
        rendered = compose_page(
            page,
            RenderOptions(include_text=include_native_text),
            page_program=capture.program,
        )
    compose_seconds = time.perf_counter() - compose_started
    if crop is None:
        raster_area = max(1.0, float(page.width) * float(page.height))
    else:
        raster_area = max(1.0, (crop[2] - crop[0]) * (crop[3] - crop[1]))
    safe_scale = math.sqrt(max_pixels / raster_area) * 0.999
    scale = min(requested_scale, safe_scale)
    raster_started = time.perf_counter()
    width, height = rendered.raster_size(scale)
    try:
        data = rendered.rasterize(
            background=(255, 255, 255, 255),
            scale=scale,
            max_pixels=max_pixels,
            crop=crop,
            cache=cache,
        )
    except IndexError as error:
        # A malformed embedded image can produce a source sample outside its
        # decoded raster during compositing.  Keep native extraction usable and
        # let OCR continue without the rendered-page fallback.
        page.extraction_cache["ocr_render_error"] = str(error)
        return None
    page.extraction_cache["ocr_render_timings"] = {
        "compose_seconds": compose_seconds,
        "rasterize_seconds": time.perf_counter() - raster_started,
        "raster_mode": "region" if crop is not None else "page",
        "crop": crop,
        "raster_pixels": width * height,
        "pixel_budget": max_pixels,
        "include_native_text": include_native_text,
        "image_timings": rendered.metadata.get("__core_pdf_raster_image_timings__", {}),
        "display_items": len(rendered.display_list.items),
        "display_item_kinds": dict(
            Counter(
                str(getattr(item, "kind", type(item).__name__))
                for item in rendered.display_list.items
            )
        ),
        "image_filters": tuple(
            str(((getattr(item, "data", None) or {}).get("dictionary") or {}).get("Filter"))
            for item in rendered.display_list.items
            if getattr(item, "kind", None) in {"image", "inline-image"}
        ),
    }
    return internal_Raster(data, max(70, int(round(72.0 * scale))))


STROKED_VECTOR_PACK_WIDTH = 240.0
STROKED_VECTOR_PACK_HORIZONTAL_PADDING = 4.0
STROKED_VECTOR_PACK_VERTICAL_PADDING = 4.0
STROKED_VECTOR_PACK_DENSE_VERTICAL_PADDING = 2.0
STROKED_VECTOR_PACK_DENSE_MIN_CELLS = 96
STROKED_VECTOR_PACK_REMAP_TOLERANCE = 4.0
STROKED_VECTOR_PACK_MIN_ALIGNED_SEEDS = 12
STROKED_VECTOR_PACK_MIN_LEARNED_SIGNATURES = 16
STROKED_VECTOR_PACK_MIN_DECODED_RUNS = 16


@dataclass(frozen=True, slots=True)
class internal_CachedStrokedTextProfile:
    drawings: tuple[Any, ...]
    drawing_indexes: tuple[int, ...]
    profile: StrokedTextProfile


def internal_stroked_text_profile(capture: CapturedPage) -> StrokedTextProfile:
    """Return the single structural glyph profile shared by OCR and document reuse."""
    evidence = capture.evidence.stroked_vector_text
    cache = capture.page.extraction_cache
    cached = cache.get("_stroked_text_profile")
    if (
        isinstance(cached, internal_CachedStrokedTextProfile)
        and cached.drawings is capture.drawings
        and cached.drawing_indexes == evidence.drawing_indexes
    ):
        return cached.profile
    profile = profile_stroked_text(capture.drawings, evidence.drawing_indexes)
    cache["_stroked_text_profile"] = internal_CachedStrokedTextProfile(
        capture.drawings,
        evidence.drawing_indexes,
        profile,
    )
    return profile


def internal_pack_stroked_text_runs(
    runs: tuple[StrokedTextRun, ...],
    *,
    width: float = STROKED_VECTOR_PACK_WIDTH,
    horizontal_padding: float = STROKED_VECTOR_PACK_HORIZONTAL_PADDING,
    vertical_padding: float = STROKED_VECTOR_PACK_VERTICAL_PADDING,
) -> tuple[tuple[internal_StrokedTextCell, ...], float]:
    """Shelf-pack vector words without scaling their glyph geometry."""
    if not runs:
        return (), 0.0
    ordered = sorted(
        runs,
        key=lambda run: (
            -(run.bbox[3] - run.bbox[1]),
            -(run.bbox[2] - run.bbox[0]),
            run.drawing_indexes[0],
        ),
    )
    x = horizontal_padding
    y = vertical_padding
    row_height = 0.0
    cells: list[internal_StrokedTextCell] = []
    for run in ordered:
        source = run.bbox
        run_width = source[2] - source[0]
        run_height = source[3] - source[1]
        cell_width = run_width + horizontal_padding * 2.0
        cell_height = run_height + vertical_padding * 2.0
        if x > horizontal_padding and x + cell_width > width:
            y += row_height
            x = horizontal_padding
            row_height = 0.0
        packed = (
            x + horizontal_padding,
            y + vertical_padding,
            x + horizontal_padding + run_width,
            y + vertical_padding + run_height,
        )
        cells.append(
            internal_StrokedTextCell(
                source_box=source,
                packed_box=packed,
                drawing_indexes=run.drawing_indexes,
            )
        )
        x += cell_width
        row_height = max(row_height, cell_height)
    return tuple(cells), y + row_height + vertical_padding


def internal_stroked_vector_text_raster(
    capture: CapturedPage,
    requested_scale: float,
    *,
    max_pixels: int = MAX_OCR_PIXELS,
) -> internal_PackedStrokedTextRaster | None:
    """Pack vector words into a compact seed raster with piecewise page mapping."""
    evidence = capture.evidence.stroked_vector_text
    if not evidence.trusted or not evidence.drawing_indexes:
        return None
    runs = internal_stroked_text_profile(capture).seed_runs
    vertical_padding = (
        STROKED_VECTOR_PACK_DENSE_VERTICAL_PADDING
        if len(runs) >= STROKED_VECTOR_PACK_DENSE_MIN_CELLS
        else STROKED_VECTOR_PACK_VERTICAL_PADDING
    )
    cells, packed_height = internal_pack_stroked_text_runs(
        runs,
        vertical_padding=vertical_padding,
    )
    if not cells or packed_height <= 0.0:
        return None
    packed_width = STROKED_VECTOR_PACK_WIDTH
    area = max(1.0, packed_width * packed_height)
    safe_scale = math.sqrt(max_pixels / area) * 0.999
    scale = min(requested_scale, safe_scale)
    page = capture.page
    cache = getattr(page, "extraction_cache", None)
    cache_key = ("packed_stroked_vector_text_raster_v5", scale, max_pixels)
    if cache is not None:
        cached = cache.get(cache_key)
        if isinstance(cached, internal_PackedStrokedTextRaster):
            return cached

    compose_started = time.perf_counter()
    display_list = DisplayList(width=packed_width, height=packed_height)
    for cell in cells:
        tx = cell.packed_box[0] - cell.source_box[0]
        ty = cell.packed_box[1] - cell.source_box[1]
        for index in cell.drawing_indexes:
            drawing = capture.drawings[index]
            drawing_box = rect_tuple(getattr(drawing, "rect", None))
            path = getattr(drawing, "path", None)
            if drawing_box is None or path is None:
                continue
            # Packed cells preserve capture order and paint style. Reuse the
            # display-list stroke coalescer so thousands of tiny glyph paths do
            # not become thousands of renderer dispatches.
            display_list.append_captured_drawing(
                drawing.replace(
                    bbox=None,
                    path=path.translated(tx, ty),
                    fill_pattern=None,
                    stroke_pattern=None,
                    fill=(0.0, 0.0, 0.0),
                    stroke_color=(0.0, 0.0, 0.0),
                )
            )
    rendered = RenderedPage(
        page_number=int(getattr(page, "page_number", 0)),
        width=packed_width,
        height=packed_height,
        rotate=0,
        display_list=display_list,
    )
    compose_seconds = time.perf_counter() - compose_started
    raster_started = time.perf_counter()
    fast_path = bool(display_list.items) and all(
        type(item) is PathPaintItem
        and item.paint_kind is PathPaintKind.STROKE
        and not (item.dash_pattern and item.dash_pattern[0])
        and item.blend_mode is None
        and item.soft_mask_alpha is None
        and (item.stroke_opacity is None or float(item.stroke_opacity) >= 0.999)
        and int(item.line_cap or 0) == 1
        and int(item.line_join or 0) == 1
        for item in display_list.items
    )
    if fast_path:
        data = rasterize_packed_stroked_paths(
            tuple(display_list.items),
            packed_width,
            packed_height,
            scale,
        )
    else:
        data = rendered.rasterize(
            background=(255, 255, 255, 255),
            scale=scale,
            max_pixels=max_pixels,
            cache=False,
        )
    raster_seconds = time.perf_counter() - raster_started
    raster = internal_Raster(data, max(70, int(round(72.0 * scale))))
    packed = internal_PackedStrokedTextRaster(
        raster=raster,
        packed_box=(0.0, 0.0, packed_width, packed_height),
        cells=cells,
    )
    if cache is not None:
        cache[cache_key] = packed
        cache["ocr_render_timings"] = {
            "compose_seconds": compose_seconds,
            "rasterize_seconds": raster_seconds,
            "raster_mode": "packed-stroked-vector-text",
            "raster_kernel": "wu" if fast_path else "general",
            "crop": packed.packed_box,
            "raster_pixels": raster.width * raster.height,
            "pixel_budget": max_pixels,
            "include_native_text": False,
            "image_timings": {},
            "display_items": len(display_list.items),
            "display_item_kinds": {"compact-stroke": len(display_list.items)},
            "image_filters": (),
            "packed_cells": len(cells),
            "horizontal_padding": STROKED_VECTOR_PACK_HORIZONTAL_PADDING,
            "vertical_padding": vertical_padding,
            "source_bbox": evidence.bbox,
        }
    return packed


def internal_full_stroked_vector_text_raster(
    capture: CapturedPage,
    requested_scale: float,
    *,
    max_pixels: int = MAX_OCR_PIXELS,
) -> internal_RasterRegion | None:
    """Render the full compact-stroke layer when packed seed OCR is insufficient."""
    evidence = capture.evidence.stroked_vector_text
    if not evidence.trusted or evidence.bbox is None or not evidence.drawing_indexes:
        return None
    page = capture.page
    page_width = float(page.width)
    page_height = float(page.height)
    padding = 4.0
    bbox = evidence.bbox
    crop = (
        max(0.0, bbox[0] - padding),
        max(0.0, bbox[1] - padding),
        min(page_width, bbox[2] + padding),
        min(page_height, bbox[3] + padding),
    )
    area = max(1.0, (crop[2] - crop[0]) * (crop[3] - crop[1]))
    safe_scale = math.sqrt(max_pixels / area) * 0.999
    scale = min(requested_scale, safe_scale)
    cache = getattr(page, "extraction_cache", None)
    cache_key = ("stroked_vector_text_raster_v1", scale, max_pixels)
    if cache is not None:
        cached = cache.get(cache_key)
        if isinstance(cached, internal_RasterRegion):
            return cached

    compose_started = time.perf_counter()
    display_list = DisplayList(width=page_width, height=page_height)
    for index in evidence.drawing_indexes:
        drawing = capture.drawings[index]
        display_list.append(
            drawing.kind,
            drawing.seqno,
            bbox=drawing.rect,
            path=drawing.path,
            fill=(0.0, 0.0, 0.0),
            fill_opacity=drawing.fill_opacity,
            stroke_color=(0.0, 0.0, 0.0),
            stroke_opacity=drawing.stroke_opacity,
            line_width=drawing.line_width,
            line_cap=drawing.line_cap,
            line_join=drawing.line_join,
            dash_pattern=drawing.dash_pattern,
            fill_rule=drawing.fill_rule,
            blend_mode=drawing.blend_mode,
            soft_mask_alpha=drawing.soft_mask_alpha,
        )
    rendered = RenderedPage(
        page_number=int(getattr(page, "page_number", 0)),
        width=page_width,
        height=page_height,
        rotate=0,
        display_list=display_list,
    )
    compose_seconds = time.perf_counter() - compose_started
    raster_started = time.perf_counter()
    data = rendered.rasterize(
        background=(255, 255, 255, 255),
        scale=scale,
        max_pixels=max_pixels,
        crop=crop,
        cache=False,
    )
    raster = internal_Raster(data, max(70, int(round(72.0 * scale))))
    region = internal_RasterRegion(raster, crop)
    if cache is not None:
        cache[cache_key] = region
        cache["ocr_render_timings"] = {
            "compose_seconds": compose_seconds,
            "rasterize_seconds": time.perf_counter() - raster_started,
            "raster_mode": "stroked-vector-text-fallback",
            "crop": crop,
            "raster_pixels": raster.width * raster.height,
            "pixel_budget": max_pixels,
            "include_native_text": False,
            "image_timings": {},
            "display_items": len(display_list.items),
            "display_item_kinds": {"compact-stroke": len(display_list.items)},
            "image_filters": (),
        }
    return region


def internal_remap_stroked_vector_observations(
    observations: ObservationBatch,
    packed: internal_PackedStrokedTextRaster,
) -> tuple[ObservationBatch, int]:
    """Translate montage observations back through their containing cells."""
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    polygons: list[tuple[float, ...]] = []
    confidences: list[float] = []
    sequences: list[int] = []
    references: list[Any | None] = []
    for index, packed_box in enumerate(observations.bbox):
        box = cast(tuple[float, float, float, float], tuple(float(value) for value in packed_box))
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        cells = tuple(
            cell
            for cell in packed.cells
            if cell.packed_box[0] - STROKED_VECTOR_PACK_REMAP_TOLERANCE
            <= center_x
            <= cell.packed_box[2] + STROKED_VECTOR_PACK_REMAP_TOLERANCE
            and cell.packed_box[1] - STROKED_VECTOR_PACK_REMAP_TOLERANCE
            <= center_y
            <= cell.packed_box[3] + STROKED_VECTOR_PACK_REMAP_TOLERANCE
        )
        if len(cells) != 1:
            continue
        cell = cells[0]
        dx = cell.source_box[0] - cell.packed_box[0]
        dy = cell.source_box[1] - cell.packed_box[1]
        mapped = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
        texts.append(observations.text[index])
        boxes.append(mapped)
        polygons.append(
            (
                mapped[0],
                mapped[1],
                mapped[2],
                mapped[1],
                mapped[2],
                mapped[3],
                mapped[0],
                mapped[3],
            )
        )
        confidences.append(float(observations.confidence[index]))
        sequences.append(cell.drawing_indexes[0])
        references.append(observations.references[index])
    return (
        ObservationBatch.from_columns(
            texts,
            boxes,
            polygon=polygons,
            source=ObservationSource.OCR,
            confidence=confidences,
            sequence=sequences,
            rotation=(0 for _ in texts),
            font_size=(box[3] - box[1] for box in boxes),
            line_break_before=(True for _ in texts),
            references=references,
        ),
        len(observations) - len(texts),
    )


def internal_remap_stroked_vector_candidate(
    candidate: internal_Candidate,
    packed: internal_PackedStrokedTextRaster,
) -> tuple[internal_Candidate, int]:
    """Translate montage OCR words and symbols into their source page cells."""
    remapped, unmapped = internal_remap_stroked_vector_observations(
        candidate.observations,
        packed,
    )
    remapped_symbols, _unmapped_symbols = internal_remap_stroked_vector_observations(
        candidate.symbols,
        packed,
    )
    return (
        internal_candidate(
            candidate.mode,
            remapped,
            symbols=remapped_symbols,
            api_seconds=candidate.api_seconds,
            setup_seconds=candidate.setup_seconds,
            recognition_seconds=candidate.recognition_seconds,
            iterator_seconds=candidate.iterator_seconds,
            cleanup_seconds=candidate.cleanup_seconds,
            candidate_seconds=candidate.candidate_seconds,
            recognition_status=candidate.recognition_status,
            median_text_height=candidate.metrics.median_text_height,
        ),
        unmapped,
    )


def internal_safe_image_crop(capture: CapturedPage) -> tuple[float, float, float, float] | None:
    """Return a useful crop when OCR is known to be image-dominated.

    A crop is only safe when the image coverage is substantial.  Sparse images
    must not hide page text outside the image bounds from the page OCR path.
    """
    evidence = capture.evidence
    if not evidence.image_boxes or not (
        evidence.full_page_image or evidence.image_area_ratio >= 0.65
    ):
        return None
    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    x0 = min(box[0] for box in evidence.image_boxes)
    y0 = min(box[1] for box in evidence.image_boxes)
    x1 = max(box[2] for box in evidence.image_boxes)
    y1 = max(box[3] for box in evidence.image_boxes)
    crop = (max(0.0, x0), max(0.0, y0), min(page_width, x1), min(page_height, y1))
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        return None
    crop_area = (crop[2] - crop[0]) * (crop[3] - crop[1])
    if crop_area >= max(1.0, page_width * page_height * 0.90):
        return None
    return crop


def internal_ocr_region_batch(
    regions: tuple[internal_OcrRegion, ...],
    ocr_pass: OcrPass,
    *,
    expanded: bool,
    page_area: float,
) -> tuple[internal_OcrRegion, ...]:
    count_limit = max(
        ocr_pass.max_regions,
        OCR_REGION_MAX_COUNT if expanded else OCR_REGION_INITIAL_COUNT,
    )
    area_limit = OCR_REGION_MAX_AREA_RATIO if expanded else OCR_REGION_INITIAL_AREA_RATIO
    selected: list[internal_OcrRegion] = []
    area = 0.0
    page_area = max(1.0, page_area)
    if page_area <= 0.0:
        return ()
    for region in regions:
        if len(selected) >= count_limit:
            break
        if selected and area + region.area > page_area * area_limit:
            continue
        selected.append(region)
        area += region.area
    return tuple(selected)


def internal_candidate_region_tasks(
    capture: CapturedPage,
    regions: tuple[internal_OcrRegion, ...],
    ocr_pass: OcrPass,
    *,
    rendered: Any | None,
    compact_image: bool | str,
) -> tuple[
    tuple[internal_OcrTask, ...], int, Any | None, tuple[tuple[float, float, float, float], ...]
]:
    direct_regions = internal_page_image_regions(
        capture,
        minimum_area_ratio=0.02,
        max_pixels=ocr_pass.pixel_budget,
    )
    if not direct_regions:
        dominant = internal_dominant_image_region(
            capture,
            max_pixels=ocr_pass.pixel_budget,
        )
        if dominant is not None:
            direct_regions = (dominant,)
    tasks: list[internal_OcrTask] = []
    raster_pixels = 0
    rendered_boxes: list[tuple[float, float, float, float]] = []
    region_pass = replace(ocr_pass, scope=OcrPassScope.TILES, tiles=1)
    direct_region_index = (
        SpatialIndex(((index, region.page_box) for index, region in enumerate(direct_regions)))
        if len(direct_regions) > 4
        else None
    )
    for region in regions:
        raster: internal_Raster | None
        direct_candidates = (
            (direct_regions[index] for index in direct_region_index.intersecting(region.page_box))
            if direct_region_index is not None
            else iter(direct_regions)
        )
        matching_direct = tuple(
            candidate
            for candidate in direct_candidates
            # Region proposals include padding, so a source image need not cover the
            # entire box. It must still cover most of the requested target: otherwise
            # a narrow banner can incorrectly replace a broad compositor render.
            if internal_ocr_region_coverage(region.page_box, candidate.page_box)
            >= OCR_DIRECT_REGION_MIN_COVERAGE
        )
        layered_scan = any(
            internal_ocr_region_overlap(left.page_box, right.page_box) >= 0.90
            for index, left in enumerate(matching_direct)
            for right in matching_direct[:index]
        )
        direct = (
            None
            if layered_scan
            else max(
                matching_direct,
                key=lambda candidate: candidate.raster.width * candidate.raster.height,
                default=None,
            )
        )
        if direct is not None:
            raster = direct.raster
            raster_box = direct.page_box
        else:
            if rendered is None:
                rendered = compose_page(
                    capture.page,
                    RenderOptions(include_text=ocr_pass.include_native_text),
                    page_program=capture.program,
                )
            raster = internal_rendered_page_raster(
                capture,
                ocr_pass.scale,
                crop=region.page_box,
                rendered=rendered,
                cache=True,
                max_pixels=ocr_pass.pixel_budget,
                include_native_text=ocr_pass.include_native_text,
            )
            raster_box = region.page_box
        if raster is None:
            continue
        rendered_boxes.append(raster_box)
        raster_pixels += raster.width * raster.height
        full_page_region = (
            ocr_pass.scope is OcrPassScope.PAGE
            and len(regions) == 1
            and region.area
            >= getattr(
                getattr(capture, "evidence", None),
                "page_area",
                float(capture.page.width) * float(capture.page.height),
            )
            * 0.75
            and internal_ocr_region_coverage(
                region.page_box,
                (0.0, 0.0, float(capture.page.width), float(capture.page.height)),
            )
            >= 0.90
            and getattr(getattr(capture, "evidence", None), "vector_complexity", 0)
            >= OCR_PARALLEL_TILE_MIN_VECTOR_COMPLEXITY
            and 4_000_000 <= raster.width * raster.height <= PRIMARY_OCR_PIXELS
        )
        tile_count = ocr_pass.parallel_tiles if full_page_region else 1
        task_pass = (
            replace(
                region_pass,
                tiles=max(1, tile_count),
                recognize_words=True,
            )
            if layered_scan
            else replace(region_pass, tiles=max(1, tile_count))
        )
        tasks.extend(
            internal_tile_tasks(
                raster,
                raster_box,
                task_pass,
                compact_image=compact_image,
            )
        )
    return tuple(tasks), raster_pixels, rendered, tuple(rendered_boxes)


def internal_raster_rectangle_page_box(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    rectangle: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Map a top-left raster rectangle into bottom-left PDF page space."""
    x, y, width, height = rectangle
    page_x0, page_y0, page_x1, page_y1 = page_box
    page_width = page_x1 - page_x0
    page_height = page_y1 - page_y0
    return (
        page_x0 + x * page_width / raster.width,
        page_y1 - (y + height) * page_height / raster.height,
        page_x0 + (x + width) * page_width / raster.width,
        page_y1 - y * page_height / raster.height,
    )


def internal_high_resolution_weak_region_tasks(
    capture: CapturedPage,
    source_tasks: tuple[internal_OcrTask, ...],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
    *,
    rendered: Any | None,
    compact_image: bool | str,
) -> tuple[
    tuple[internal_OcrTask, ...], int, Any | None, tuple[tuple[float, float, float, float], ...]
]:
    """Rasterize only weak cells at rescue resolution instead of the whole page."""
    source_rasters: dict[tuple[int, tuple[float, float, float, float], int], internal_Raster] = {}
    for task in source_tasks:
        source_rasters.setdefault(
            (id(task.image), task.page_box, task.resolution),
            internal_Raster(task.image, task.resolution),
        )
    weak_regions: list[internal_OcrRegion] = []
    for (_, page_box, _), source_raster in source_rasters.items():
        for rectangle in internal_weak_region_rectangles(
            source_raster,
            page_box,
            ocr_pass,
            primary,
        ):
            weak_regions.append(
                internal_OcrRegion(
                    internal_raster_rectangle_page_box(source_raster, page_box, rectangle),
                    1.0,
                    ("adaptive-weak-region",),
                )
            )
    regions = internal_merge_ocr_regions(weak_regions)
    if not regions:
        return (), 0, rendered, ()
    if rendered is None:
        rendered = compose_page(
            capture.page,
            RenderOptions(include_text=ocr_pass.include_native_text),
            page_program=capture.program,
        )
    region_pass = replace(ocr_pass, scope=OcrPassScope.TILES, tiles=1)
    tasks: list[internal_OcrTask] = []
    raster_pixels = 0
    boxes: list[tuple[float, float, float, float]] = []
    for region in regions:
        raster = internal_rendered_page_raster(
            capture,
            ocr_pass.scale,
            crop=region.page_box,
            rendered=rendered,
            cache=True,
            max_pixels=ocr_pass.pixel_budget,
            include_native_text=ocr_pass.include_native_text,
        )
        if raster is None:
            continue
        boxes.append(region.page_box)
        raster_pixels += raster.width * raster.height
        tasks.extend(
            internal_tile_tasks(
                raster,
                region.page_box,
                region_pass,
                compact_image=compact_image,
            )
        )
    return tuple(tasks), raster_pixels, rendered, tuple(boxes)


STROKED_VECTOR_DECODE_MIN_OVERLAP = 0.55
STROKED_VECTOR_MULTI_EDIT_MIN_OVERLAP = 0.90
STROKED_VECTOR_MULTI_EDIT_MAX_CONFIDENCE = 85.0


def internal_stroked_vector_decoded_batch(
    observations: tuple[StrokedTextObservation, ...],
) -> ObservationBatch:
    boxes = tuple(observation.bbox for observation in observations)
    return ObservationBatch.from_columns(
        (observation.text for observation in observations),
        boxes,
        polygon=((box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]) for box in boxes),
        source=ObservationSource.STRUCTURE,
        confidence=(observation.confidence for observation in observations),
        sequence=(observation.first_drawing for observation in observations),
        rotation=(0 for _ in observations),
        font_size=(max(0.0, box[3] - box[1]) for box in boxes),
        line_break_before=(True for _ in observations),
    )


def internal_single_character_substitution(left: str, right: str) -> bool:
    return len(left) == len(right) and sum(a != b for a, b in zip(left, right, strict=True)) == 1


def internal_bounded_edit_distance(left: str, right: str, maximum: int) -> int:
    """Return a small Levenshtein distance, stopping once the bound is exceeded."""
    if abs(len(left) - len(right)) > maximum:
        return maximum + 1
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, 1):
        current = [row]
        for column, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        if min(current) > maximum:
            return maximum + 1
        previous = current
    return previous[-1]


def internal_stroked_vector_substitution(
    recognized: str,
    decoded: str,
    *,
    confidence: float,
    overlap: float,
) -> bool:
    if internal_single_character_substitution(recognized, decoded):
        return True
    left = recognized.casefold()
    right = decoded.casefold()
    return bool(
        confidence < STROKED_VECTOR_MULTI_EDIT_MAX_CONFIDENCE
        and overlap >= STROKED_VECTOR_MULTI_EDIT_MIN_OVERLAP
        and 2 <= len(left) <= 12
        and 2 <= len(right) <= 12
        and abs(len(left) - len(right)) <= 1
        and left[0] == right[0]
        and any(character.isalnum() for character in left)
        and any(character.isalnum() for character in right)
        and internal_bounded_edit_distance(left, right, 2) <= 2
    )


def internal_stroked_vector_symbol_seeds(
    capture: CapturedPage,
    symbols: ObservationBatch,
) -> tuple[StrokedTextSeed, ...]:
    """Join character boxes only when they exactly fill one known vector run."""
    if not len(symbols):
        return ()
    runs_by_sequence = {
        run.drawing_indexes[0]: run for run in internal_stroked_text_profile(capture).seed_runs
    }
    grouped: dict[
        int,
        list[tuple[float, str, float]],
    ] = defaultdict(list)
    for text, box, confidence, raw_sequence in zip(
        symbols.text,
        symbols.bbox,
        symbols.confidence,
        symbols.sequence,
        strict=True,
    ):
        character = text.strip()
        sequence = int(raw_sequence)
        if len(character) != 1 or sequence not in runs_by_sequence:
            continue
        grouped[sequence].append((float(box[0]), character, float(confidence)))

    seeds: list[StrokedTextSeed] = []
    for sequence, items in grouped.items():
        run = runs_by_sequence[sequence]
        if len(items) != run.glyph_count:
            continue
        ordered = sorted(items)
        seeds.append(
            StrokedTextSeed(
                text="".join(character for ignored_x, character, ignored_confidence in ordered),
                bbox=run.bbox,
                confidence=min(confidence for ignored_x, ignored_character, confidence in ordered),
                sequence=sequence,
            )
        )
    return tuple(seeds)


def internal_decode_stroked_vector_text(
    capture: CapturedPage,
    ocr: ObservationBatch,
    symbols: ObservationBatch | None = None,
) -> StrokedTextDecode:
    evidence = capture.evidence.stroked_vector_text
    if not evidence.trusted or not evidence.drawing_indexes or not len(ocr):
        return StrokedTextDecode()
    profile = internal_stroked_text_profile(capture)
    word_seeds = tuple(
        StrokedTextSeed(
            text=text,
            bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            confidence=float(confidence),
            sequence=int(sequence),
        )
        for text, box, confidence, sequence in zip(
            ocr.text,
            ocr.bbox,
            ocr.confidence,
            ocr.sequence,
            strict=True,
        )
    )
    symbol_seeds = internal_stroked_vector_symbol_seeds(
        capture,
        symbols if symbols is not None else ObservationBatch.empty(),
    )
    if not symbol_seeds:
        return decode_stroked_text_profile(profile, word_seeds)
    return decode_stroked_text_profile_with_supplemental_seeds(
        profile,
        word_seeds,
        symbol_seeds,
    )


def internal_packed_stroked_vector_decode_gate(
    decoded: StrokedTextDecode,
    cell_count: int,
) -> tuple[bool, dict[str, int | bool]]:
    """Require enough learned geometry before skipping the full-layer OCR fallback."""
    aligned_required = min(
        STROKED_VECTOR_PACK_MIN_ALIGNED_SEEDS,
        max(4, cell_count // 4),
    )
    learned_required = min(
        STROKED_VECTOR_PACK_MIN_LEARNED_SIGNATURES,
        max(8, cell_count // 3),
    )
    decoded_required = min(
        STROKED_VECTOR_PACK_MIN_DECODED_RUNS,
        max(8, cell_count // 3),
    )
    accepted = bool(
        decoded.aligned_seeds >= aligned_required
        and decoded.learned_signatures >= learned_required
        and len(decoded.observations) >= decoded_required
    )
    return accepted, {
        "accepted": accepted,
        "cells": cell_count,
        "aligned_seeds": decoded.aligned_seeds,
        "aligned_required": aligned_required,
        "learned_signatures": decoded.learned_signatures,
        "learned_required": learned_required,
        "decoded_runs": len(decoded.observations),
        "decoded_required": decoded_required,
    }


def internal_recover_stroked_vector_text(
    capture: CapturedPage,
    ocr: ObservationBatch,
) -> ObservationBatch:
    """Augment one OCR pass with text decoded from repeated vector glyphs."""
    evidence = capture.evidence.stroked_vector_text
    if not evidence.trusted or not evidence.drawing_indexes or not len(ocr):
        return ocr
    started = time.perf_counter()
    cache = capture.page.extraction_cache
    cached_decode = cache.pop("_stroked_vector_decode_preview", None)
    if (
        isinstance(cached_decode, tuple)
        and len(cached_decode) == 3
        and cached_decode[0] == id(ocr)
        and isinstance(cached_decode[1], StrokedTextDecode)
    ):
        decoded = cached_decode[1]
        prior_decode_seconds = float(cached_decode[2])
    else:
        decoded = internal_decode_stroked_vector_text(capture, ocr)
        prior_decode_seconds = 0.0
    ocr_index = SpatialIndex.from_boxes(ocr.bbox)
    replacements: set[int] = set()
    accepted: list[StrokedTextObservation] = []
    additions = 0
    corrections = 0
    for observation in decoded.observations:
        candidate_area = max(
            0.01,
            (observation.bbox[2] - observation.bbox[0])
            * (observation.bbox[3] - observation.bbox[1]),
        )
        overlaps: list[tuple[float, int]] = []
        for hit in ocr_index.intersecting_hits(observation.bbox):
            hit_area = max(0.01, (hit.bbox[2] - hit.bbox[0]) * (hit.bbox[3] - hit.bbox[1]))
            overlap = bbox_intersection_area(observation.bbox, hit.bbox) / min(
                candidate_area, hit_area
            )
            if overlap >= STROKED_VECTOR_DECODE_MIN_OVERLAP:
                overlaps.append((overlap, int(hit.item)))
        if not overlaps:
            accepted.append(observation)
            additions += 1
            continue
        best_overlap, best_index = max(overlaps)
        recognized_text = ocr.text[best_index].strip()
        if recognized_text == observation.text:
            continue
        if best_index not in replacements and internal_stroked_vector_substitution(
            recognized_text,
            observation.text,
            confidence=float(ocr.confidence[best_index]),
            overlap=best_overlap,
        ):
            replacements.add(best_index)
            accepted.append(observation)
            corrections += 1

    cache["stroked_vector_decode"] = {
        "seconds": prior_decode_seconds + time.perf_counter() - started,
        "eligible_seeds": decoded.eligible_seeds,
        "aligned_seeds": decoded.aligned_seeds,
        "accepted_seeds": decoded.accepted_seeds,
        "initial_signatures": decoded.initial_signatures,
        "learned_signatures": decoded.learned_signatures,
        "approximate_signatures": decoded.approximate_signatures,
        "candidate_runs": decoded.candidate_runs,
        "decoded_candidate_runs": decoded.decoded_candidate_runs,
        "candidate_run_coverage": decoded.candidate_run_coverage,
        "candidate_glyph_coverage": decoded.candidate_glyph_coverage,
        "decoded_runs": len(decoded.observations),
        "additions": additions,
        "corrections": corrections,
    }
    cache["_stroked_vector_alphabet"] = decoded.alphabet
    if not accepted:
        return ocr
    retained = ocr.take(tuple(index for index in range(len(ocr)) if index not in replacements))
    return ObservationBatch.concatenate(
        retained,
        internal_stroked_vector_decoded_batch(tuple(accepted)),
    )


def recognize_page(
    capture: CapturedPage,
    plan: WorkPlan,
    context: TaskScope,
) -> ObservationBatch:
    if not plan.ocr_passes:
        return ObservationBatch.empty()
    with context.reserve_raster(MAX_OCR_RASTER_BYTES):
        context.raise_if_cancelled()
        observations = internal_recognize_page_with_reserved_raster(capture, plan, context)
    return internal_recover_stroked_vector_text(capture, observations)


def internal_recognize_page_with_reserved_raster(
    capture: CapturedPage,
    plan: WorkPlan,
    context: TaskScope,
) -> ObservationBatch:
    page = capture.page
    page_box = (0.0, 0.0, float(page.width), float(page.height))
    compact_image: bool | str = True
    if capture.evidence.full_page_image:
        image_filters = preflight_page(page, capture).features.image_filters
        if any("JPX" in str(filter_name).upper() for filter_name in image_filters):
            compact_image = "grayscale"
    dominant_regions: dict[int, internal_RasterRegion | None] = {}
    rendered_rasters: dict[tuple[float, int, bool], internal_Raster | None] = {}
    rendered_page: Any | None = None
    candidate_regions: tuple[internal_OcrRegion, ...] | None = None
    candidates: list[tuple[str, internal_Candidate]] = []
    pass_diagnostics: list[dict[str, object]] = []
    selected_name = ""
    selected: internal_Candidate | None = None
    selected_tasks: tuple[internal_OcrTask, ...] = ()
    previous_region_additions = 0
    seeded_region_selected = False
    adaptive_rescue_used = False

    def recognize_tasks(tasks: tuple[internal_OcrTask, ...]) -> tuple[internal_Candidate, ...]:
        groups = internal_ocr_task_groups(tasks)
        results = context.map_ordered(internal_recognize_group, groups, stage=WorkStage.OCR)
        return tuple(candidate for group in results for candidate in group)

    if plan.verify_hidden_text:
        context.raise_if_cancelled()
        started = time.perf_counter()
        verification_pass = OcrPass(
            "hidden-text-verification",
            OcrPassScope.PAGE,
            1.0,
            (PSM_SPARSE_TEXT,),
            minimum_confidence=HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE,
            pixel_budget=HIDDEN_TEXT_VERIFY_PIXELS,
            recognize_words=True,
            region_first=False,
        )
        verification_region = internal_dominant_image_region(
            capture,
            max_pixels=HIDDEN_TEXT_VERIFY_PIXELS,
        )
        verification_tasks = (
            internal_tile_tasks(
                verification_region.raster,
                verification_region.page_box,
                verification_pass,
                compact_image=compact_image,
            )
            if verification_region is not None
            else ()
        )
        verification_candidates = recognize_tasks(verification_tasks)
        verification_candidate = internal_merge_candidate_batches(verification_candidates)
        verification = internal_hidden_text_verification(
            capture.observations,
            verification_candidate.observations,
        )
        raster_pixels = (
            verification_region.raster.width * verification_region.raster.height
            if verification_region is not None
            else 0
        )
        verification_record: dict[str, object] = {
            "name": verification_pass.name,
            "scope": verification_pass.scope.value,
            "scale": verification_pass.scale,
            "modes": verification_pass.modes,
            "recognize_words": verification_pass.recognize_words,
            "character_confidence_threshold": None,
            "task_count": len(verification_tasks),
            "raster_pixels": raster_pixels,
            "region_stage": "dominant-image-preview",
            "region_boxes": (
                (verification_region.page_box,) if verification_region is not None else ()
            ),
            "full_page_fallback": False,
            "elapsed_seconds": time.perf_counter() - started,
            "render_timings": capture.page.extraction_cache.get("ocr_render_timings", {}),
            "recognition_seconds": sum(
                candidate.recognition_seconds for candidate in verification_candidates
            ),
            "setup_seconds": sum(candidate.setup_seconds for candidate in verification_candidates),
            "api_seconds": sum(candidate.api_seconds for candidate in verification_candidates),
            "iterator_seconds": sum(
                candidate.iterator_seconds for candidate in verification_candidates
            ),
            "cleanup_seconds": sum(
                candidate.cleanup_seconds for candidate in verification_candidates
            ),
            "candidate_seconds": sum(
                candidate.candidate_seconds for candidate in verification_candidates
            ),
            "recognition_statuses": tuple(
                candidate.recognition_status for candidate in verification_candidates
            ),
            "accepted_additions": 0,
            "adaptive_retry_scale": None,
            "adaptive_preflight": None,
            "adaptive_rescue_decision": None,
            "adaptive_rescue": None,
            "pixel_budget": verification_pass.pixel_budget,
            "rectangles": tuple(task.rectangle for task in verification_tasks),
            "selected": verification.accepted,
            **verification_candidate.metrics.as_record(),
            **verification.as_record(),
        }
        pass_diagnostics.append(verification_record)
        capture.page.extraction_cache["hidden_text_verification"] = {
            "raster_pixels": raster_pixels,
            **verification.as_record(),
        }
        if verification.accepted:
            capture.page.extraction_cache["ocr_pass_diagnostics"] = tuple(pass_diagnostics)
            return internal_promoted_hidden_observations(capture)

    for ocr_pass in plan.ocr_passes:
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.run_if_characters_below is not None
            and internal_primary_text_is_sufficient(selected)
        ):
            continue
        if (
            selected is not None
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= ocr_pass.run_if_characters_below
        ):
            continue
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.IMAGE_REGIONS
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= 28
            and selected.metrics.mean_confidence >= 97.0
        ):
            continue
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.IMAGE_REGIONS
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= 1500
            and selected.metrics.mean_confidence >= 98.0
        ):
            continue
        if (
            ocr_pass.run_if_additions_below is not None
            and previous_region_additions >= ocr_pass.run_if_additions_below
        ):
            continue
        if (
            ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.run_if_additions_below is not None
            and previous_region_additions == 0
            and selected is None
            and capture.evidence.visible_native_characters >= 3_000
        ):
            continue
        if (
            ocr_pass.scope is OcrPassScope.WEAK_REGIONS
            and ocr_pass.run_if_additions_below is not None
            and previous_region_additions == 0
            and selected is not None
            and selected.metrics.characters >= 32
            and selected.metrics.mean_confidence >= 90.0
        ):
            continue
        if (
            ocr_pass.scope is OcrPassScope.PAGE
            and seeded_region_selected
            and ocr_pass.run_if_additions_below is not None
        ):
            selected = None
            selected_name = ""
            selected_tasks = ()
            seeded_region_selected = False
        context.raise_if_cancelled()
        started = time.perf_counter()
        adaptive_preflight: dict[str, object] | None = None
        vector_preview = bool(
            capture.evidence.image_count == 0
            and capture.evidence.vector_complexity >= 100_000
            and capture.evidence.text_coverage < 0.05
        )
        if (
            ocr_pass.adaptive_scale
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.pixel_budget == PRIMARY_OCR_PIXELS
            and (capture.evidence.full_page_image or vector_preview)
        ):
            preview_raster: internal_Raster | None = None
            if capture.evidence.full_page_image:
                if OCR_PREFLIGHT_PIXELS not in dominant_regions:
                    dominant_regions[OCR_PREFLIGHT_PIXELS] = internal_dominant_image_region(
                        capture,
                        max_pixels=OCR_PREFLIGHT_PIXELS,
                    )
                preview_region = dominant_regions[OCR_PREFLIGHT_PIXELS]
                preview_raster = preview_region.raster if preview_region is not None else None
            else:
                if rendered_page is None:
                    rendered_page = compose_page(
                        capture.page,
                        RenderOptions(include_text=ocr_pass.include_native_text),
                        page_program=capture.program,
                    )
                preview_raster = internal_rendered_page_raster(
                    capture,
                    ocr_pass.scale,
                    rendered=rendered_page,
                    cache=True,
                    max_pixels=OCR_PREFLIGHT_PIXELS,
                    include_native_text=ocr_pass.include_native_text,
                )
            if preview_raster is not None:
                preview_height = internal_estimated_text_height(preview_raster)
                projected_height = preview_height * math.sqrt(
                    ocr_pass.pixel_budget / max(1, preview_raster.width * preview_raster.height)
                )
                projected_limit = 22.0 if vector_preview else 20.0
                if 12.0 <= projected_height < projected_limit:
                    original_scale = ocr_pass.scale
                    ocr_pass = replace(
                        ocr_pass,
                        scale=min(
                            8.0,
                            max(
                                original_scale + 0.5,
                                original_scale * 32.0 / projected_height,
                            ),
                        ),
                        pixel_budget=MAX_OCR_PIXELS,
                    )
                    adaptive_preflight = {
                        "preview_pixels": preview_raster.width * preview_raster.height,
                        "preview_text_height": preview_height,
                        "projected_primary_text_height": projected_height,
                        "selected_scale": ocr_pass.scale,
                        "source": "vector-render" if vector_preview else "dominant-image",
                    }
        tasks: tuple[internal_OcrTask, ...]
        packed_stroked: internal_PackedStrokedTextRaster | None = None
        raster_pixels = 0
        skipped_raster_pixels = 0
        image_text_preflight: tuple[dict[str, object], ...] = ()
        skipped_region_boxes: tuple[tuple[float, float, float, float], ...] = ()
        region_stage = "page"
        region_boxes: tuple[tuple[float, float, float, float], ...] = ()
        if (
            ocr_pass.region_first
            and ocr_pass.scope in {OcrPassScope.PAGE, OcrPassScope.WEAK_REGIONS}
            and (
                ocr_pass.scope is not OcrPassScope.WEAK_REGIONS
                or selected is not None
                or ocr_pass.seed_with_native
            )
        ):
            if candidate_regions is None:
                candidate_regions = internal_candidate_ocr_regions(capture)
            distributed_outline_text = bool(
                ocr_pass.scope is OcrPassScope.PAGE
                and internal_has_distributed_outline_text(capture)
            )
            region_batch = (
                (
                    internal_OcrRegion(
                        page_box,
                        float("inf"),
                        ("distributed-outline-text",),
                    ),
                )
                if distributed_outline_text
                else internal_ocr_region_batch(
                    candidate_regions,
                    ocr_pass,
                    expanded=False,
                    page_area=max(1.0, float(page.width) * float(page.height)),
                )
            )
            tasks, raster_pixels, rendered_page, region_boxes = internal_candidate_region_tasks(
                capture,
                region_batch,
                ocr_pass,
                rendered=rendered_page,
                compact_image=compact_image,
            )
            region_stage = (
                "distributed-outline-page" if distributed_outline_text else "initial-regions"
            )
            if len(region_batch) == 1 and "page-fallback" in region_batch[0].reasons:
                region_stage = "page"
        elif ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            if selected is None and not ocr_pass.seed_with_native:
                continue
            if selected is not None and selected_tasks:
                tasks, raster_pixels, rendered_page, region_boxes = (
                    internal_high_resolution_weak_region_tasks(
                        capture,
                        selected_tasks,
                        ocr_pass,
                        selected.observations,
                        rendered=rendered_page,
                        compact_image=compact_image,
                    )
                )
                region_stage = "weak-region-crops"
            else:
                if ocr_pass.pixel_budget not in dominant_regions:
                    dominant_regions[ocr_pass.pixel_budget] = internal_dominant_image_region(
                        capture,
                        max_pixels=ocr_pass.pixel_budget,
                    )
                direct_region = dominant_regions[ocr_pass.pixel_budget]
                raster = direct_region.raster if direct_region is not None else None
                raster_page_box = direct_region.page_box if direct_region is not None else page_box
                if raster is None:
                    raster_key = (
                        ocr_pass.scale,
                        ocr_pass.pixel_budget,
                        ocr_pass.include_native_text,
                    )
                    if raster_key not in rendered_rasters:
                        rendered_rasters[raster_key] = internal_rendered_page_raster(
                            capture,
                            ocr_pass.scale,
                            max_pixels=ocr_pass.pixel_budget,
                            include_native_text=ocr_pass.include_native_text,
                        )
                    raster = rendered_rasters[raster_key]
                    raster_page_box = page_box
                tasks = (
                    internal_weak_region_tasks(
                        raster,
                        raster_page_box,
                        ocr_pass,
                        selected.observations if selected is not None else capture.observations,
                        compact_image=compact_image,
                    )
                    if raster is not None
                    else ()
                )
                raster_pixels = (
                    sum(task.rectangle[2] * task.rectangle[3] for task in tasks)
                    if raster is not None
                    else 0
                )
        elif ocr_pass.scope is OcrPassScope.STROKED_VECTOR_TEXT:
            packed_stroked = internal_stroked_vector_text_raster(
                capture,
                ocr_pass.scale,
                max_pixels=ocr_pass.pixel_budget,
            )
            if packed_stroked is not None:
                region_stage = "packed-stroked-vector-text"
                region_boxes = (
                    (capture.evidence.stroked_vector_text.bbox,)
                    if capture.evidence.stroked_vector_text.bbox is not None
                    else ()
                )
                tasks = internal_tile_tasks(
                    packed_stroked.raster,
                    packed_stroked.packed_box,
                    replace(ocr_pass, recognize_words=True, collect_symbols=True),
                    compact_image=compact_image,
                )
                raster_pixels = packed_stroked.raster.width * packed_stroked.raster.height
            else:
                fallback_region = internal_full_stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    max_pixels=ocr_pass.pixel_budget,
                )
                region_stage = "stroked-vector-text-fallback"
                region_boxes = (fallback_region.page_box,) if fallback_region is not None else ()
                tasks = (
                    internal_tile_tasks(
                        fallback_region.raster,
                        fallback_region.page_box,
                        ocr_pass,
                        compact_image=compact_image,
                    )
                    if fallback_region is not None
                    else ()
                )
                raster_pixels = (
                    fallback_region.raster.width * fallback_region.raster.height
                    if fallback_region is not None
                    else 0
                )
                capture.page.extraction_cache["stroked_vector_packed"] = {
                    "accepted": False,
                    "cells": 0,
                    "raster_pixels": 0,
                    "unmapped_observations": 0,
                    "fallback_used": bool(tasks),
                }
        elif ocr_pass.scope is OcrPassScope.IMAGE_REGIONS:
            regions = internal_page_image_regions(
                capture,
                minimum_area_ratio=0.02,
                max_pixels=ocr_pass.pixel_budget,
                maximum_axis_deviation=OCR_IMAGE_REGIONS_MAX_AXIS_DEVIATION,
            )
            if regions:
                region_signals = tuple(
                    (region, internal_raster_text_signal(region.raster.image)) for region in regions
                )
                image_text_preflight = tuple(
                    {
                        "page_box": region.page_box,
                        "raster_pixels": region.raster.width * region.raster.height,
                        **signal.as_record(),
                    }
                    for region, signal in region_signals
                )
                eligible_regions = tuple(
                    region for region, signal in region_signals if signal.likely_text
                )
                skipped_regions = tuple(
                    region for region, signal in region_signals if not signal.likely_text
                )
                skipped_raster_pixels = sum(
                    region.raster.width * region.raster.height for region in skipped_regions
                )
                skipped_region_boxes = tuple(region.page_box for region in skipped_regions)
                region_boxes = tuple(region.page_box for region in eligible_regions)
                region_stage = "direct-image-regions"
                tasks = tuple(
                    task
                    for region in eligible_regions
                    for task in internal_tile_tasks(
                        region.raster,
                        region.page_box,
                        ocr_pass,
                        compact_image=compact_image,
                    )
                )
                raster_pixels = sum(
                    region.raster.width * region.raster.height for region in eligible_regions
                )
            else:
                fallback_scale = max(2.0, ocr_pass.scale)
                image_crop = internal_safe_image_crop(capture)
                raster = internal_rendered_page_raster(
                    capture,
                    fallback_scale,
                    crop=image_crop,
                    max_pixels=ocr_pass.pixel_budget,
                    include_native_text=ocr_pass.include_native_text,
                )
                raster_page_box = image_crop or page_box
                tasks = (
                    internal_tile_tasks(
                        raster,
                        raster_page_box,
                        ocr_pass,
                        compact_image=compact_image,
                    )
                    if raster is not None
                    else ()
                )
                raster_pixels = raster.width * raster.height if raster is not None else 0
        else:
            if ocr_pass.pixel_budget not in dominant_regions:
                dominant_regions[ocr_pass.pixel_budget] = internal_dominant_image_region(
                    capture,
                    max_pixels=ocr_pass.pixel_budget,
                )
            direct_region = dominant_regions[ocr_pass.pixel_budget]
            raster = direct_region.raster if direct_region is not None else None
            raster_page_box = direct_region.page_box if direct_region is not None else page_box
            if raster is None:
                raster_key = (
                    ocr_pass.scale,
                    ocr_pass.pixel_budget,
                    ocr_pass.include_native_text,
                )
                if raster_key not in rendered_rasters:
                    rendered_rasters[raster_key] = internal_rendered_page_raster(
                        capture,
                        ocr_pass.scale,
                        max_pixels=ocr_pass.pixel_budget,
                        include_native_text=ocr_pass.include_native_text,
                    )
                raster = rendered_rasters[raster_key]
                raster_page_box = page_box
            task_raster = (
                internal_adaptive_ocr_raster(raster)
                if raster is not None and ocr_pass.name == "adaptive-page"
                else raster
            )
            tasks = (
                internal_tile_tasks(
                    task_raster,
                    raster_page_box,
                    ocr_pass,
                    compact_image=compact_image,
                )
                if task_raster is not None
                else ()
            )
            raster_pixels = raster.width * raster.height if raster is not None else 0
        if not tasks:
            if not image_text_preflight:
                continue
            region_stage = "image-text-preflight"

        candidate_source_tasks = tasks
        task_candidates = recognize_tasks(tasks)
        if packed_stroked is not None:
            remapped_with_counts = tuple(
                internal_remap_stroked_vector_candidate(candidate, packed_stroked)
                for candidate in task_candidates
            )
            task_candidates = tuple(item[0] for item in remapped_with_counts)
            unmapped_observations = sum(item[1] for item in remapped_with_counts)
            packed_candidate = internal_merge_candidate_batches(task_candidates)
            decode_started = time.perf_counter()
            packed_decode = internal_decode_stroked_vector_text(
                capture,
                packed_candidate.observations,
                packed_candidate.symbols,
            )
            decode_seconds = time.perf_counter() - decode_started
            packed_accepted, packed_gate = internal_packed_stroked_vector_decode_gate(
                packed_decode,
                len(packed_stroked.cells),
            )
            packed_pixels = raster_pixels
            fallback_used = False
            if packed_accepted:
                capture.page.extraction_cache["_stroked_vector_decode_preview"] = (
                    id(packed_candidate.observations),
                    packed_decode,
                    decode_seconds,
                )
            else:
                fallback_region = internal_full_stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    max_pixels=ocr_pass.pixel_budget,
                )
                fallback_tasks = (
                    internal_tile_tasks(
                        fallback_region.raster,
                        fallback_region.page_box,
                        replace(ocr_pass, recognize_words=False),
                        compact_image=compact_image,
                    )
                    if fallback_region is not None
                    else ()
                )
                if fallback_tasks:
                    fallback_used = True
                    fallback_candidates = recognize_tasks(fallback_tasks)
                    task_candidates = (*task_candidates, *fallback_candidates)
                    candidate_source_tasks = (*candidate_source_tasks, *fallback_tasks)
                    tasks = (*tasks, *fallback_tasks)
                    packed_candidate = internal_merge_candidate_batches(fallback_candidates)
                    raster_pixels += (
                        fallback_region.raster.width * fallback_region.raster.height
                        if fallback_region is not None
                        else 0
                    )
                    region_stage = "stroked-vector-text-fallback"
                    region_boxes = (
                        (fallback_region.page_box,) if fallback_region is not None else region_boxes
                    )
            capture.page.extraction_cache["stroked_vector_packed"] = {
                **packed_gate,
                "raster_pixels": packed_pixels,
                "unmapped_observations": unmapped_observations,
                "symbol_observations": len(packed_candidate.symbols),
                "fallback_used": fallback_used,
            }
            candidate = packed_candidate
        else:
            candidate = internal_merge_candidate_batches(task_candidates)
        adaptive_retry_scale: float | None = None
        adaptive_rescue: dict[str, object] | None = None
        adaptive_rescue_decision: dict[str, object] | None = None
        median_height = candidate.metrics.median_text_height
        rescue_eligible = bool(
            ocr_pass.adaptive_scale
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.pixel_budget < MAX_OCR_PIXELS
            and not adaptive_rescue_used
            and candidate.metrics.characters >= ocr_pass.minimum_characters_for_rescue
            and (candidate.metrics.characters < 32 or 0.0 < median_height < 24.0)
        )
        run_rescue = False
        if rescue_eligible:
            adaptive_rescue_used = True
            run_rescue, adaptive_rescue_decision = internal_adaptive_rescue_decision(
                candidate,
                candidate_source_tasks,
                ocr_pass,
            )
        if run_rescue:
            factor = 1.5 if median_height <= 0.0 else min(2.5, max(1.25, 32.0 / median_height))
            adaptive_retry_scale = min(8.0, max(ocr_pass.scale + 0.5, ocr_pass.scale * factor))
            retry_pass = replace(
                ocr_pass,
                name="adaptive-rescue",
                scale=adaptive_retry_scale,
                pixel_budget=MAX_OCR_PIXELS,
                region_first=False,
            )
            retry_scope = (
                "page"
                if candidate.metrics.characters < 32 or median_height < 18.0
                else "weak-regions"
            )
            retry_boxes: tuple[tuple[float, float, float, float], ...] = ()
            if retry_scope == "page":
                retry_raster = internal_rendered_page_raster(
                    capture,
                    adaptive_retry_scale,
                    max_pixels=MAX_OCR_PIXELS,
                    include_native_text=ocr_pass.include_native_text,
                )
                retry_tasks = (
                    internal_tile_tasks(
                        retry_raster,
                        page_box,
                        retry_pass,
                        compact_image=compact_image,
                    )
                    if retry_raster is not None
                    else ()
                )
                rescue_pixels = (
                    retry_raster.width * retry_raster.height if retry_raster is not None else 0
                )
            else:
                retry_pass = replace(
                    retry_pass,
                    scope=OcrPassScope.WEAK_REGIONS,
                    tiles=max(6, retry_pass.tiles),
                    region_columns=max(3, retry_pass.region_columns),
                    max_regions=max(8, retry_pass.max_regions),
                )
                retry_tasks, rescue_pixels, rendered_page, retry_boxes = (
                    internal_high_resolution_weak_region_tasks(
                        capture,
                        tasks,
                        retry_pass,
                        candidate.observations,
                        rendered=rendered_page,
                        compact_image=compact_image,
                    )
                )
            if retry_tasks:
                candidate_source_tasks = (*candidate_source_tasks, *retry_tasks)
                retry_candidates = recognize_tasks(retry_tasks)
                retry_candidate = internal_merge_candidate_batches(retry_candidates)
                augmented_candidate, rescue_additions = internal_augment_candidate(
                    candidate,
                    retry_candidate,
                    minimum_confidence=ocr_pass.minimum_confidence,
                )
                if retry_candidate.metrics.utility > augmented_candidate.metrics.utility * 1.05:
                    candidate = retry_candidate
                elif augmented_candidate.metrics.utility > candidate.metrics.utility:
                    candidate = augmented_candidate
                task_candidates = (*task_candidates, *retry_candidates)
                raster_pixels += rescue_pixels
                adaptive_rescue = {
                    "scope": retry_scope,
                    "scale": adaptive_retry_scale,
                    "raster_pixels": rescue_pixels,
                    "task_count": len(retry_tasks),
                    "accepted_additions": rescue_additions,
                    "region_boxes": retry_boxes,
                }
        additions = 0
        if ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            used_native_seed = selected is None
            if selected is not None:
                candidate, additions = internal_augment_candidate(
                    selected,
                    candidate,
                    minimum_confidence=ocr_pass.minimum_confidence,
                )
            else:
                additions = len(candidate.observations)
        candidates.append((ocr_pass.name, candidate))
        elapsed = time.perf_counter() - started
        pass_diagnostics.append(
            {
                "name": ocr_pass.name,
                "scope": ocr_pass.scope.value,
                "scale": ocr_pass.scale,
                "modes": ocr_pass.modes,
                "recognize_words": any(task.recognize_words for task in tasks),
                "character_confidence_threshold": ocr_pass.character_confidence_threshold,
                "task_count": len(tasks),
                "raster_pixels": raster_pixels,
                "skipped_raster_pixels": skipped_raster_pixels,
                "image_text_preflight": image_text_preflight,
                "region_stage": region_stage,
                "region_boxes": region_boxes,
                "skipped_region_boxes": skipped_region_boxes,
                "full_page_fallback": (
                    region_stage == "page" and ocr_pass.scope is OcrPassScope.PAGE
                ),
                "elapsed_seconds": elapsed,
                "render_timings": capture.page.extraction_cache.get("ocr_render_timings", {}),
                "recognition_seconds": sum(
                    task_candidate.recognition_seconds for task_candidate in task_candidates
                ),
                "setup_seconds": sum(
                    task_candidate.setup_seconds for task_candidate in task_candidates
                ),
                "api_seconds": sum(
                    task_candidate.api_seconds for task_candidate in task_candidates
                ),
                "iterator_seconds": sum(
                    task_candidate.iterator_seconds for task_candidate in task_candidates
                ),
                "cleanup_seconds": sum(
                    task_candidate.cleanup_seconds for task_candidate in task_candidates
                ),
                "candidate_seconds": sum(
                    task_candidate.candidate_seconds for task_candidate in task_candidates
                ),
                "recognition_statuses": tuple(
                    task_candidate.recognition_status for task_candidate in task_candidates
                ),
                "accepted_additions": additions,
                "adaptive_retry_scale": adaptive_retry_scale,
                "adaptive_preflight": adaptive_preflight,
                "adaptive_rescue_decision": adaptive_rescue_decision,
                "adaptive_rescue": adaptive_rescue,
                "pixel_budget": ocr_pass.pixel_budget,
                "rectangles": tuple(task.rectangle for task in tasks),
                "selected": False,
                **candidate.metrics.as_record(),
            }
        )
        if not tasks:
            continue
        if ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            previous_region_additions = additions
            if additions:
                selected_name = ocr_pass.name
                selected = candidate
                selected_tasks = (*selected_tasks, *candidate_source_tasks)
                seeded_region_selected = used_native_seed and ocr_pass.seed_with_native
            continue
        if selected is None or candidate.metrics.utility > (
            selected.metrics.utility * ocr_pass.minimum_utility_gain
        ):
            selected_name = ocr_pass.name
            selected = candidate
            selected_tasks = candidate_source_tasks

    if selected is None:
        capture.page.extraction_cache["ocr_pass_diagnostics"] = tuple(pass_diagnostics)
        internal_record_candidates(capture, tuple(candidates), selected_name)
        return ObservationBatch.empty()
    for diagnostic in pass_diagnostics:
        diagnostic["selected"] = diagnostic["name"] == selected_name
    capture.page.extraction_cache["ocr_pass_diagnostics"] = tuple(pass_diagnostics)
    internal_record_candidates(capture, tuple(candidates), selected_name)
    return selected.observations


# ===== tables =====


MAX_GRID_INTERSECTIONS = 4_000_000
AXIS_TOLERANCE = 1.5
COLUMN_TOLERANCE = 8.0
TABLE_REGION_GAP = 12.0
TABLE_MERGE_GAP = 18.0
internal_CHART_NUMERIC_TOKEN = re.compile(r"^[+-]?(?:\d[\d,./%\-]*|\d[\d,./%\-]*\s+\d+)$")


def internal_chart_cell_texts(text: str) -> tuple[str, ...]:
    """Split dense OCR axis/value lines while keeping prose intact."""
    tokens = tuple(part for part in text.split() if part)
    numeric_count = sum(bool(internal_CHART_NUMERIC_TOKEN.fullmatch(part)) for part in tokens)
    if len(tokens) >= 4 and numeric_count >= 3:
        return tokens
    return (text,)


def extract_chart_table(capture: CapturedPage, observations: ObservationBatch) -> Table | None:
    """Represent OCR text recovered from vector artwork as one chart region.

    Vector charts frequently paint labels and values without table ruling.  The
    normal table detector correctly ignores them, but downstream parsers then
    lose the association between the recovered labels and values.  A compact
    synthetic row gives consumers a structured region while leaving ordinary
    pages untouched.
    """
    if (capture.evidence.uncovered_vector_area or 0.0) < 20_000.0:
        return None
    ocr_indexes = numpy.flatnonzero(observations.source == int(ObservationSource.OCR))
    if len(ocr_indexes) < 3:
        return None

    cells: list[TableCell] = []
    boxes: list[tuple[float, float, float, float]] = []
    seen: set[str] = set()
    column = 0
    for index in sorted(ocr_indexes, key=lambda item: observations.bbox[item, 0]):
        text = observations.text[int(index)].strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        box = cast(
            tuple[float, float, float, float],
            tuple(float(value) for value in observations.bbox[int(index)]),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        parts = internal_chart_cell_texts(text)
        if len(parts) == 1:
            boxes.append(box)
            cells.append(TableCell(row=0, column=column, text=text, bbox=box))
            column += 1
            continue
        width = (box[2] - box[0]) / len(parts)
        for offset, part in enumerate(parts):
            part_box = (box[0] + width * offset, box[1], box[0] + width * (offset + 1), box[3])
            boxes.append(part_box)
            cells.append(TableCell(row=0, column=column, text=part, bbox=part_box))
            column += 1
    if len(cells) < 3 or not boxes:
        return None
    row_tolerance = max(6.0, float(capture.page.height) * 0.008)
    row_groups: list[tuple[float, list[TableCell]]] = []
    for cell in sorted(
        cells,
        key=lambda item: (-(item.bbox or (0, 0, 0, 0))[1], item.column),
    ):
        cell_box = cell.bbox or (0.0, 0.0, 0.0, 0.0)
        center_y = (cell_box[1] + cell_box[3]) / 2
        group = next(
            (
                candidate
                for candidate in row_groups
                if abs(candidate[0] - center_y) <= row_tolerance
            ),
            None,
        )
        if group is None:
            row_groups.append((center_y, [cell]))
        else:
            group[1].append(cell)
    rows = tuple(
        tuple(
            sorted(
                (replace(cell, row=row_index) for cell in group),
                key=lambda item: item.column,
            )
        )
        for row_index, (internal_center_y, group) in enumerate(row_groups)
    )
    return Table(
        order=-1,
        rows=rows,
        bbox=(
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ),
        confidence=0.35,
        metadata={"source": "chart-ocr", "synthetic": True},
    )


class internal_DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def internal_cluster_positions(values: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    if values.size == 0:
        return numpy.empty(0, dtype=numpy.float32)
    ordered = numpy.sort(values.astype(numpy.float32, copy=False))
    breaks = numpy.empty(len(ordered), dtype=numpy.bool_)
    breaks[0] = True
    breaks[1:] = numpy.diff(ordered) > AXIS_TOLERANCE
    groups = numpy.cumsum(breaks) - 1
    counts = numpy.bincount(groups)
    sums = numpy.bincount(groups, weights=ordered)
    clustered = (sums / counts).astype(numpy.float32)
    if len(clustered) < 2:
        return clustered
    keep = numpy.empty(len(clustered), dtype=numpy.bool_)
    keep[0] = True
    keep[1:] = numpy.diff(clustered) >= 2.0
    return clustered[keep]


def internal_axis_segments(
    capture: CapturedPage,
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]:
    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    lines = capture.grid_lines
    if not lines:
        empty = numpy.empty((0, 3), dtype=numpy.float32)
        return empty, empty

    # Materialize the four coordinates once, then classify and normalize all
    # segments with array operations.  The old loop repeatedly performed the
    # same Python branching and tuple construction for every grid line.
    coordinates = numpy.fromiter(
        (value for line in lines for value in (line.x0, line.y0, line.x1, line.y1)),
        dtype=numpy.float64,
        count=len(lines) * 4,
    ).reshape((-1, 4))
    x0, y0, x1, y1 = coordinates.T
    horizontal_mask = (numpy.abs(y1 - y0) <= AXIS_TOLERANCE) & (
        numpy.abs(x1 - x0) >= page_width * 0.02
    )
    vertical_mask = (
        ~horizontal_mask
        & (numpy.abs(x1 - x0) <= AXIS_TOLERANCE)
        & (numpy.abs(y1 - y0) >= page_height * 0.015)
    )
    horizontal = numpy.column_stack(
        (numpy.minimum(x0, x1), numpy.maximum(x0, x1), (y0 + y1) * 0.5)
    )[horizontal_mask].astype(numpy.float32, copy=False)
    vertical = numpy.column_stack(((x0 + x1) * 0.5, numpy.minimum(y0, y1), numpy.maximum(y0, y1)))[
        vertical_mask
    ].astype(numpy.float32, copy=False)
    return horizontal.reshape((-1, 3)), vertical.reshape((-1, 3))


def internal_merge_collinear_segments(
    segments: numpy.ndarray[Any, Any],
    *,
    coordinate: int,
    start: int,
    end: int,
) -> numpy.ndarray[Any, Any]:
    if len(segments) < 2:
        return segments
    order = numpy.lexsort((segments[:, start], segments[:, coordinate]))
    sorted_segs = segments[order]
    diff_coord = (
        numpy.abs(sorted_segs[1:, coordinate] - sorted_segs[:-1, coordinate]) <= AXIS_TOLERANCE
    )
    overlap = sorted_segs[1:, start] <= (sorted_segs[:-1, end] + AXIS_TOLERANCE * 2.0)
    can_merge = diff_coord & overlap
    if not numpy.any(can_merge):
        return sorted_segs.astype(numpy.float32, copy=False)

    merged: list[list[float]] = []
    for values in sorted_segs:
        current = [float(value) for value in values]
        if merged:
            previous = merged[-1]
            if (
                abs(current[coordinate] - previous[coordinate]) <= AXIS_TOLERANCE
                and current[start] <= previous[end] + AXIS_TOLERANCE * 2.0
            ):
                previous[start] = min(previous[start], current[start])
                previous[end] = max(previous[end], current[end])
                previous[coordinate] = (previous[coordinate] + current[coordinate]) * 0.5
                continue
        merged.append(current)
    return numpy.asarray(merged, dtype=numpy.float32).reshape((-1, 3))


def internal_grid_components(
    horizontal: numpy.ndarray[Any, Any],
    vertical: numpy.ndarray[Any, Any],
) -> tuple[tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]], ...]:
    if len(horizontal) < 2 or len(vertical) < 2:
        return ()
    v_boxes = numpy.column_stack(
        (
            vertical[:, 0] - AXIS_TOLERANCE,
            vertical[:, 1] - AXIS_TOLERANCE,
            vertical[:, 0] + AXIS_TOLERANCE,
            vertical[:, 2] + AXIS_TOLERANCE,
        )
    )
    vertical_index = SpatialIndex((i, v_boxes[i]) for i in range(len(vertical)))
    pairs: list[tuple[int, int]] = []
    for h_index, segment in enumerate(horizontal):
        h_box = (
            float(segment[0]) - AXIS_TOLERANCE,
            float(segment[2]) - AXIS_TOLERANCE,
            float(segment[1]) + AXIS_TOLERANCE,
            float(segment[2]) + AXIS_TOLERANCE,
        )
        for v_index in vertical_index.intersecting(h_box):
            pairs.append((h_index, int(v_index)))
    if not pairs:
        return ()
    disjoint = internal_DisjointSet(len(horizontal) + len(vertical))
    for h_index, v_index in pairs:
        disjoint.union(h_index, len(horizontal) + v_index)
    grouped_h: dict[int, list[int]] = defaultdict(list)
    grouped_v: dict[int, list[int]] = defaultdict(list)
    for index in sorted({h_index for h_index, internal_v_index in pairs}):
        grouped_h[disjoint.find(index)].append(index)
    for index in sorted({v_index for internal_h_index, v_index in pairs}):
        grouped_v[disjoint.find(len(horizontal) + index)].append(index)
    return tuple(
        (horizontal[grouped_h[root]], vertical[grouped_v[root]])
        for root in grouped_h.keys() & grouped_v.keys()
    )


def internal_split_grid_component(
    horizontal: numpy.ndarray[Any, Any],
    vertical: numpy.ndarray[Any, Any],
) -> tuple[tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]], ...]:
    """Split a connected ruled component into vertically separated table regions."""
    positions = numpy.unique(horizontal[:, 2])
    if len(positions) < 3:
        return ((horizontal, vertical),)
    group_breaks = numpy.empty(len(positions), dtype=numpy.bool_)
    group_breaks[0] = True
    group_breaks[1:] = numpy.diff(positions) > TABLE_REGION_GAP
    position_groups = numpy.cumsum(group_breaks) - 1
    horizontal_groups = position_groups[numpy.searchsorted(positions, horizontal[:, 2])]
    regions = []
    for group_index in range(int(position_groups[-1]) + 1):
        group_positions = positions[position_groups == group_index]
        y0, y1 = group_positions[0], group_positions[-1]
        region_horizontal = horizontal[horizontal_groups == group_index]
        region_vertical = vertical[
            (vertical[:, 1] <= y1 + AXIS_TOLERANCE) & (vertical[:, 2] >= y0 - AXIS_TOLERANCE)
        ]
        if len(region_horizontal) >= 2 and len(region_vertical) >= 2:
            regions.append((region_horizontal, region_vertical))
    return tuple(regions) or ((horizontal, vertical),)


def internal_cell_text(observations: ObservationBatch, indexes: list[int]) -> str:
    ordered = sorted(
        indexes,
        key=lambda index: (
            -float((observations.bbox[index, 1] + observations.bbox[index, 3]) * 0.5),
            float(observations.bbox[index, 0]),
            int(observations.sequence[index]),
        ),
    )
    parts = []
    for index in ordered:
        part = " ".join(observations.text[index].split())
        if part and not (len(part) >= 4 and set(part) <= {".", "…"}):
            parts.append(part)
    return " ".join(parts)


def internal_vertical_boundary_present(
    vertical: numpy.ndarray[Any, Any],
    x: float,
    y0: float,
    y1: float,
) -> bool:
    return bool(
        numpy.any(
            (numpy.abs(vertical[:, 0] - x) <= AXIS_TOLERANCE)
            & (vertical[:, 1] <= y0 + AXIS_TOLERANCE)
            & (vertical[:, 2] >= y1 - AXIS_TOLERANCE)
        )
    )


def internal_horizontal_boundary_present(
    horizontal: numpy.ndarray[Any, Any],
    y: float,
    x0: float,
    x1: float,
) -> bool:
    return bool(
        numpy.any(
            (numpy.abs(horizontal[:, 2] - y) <= AXIS_TOLERANCE)
            & (horizontal[:, 0] <= x0 + AXIS_TOLERANCE)
            & (horizontal[:, 1] >= x1 - AXIS_TOLERANCE)
        )
    )


def internal_merge_grid_cells(
    rows: list[list[TableCell]],
    horizontal: numpy.ndarray[Any, Any],
    vertical: numpy.ndarray[Any, Any],
    x_edges: numpy.ndarray[Any, Any],
    y_edges: numpy.ndarray[Any, Any],
) -> list[tuple[TableCell, ...]]:
    """Collapse grid cells across absent rules into row/column-spanning cells."""
    row_count = len(rows)
    column_count = max((len(row) for row in rows), default=0)
    if row_count == 0 or column_count == 0:
        return []
    vertical_index = SpatialIndex(
        (
            (
                index,
                (
                    float(segment[0]) - AXIS_TOLERANCE,
                    float(segment[1]) - AXIS_TOLERANCE,
                    float(segment[0]) + AXIS_TOLERANCE,
                    float(segment[2]) + AXIS_TOLERANCE,
                ),
            )
            for index, segment in enumerate(vertical)
        )
    )
    horizontal_index = SpatialIndex(
        (
            (
                index,
                (
                    float(segment[0]) - AXIS_TOLERANCE,
                    float(segment[2]) - AXIS_TOLERANCE,
                    float(segment[1]) + AXIS_TOLERANCE,
                    float(segment[2]) + AXIS_TOLERANCE,
                ),
            )
            for index, segment in enumerate(horizontal)
        )
    )

    def vertical_boundary_present(x: float, y0: float, y1: float) -> bool:
        query = (x - AXIS_TOLERANCE, y0 - AXIS_TOLERANCE, x + AXIS_TOLERANCE, y1 + AXIS_TOLERANCE)
        return any(
            abs(float(vertical[index, 0]) - x) <= AXIS_TOLERANCE
            and float(vertical[index, 1]) <= y0 + AXIS_TOLERANCE
            and float(vertical[index, 2]) >= y1 - AXIS_TOLERANCE
            for index in vertical_index.intersecting(query)
        )

    def horizontal_boundary_present(y: float, x0: float, x1: float) -> bool:
        query = (x0 - AXIS_TOLERANCE, y - AXIS_TOLERANCE, x1 + AXIS_TOLERANCE, y + AXIS_TOLERANCE)
        return any(
            abs(float(horizontal[index, 2]) - y) <= AXIS_TOLERANCE
            and float(horizontal[index, 0]) <= x0 + AXIS_TOLERANCE
            and float(horizontal[index, 1]) >= x1 - AXIS_TOLERANCE
            for index in horizontal_index.intersecting(query)
        )

    disjoint = internal_DisjointSet(row_count * column_count)
    for row in range(row_count):
        y0, y1 = float(y_edges[row + 1]), float(y_edges[row])
        for column in range(column_count - 1):
            if not vertical_boundary_present(float(x_edges[column + 1]), y0, y1):
                disjoint.union(row * column_count + column, row * column_count + column + 1)
    for row in range(row_count - 1):
        y = float(y_edges[row + 1])
        for column in range(column_count):
            x0, x1 = float(x_edges[column]), float(x_edges[column + 1])
            if not horizontal_boundary_present(y, x0, x1):
                disjoint.union(row * column_count + column, (row + 1) * column_count + column)

    members: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in range(row_count):
        for column in range(column_count):
            members[disjoint.find(row * column_count + column)].append((row, column))
    merged: list[list[TableCell]] = [[] for _ in range(row_count)]
    for cells in members.values():
        min_row = min(row for row, internal_column in cells)
        max_row = max(row for row, internal_column in cells)
        min_column = min(column for internal_row, column in cells)
        max_column = max(column for internal_row, column in cells)
        source_cells = [rows[row][column] for row, column in cells]
        text = " ".join(cell.text for cell in source_cells if cell.text).strip()
        boxes = [cell.bbox for cell in source_cells if cell.bbox is not None]
        bbox = (
            (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
            if boxes
            else None
        )
        merged[min_row].append(
            TableCell(
                row=min_row,
                column=min_column,
                text=text,
                row_span=max_row - min_row + 1,
                column_span=max_column - min_column + 1,
                bbox=bbox,
            )
        )
    return [tuple(sorted(row, key=lambda cell: cell.column)) for row in merged]


def internal_text_rows(observations: ObservationBatch) -> list[list[int]]:
    visible = tuple(
        index
        for index, text in enumerate(observations.text)
        if bool(observations.visible[index])
        and text.strip()
        and int(observations.rotation[index]) == 0
    )
    if not visible:
        return []
    ordered = sorted(
        visible,
        key=lambda index: (
            -float((observations.bbox[index, 1] + observations.bbox[index, 3]) * 0.5),
            float(observations.bbox[index, 0]),
            int(observations.sequence[index]),
        ),
    )
    rows: list[list[int]] = []
    centers: list[float] = []
    heights: list[float] = []
    for index in ordered:
        center = float((observations.bbox[index, 1] + observations.bbox[index, 3]) * 0.5)
        height = max(
            1.0,
            float(observations.bbox[index, 3] - observations.bbox[index, 1]),
        )
        if rows and abs(center - centers[-1]) <= max(2.0, min(height, heights[-1]) * 0.5):
            count = len(rows[-1])
            rows[-1].append(index)
            centers[-1] = (centers[-1] * count + center) / (count + 1)
            heights[-1] = (heights[-1] * count + height) / (count + 1)
        else:
            rows.append([index])
            centers.append(center)
            heights.append(height)
    return [
        sorted(
            row,
            key=lambda index: (
                float(observations.bbox[index, 0]),
                int(observations.sequence[index]),
            ),
        )
        for row in rows
    ]


def internal_row_center(observations: ObservationBatch, row: list[int]) -> float:
    return sum(
        float((observations.bbox[index, 1] + observations.bbox[index, 3]) * 0.5) for index in row
    ) / len(row)


def internal_aligned_column_clusters(
    observations: ObservationBatch,
    rows: list[list[int]],
    page_width: float,
    *,
    minimum_rows: int = 3,
) -> list[list[tuple[int, int]]]:
    tolerance = max(COLUMN_TOLERANCE, min(28.0, page_width * 0.05))
    positions = [
        (float(observations.bbox[index, 0]), row_index, index)
        for row_index, row in enumerate(rows)
        for index in row
    ]
    positions.sort(key=lambda item: (item[0], item[1], int(observations.sequence[item[2]])))
    clusters: list[list[tuple[int, int]]] = []
    means: list[float] = []
    for x, row_index, index in positions:
        if clusters and abs(x - means[-1]) <= tolerance:
            count = len(clusters[-1])
            clusters[-1].append((row_index, index))
            means[-1] = (means[-1] * count + x) / (count + 1)
        else:
            clusters.append([(row_index, index)])
            means.append(x)
    candidates = []
    for cluster in clusters:
        row_support = {row_index for row_index, internal_index in cluster}
        widths = [
            float(observations.bbox[index, 2] - observations.bbox[index, 0])
            for internal_row_index, index in cluster
        ]
        alphanumeric = sum(
            any(character.isalnum() for character in observations.text[index])
            for internal_row_index, index in cluster
        )
        if (
            len(row_support) >= minimum_rows
            and float(numpy.median(widths)) <= page_width * 0.48
            and alphanumeric * 2 >= len(cluster)
        ):
            candidates.append(cluster)
    return candidates


def internal_split_support_rows(
    observations: ObservationBatch,
    rows: list[list[int]],
    indexes: list[int],
    *,
    minimum_rows: int = 3,
) -> list[list[int]]:
    if not indexes:
        return []
    groups = [[indexes[0]]]
    for index in indexes[1:]:
        previous = groups[-1][-1]
        previous_height = max(
            float(observations.bbox[item, 3] - observations.bbox[item, 1])
            for item in rows[previous]
        )
        current_height = max(
            float(observations.bbox[item, 3] - observations.bbox[item, 1]) for item in rows[index]
        )
        allowed_gap = max(55.0, max(previous_height, current_height) * 4.0)
        if (
            internal_row_center(observations, rows[previous])
            - internal_row_center(observations, rows[index])
            > allowed_gap
        ):
            groups.append([])
        groups[-1].append(index)
    return [group for group in groups if len(group) >= minimum_rows]


def internal_stream_table(
    order: int,
    observations: ObservationBatch,
    rows: list[list[int]],
    support: list[int],
    columns: list[list[tuple[int, int]]],
    *,
    minimum_rows: int = 3,
) -> Table | None:
    support_set = set(support)
    columns = [
        column
        for column in columns
        if len({row_index for row_index, internal_index in column}.intersection(support_set))
        >= minimum_rows
    ]
    if len(columns) < 2:
        return None
    column_centers = numpy.asarray(
        [
            numpy.median(
                [float(observations.bbox[index, 0]) for internal_row_index, index in column]
            )
            for column in columns
        ],
        dtype=numpy.float32,
    )
    column_order = numpy.argsort(column_centers)
    column_centers = column_centers[column_order]
    columns = [columns[int(index)] for index in column_order]
    top = internal_row_center(observations, rows[support[0]])
    bottom = internal_row_center(observations, rows[support[-1]])
    selected = [
        row for row in rows if bottom - 1.0 <= internal_row_center(observations, row) <= top + 1.0
    ]
    if len(selected) < minimum_rows:
        return None
    edges = numpy.empty(len(columns) + 1, dtype=numpy.float32)
    edges[1:-1] = (column_centers[:-1] + column_centers[1:]) * 0.5
    edges[0] = min(float(observations.bbox[index, 0]) for row in selected for index in row)
    edges[-1] = max(
        float(observations.bbox[index, 2])
        for column in columns
        for row_index, index in column
        if support[0] <= row_index <= support[-1]
    )
    if numpy.any(numpy.diff(edges) <= 2.0):
        return None

    table_rows: list[tuple[TableCell, ...]] = []
    populated = 0
    numeric_by_column = [0] * len(columns)
    text_lengths = 0
    for row_index, row in enumerate(selected):
        cells: list[list[int]] = [[] for internal_column in columns]
        for index in row:
            x0 = float(observations.bbox[index, 0])
            x1 = float(observations.bbox[index, 2])
            x_center = (x0 + x1) * 0.5
            column = int(numpy.searchsorted(edges, x_center, side="right") - 1)
            # Wide description cells whose midpoint falls left of the first clustered
            # column edge would get column=-1.  Snap them to column 0 so they are
            # captured rather than silently dropped.
            if column < 0 and x0 <= float(edges[-1]) + COLUMN_TOLERANCE:
                column = 0
            if 0 <= column < len(columns) and x0 <= float(edges[-1]) + COLUMN_TOLERANCE:
                cells[column].append(index)
        texts = [internal_cell_text(observations, cell) for cell in cells]
        if not any(texts):
            continue
        populated += sum(bool(text) for text in texts)
        numeric_cells = [internal_numeric_cell(text) for text in texts]
        for column, is_numeric in enumerate(numeric_cells):
            numeric_by_column[column] += int(is_numeric)
        text_lengths += sum(len(text) for text in texts)
        y0 = min(float(observations.bbox[index, 1]) for index in row)
        y1 = max(float(observations.bbox[index, 3]) for index in row)
        table_rows.append(
            tuple(
                TableCell(
                    row=len(table_rows),
                    column=column,
                    text=text,
                    bbox=(float(edges[column]), y0, float(edges[column + 1]), y1),
                )
                for column, text in enumerate(texts)
            )
        )
    if len(table_rows) < minimum_rows or populated < minimum_rows * 2:
        return None
    density = populated / (len(table_rows) * len(columns))
    average_text = text_lengths / max(1, populated)
    minimum_density = 0.75 if minimum_rows == 2 else 0.35
    if density < minimum_density:
        return None
    numeric_total = sum(numeric_by_column)
    filled_texts = [cell.text.strip() for row in table_rows for cell in row if cell.text.strip()]
    long_text_cells = sum(len(text) > 18 for text in filled_texts)
    sentence_like_cells = sum(
        any(mark in text for mark in (". ", ", ", "; ", ": ")) for text in filled_texts
    )
    character_spaced_cells = sum(internal_character_spaced_cell(text) for text in filled_texts)
    if (
        minimum_rows >= 3
        and len(table_rows) >= 5
        and len(columns) >= 4
        and numeric_total <= 1
        and filled_texts
        and long_text_cells / len(filled_texts) >= 0.35
        and sentence_like_cells / len(filled_texts) >= 0.20
    ):
        return None
    if (
        len(columns) >= 6
        and filled_texts
        and numeric_total / len(filled_texts) < 0.12
        and character_spaced_cells / len(filled_texts) >= 0.50
    ):
        return None
    if minimum_rows == 2:
        if max(numeric_by_column, default=0) < 1 and average_text > 24.0:
            return None
    elif max(numeric_by_column, default=0) < 3 and average_text > 12.0:
        return None
    bbox = (
        float(edges[0]),
        min(cell.bbox[1] for row in table_rows for cell in row if cell.bbox is not None),
        float(edges[-1]),
        max(cell.bbox[3] for row in table_rows for cell in row if cell.bbox is not None),
    )
    return Table(
        order=order,
        rows=tuple(table_rows),
        bbox=bbox,
        confidence=0.75,
        metadata={
            "source": "stream",
            "rows": len(table_rows),
            "columns": len(columns),
            "density": round(density, 4),
            "average_text": round(average_text, 2),
            "numeric_cells": numeric_total,
        },
    )


def internal_compact_stream_table(
    order: int,
    observations: ObservationBatch,
    rows: list[list[int]],
    page_width: float,
) -> Table | None:
    """Recover compact tables whose rows are interleaved with nearby prose."""
    candidates = [
        row
        for row in rows
        if len(row) >= 3
        and max(float(observations.bbox[index, 2]) for index in row) <= page_width * 0.55
        and sum(len(observations.text[index].strip()) for index in row) <= 110
    ]
    if len(candidates) < 4:
        return None

    anchor_rows = [row for row in candidates if len(row) == 3]
    if len(anchor_rows) < 2:
        return None
    anchors = numpy.median(
        numpy.asarray(
            [[float(observations.bbox[index, 0]) for index in row] for row in anchor_rows],
            dtype=numpy.float32,
        ),
        axis=0,
    )
    if numpy.any(numpy.diff(anchors) < 30.0):
        return None

    table_rows: list[tuple[TableCell, ...]] = []
    numeric_cells = 0
    for row in candidates:
        cell_indexes: list[list[int]] = [[] for _ in anchors]
        for index in row:
            column = int(numpy.argmin(numpy.abs(anchors - observations.bbox[index, 0])))
            cell_indexes[column].append(index)
        texts = [internal_cell_text(observations, indexes) for indexes in cell_indexes]
        if not any(texts):
            continue
        numeric_cells += sum(
            internal_numeric_cell(text) or any(character.isdigit() for character in text)
            for text in texts
        )
        y0 = min(float(observations.bbox[index, 1]) for index in row)
        y1 = max(float(observations.bbox[index, 3]) for index in row)
        edges = [
            min(float(observations.bbox[index, 0]) for index in row),
            *(float((anchors[column] + anchors[column + 1]) * 0.5) for column in range(2)),
            max(float(observations.bbox[index, 2]) for index in row),
        ]
        table_rows.append(
            tuple(
                TableCell(
                    row=len(table_rows),
                    column=column,
                    text=text,
                    bbox=(edges[column], y0, edges[column + 1], y1),
                )
                for column, text in enumerate(texts)
            )
        )
    if len(table_rows) < 4 or numeric_cells < 2:
        return None
    boxes = [cell.bbox for row in table_rows for cell in row if cell.bbox is not None]
    if not boxes:
        return None
    return Table(
        order=order,
        rows=tuple(table_rows),
        bbox=(
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ),
        confidence=0.7,
        metadata={"source": "stream", "compact": True},
    )


def internal_numeric_cell(text: str) -> bool:
    alphanumeric = sum(character.isalnum() for character in text)
    digits = sum(character.isdigit() for character in text)
    return bool(digits and digits * 2 >= max(1, alphanumeric))


def internal_character_spaced_cell(text: str) -> bool:
    tokens = [token for token in text.split() if any(character.isalpha() for character in token)]
    if len(tokens) < 8:
        return False
    single_character = sum(len(token) == 1 for token in tokens)
    return single_character / len(tokens) >= 0.80


def internal_stream_tables(
    capture: CapturedPage,
    observations: ObservationBatch,
    start_order: int,
) -> tuple[Table, ...]:
    rows = internal_text_rows(observations)
    tables: list[Table] = []
    for minimum_rows in (3, 2):
        columns = internal_aligned_column_clusters(
            observations,
            rows,
            float(capture.page.width),
            minimum_rows=minimum_rows,
        )
        if len(columns) < 2:
            continue
        row_columns: dict[int, set[int]] = defaultdict(set)
        for column_index, column in enumerate(columns):
            for row_index, internal_index in column:
                row_columns[row_index].add(column_index)
        pair_counts: Counter[tuple[int, int]] = Counter()
        for present in row_columns.values():
            pair_counts.update(combinations(sorted(present), 2))
        disjoint = internal_DisjointSet(len(columns))
        for pair, count in pair_counts.items():
            if count >= minimum_rows:
                disjoint.union(*pair)
        components: dict[int, list[int]] = defaultdict(list)
        for column_index in range(len(columns)):
            components[disjoint.find(column_index)].append(column_index)

        for component in components.values():
            if len(component) < 2 or len(component) > 20:
                continue
            required = max(2, (len(component) + 1) // 2)
            support = sorted(
                row_index
                for row_index, present in row_columns.items()
                if len(present.intersection(component)) >= required
            )
            for group in internal_split_support_rows(
                observations,
                rows,
                support,
                minimum_rows=minimum_rows,
            ):
                table = internal_stream_table(
                    start_order + len(tables),
                    observations,
                    rows,
                    group,
                    [columns[index] for index in component],
                    minimum_rows=minimum_rows,
                )
                if table is not None:
                    tables.append(table)
    if not tables:
        compact = internal_compact_stream_table(
            start_order,
            observations,
            rows,
            float(capture.page.width),
        )
        if compact is not None:
            tables.append(compact)
    unique: list[Table] = []
    for table in tables:
        if any(
            table.bbox is not None
            and existing.bbox is not None
            and internal_bbox_overlap_ratio(table.bbox, existing.bbox) >= 0.8
            for existing in unique
        ):
            continue
        unique.append(table)
    return tuple(unique)


def internal_table_quality(table: Table) -> tuple[int, int, float, int, int]:
    rows = len(table.rows)
    columns = max((len(row) for row in table.rows), default=0)
    populated = sum(bool(cell.text.strip()) for row in table.rows for cell in row)
    density = populated / max(1, rows * columns)
    return (int(2 <= columns <= 12), populated, density, rows, -columns)


def internal_merge_adjacent_tables(tables: list[Table]) -> list[Table]:
    ordered = sorted(tables, key=lambda table: -(table.bbox or (0.0, 0.0, 0.0, 0.0))[3])
    merged: list[Table] = []
    for table in ordered:
        if not merged or table.bbox is None or merged[-1].bbox is None:
            merged.append(table)
            continue
        previous = merged[-1]
        previous_bbox = previous.bbox
        table_bbox = table.bbox
        if previous_bbox is None or table_bbox is None:
            merged.append(table)
            continue
        previous_columns = max((len(row) for row in previous.rows), default=0)
        columns = max((len(row) for row in table.rows), default=0)
        horizontal_overlap = max(
            0.0,
            min(previous_bbox[2], table_bbox[2]) - max(previous_bbox[0], table_bbox[0]),
        )
        minimum_width = max(
            1.0,
            min(previous_bbox[2] - previous_bbox[0], table_bbox[2] - table_bbox[0]),
        )
        vertical_gap = previous_bbox[1] - table_bbox[3]
        if (
            columns != previous_columns
            or not 2 <= columns <= 12
            or horizontal_overlap / minimum_width < 0.8
            or not -5.0 <= vertical_gap <= TABLE_MERGE_GAP
        ):
            merged.append(table)
            continue
        combined_rows: list[tuple[TableCell, ...]] = []
        for row in (*previous.rows, *table.rows):
            combined_rows.append(
                tuple(
                    TableCell(
                        row=len(combined_rows),
                        column=cell.column,
                        text=cell.text,
                        row_span=cell.row_span,
                        column_span=cell.column_span,
                        bbox=cell.bbox,
                    )
                    for cell in row
                )
            )
        merged[-1] = Table(
            order=previous.order,
            rows=tuple(combined_rows),
            bbox=(
                min(previous_bbox[0], table_bbox[0]),
                min(previous_bbox[1], table_bbox[1]),
                max(previous_bbox[2], table_bbox[2]),
                max(previous_bbox[3], table_bbox[3]),
            ),
            confidence=min(previous.confidence or 1.0, table.confidence or 1.0),
            metadata=previous.metadata,
        )
    return merged


def internal_semantic_header_row(row: tuple[TableCell, ...]) -> bool:
    populated = [cell for cell in row if cell.text.strip()]
    if not populated:
        return False
    if len(populated) == 1:
        return populated[0].column_span > 1
    numeric = sum(internal_numeric_cell(cell.text) for cell in populated)
    return numeric == 0 and len(populated) >= 2


def internal_numeric_density(table: Table) -> float:
    cells = [cell for row in table.rows for cell in row if cell.text.strip()]
    return sum(internal_numeric_cell(cell.text) for cell in cells) / max(1, len(cells))


def internal_structured_stream_table(table: Table) -> bool:
    """Identify stream tables with enough structured values to preserve."""
    if table.metadata.get("source") != "stream":
        return False
    numeric_cells = table.metadata.get("numeric_cells", 0)
    return internal_numeric_density(table) >= 0.10 or (
        isinstance(numeric_cells, int) and numeric_cells >= 2
    )


def internal_split_semantic_table(table: Table) -> tuple[Table, ...]:
    """Split long grid regions at repeated section-header rows."""
    if len(table.rows) < 6 or internal_numeric_density(table) < 0.3:
        return (table,)
    boundaries = [
        index
        for index, row in enumerate(table.rows[1:], start=1)
        if (
            internal_semantic_header_row(row)
            and index >= 2
            and index + 1 < len(table.rows)
            and any(internal_numeric_cell(cell.text) for cell in table.rows[index + 1])
        )
    ]
    if not boundaries:
        return (table,)
    signatures = {
        tuple(index for index, cell in enumerate(table.rows[index]) if cell.text.strip())
        for index in boundaries
    }
    labels = {
        " ".join(item.text for item in table.rows[index] if item.text.strip()).casefold()
        for index in boundaries
        if any(item.text.strip() for item in table.rows[index])
    }
    if len(table.rows) > 8 and len(boundaries) > 1 and len(signatures) == 1 and len(labels) == 1:
        return (table,)
    starts = [0, *boundaries]
    segments: list[Table] = []
    for segment_index, start in enumerate(starts):
        end = starts[segment_index + 1] if segment_index + 1 < len(starts) else len(table.rows)
        rows = table.rows[start:end]
        if len(rows) < 2:
            continue
        boxes = [cell.bbox for row in rows for cell in row if cell.bbox is not None]
        bbox = (
            (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
            if boxes
            else None
        )
        segments.append(
            Table(
                order=table.order + segment_index,
                rows=rows,
                bbox=bbox,
                confidence=table.confidence,
                metadata=table.metadata,
            )
        )
    return tuple(segments) or (table,)


def internal_table_character_spaced_prose(table: Table) -> bool:
    if table.metadata.get("source") != "stream":
        return False
    columns = max((len(row) for row in table.rows), default=0)
    if columns < 6:
        return False
    filled_texts = [cell.text.strip() for row in table.rows for cell in row if cell.text.strip()]
    if not filled_texts:
        return False
    numeric_cells = sum(internal_numeric_cell(text) for text in filled_texts)
    character_spaced_cells = sum(internal_character_spaced_cell(text) for text in filled_texts)
    return (
        numeric_cells / len(filled_texts) < 0.12
        and character_spaced_cells / len(filled_texts) >= 0.50
    )


def internal_merge_stream_text_columns(table: Table) -> Table:
    """Merge word-aligned columns that form two wrapped text columns."""
    columns = max((len(row) for row in table.rows), default=0)
    if (
        table.metadata.get("source") != "stream"
        or columns < 6
        or columns % 2
        or len(table.rows) < 4
        or internal_numeric_density(table) >= 0.25
    ):
        return table
    group_size = columns // 2
    merged_rows: list[tuple[TableCell, ...]] = []
    for row_index, row in enumerate(table.rows):
        merged: list[TableCell] = []
        for group in range(2):
            cells = row[group * group_size : (group + 1) * group_size]
            text = " ".join(cell.text for cell in cells if cell.text).strip()
            boxes = [cell.bbox for cell in cells if cell.bbox is not None]
            if not text and row_index == 0:
                continue
            bbox = (
                (
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                )
                if boxes
                else None
            )
            merged.append(
                TableCell(
                    row=row_index,
                    column=len(merged),
                    text=text,
                    column_span=group_size if row_index == 0 and len(merged) == 0 else 1,
                    bbox=bbox,
                )
            )
        if merged:
            merged_rows.append(tuple(merged))
    return replace(table, rows=tuple(merged_rows), metadata={**table.metadata, "merged": True})


def internal_table_from_component(
    order: int,
    horizontal: numpy.ndarray[Any, Any],
    vertical: numpy.ndarray[Any, Any],
    observations: ObservationBatch,
    observation_index: SpatialIndex[int] | None = None,
) -> Table | None:
    x_edges = internal_cluster_positions(vertical[:, 0])
    y_edges = internal_cluster_positions(horizontal[:, 2])[::-1]
    columns = len(x_edges) - 1
    row_count = len(y_edges) - 1
    if columns < 2 or row_count < 1 or columns * row_count > 1_000:
        return None
    x0, x1 = float(x_edges[0]), float(x_edges[-1])
    y0, y1 = float(y_edges[-1]), float(y_edges[0])
    cell_observations: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidate_indexes = (
        observation_index.intersecting((x0, y0, x1, y1))
        if observation_index is not None
        else range(len(observations))
    )
    for index in candidate_indexes:
        index = int(index)
        if observation_index is None and not bool(observations.visible[index]):
            continue
        center_x = float((observations.bbox[index, 0] + observations.bbox[index, 2]) * 0.5)
        center_y = float((observations.bbox[index, 1] + observations.bbox[index, 3]) * 0.5)
        if not (x0 <= center_x <= x1 and y0 <= center_y <= y1):
            continue
        column = int(numpy.searchsorted(x_edges, center_x, side="right") - 1)
        row = int(numpy.searchsorted(-y_edges, -center_y, side="right") - 1)
        if column == columns:
            column -= 1
        if row == row_count:
            row -= 1
        if 0 <= row < row_count and 0 <= column < columns:
            cell_observations[(row, column)].append(index)
    populated = sum(bool(value) for value in cell_observations.values())
    if populated < 2:
        return None
    density = populated / max(1, columns * row_count)
    # Wide, sparsely populated ruled grids are usually decorative form/layout
    # geometry rather than tables.  Keep narrow sparse tables supported while
    # requiring broad grids to contain enough text to justify table structure.
    if columns >= 6 and density < 0.5:
        return None
    rows: list[list[TableCell]] = []
    for row in range(row_count):
        cells = []
        for column in range(columns):
            bbox = (
                float(x_edges[column]),
                float(y_edges[row + 1]),
                float(x_edges[column + 1]),
                float(y_edges[row]),
            )
            cells.append(
                TableCell(
                    row=row,
                    column=column,
                    text=internal_cell_text(observations, cell_observations[(row, column)]),
                    bbox=bbox,
                )
            )
        rows.append(cells)
    merged_rows = internal_merge_grid_cells(rows, horizontal, vertical, x_edges, y_edges)
    return Table(
        order=order,
        rows=tuple(tuple(row) for row in merged_rows),
        bbox=(x0, y0, x1, y1),
        confidence=1.0,
    )


def extract_tables(
    capture: CapturedPage,
    observations: ObservationBatch,
) -> tuple[Table, ...]:
    horizontal, vertical = internal_axis_segments(capture)
    horizontal = internal_merge_collinear_segments(horizontal, coordinate=2, start=0, end=1)
    vertical = internal_merge_collinear_segments(vertical, coordinate=0, start=1, end=2)
    components = internal_grid_components(horizontal, vertical)
    visible_indices = numpy.flatnonzero(observations.visible)
    observation_index = SpatialIndex((int(idx), observations.bbox[idx]) for idx in visible_indices)
    ruled_tables: list[Table] = []
    for component in components:
        for component_part in internal_split_grid_component(*component):
            table = internal_table_from_component(
                len(ruled_tables),
                *component_part,
                observations,
                observation_index,
            )
            if table is not None:
                ruled_tables.append(table)
    ruled = tuple(ruled_tables)
    tables = list(ruled)
    for stream in internal_stream_tables(capture, observations, len(tables)):
        conflicts = [
            table
            for table in tables
            if stream.bbox is not None
            and table.bbox is not None
            and internal_bbox_overlap_ratio(stream.bbox, table.bbox) >= 0.5
        ]
        if conflicts and internal_table_quality(stream) < max(
            map(internal_table_quality, conflicts)
        ):
            continue
        for conflict in conflicts:
            tables.remove(conflict)
        tables.append(internal_merge_stream_text_columns(stream))
    tables = [
        segment
        for table in internal_merge_adjacent_tables(tables)
        for segment in internal_split_semantic_table(table)
        if not internal_table_character_spaced_prose(segment)
    ]
    for order, table in enumerate(tables):
        if table.order != order:
            tables[order] = Table(
                order=order,
                rows=table.rows,
                bbox=table.bbox,
                confidence=table.confidence,
                metadata=table.metadata,
            )
    return tuple(sorted(tables, key=lambda table: -(table.bbox or (0.0, 0.0, 0.0, 0.0))[3]))


# ===== layout =====


internal_NATIVE_DOTTED_LEADER_RE = re.compile(r"\.{2,}")
internal_NATIVE_DASH_RULE_RE = re.compile(r"(?:\s*-\s*){2,}")


def internal_line_group_indexes(observations: ObservationBatch) -> list[list[int]]:
    if not len(observations):
        return []
    visible_indexes = numpy.flatnonzero(observations.visible)
    indexes = (
        visible_indexes
        if len(visible_indexes)
        else numpy.arange(len(observations), dtype=numpy.int64)
    )
    boxes = observations.bbox[indexes]
    heights = numpy.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    centers = (boxes[:, 1] + boxes[:, 3]) * 0.5
    rotations = observations.rotation[indexes]
    explicit = observations.line_break_before[indexes]
    breaks = numpy.zeros(len(indexes), dtype=numpy.bool_)
    breaks[0] = True
    if len(indexes) > 1:
        tolerance = numpy.maximum(2.0, numpy.minimum(heights[:-1], heights[1:]) * 0.65)
        breaks[1:] = (
            explicit[1:]
            | (rotations[1:] != rotations[:-1])
            | (numpy.abs(centers[1:] - centers[:-1]) > tolerance)
        )
    # Split at the already-vectorized break positions instead of rebuilding
    # every group through a Python loop over all observations.
    return [group.tolist() for group in numpy.split(indexes, numpy.flatnonzero(breaks)[1:])]


def internal_group_text(observations: ObservationBatch, indexes: list[int]) -> str:
    if any(int(observations.source[index]) == int(ObservationSource.OCR) for index in indexes):
        indexes = sorted(indexes, key=lambda index: float(observations.bbox[index, 0]))
    references = tuple(observations.references[index] for index in indexes)
    if references and all(reference is not None for reference in references):
        runs = cast(list[TextRun], list(references))
        return reconstruct_layout_line_text(runs).text.strip()
    parts: list[str] = []
    for index in indexes:
        text = observations.text[index].strip()
        if not text:
            continue
        if (
            parts
            and not parts[-1].endswith((" ", "-", "/"))
            and not text.startswith((".", ",", ":", ";", ")", "]", "}"))
        ):
            parts.append(" ")
        parts.append(text)
    return "".join(parts)


def internal_looks_like_native_artifact(text: str) -> bool:
    """Reject symbol-heavy native lines produced by damaged text layers.

    Some PDFs expose decorative rules, malformed glyph mappings, and dotted
    leaders as ordinary text runs.  They are not OCR observations and are
    therefore safe to reject only after line reconstruction, where the whole
    artifact is visible.  Requiring a small alphanumeric count keeps compact
    identifiers and schematic labels intact.
    """
    # Unicode punctuation and scripts can be valid standalone text runs.  The
    # damaged mappings this targets are emitted as ASCII-looking rules and
    # dotted leaders, so leave non-ASCII lines untouched.
    if any(ord(character) > 127 for character in text):
        return False
    nonspace = [character for character in text if not character.isspace()]
    if not nonspace:
        return False
    alphanumeric = sum(character.isalnum() for character in nonspace)
    if alphanumeric >= 12:
        return False
    return (len(nonspace) - alphanumeric) / len(nonspace) >= 0.60


def internal_repeated_native_label_tokens(
    observations: ObservationBatch,
    indexes: list[int],
) -> frozenset[str]:
    counts: Counter[str] = Counter()
    for index in indexes:
        text = " ".join(observations.text[index].split())
        if len(text) == 1 and text.isascii() and text.isalpha():
            counts[text.casefold()] += 1
    return frozenset(token for token, count in counts.items() if count >= 4)


def internal_is_repeated_native_label(text: str, repeated_tokens: frozenset[str]) -> bool:
    parts = text.casefold().split()
    return bool(parts) and all(len(part) == 1 and part in repeated_tokens for part in parts)


def internal_clean_native_punctuation_runs(text: str) -> str:
    text = internal_NATIVE_DOTTED_LEADER_RE.sub(" ", text)
    text = internal_NATIVE_DASH_RULE_RE.sub(" ", text)
    return " ".join(text.split())


def internal_color_is_emphasis(color: object) -> bool:
    if not isinstance(color, (tuple, list)) or len(color) < 3:
        return False
    components: list[float] = []
    for component in color[:3]:
        if not isinstance(component, (int, float)):
            return False
        components.append(float(component))
    return max(components) - min(components) >= 0.15


def internal_build_lines(observations: ObservationBatch) -> tuple[ParsedLine, ...]:
    line_groups = internal_line_group_indexes(observations)
    repeated_native_labels = internal_repeated_native_label_tokens(
        observations,
        [index for group in line_groups for index in group],
    )
    output: list[ParsedLine] = []
    for sequence, indexes in enumerate(line_groups):
        text = internal_group_text(observations, indexes)
        if not text:
            continue
        if all(
            int(source) == int(ObservationSource.NATIVE) for source in observations.source[indexes]
        ) and internal_looks_like_native_artifact(text):
            continue
        if (
            repeated_native_labels
            and all(
                int(source) == int(ObservationSource.NATIVE)
                for source in observations.source[indexes]
            )
            and internal_is_repeated_native_label(text, repeated_native_labels)
        ):
            continue
        if all(
            int(source) == int(ObservationSource.NATIVE) for source in observations.source[indexes]
        ):
            text = internal_clean_native_punctuation_runs(text)
            if not text:
                continue
        boxes = observations.bbox[indexes]
        confidences = observations.confidence[indexes]
        font_sizes = observations.font_size[indexes]
        finite_confidences = confidences[numpy.isfinite(confidences)]
        finite_font_sizes = font_sizes[numpy.isfinite(font_sizes) & (font_sizes > 0)]
        native_references = [
            reference
            for reference in (observations.references[index] for index in indexes)
            if reference is not None
        ]

        def style_enabled(reference: object, name: str) -> bool:
            value = getattr(reference, name, False)
            return bool(value() if callable(value) else value)

        bold = bool(native_references) and sum(
            style_enabled(reference, "is_bold") for reference in native_references
        ) * 2 >= len(native_references)
        italic = bool(native_references) and sum(
            style_enabled(reference, "is_italic") for reference in native_references
        ) * 2 >= len(native_references)
        span_values: list[TextSpan] = []
        pending_space = False
        for reference in native_references:
            reference_text = reference.text.strip()
            if not reference_text:
                pending_space = True
                continue
            prefix = ""
            if (
                pending_space
                and span_values
                and not span_values[-1].text.endswith(("(", "[", "{", "/", "-"))
                and not reference_text.startswith((".", ",", ";", ":", "!", "?", ")", "]", "}"))
            ):
                prefix = " "
            span_values.append(
                TextSpan(
                    text=prefix + reference_text,
                    bold=style_enabled(reference, "is_bold"),
                    italic=style_enabled(reference, "is_italic"),
                    mark=internal_color_is_emphasis(getattr(reference, "fill_color", None)),
                )
            )
            pending_space = reference.text.endswith((" ", "\t", "\n"))
        spans = tuple(span_values)
        if not spans or "".join(span.text for span in spans) != text:
            spans = ()
        sources = {int(value) for value in observations.source[indexes]}
        if sources == {int(ObservationSource.NATIVE)}:
            source = "native"
        elif sources == {int(ObservationSource.OCR)}:
            source = "ocr"
        else:
            source = "hybrid"
        output.append(
            ParsedLine(
                text=text,
                bbox=(
                    float(numpy.min(boxes[:, 0])),
                    float(numpy.min(boxes[:, 1])),
                    float(numpy.max(boxes[:, 2])),
                    float(numpy.max(boxes[:, 3])),
                ),
                source=source,
                confidence=(
                    float(numpy.mean(finite_confidences)) if len(finite_confidences) else None
                ),
                sequence=sequence,
                rotation=int(observations.rotation[indexes[0]]),
                font_size=(
                    float(numpy.median(finite_font_sizes)) if len(finite_font_sizes) else None
                ),
                bold=bold,
                italic=italic,
                spans=spans,
            )
        )
    return tuple(output)


def internal_best_projection_gap(
    boxes: numpy.ndarray,
    axis: int,
    minimum_gap: float,
) -> tuple[float, float] | None:
    starts = boxes[:, axis]
    ends = boxes[:, axis + 2]
    order = numpy.argsort(starts, kind="stable")
    sorted_starts = starts[order]
    sorted_ends = ends[order]
    previous_ends = numpy.maximum.accumulate(sorted_ends)[:-1]
    gaps = sorted_starts[1:] - previous_ends
    if not len(gaps):
        return None
    # PDF page space is bottom-left based.  For equal horizontal whitespace,
    # split at the uppermost gap first so a full-width header is detached before
    # column detection runs on the body below it.
    best_index = (
        len(gaps) - 1 - int(numpy.argmax(gaps[::-1])) if axis == 1 else int(numpy.argmax(gaps))
    )
    best_gap = float(gaps[best_index])
    best_cut = float((sorted_starts[best_index + 1] + previous_ends[best_index]) * 0.5)
    return (best_gap, best_cut) if best_gap >= minimum_gap else None


def internal_row_order_indexes(indexes: numpy.ndarray, boxes: numpy.ndarray) -> numpy.ndarray:
    region = boxes[indexes]
    heights = numpy.maximum(1.0, region[:, 3] - region[:, 1])
    row_quantum = max(2.0, float(numpy.median(heights)) * 0.75)
    if not math.isfinite(row_quantum):
        row_quantum = 2.0
    rows = numpy.rint(region[:, 1] / row_quantum).astype(numpy.int64)
    return indexes[numpy.lexsort((region[:, 0], -rows))]


def internal_xy_leaf_order(indexes: numpy.ndarray, boxes: numpy.ndarray) -> numpy.ndarray:
    region = boxes[indexes]
    return indexes[numpy.lexsort((region[:, 0], -region[:, 1]))]


def internal_obstacle_partition(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    obstacles: tuple[tuple[float, float, float, float], ...],
    obstacle_index: SpatialIndex[int] | None = None,
    used_obstacles: frozenset[int] = frozenset(),
) -> tuple[tuple[numpy.ndarray, ...], int] | None:
    if not obstacles or len(indexes) < 3:
        return None
    region = boxes[indexes]
    region_box = (
        float(numpy.min(region[:, 0])),
        float(numpy.min(region[:, 1])),
        float(numpy.max(region[:, 2])),
        float(numpy.max(region[:, 3])),
    )
    region_width = max(1.0, region_box[2] - region_box[0])
    region_height = max(1.0, region_box[3] - region_box[1])
    centers_x = (region[:, 0] + region[:, 2]) * 0.5
    centers_y = (region[:, 1] + region[:, 3]) * 0.5
    obstacle_indexes: Iterable[int] = (
        obstacle_index.intersecting(region_box)
        if obstacle_index is not None
        else range(len(obstacles))
    )
    for raw_obstacle_index in obstacle_indexes:
        current_obstacle_index = int(raw_obstacle_index)
        if current_obstacle_index in used_obstacles:
            continue
        obstacle = obstacles[current_obstacle_index]
        x0, y0, x1, y1 = obstacle
        obstacle_width = max(0.0, x1 - x0)
        obstacle_height = max(0.0, y1 - y0)
        if obstacle_width / region_width >= 0.70:
            groups = (
                indexes[centers_y > y1],
                indexes[(centers_y >= y0) & (centers_y <= y1)],
                indexes[centers_y < y0],
            )
        elif obstacle_height / region_height >= 0.70:
            groups = (
                indexes[centers_x < x0],
                indexes[(centers_x >= x0) & (centers_x <= x1)],
                indexes[centers_x > x1],
            )
        else:
            continue
        populated = tuple(group for group in groups if len(group))
        if len(populated) >= 2:
            return populated, current_obstacle_index
    return None


def internal_xy_cut_regions(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    obstacles: tuple[tuple[float, float, float, float], ...],
    median_height: float,
    *,
    depth: int = 0,
    obstacle_index: SpatialIndex[int] | None = None,
    used_obstacles: frozenset[int] = frozenset(),
) -> list[numpy.ndarray]:
    if len(indexes) <= 2 or depth >= 32:
        return [internal_xy_leaf_order(indexes, boxes)]

    obstacle_partition = internal_obstacle_partition(
        indexes,
        boxes,
        obstacles,
        obstacle_index,
        used_obstacles,
    )
    if obstacle_partition is not None:
        groups, used_obstacle = obstacle_partition
        next_used_obstacles = used_obstacles | {used_obstacle}
        return [
            region
            for group in groups
            for region in internal_xy_cut_regions(
                group,
                boxes,
                obstacles,
                median_height,
                depth=depth + 1,
                obstacle_index=obstacle_index,
                used_obstacles=next_used_obstacles,
            )
        ]

    region_boxes = boxes[indexes]
    horizontal = internal_best_projection_gap(region_boxes, 1, max(3.0, median_height * 0.90))
    vertical = internal_best_projection_gap(region_boxes, 0, max(12.0, median_height * 1.5))
    candidates: list[tuple[float, int, float]] = []
    if horizontal is not None:
        candidates.append((horizontal[0] / median_height, 1, horizontal[1]))
    if vertical is not None:
        candidates.append((vertical[0] / median_height, 0, vertical[1]))
    if not candidates:
        return [internal_xy_leaf_order(indexes, boxes)]

    internal_score, axis, cut = max(candidates, key=lambda item: item[0])
    centers = (region_boxes[:, axis] + region_boxes[:, axis + 2]) * 0.5
    first = indexes[centers < cut]
    second = indexes[centers >= cut]
    if not len(first) or not len(second):
        return [internal_xy_leaf_order(indexes, boxes)]
    ordered_groups = (second, first) if axis == 1 else (first, second)
    return [
        region
        for group in ordered_groups
        for region in internal_xy_cut_regions(
            group,
            boxes,
            obstacles,
            median_height,
            depth=depth + 1,
            obstacle_index=obstacle_index,
            used_obstacles=used_obstacles,
        )
    ]


def internal_block_bbox(lines: tuple[ParsedLine, ...]) -> tuple[float, float, float, float]:
    boxes = numpy.asarray(tuple(line.bbox for line in lines), dtype=numpy.float32)
    return (
        float(numpy.min(boxes[:, 0])),
        float(numpy.min(boxes[:, 1])),
        float(numpy.max(boxes[:, 2])),
        float(numpy.max(boxes[:, 3])),
    )


def internal_assign_columns(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    if len(blocks) < 2:
        return blocks
    page_x0 = min(block.bbox[0] for block in blocks)
    page_x1 = max(block.bbox[2] for block in blocks)
    page_width = max(1.0, page_x1 - page_x0)
    bands: list[list[float]] = []
    assignments: list[int | None] = []
    for block in blocks:
        x0, internal_y0, x1, internal_y1 = block.bbox
        width = x1 - x0
        if width / page_width >= 0.70:
            assignments.append(None)
            continue
        best_band: int | None = None
        best_overlap = 0.0
        for band_index, (band_x0, band_x1) in enumerate(bands):
            overlap = max(0.0, min(x1, band_x1) - max(x0, band_x0))
            overlap_ratio = overlap / max(1.0, min(width, band_x1 - band_x0))
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_band = band_index
        if best_band is None or best_overlap < 0.50:
            bands.append([x0, x1])
            assignments.append(len(bands) - 1)
        else:
            bands[best_band][0] = min(bands[best_band][0], x0)
            bands[best_band][1] = max(bands[best_band][1], x1)
            assignments.append(best_band)
    ranked_bands = {
        old_index: new_index
        for new_index, old_index in enumerate(
            sorted(range(len(bands)), key=lambda index: bands[index][0])
        )
    }
    return [
        ParsedBlock(
            lines=block.lines,
            bbox=block.bbox,
            column_index=(ranked_bands[assignment] if assignment is not None else None),
            kind=block.kind,
        )
        for block, assignment in zip(blocks, assignments, strict=True)
    ]


def internal_classify_blocks(
    blocks: list[ParsedBlock],
    *,
    body_font_size: float | None,
) -> list[ParsedBlock]:
    """Add conservative semantic roles using typography and stable text cues."""
    classified: list[ParsedBlock] = []
    heading_sizes = sorted(
        {
            line.font_size
            for block in blocks
            for line in block.lines
            if line.font_size is not None and line.font_size > 0
        },
        reverse=True,
    )
    for block in blocks:
        text = " ".join(line.text for line in block.lines)
        normalized = " ".join(text.split())
        kind = "paragraph"
        level: int | None = None
        lowered = normalized.casefold()
        if re.match(r"^(?:figure|fig\.|table|chart|exhibit)\s+\d+\b", lowered):
            kind = "caption"
        elif block.lines and all(
            re.match(r"^(?:[-*•]|\d+[.)])\s+", line.text.strip()) for line in block.lines
        ):
            kind = "list"
        elif (
            body_font_size is not None
            and len(block.lines) <= 3
            and len(normalized) <= 240
            and max((line.font_size or 0.0) for line in block.lines) >= body_font_size * 1.2
        ):
            kind = "heading"
            size = max((line.font_size or 0.0) for line in block.lines)
            level = min(3, heading_sizes.index(size) + 1) if size in heading_sizes else 1
        classified.append(replace(block, kind=kind, level=level))
    return classified


def internal_semantic_body_font_size(lines: tuple[ParsedLine, ...]) -> float | None:
    sizes = numpy.asarray(
        [line.font_size for line in lines if line.font_size is not None and line.font_size > 0],
        dtype=numpy.float32,
    )
    return float(numpy.median(sizes)) if len(sizes) else None


def layout_blocks(
    observations: ObservationBatch,
    *,
    obstacles: tuple[tuple[float, float, float, float], ...] = (),
    use_xy_cut: bool = True,
) -> tuple[ParsedBlock, ...]:
    """Reduce fused observations into geometrically ordered, structured blocks."""
    lines = internal_build_lines(observations)
    if not lines:
        return ()
    boxes = numpy.asarray(tuple(line.bbox for line in lines), dtype=numpy.float32)
    if not use_xy_cut:
        indexes = internal_row_order_indexes(
            numpy.arange(len(lines), dtype=numpy.int64),
            boxes,
        )
        blocks = [
            ParsedBlock(lines=(lines[int(index)],), bbox=lines[int(index)].bbox)
            for index in indexes
        ]
        return tuple(
            internal_classify_blocks(
                internal_assign_columns(blocks),
                body_font_size=internal_semantic_body_font_size(lines),
            )
        )
    heights = numpy.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    median_height = max(1.0, float(numpy.median(heights)))
    obstacle_index = (
        SpatialIndex(((index, obstacle) for index, obstacle in enumerate(obstacles)))
        if obstacles
        else None
    )
    regions = internal_xy_cut_regions(
        numpy.arange(len(lines), dtype=numpy.int64),
        boxes,
        obstacles,
        median_height,
        obstacle_index=obstacle_index,
    )
    blocks = [
        ParsedBlock(
            lines=tuple(lines[int(index)] for index in region),
            bbox=internal_block_bbox(tuple(lines[int(index)] for index in region)),
        )
        for region in regions
    ]
    return tuple(
        internal_classify_blocks(
            internal_assign_columns(blocks), body_font_size=internal_semantic_body_font_size(lines)
        )
    )


def layout_lines(observations: ObservationBatch) -> tuple[ParsedLine, ...]:
    return tuple(line for block in layout_blocks(observations) for line in block.lines)


def order_lines(lines: tuple[ParsedLine, ...]) -> tuple[ParsedLine, ...]:
    if len(lines) < 2:
        return lines
    boxes = numpy.asarray(tuple(line.bbox for line in lines), dtype=numpy.float32)
    heights = numpy.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    regions = internal_xy_cut_regions(
        numpy.arange(len(lines), dtype=numpy.int64),
        boxes,
        (),
        max(1.0, float(numpy.median(heights))),
    )
    return tuple(lines[int(index)] for region in regions for index in region)


def layout_element_order(
    boxes: tuple[tuple[float, float, float, float], ...],
) -> tuple[int, ...]:
    """Return reading order for arbitrary page elements represented by boxes."""
    if len(boxes) < 2:
        return tuple(range(len(boxes)))
    values = numpy.asarray(boxes, dtype=numpy.float32)
    heights = numpy.maximum(1.0, values[:, 3] - values[:, 1])
    regions = internal_xy_cut_regions(
        numpy.arange(len(boxes), dtype=numpy.int64),
        values,
        (),
        max(1.0, float(numpy.median(heights))),
    )
    return tuple(int(index) for region in regions for index in region)


def internal_has_repeated_block_columns(blocks: tuple[ParsedBlock, ...]) -> bool:
    """Identify pages whose blocks form a repeated multi-column grid."""
    bounded = tuple(block.bbox for block in blocks if block.bbox is not None)
    if len(bounded) < 6:
        return False
    top = max(box[3] for box in bounded)
    bottom = min(box[1] for box in bounded)
    cutoff = top - (top - bottom) * 0.55
    starts = sorted(box[0] for box in bounded if box[3] >= cutoff)
    if len(starts) < 6:
        return False
    clusters: list[list[float]] = []
    for start in starts:
        if clusters and start - clusters[-1][-1] <= 16.0:
            clusters[-1].append(start)
        else:
            clusters.append([start])
    return sum(len(cluster) >= 3 for cluster in clusters) >= 3


# ===== emit =====


def internal_horizontal_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    return intersection / max(1.0, min(left[2] - left[0], right[2] - right[0]))


def internal_caption_for(
    caption_blocks: tuple[Block, ...],
    target_bbox: tuple[float, float, float, float] | None,
) -> Block | None:
    if target_bbox is None:
        return None
    candidates: list[tuple[float, Block]] = []
    for caption in caption_blocks:
        if caption.bbox is None or internal_horizontal_overlap(caption.bbox, target_bbox) < 0.3:
            continue
        if caption.bbox[3] <= target_bbox[1]:
            gap = target_bbox[1] - caption.bbox[3]
        elif target_bbox[3] <= caption.bbox[1]:
            gap = caption.bbox[1] - target_bbox[3]
        else:
            continue
        caption_height = max(1.0, caption.bbox[3] - caption.bbox[1])
        if gap <= max(24.0, caption_height * 2.5):
            candidates.append((gap, caption))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def internal_attach_semantic_context(
    blocks: tuple[Block, ...],
    tables: list[Table],
    figures: list[Figure],
) -> tuple[list[Table], list[Figure]]:
    captions = tuple(block for block in blocks if block.kind is BlockKind.CAPTION)
    headings = tuple(block for block in blocks if block.kind is BlockKind.HEADING)

    def context(order: int, bbox: tuple[float, float, float, float] | None) -> dict[str, object]:
        metadata: dict[str, object] = {}
        caption = internal_caption_for(captions, bbox)
        if caption is not None:
            metadata["caption"] = caption.text
            metadata["caption_order"] = caption.order
        preceding = [
            heading
            for heading in headings
            if heading.order < order
            or (bbox is not None and heading.bbox is not None and heading.bbox[1] >= bbox[3])
        ]
        if preceding:
            heading = min(
                preceding,
                key=lambda item: (
                    abs((item.bbox or (0.0, 0.0, 0.0, 0.0))[1] - (bbox or (0.0, 0.0, 0.0, 0.0))[3]),
                    -item.order,
                ),
            )
            metadata["section"] = heading.text
            metadata["section_level"] = heading.level or 1
        return metadata

    tables = [
        replace(table, metadata={**table.metadata, **context(table.order, table.bbox)})
        for table in tables
    ]
    figures = [
        replace(figure, metadata={**figure.metadata, **context(figure.order, figure.bbox)})
        for figure in figures
    ]
    return tables, figures


internal_EMITTED_TEXT_TOKEN_RE = re.compile(r"\w+")


def internal_emitted_text_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in internal_EMITTED_TEXT_TOKEN_RE.finditer(text)]


def internal_table_text(table: Table) -> str:
    return " ".join(cell.text for row in table.rows for cell in row if cell.text)


def internal_table_character_spaced_ratio(table: Table) -> float:
    filled_texts = [cell.text.strip() for row in table.rows for cell in row if cell.text.strip()]
    if not filled_texts:
        return 0.0
    return sum(internal_character_spaced_cell(text) for text in filled_texts) / len(filled_texts)


def internal_overlapping_block_token_coverage(table: Table, blocks: list[Block]) -> float:
    if table.bbox is None:
        return 0.0
    table_tokens = internal_emitted_text_tokens(internal_table_text(table))
    if len(table_tokens) < 16:
        return 0.0
    block_tokens: list[str] = []
    for block in blocks:
        if block.bbox is None or not block.text:
            continue
        if internal_bbox_overlap_ratio(block.bbox, table.bbox) >= 0.45:
            block_tokens.extend(internal_emitted_text_tokens(block.text))
    if not block_tokens:
        return 0.0
    block_counts = Counter(block_tokens)
    matched = 0
    for token in table_tokens:
        if block_counts[token] > 0:
            matched += 1
            block_counts[token] -= 1
    return matched / len(table_tokens)


def internal_stream_table_duplicated_by_blocks(table: Table, blocks: list[Block]) -> bool:
    table_tokens = internal_emitted_text_tokens(internal_table_text(table))
    token_count = len(table_tokens)
    if token_count < 16:
        return False
    # Keep structured numeric tables even when the layout pass also emits the
    # same glyphs as paragraph text.  ``internal_remove_table_duplicate_blocks``
    # removes those overlapping blocks after this decision; dropping the table
    # here loses the structure needed by downstream consumers.
    if internal_structured_stream_table(table):
        return False
    coverage = internal_overlapping_block_token_coverage(table, blocks)
    if coverage >= 0.80:
        return True
    return token_count >= 500 and coverage >= 0.35


def internal_table_duplicated_by_blocks(table: Table, blocks: list[Block]) -> bool:
    table_tokens = internal_emitted_text_tokens(internal_table_text(table))
    if len(table_tokens) < 24:
        return False
    if internal_structured_stream_table(table):
        return False
    return internal_overlapping_block_token_coverage(table, blocks) >= 0.90


def internal_small_table_duplicated_by_page_text(table: Table, blocks: list[Block]) -> bool:
    if table.metadata.get("source") == "stream":
        return False
    table_tokens = internal_emitted_text_tokens(internal_table_text(table))
    if not 4 <= len(table_tokens) < 24:
        return False
    block_counts: Counter[str] = Counter()
    for block in blocks:
        block_counts.update(internal_emitted_text_tokens(block.text))
    if not block_counts:
        return False
    matched = 0
    for token in table_tokens:
        if block_counts[token] > 0:
            matched += 1
            block_counts[token] -= 1
    return matched / len(table_tokens) >= 0.90


def internal_covers_synthetic_chart_table(table: Table, tables: tuple[Table, ...]) -> bool:
    return any(
        other is not table
        and other.metadata.get("source") == "chart-ocr"
        and other.metadata.get("synthetic")
        and internal_table_token_coverage(other, table) >= 0.95
        for other in tables
    )


def internal_fragmented_stream_table(table: Table) -> bool:
    if table.metadata.get("source") != "stream":
        return False
    tokens = internal_emitted_text_tokens(internal_table_text(table))
    if len(tokens) < 80:
        return False
    single_character = sum(len(token) == 1 for token in tokens)
    return single_character / len(tokens) >= 0.70


def internal_noisy_stream_table(table: Table) -> bool:
    if table.metadata.get("source") != "stream":
        return False
    tokens = internal_emitted_text_tokens(internal_table_text(table))
    if len(tokens) < 16:
        return False
    single_character = sum(len(token) == 1 for token in tokens)
    wordlike = sum(
        token.isalpha() and len(token) >= 3 and any(character in "aeiou" for character in token)
        for token in tokens
    )
    return single_character / len(tokens) >= 0.55 and wordlike / len(tokens) < 0.30


def internal_remove_block_duplicate_tables(
    blocks: list[Block],
    tables: tuple[Table, ...],
) -> tuple[Table, ...]:
    if not blocks or not tables:
        return tables
    filtered: list[Table] = []
    block_boxes = tuple(block.bbox for block in blocks if block.bbox is not None and block.text)
    for table in tables:
        if (
            (
                (
                    internal_table_duplicated_by_blocks(table, blocks)
                    or internal_small_table_duplicated_by_page_text(table, blocks)
                )
                and not internal_covers_synthetic_chart_table(table, tables)
            )
            or table.metadata.get("source") == "stream"
            and table.bbox is not None
            and (
                internal_fragmented_stream_table(table)
                or internal_noisy_stream_table(table)
                or internal_covers_synthetic_chart_table(table, tables)
                or (
                    internal_table_character_spaced_ratio(table) >= 0.20
                    and any(
                        internal_bbox_overlap_ratio(block_box, table.bbox) >= 0.85
                        for block_box in block_boxes
                    )
                )
                or internal_stream_table_duplicated_by_blocks(table, blocks)
            )
        ):
            continue
        filtered.append(table)
    return tuple(filtered)


def internal_table_token_coverage(candidate: Table, reference: Table) -> float:
    candidate_tokens = internal_emitted_text_tokens(internal_table_text(candidate))
    if not candidate_tokens:
        return 0.0
    reference_counts = Counter(internal_emitted_text_tokens(internal_table_text(reference)))
    matched = 0
    for token in candidate_tokens:
        if reference_counts[token] > 0:
            matched += 1
            reference_counts[token] -= 1
    return matched / len(candidate_tokens)


def internal_remove_duplicate_tables(tables: tuple[Table, ...]) -> tuple[Table, ...]:
    filtered: list[Table] = []
    for index, table in enumerate(tables):
        tokens = internal_emitted_text_tokens(internal_table_text(table))
        if not table.metadata and 0 < len(tokens) <= 8 and set(tokens) <= {"b", "i"}:
            continue
        if len(tables) < 2:
            filtered.append(table)
            continue
        if (
            table.metadata.get("source") == "chart-ocr"
            and table.metadata.get("synthetic")
            and 0 < len(tokens) <= 24
            and any(
                other_index != index and internal_table_token_coverage(table, other) >= 0.95
                for other_index, other in enumerate(tables)
            )
        ):
            continue
        filtered.append(table)
    return tuple(filtered)


def internal_corrupt_native_block(block: Block) -> bool:
    if "native" not in block.provenance:
        return False
    tokens = internal_emitted_text_tokens(block.text)
    nonspace = [character for character in block.text if not character.isspace()]
    if not nonspace:
        return False
    alphabetic = [character for character in nonspace if character.isalpha()]
    non_latin_alphabetic = [
        character for character in alphabetic if not ("a" <= character.casefold() <= "z")
    ]
    if non_latin_alphabetic and len(non_latin_alphabetic) / len(alphabetic) >= 0.50:
        return False
    alphanumeric = sum(character.isalnum() for character in nonspace)
    if not alphanumeric:
        return False
    non_ascii = sum(ord(character) > 127 for character in nonspace)
    symbol_ratio = 1.0 - alphanumeric / len(nonspace)
    non_ascii_ratio = non_ascii / len(nonspace)
    wordlike = sum(
        token.isalpha() and len(token) >= 3 and any(character in "aeiou" for character in token)
        for token in tokens
    )
    digit_bearing = sum(any(character.isdigit() for character in token) for token in tokens)
    if len(tokens) < 24:
        return (
            wordlike == 0
            and (symbol_ratio > 0.30 or non_ascii_ratio > 0.10)
            or non_ascii_ratio > 0.02
            and symbol_ratio > 0.10
            and digit_bearing / max(1, len(tokens)) >= 0.30
        )
    if wordlike / len(tokens) >= 0.12:
        return False
    if digit_bearing / len(tokens) < 0.35:
        return False
    return symbol_ratio > 0.25 or non_ascii_ratio > 0.02


def internal_corrupt_ocr_block(block: Block) -> bool:
    if block.provenance != ("ocr",):
        return False
    text = block.text.strip()
    if not text:
        return True
    nonspace = [character for character in text if not character.isspace()]
    return len(nonspace) <= 2 and not any(character.isalnum() for character in nonspace)


def internal_remove_corrupt_native_blocks(blocks: list[Block]) -> list[Block]:
    return [
        block
        for block in blocks
        if not internal_corrupt_native_block(block) and not internal_corrupt_ocr_block(block)
    ]


def internal_block_inside_page(block: Block, width: float, height: float) -> bool:
    if block.bbox is None:
        return True
    x0, y0, x1, y1 = block.bbox
    return min(width, x1) > max(0.0, x0) and min(height, y1) > max(0.0, y0)


def internal_remove_off_page_blocks(
    blocks: list[Block], width: float, height: float
) -> list[Block]:
    return [block for block in blocks if internal_block_inside_page(block, width, height)]


def internal_remove_table_duplicate_blocks(
    blocks: list[Block], tables: tuple[Table, ...]
) -> list[Block]:
    if not blocks or not tables:
        return blocks
    table_tokens = [
        (table.bbox, set(internal_emitted_text_tokens(internal_table_text(table))))
        for table in tables
        if table.bbox is not None
    ]
    if not table_tokens:
        return blocks
    deduplicated: list[Block] = []
    for block in blocks:
        if block.bbox is None:
            deduplicated.append(block)
            continue
        block_tokens = internal_emitted_text_tokens(block.text)
        if not block_tokens:
            deduplicated.append(block)
            continue
        duplicate = False
        for table_bbox, tokens in table_tokens:
            if (
                internal_bbox_overlap_ratio(block.bbox, table_bbox) >= 0.9
                and sum(token in tokens for token in block_tokens) / len(block_tokens) >= 0.85
            ):
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(block)
    return deduplicated


def internal_remove_tiny_table_duplicate_blocks(
    blocks: list[Block],
    tables: tuple[Table, ...],
) -> list[Block]:
    if not blocks or not tables:
        return blocks
    table_token_counts: Counter[str] = Counter()
    for table in tables:
        table_token_counts.update(internal_emitted_text_tokens(internal_table_text(table)))
    if not table_token_counts:
        return blocks
    filtered: list[Block] = []
    for block in blocks:
        tokens = internal_emitted_text_tokens(block.text)
        if (
            block.provenance == ("ocr",)
            and 0 < len(tokens) <= 3
            and all(table_token_counts[token] >= count for token, count in Counter(tokens).items())
        ):
            continue
        filtered.append(block)
    return filtered


internal_ARABIC_INDIC_DIGITS = str.maketrans(
    {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)


internal_NUMERIC_PIPE_TOKEN = re.compile(r"^[($+-]?\d+(?:[.,]\d+)?[%)]?$")
internal_STANDALONE_ARTIFACT_TOKENS = frozenset({"]", "_", "□", "☐", "☒", "❖"})
internal_LINE_INITIAL_OCR_SUFFIX_FRAGMENTS = frozenset(
    {
        "able",
        "ating",
        "ducted",
        "ence",
        "ical",
        "ing",
        "lation",
        "ment",
        "ments",
        "tion",
        "tions",
        "ture",
    }
)


def internal_numeric_pipe_token(token: str) -> bool:
    return bool(internal_NUMERIC_PIPE_TOKEN.match(token.strip()))


def internal_wordlike_pipe_token(token: str) -> bool:
    letters = [character for character in token.casefold() if character.isalpha()]
    return len(letters) >= 3 and any(character in "aeiou" for character in letters)


def internal_ocr_artifact_token(token: str, line_tokens: list[str]) -> bool:
    if token in {"'", "[", "!"}:
        return True
    if (
        len(line_tokens) <= 2
        and len(token) == 2
        and token.startswith("0")
        and token[1].isdigit()
        and not any(internal_wordlike_pipe_token(line_token) for line_token in line_tokens)
    ):
        return True
    return token == "•" and not any(
        internal_wordlike_pipe_token(line_token) for line_token in line_tokens
    )


def internal_remove_line_initial_suffix_fragments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        tokens = line.split()
        if (
            len(tokens) >= 2
            and tokens[0].casefold() in internal_LINE_INITIAL_OCR_SUFFIX_FRAGMENTS
            and any(internal_wordlike_pipe_token(token) for token in tokens[1:3])
        ):
            tokens = tokens[1:]
            lines.append(" ".join(tokens))
            continue
        lines.append(line)
    return "\n".join(lines)


def internal_remove_sparse_ocr_artifacts(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        tokens = line.split()
        original_tokens = tokens
        if len(tokens) == 2 and tokens[0] == ">" and re.fullmatch(r"\d+(?:[.,]\d+)?", tokens[1]):
            lines.append(tokens[1])
            continue
        if any(internal_ocr_artifact_token(token, tokens) for token in tokens):
            tokens = [token for token in tokens if not internal_ocr_artifact_token(token, tokens)]
        if "|" not in tokens:
            lines.append(" ".join(tokens) if tokens != original_tokens else line)
            continue
        non_pipe_tokens = [token for token in tokens if token != "|"]
        if not non_pipe_tokens:
            continue
        if (
            all(internal_numeric_pipe_token(token) for token in non_pipe_tokens)
            or sum(internal_wordlike_pipe_token(token) for token in non_pipe_tokens) <= 1
        ):
            lines.append(" ".join(non_pipe_tokens))
            continue
        lines.append(line)
    return "\n".join(lines)


def internal_remove_standalone_artifact_tokens(text: str) -> str:
    return "\n".join(
        " ".join(token for token in line.split() if not internal_standalone_artifact_token(token))
        for line in text.splitlines()
    )


def internal_remove_nonword_bullet_lines(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        tokens = line.split()
        if "•" not in tokens or any(internal_wordlike_pipe_token(token) for token in tokens):
            lines.append(line)
            continue
        lines.append(" ".join(token for token in tokens if token != "•"))
    return "\n".join(lines)


def internal_standalone_artifact_token(token: str) -> bool:
    if token in internal_STANDALONE_ARTIFACT_TOKENS:
        return True
    if token == '"':
        return True
    if "�" in token:
        return True
    return ";" in token and not any(character.isalnum() for character in token)


def internal_normalize_latin_confusables(text: str) -> str:
    if not text:
        return text
    if any(token in text for token in (*internal_STANDALONE_ARTIFACT_TOKENS, ";", "�")):
        text = internal_remove_standalone_artifact_tokens(text)
    if "•" in text:
        text = internal_remove_nonword_bullet_lines(text)
    if any(fragment in text for fragment in internal_LINE_INITIAL_OCR_SUFFIX_FRAGMENTS):
        text = internal_remove_line_initial_suffix_fragments(text)
    latin_letters = sum("a" <= character.casefold() <= "z" for character in text)
    if latin_letters < 3:
        return text
    normalized = text.translate(internal_ARABIC_INDIC_DIGITS).replace("؛", "")
    normalized = re.sub(
        r"(?<=[0-9A-Za-z])Η(?=[0-9A-Za-z])",
        "H",
        normalized,
    )
    return normalized


def internal_normalize_intrusive_punctuation(text: str) -> str:
    if not text or not any(character in text for character in "!["):
        return text
    normalized = (
        re.sub(r"(?<=[0-9A-Za-z])\[(?=[0-9A-Za-z])", "", text) if text.count("[") == 1 else text
    )
    normalized = re.sub(r"(?<=[%([])!(?=\s|$)", "", normalized)
    return normalized


def internal_normalize_emitted_text(text: str, source: str) -> str:
    normalized = internal_normalize_latin_confusables(text)
    normalized = internal_normalize_intrusive_punctuation(normalized)
    if source == "native" and '"' in normalized:
        normalized = internal_remove_standalone_artifact_tokens(normalized)
    if source == "ocr":
        normalized = internal_remove_sparse_ocr_artifacts(normalized)
    return normalized


def internal_line_decoration_flags(
    line: ParsedLine,
    drawings: tuple[Any, ...],
    *,
    decoration_index: SpatialIndex[tuple[float, float, float, float]] | None = None,
) -> dict[str, bool]:
    """Infer simple text decorations from nearby, thin PDF paths."""
    if line.bbox is None:
        return {}
    x0, y0, x1, y1 = line.bbox
    line_height = max(1.0, y1 - y0)
    flags = {"underline": False, "strikeout": False}
    query = (x0, y0 - 3.0, x1, y0 + line_height * 0.75)
    if decoration_index is None:
        candidates = (
            bbox
            for drawing in drawings
            if (bbox := internal_line_decoration_bbox(drawing)) is not None
            and getattr(drawing, "kind", None) in {"fill", "fillstroke", "stroke"}
        )
    else:
        candidates = (hit.item for hit in decoration_index.candidate_hits(query))
    for bbox in candidates:
        dx0, dy0, dx1, dy1 = bbox
        width = dx1 - dx0
        height = dy1 - dy0
        if width < 2.0 or height > 2.5:
            continue
        overlap = max(0.0, min(x1, dx1) - max(x0, dx0)) / width
        if overlap < 0.75:
            continue
        center_y = (dy0 + dy1) * 0.5
        if y0 - 3.0 <= center_y <= y0 + 1.5:
            flags["underline"] = True
        elif y0 + line_height * 0.25 <= center_y <= y0 + line_height * 0.75:
            flags["strikeout"] = True
        if flags["underline"] and flags["strikeout"]:
            break
    return flags


def internal_line_decoration_bbox(drawing: Any) -> tuple[float, float, float, float] | None:
    """Return a drawing bbox, materializing path geometry at most once."""
    bbox = getattr(drawing, "bbox", None)
    if bbox is None:
        rect = getattr(drawing, "rect", None)
        bbox = rect
        if bbox is None:
            path = getattr(drawing, "path", None)
            bbox_method = getattr(path, "bbox", None)
            bbox = bbox_method() if callable(bbox_method) else None
    return rect_tuple(bbox)


def internal_line_decoration_index(
    drawings: tuple[Any, ...],
) -> SpatialIndex[tuple[float, float, float, float]]:
    """Build the broad-phase index for thin path decoration candidates once."""
    entries = []
    for drawing in drawings:
        if getattr(drawing, "kind", None) not in {"fill", "fillstroke", "stroke"}:
            continue
        bbox = internal_line_decoration_bbox(drawing)
        if bbox is None:
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width >= 2.0 and height <= 2.5:
            entries.append((bbox, bbox))
    return SpatialIndex(entries, target_cell_count=max(64, len(entries) // 8 or 1))


def internal_remove_soft_line_end_hyphens(lines: list[str]) -> list[str]:
    if len(lines) < 2:
        return lines
    cleaned = list(lines)
    for index, text in enumerate(lines[:-1]):
        current = text.rstrip()
        next_text = lines[index + 1].lstrip()
        if (
            current.endswith("-")
            and len(current) >= 3
            and current[-2].islower()
            and next_text[:1].islower()
        ):
            cleaned[index] = f"{current[:-1]}{text[len(current) :]}"
    return cleaned


def emit_page(
    parsed: ParsedPage,
    drawings: tuple[CapturedDrawing, ...] = (),
) -> Page:
    blocks: list[Block] = []
    decoration_index = internal_line_decoration_index(drawings)
    for index, parsed_block in enumerate(parsed.blocks):
        confidences = tuple(
            line.confidence
            for line in parsed_block.lines
            if line.confidence is not None and math.isfinite(line.confidence)
        )
        sources = tuple(dict.fromkeys(line.source for line in parsed_block.lines))
        normalized_line_texts = internal_remove_soft_line_end_hyphens(
            [internal_normalize_emitted_text(line.text, line.source) for line in parsed_block.lines]
        )
        decorated_lines: list[ParsedLine] = []
        for line in parsed_block.lines:
            flags = internal_line_decoration_flags(
                line,
                drawings,
                decoration_index=decoration_index,
            )
            decorated_lines.append(
                replace(
                    line,
                    underline=flags["underline"],
                    strikeout=flags["strikeout"],
                )
            )
        blocks.append(
            Block(
                order=index,
                kind=BlockKind(parsed_block.kind),
                lines=tuple(
                    TextLine(
                        text,
                        bbox=line.bbox,
                        source=line.source,
                        confidence=line.confidence,
                        contributing_sources=(line.source,),
                        bold=line.bold,
                        italic=line.italic,
                        underline=line.underline,
                        strikeout=line.strikeout,
                        mark=line.mark,
                        superscript=line.superscript,
                        subscript=line.subscript,
                        spans=line.spans,
                    )
                    for line, text in zip(decorated_lines, normalized_line_texts, strict=True)
                ),
                bbox=parsed_block.bbox,
                column_index=parsed_block.column_index,
                rotation=(parsed_block.lines[0].rotation if parsed_block.lines else 0),
                confidence=(sum(confidences) / len(confidences) if confidences else None),
                level=parsed_block.level,
                provenance=sources,
            )
        )
    blocks = internal_remove_off_page_blocks(
        internal_remove_corrupt_native_blocks(blocks),
        parsed.width,
        parsed.height,
    )
    tables = internal_remove_duplicate_tables(
        internal_remove_block_duplicate_tables(blocks, parsed.tables)
    )
    blocks = internal_remove_table_duplicate_blocks(blocks, tables)
    blocks = internal_remove_tiny_table_duplicate_blocks(blocks, tables)
    elements: list[tuple[str, object, tuple[float, float, float, float]]] = [
        ("block", block, block.bbox or (0.0, 0.0, 0.0, 0.0)) for block in blocks
    ]
    elements.extend(("table", table, table.bbox or (0.0, 0.0, 0.0, 0.0)) for table in tables)
    elements.extend(
        ("figure", figure, figure.bbox or (0.0, 0.0, 0.0, 0.0)) for figure in parsed.figures
    )
    ordered_blocks: list[Block] = []
    ordered_tables: list[Table] = []
    ordered_figures: list[Figure] = []
    element_boxes = tuple(item[2] for item in elements)
    if (
        parsed.metrics.get("full_page_image")
        and len(element_boxes) > 1
        and internal_has_repeated_block_columns(parsed.blocks)
    ):
        element_order = tuple(
            sorted(
                range(len(element_boxes)),
                key=lambda index: (-element_boxes[index][3], element_boxes[index][0]),
            )
        )
    else:
        element_order = layout_element_order(element_boxes)
    for order, index in enumerate(element_order):
        kind, element, internal_bbox = elements[index]
        if kind == "block":
            assert isinstance(element, Block)
            ordered_blocks.append(replace(element, order=order))
        elif kind == "table":
            assert isinstance(element, Table)
            ordered_tables.append(replace(element, order=order))
        else:
            assert isinstance(element, Figure)
            ordered_figures.append(replace(element, order=order))
    ordered_tables, ordered_figures = internal_attach_semantic_context(
        tuple(ordered_blocks), ordered_tables, ordered_figures
    )
    header_parts = [
        block.text
        for block in ordered_blocks
        if block.bbox is not None
        and block.bbox[3] >= parsed.height * 0.88
        and block.bbox[3] - block.bbox[1] <= parsed.height * 0.08
        and len(block.text) <= 240
    ]
    footer_parts = [
        block.text
        for block in ordered_blocks
        if block.bbox is not None
        and block.bbox[1] <= parsed.height * 0.12
        and block.bbox[3] - block.bbox[1] <= parsed.height * 0.08
        and len(block.text) <= 240
    ]
    return Page(
        page_number=parsed.page_number,
        width=parsed.width,
        height=parsed.height,
        rotation=parsed.rotation,
        blocks=tuple(ordered_blocks),
        page_class=parsed.route.value,
        base_route=parsed.route.value,
        tables=tuple(ordered_tables),
        figures=tuple(ordered_figures),
        header="\n".join(header_parts),
        footer="\n".join(footer_parts),
    )


# ===== pipeline =====


PARSED_PAGE_CACHE_KEY = "parsed_page_v4"
EMITTED_PAGE_CACHE_KEY = "emitted_page_v4"
PAGE_EXTRACTION_CACHE_KEY = "page_extraction_v2"
CAPTURED_PAGE_CACHE_KEY = "captured_page_program_v3"


def internal_plan_cache_record(plan: WorkPlan) -> dict[str, object]:
    return {
        "route": plan.route.value,
        "reason": plan.reason,
        "verify_hidden_text": plan.verify_hidden_text,
        "ocr_passes": tuple(
            {
                "name": ocr_pass.name,
                "scope": ocr_pass.scope.value,
                "scale": ocr_pass.scale,
                "modes": ocr_pass.modes,
                "tiles": ocr_pass.tiles,
                "region_columns": ocr_pass.region_columns,
                "max_regions": ocr_pass.max_regions,
                "minimum_confidence": ocr_pass.minimum_confidence,
                "run_if_characters_below": ocr_pass.run_if_characters_below,
                "minimum_utility_gain": ocr_pass.minimum_utility_gain,
                "adaptive_scale": ocr_pass.adaptive_scale,
                "minimum_characters_for_rescue": ocr_pass.minimum_characters_for_rescue,
                "character_confidence_threshold": ocr_pass.character_confidence_threshold,
                "run_if_additions_below": ocr_pass.run_if_additions_below,
                "seed_with_native": ocr_pass.seed_with_native,
                "region_first": ocr_pass.region_first,
                "preprocess": ocr_pass.preprocess,
                "pixel_budget": ocr_pass.pixel_budget,
                "include_native_text": ocr_pass.include_native_text,
                "recognize_words": ocr_pass.recognize_words,
            }
            for ocr_pass in plan.ocr_passes
        ),
    }


class internal_PageExtraction:
    """Lazily materialized extraction products for one page."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self.started = time.perf_counter()
        self.internal_preflight: PagePreflight | None = None
        self.internal_preflighted_at: float | None = None
        self.internal_capture: CapturedPage | None = None
        self.internal_captured_at: float | None = None
        self.internal_plan: WorkPlan | None = None
        self.internal_planned_at: float | None = None
        self.internal_route_mismatches: tuple[int, int, int] | None = None
        self.internal_ocr: ObservationBatch | None = None
        self.internal_recognized_at: float | None = None
        self.internal_observations: ObservationBatch | None = None
        self.internal_fused_at: float | None = None
        self.internal_tables: tuple[Table, ...] | None = None
        self.internal_tabled_at: float | None = None
        self.internal_layout: tuple[tuple[ParsedBlock, ...], str] | None = None
        self.internal_layout_finished_at: float | None = None

    def preflight(self) -> PagePreflight:
        if self.internal_preflight is None:
            self.internal_preflight = preflight_page(self.page, self.capture())
            self.internal_preflighted_at = time.perf_counter()
        return self.internal_preflight

    def capture(self) -> CapturedPage:
        if self.internal_capture is not None:
            return self.internal_capture
        cache = self.page.extraction_cache
        program = self.page.get_page_program()
        cached = cache.get(CAPTURED_PAGE_CACHE_KEY)
        if isinstance(cached, CapturedPage) and cached.program is program:
            capture = cached
        else:
            capture = internal_capture_from_program(self.page, program)
            cache[CAPTURED_PAGE_CACHE_KEY] = capture
        self.internal_capture = capture
        self.internal_captured_at = time.perf_counter()
        return capture

    def internal_mismatches_for(
        self, preflight: PagePreflight, plan: WorkPlan
    ) -> tuple[int, int, int]:
        native = int(
            preflight.recommendation.page_class is PagePreflightClass.NATIVE_TEXT
            and plan.route is not PageRoute.NATIVE
        )
        image = int(
            preflight.recommendation.page_class is PagePreflightClass.IMAGE_ONLY
            and plan.route is not PageRoute.OCR
        )
        vector = int(
            preflight.recommendation.page_class is PagePreflightClass.VECTOR_DIAGRAM
            and plan.route is PageRoute.NATIVE
            and plan.reason != "newstroke-vector-text"
        )
        return native, image, vector

    def plan(self) -> WorkPlan:
        if self.internal_plan is not None:
            return self.internal_plan
        preflight = self.preflight()
        capture = self.capture()
        plan = plan_page(capture)
        native, image, vector = self.internal_mismatches_for(preflight, plan)
        self.internal_plan = plan
        self.internal_route_mismatches = (native, image, vector)
        self.internal_planned_at = time.perf_counter()
        self.page.extraction_cache["parse_plan"] = internal_plan_cache_record(plan)
        return plan

    def ocr(self, context: TaskScope) -> ObservationBatch:
        if self.internal_ocr is None:
            self.internal_ocr = recognize_page(self.capture(), self.plan(), context)
            self.internal_recognized_at = time.perf_counter()
        return self.internal_ocr

    def observations(self, context: TaskScope) -> ObservationBatch:
        if self.internal_observations is None:
            capture = self.capture()
            self.internal_observations = fuse_observations(
                capture.observations,
                self.ocr(context),
                self.plan(),
            )
            self.internal_fused_at = time.perf_counter()
        return self.internal_observations

    def tables(self, context: TaskScope) -> tuple[Table, ...]:
        if self.internal_tables is None:
            observations = self.observations(context)
            capture = self.capture()
            # Dense schematic wiring creates large false ruled tables. Vector
            # decoders already supply text geometry, and keeping those
            # observations in normal layout preserves reading order and
            # precision while avoiding another geometric pass.
            tables: tuple[Table, ...]
            if capture.evidence.vector_text_trusted or capture.evidence.stroked_vector_text.trusted:
                tables = ()
            else:
                tables = extract_tables(capture, observations)
                chart_table = extract_chart_table(capture, observations)
                if chart_table is not None:
                    tables = (*tables, chart_table)
            self.internal_tables = tables
            self.internal_tabled_at = time.perf_counter()
        return self.internal_tables

    def internal_image_obstacles(self) -> tuple[tuple[float, float, float, float], ...]:
        capture = self.capture()
        return tuple(
            box
            for box in capture.evidence.image_boxes
            if 0.01 <= ((box[2] - box[0]) * (box[3] - box[1])) / capture.evidence.page_area < 0.65
        )

    def layout(
        self,
        context: TaskScope,
        *,
        include_table_obstacles: bool,
    ) -> tuple[ParsedBlock, ...]:
        if include_table_obstacles and self.internal_layout is not None:
            return self.internal_layout[0]
        observations = self.observations(context)
        capture = self.capture()
        table_obstacles = (
            tuple(table.bbox for table in self.tables(context) if table.bbox is not None)
            if include_table_obstacles
            else ()
        )
        use_xy_cut = not (
            capture.evidence.image_count >= 8 and 0.05 <= capture.evidence.image_area_ratio < 0.65
        )
        blocks = layout_blocks(
            observations,
            obstacles=(*table_obstacles, *self.internal_image_obstacles()),
            use_xy_cut=use_xy_cut,
        )
        if include_table_obstacles:
            self.internal_layout = (blocks, "xy-cut" if use_xy_cut else "row-order")
            self.internal_layout_finished_at = time.perf_counter()
        return blocks

    def line_records(self, context: TaskScope) -> tuple[LineRecord, ...]:
        return tuple(
            LineRecord(
                text=line.text,
                break_before=1,
                source=line.source,
                bbox=line.bbox,
                confidence=line.confidence,
                contributing_sources=(line.source,),
            )
            for line in order_lines(internal_build_lines(self.observations(context)))
        )

    def parsed_page(self, context: TaskScope) -> ParsedPage:
        cache = self.page.extraction_cache
        cached = cache.get_as(PARSED_PAGE_CACHE_KEY, ParsedPage)
        if cached is not None:
            return cached
        preflight = self.preflight()
        capture = self.capture()
        plan = self.plan()
        ocr = self.ocr(context)
        observations = self.observations(context)
        tables = self.tables(context)
        blocks = self.layout(context, include_table_obstacles=True)
        figures = (
            ()
            if capture.evidence.full_page_image
            else tuple(
                Figure(order=index, bbox=box, kind="image", metadata={"source": "capture"})
                for index, box in enumerate(capture.evidence.image_boxes)
            )
        )
        finished = self.internal_layout_finished_at or time.perf_counter()
        preflight_native, preflight_image, preflight_vector = self.internal_route_mismatches or (
            0,
            0,
            0,
        )
        ocr_diagnostics = cache.get("ocr_pass_diagnostics", ())
        capture_diagnostics = cache.get("capture_diagnostics", {})
        newstroke_diagnostics = (
            capture_diagnostics.get("newstroke", {})
            if isinstance(capture_diagnostics, dict)
            else {}
        )
        stroked_decode_diagnostics = cache.get("stroked_vector_decode", {})
        if not isinstance(stroked_decode_diagnostics, dict):
            stroked_decode_diagnostics = {}
        stroked_packed_diagnostics = cache.get("stroked_vector_packed", {})
        if not isinstance(stroked_packed_diagnostics, dict):
            stroked_packed_diagnostics = {}
        document_stroked_diagnostics = cache.get("document_stroked_glyphs", {})
        if not isinstance(document_stroked_diagnostics, dict):
            document_stroked_diagnostics = {}
        ocr_raster_pixels = sum(
            int(diagnostic.get("raster_pixels", 0))
            for diagnostic in ocr_diagnostics
            if isinstance(diagnostic, dict)
        )
        ocr_full_page_fallback = int(
            any(
                bool(diagnostic.get("full_page_fallback"))
                for diagnostic in ocr_diagnostics
                if isinstance(diagnostic, dict)
            )
        )
        layout_strategy = self.internal_layout[1] if self.internal_layout is not None else "xy-cut"
        image_cache = getattr(self.page.document, "image_cache", None)
        image_cache_stats = image_cache.stats() if image_cache is not None else None
        decoder_cache = getattr(self.page.document, "decoder_cache", {})
        decoders = tuple(decoder_cache.values()) if isinstance(decoder_cache, dict) else ()
        parsed = ParsedPage(
            page_number=int(self.page.page_number),
            width=float(self.page.width),
            height=float(self.page.height),
            rotation=int(self.page.rotation),
            route=plan.route,
            blocks=blocks,
            tables=tables,
            figures=figures,
            metrics={
                "route": plan.route.value,
                "preflight_class": preflight.recommendation.page_class.value,
                "preflight_capture": preflight.recommendation.capture,
                "preflight_ocr": preflight.recommendation.ocr,
                "page_program_seconds": (self.internal_captured_at or self.started) - self.started,
                "preflight_seconds": (self.internal_preflighted_at or self.started)
                - (self.internal_captured_at or self.started),
                "preflight_native_route_mismatch": preflight_native,
                "preflight_image_route_mismatch": preflight_image,
                "preflight_vector_route_mismatch": preflight_vector,
                "page_program_events": len(capture.program.events.sequence),
                "content_stream_passes": 1,
                "capture_product_count": 1,
                "capture_seconds": (self.internal_captured_at or self.started) - self.started,
                "planning_seconds": (self.internal_planned_at or self.started)
                - (self.internal_preflighted_at or self.internal_captured_at or self.started),
                "ocr_seconds": (self.internal_recognized_at or self.started)
                - (self.internal_planned_at or self.started),
                "fusion_seconds": (self.internal_fused_at or self.started)
                - (self.internal_recognized_at or self.started),
                "table_seconds": (self.internal_tabled_at or self.started)
                - (self.internal_fused_at or self.started),
                "layout_seconds": finished - (self.internal_tabled_at or self.started),
                "native_observations": len(capture.observations),
                "ocr_observations": len(ocr),
                "ocr_raster_pixels": ocr_raster_pixels,
                "ocr_full_page_fallback": ocr_full_page_fallback,
                "image_cache_hits": image_cache_stats.hits if image_cache_stats else 0,
                "image_cache_misses": image_cache_stats.misses if image_cache_stats else 0,
                "image_cache_evictions": image_cache_stats.evictions if image_cache_stats else 0,
                "image_cache_bytes": image_cache_stats.bytes if image_cache_stats else 0,
                "image_cache_peak_bytes": image_cache_stats.peak_bytes if image_cache_stats else 0,
                "type3_charproc_cache_hits": sum(
                    int(getattr(decoder, "type3_charproc_cache_hits", 0)) for decoder in decoders
                ),
                "type3_charproc_cache_misses": sum(
                    int(getattr(decoder, "type3_charproc_cache_misses", 0)) for decoder in decoders
                ),
                "type3_charproc_compiled_programs": sum(
                    int(getattr(decoder, "type3_charproc_compiled_programs", 0))
                    for decoder in decoders
                ),
                "type3_charproc_compiled_operations": sum(
                    int(getattr(decoder, "type3_charproc_compiled_operations", 0))
                    for decoder in decoders
                ),
                "type3_charproc_unsafe_fallbacks": sum(
                    int(getattr(decoder, "type3_charproc_unsafe_fallbacks", 0))
                    for decoder in decoders
                ),
                "fused_observations": len(observations),
                "layout_strategy": layout_strategy,
                "text_coverage": capture.evidence.text_coverage,
                "painted_text_coverage": capture.evidence.painted_text_coverage or 0.0,
                "glyph_mapped_ratio": capture.evidence.glyphs.mapped_ratio,
                "glyph_unknown_ratio": capture.evidence.glyphs.unknown_ratio,
                "trusted_hidden_text": int(capture.evidence.trusted_hidden_text),
                "vector_text_characters": capture.evidence.vector_text_characters,
                "vector_text_candidate_segments": (capture.evidence.vector_text_candidate_segments),
                "vector_text_matched_segments": capture.evidence.vector_text_matched_segments,
                "vector_text_segment_coverage": (capture.evidence.vector_text_segment_coverage),
                "vector_text_sequences": capture.evidence.vector_text_sequences,
                "vector_text_maximum_error": capture.evidence.vector_text_maximum_error,
                "vector_text_seconds": float(newstroke_diagnostics.get("seconds", 0.0)),
                "vector_text_trusted": int(capture.evidence.vector_text_trusted),
                "stroked_vector_text_trusted": int(capture.evidence.stroked_vector_text.trusted),
                "stroked_vector_candidate_paths": (
                    capture.evidence.stroked_vector_text.candidate_paths
                ),
                "stroked_vector_packed_cells": int(stroked_packed_diagnostics.get("cells", 0)),
                "stroked_vector_packed_fallback": int(
                    bool(stroked_packed_diagnostics.get("fallback_used", False))
                ),
                "stroked_vector_document_reuse": int(
                    document_stroked_diagnostics.get("role") == "reuse"
                ),
                "stroked_vector_document_alphabet": int(
                    document_stroked_diagnostics.get("alphabet_size", 0)
                ),
                "stroked_vector_decode_seconds": float(
                    stroked_decode_diagnostics.get("seconds", 0.0)
                ),
                "stroked_vector_decoded_runs": int(
                    stroked_decode_diagnostics.get("decoded_runs", 0)
                ),
                "stroked_vector_decode_additions": int(
                    stroked_decode_diagnostics.get("additions", 0)
                ),
                "stroked_vector_decode_corrections": int(
                    stroked_decode_diagnostics.get("corrections", 0)
                ),
                "stroked_vector_approximate_signatures": int(
                    stroked_decode_diagnostics.get("approximate_signatures", 0)
                ),
                "verified_hidden_text": int(
                    bool(cache.get("hidden_text_verification", {}).get("accepted", False))
                ),
                "full_page_image": capture.evidence.full_page_image,
                "uncovered_vector_area": capture.evidence.uncovered_vector_area or 0.0,
            },
        )
        cache[PARSED_PAGE_CACHE_KEY] = parsed
        cache["parse_metrics"] = parsed.metrics
        return parsed

    def emitted_page(self, context: TaskScope) -> Any:
        cache = self.page.extraction_cache
        emitted = cache.get(EMITTED_PAGE_CACHE_KEY)
        if emitted is None:
            emitted = emit_page(self.parsed_page(context), self.capture().drawings)
            cache[EMITTED_PAGE_CACHE_KEY] = emitted
        return emitted


def page_extraction(page: Any) -> internal_PageExtraction:
    cache = page.extraction_cache
    with page.internal_page_lock:
        extraction = cache.get(PAGE_EXTRACTION_CACHE_KEY)
        if not isinstance(extraction, internal_PageExtraction):
            extraction = internal_PageExtraction(page)
            cache[PAGE_EXTRACTION_CACHE_KEY] = extraction
        return extraction


def parse_page(page: Any, context: TaskScope) -> ParsedPage:
    cache = page.extraction_cache
    with page.internal_page_lock:
        cached = cache.get_as(PARSED_PAGE_CACHE_KEY, ParsedPage)
        if cached is not None:
            return cached
        return internal_parse_page_locked(page, context, cache)


def internal_emit_cached_page(page: Any, parsed: ParsedPage) -> Any:
    cache = page.extraction_cache
    with page.internal_page_lock:
        emitted = cache.get(EMITTED_PAGE_CACHE_KEY)
        if emitted is None:
            emitted = emit_page(parsed, page_extraction(page).capture().drawings)
            cache[EMITTED_PAGE_CACHE_KEY] = emitted
        return emitted


def extract_page(page: Any, context: TaskScope) -> Any:
    """Return the canonical emitted page, parsing and emitting at most once."""
    with page.internal_page_lock:
        return page_extraction(page).emitted_page(context)


def internal_parse_page_locked(page: Any, context: TaskScope, cache: Any) -> ParsedPage:
    """Run one page pipeline while its single-flight lock is held."""
    return page_extraction(page).parsed_page(context)


DOCUMENT_FONT_SEED_LIMIT = 4
DOCUMENT_FONT_SEEDS_PER_DECODER = 2
DOCUMENT_STROKED_MIN_DECODED_RUNS = 20
DOCUMENT_STROKED_MIN_RUN_COVERAGE = 0.70
DOCUMENT_STROKED_MIN_GLYPH_COVERAGE = 0.70


def internal_unknown_decoder_counts(capture: CapturedPage) -> Counter[object]:
    counts: Counter[object] = Counter()
    for glyph in capture.program.products.glyphs:
        decoder = glyph.font_decoder
        if (
            decoder is None
            or not glyph.visible
            or not glyph.code_bytes
            or internal_learned_glyph_text(glyph) is not None
            or glyph.unicode_semantics
            not in {
                GlyphUnicodeSemantics.UNKNOWN_IDENTIFIER,
                GlyphUnicodeSemantics.UNSUPPORTED,
            }
            or not callable(getattr(decoder, "install_learned_unicode", None))
        ):
            continue
        counts[decoder] += 1
    return counts


def internal_document_font_seed_indexes(captures: Sequence[CapturedPage]) -> tuple[int, ...]:
    pages_by_decoder: dict[object, list[tuple[int, int]]] = defaultdict(list)
    for page_index, capture in enumerate(captures):
        for decoder, count in internal_unknown_decoder_counts(capture).items():
            if count >= 8:
                pages_by_decoder[decoder].append((page_index, count))
    page_scores: Counter[int] = Counter()
    for entries in pages_by_decoder.values():
        if len(entries) < 2 or sum(count for _, count in entries) < 32:
            continue
        for page_index, count in sorted(entries, key=lambda item: -item[1])[
            :DOCUMENT_FONT_SEEDS_PER_DECODER
        ]:
            page_scores[page_index] += count
    return tuple(
        page_index
        for page_index, ignored_score in page_scores.most_common(DOCUMENT_FONT_SEED_LIMIT)
    )


def internal_font_mapping_votes(
    capture: CapturedPage,
    ocr: ObservationBatch,
) -> dict[object, dict[bytes, Counter[str]]]:
    votes: dict[object, dict[bytes, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    glyphs = tuple(
        glyph
        for glyph in capture.program.products.glyphs
        if glyph.visible
        and glyph.code_bytes
        and len(glyph.text) == 1
        and not glyph.text.isspace()
        and int(glyph.rotation_angle) % 360 == 0
    )
    if not glyphs:
        return votes
    for text, bbox, confidence in zip(ocr.text, ocr.bbox, ocr.confidence, strict=True):
        if not math.isfinite(float(confidence)) or float(confidence) < 90.0:
            continue
        characters = tuple(character for character in text if not character.isspace())
        if len(characters) < 3:
            continue
        x0, y0, x1, y1 = (float(value) for value in bbox)
        tolerance = max(1.0, (y1 - y0) * 0.10)
        aligned = tuple(
            sorted(
                (
                    glyph
                    for glyph in glyphs
                    if x0 - tolerance
                    <= (glyph.ink_bbox[0] + glyph.ink_bbox[2]) * 0.5
                    <= x1 + tolerance
                    and y0 - tolerance
                    <= (glyph.ink_bbox[1] + glyph.ink_bbox[3]) * 0.5
                    <= y1 + tolerance
                ),
                key=lambda glyph: (glyph.ink_bbox[1], glyph.ink_bbox[0], glyph.seqno),
            )
        )
        if len(aligned) != len(characters):
            continue
        known_pairs = tuple(
            (glyph.text.casefold(), character.casefold())
            for glyph, character in zip(aligned, characters, strict=True)
            if glyph.unicode_semantics
            in {GlyphUnicodeSemantics.AUTHORITATIVE, GlyphUnicodeSemantics.HEURISTIC}
        )
        if (
            known_pairs
            and sum(left == right for left, right in known_pairs) / len(known_pairs) < 0.8
        ):
            continue
        for glyph, character in zip(aligned, characters, strict=True):
            decoder = glyph.font_decoder
            if (
                decoder is None
                or glyph.unicode_semantics
                not in {
                    GlyphUnicodeSemantics.UNKNOWN_IDENTIFIER,
                    GlyphUnicodeSemantics.UNSUPPORTED,
                }
                or not character.isprintable()
            ):
                continue
            votes[decoder][glyph.code_bytes][character] += 1
    return votes


def internal_merge_font_mapping_votes(
    destination: dict[object, dict[bytes, Counter[str]]],
    source: dict[object, dict[bytes, Counter[str]]],
) -> None:
    for decoder, by_code in source.items():
        destination_codes = destination.setdefault(decoder, {})
        for code_bytes, counts in by_code.items():
            destination_codes.setdefault(code_bytes, Counter()).update(counts)


def internal_install_document_font_mappings(
    votes: dict[object, dict[bytes, Counter[str]]],
) -> tuple[frozenset[object], int]:
    installed_decoders: set[object] = set()
    installed_characters = 0
    for decoder, by_code in votes.items():
        mapping: dict[bytes, str] = {}
        for code_bytes, counts in by_code.items():
            if not counts:
                continue
            character, count = counts.most_common(1)[0]
            total = counts.total()
            if count >= 2 and count / total >= 0.90:
                mapping[code_bytes] = character
        installer = getattr(decoder, "install_learned_unicode", None)
        if not mapping or not callable(installer):
            continue
        additions = int(installer(mapping))
        if additions:
            installed_decoders.add(decoder)
            installed_characters += additions
    return frozenset(installed_decoders), installed_characters


def internal_refresh_learned_capture(page: Any, decoders: frozenset[object]) -> None:
    extraction = page_extraction(page)
    capture = extraction.internal_capture
    if capture is None or not any(
        glyph.font_decoder in decoders for glyph in capture.program.products.glyphs
    ):
        return
    cache = page.extraction_cache
    cache.pop(CAPTURED_PAGE_CACHE_KEY, None)
    extraction.internal_capture = internal_capture_from_program(page, capture.program)
    extraction.internal_captured_at = time.perf_counter()


def internal_prepare_document_font_mappings(
    pages: tuple[Any, ...],
    captures: tuple[CapturedPage, ...],
    context: TaskScope,
) -> tuple[int, int]:
    seed_indexes = internal_document_font_seed_indexes(captures)
    if not seed_indexes:
        return 0, 0
    ocr_by_index: dict[int, ObservationBatch] = {}
    for completed in context.map_completed(
        lambda page_index: page_extraction(pages[page_index]).ocr(context),
        seed_indexes,
        stage=WorkStage.PAGE,
    ):
        ocr_by_index[seed_indexes[completed.index]] = completed.value
    votes: dict[object, dict[bytes, Counter[str]]] = {}
    for page_index, ocr in ocr_by_index.items():
        internal_merge_font_mapping_votes(
            votes,
            internal_font_mapping_votes(captures[page_index], ocr),
        )
    installed_decoders, installed_characters = internal_install_document_font_mappings(votes)
    if not installed_decoders:
        return len(seed_indexes), 0
    seed_set = frozenset(seed_indexes)
    for page_index, page in enumerate(pages):
        if page_index not in seed_set:
            internal_refresh_learned_capture(page, installed_decoders)
    for page_index in seed_indexes:
        pages[page_index].extraction_cache["document_font_learning"] = {
            "seed_pages": len(seed_indexes),
            "installed_characters": installed_characters,
        }
    return len(seed_indexes), installed_characters


def internal_merge_document_stroked_alphabet(
    destination: dict[GlyphSignature, str],
    ambiguous: set[GlyphSignature],
    source: Iterable[tuple[GlyphSignature, str]],
) -> None:
    """Merge exact glyph mappings and permanently exclude cross-page conflicts."""
    for signature, character in source:
        if signature in ambiguous:
            continue
        if signature not in destination:
            destination[signature] = character
        elif destination[signature] != character:
            destination.pop(signature)
            ambiguous.add(signature)


def internal_document_stroked_decode_is_sufficient(decoded: StrokedTextDecode) -> bool:
    return bool(
        len(decoded.observations) >= DOCUMENT_STROKED_MIN_DECODED_RUNS
        and decoded.decoded_candidate_runs >= DOCUMENT_STROKED_MIN_DECODED_RUNS
        and decoded.candidate_run_coverage >= DOCUMENT_STROKED_MIN_RUN_COVERAGE
        and decoded.candidate_glyph_coverage >= DOCUMENT_STROKED_MIN_GLYPH_COVERAGE
    )


def internal_install_document_stroked_decode(
    page: Any,
    decoded: StrokedTextDecode,
    *,
    seconds: float,
    seed_pages: tuple[int, ...],
    alphabet_size: int,
) -> bool:
    """Install deterministic cross-page vector text as this page's zero-raster OCR result."""
    with page.internal_page_lock:
        extraction = page_extraction(page)
        if extraction.internal_ocr is not None:
            return False
        internal_install_document_stroked_decode_locked(
            page,
            extraction,
            decoded,
            seconds=seconds,
            seed_pages=seed_pages,
            alphabet_size=alphabet_size,
        )
    return True


def internal_install_document_stroked_decode_locked(
    page: Any,
    extraction: internal_PageExtraction,
    decoded: StrokedTextDecode,
    *,
    seconds: float,
    seed_pages: tuple[int, ...],
    alphabet_size: int,
) -> None:
    observations = internal_stroked_vector_decoded_batch(decoded.observations)
    candidate = internal_candidate(-1, observations)
    extraction.internal_ocr = observations
    extraction.internal_recognized_at = time.perf_counter()
    cache = page.extraction_cache
    bbox = extraction.capture().evidence.stroked_vector_text.bbox
    cache["ocr_pass_diagnostics"] = (
        {
            "name": "document-stroked-glyphs",
            "scope": OcrPassScope.STROKED_VECTOR_TEXT.value,
            "scale": 0.0,
            "modes": (),
            "recognize_words": False,
            "character_confidence_threshold": None,
            "task_count": 0,
            "raster_pixels": 0,
            "skipped_raster_pixels": 0,
            "image_text_preflight": (),
            "region_stage": "document-glyph-alphabet",
            "region_boxes": (bbox,) if bbox is not None else (),
            "skipped_region_boxes": (),
            "full_page_fallback": False,
            "elapsed_seconds": seconds,
            "render_timings": {},
            "recognition_seconds": 0.0,
            "setup_seconds": 0.0,
            "api_seconds": 0.0,
            "iterator_seconds": 0.0,
            "cleanup_seconds": 0.0,
            "candidate_seconds": 0.0,
            "recognition_statuses": (),
            "accepted_additions": len(observations),
            "adaptive_retry_scale": None,
            "adaptive_preflight": None,
            "adaptive_rescue_decision": None,
            "adaptive_rescue": None,
            "pixel_budget": 0,
            "rectangles": (),
            "selected": True,
            **candidate.metrics.as_record(),
        },
    )
    cache["stroked_vector_decode"] = {
        "seconds": seconds,
        "eligible_seeds": 0,
        "aligned_seeds": 0,
        "accepted_seeds": 0,
        "initial_signatures": decoded.initial_signatures,
        "learned_signatures": decoded.learned_signatures,
        "approximate_signatures": decoded.approximate_signatures,
        "candidate_runs": decoded.candidate_runs,
        "decoded_candidate_runs": decoded.decoded_candidate_runs,
        "candidate_run_coverage": decoded.candidate_run_coverage,
        "candidate_glyph_coverage": decoded.candidate_glyph_coverage,
        "decoded_runs": len(decoded.observations),
        "additions": len(decoded.observations),
        "corrections": 0,
        "document_reuse": True,
    }
    cache["stroked_vector_packed"] = {
        "accepted": True,
        "cells": 0,
        "raster_pixels": 0,
        "unmapped_observations": 0,
        "fallback_used": False,
        "document_reuse": True,
    }
    cache["document_stroked_glyphs"] = {
        "role": "reuse",
        "seed_pages": seed_pages,
        "alphabet_size": alphabet_size,
        "candidate_run_coverage": decoded.candidate_run_coverage,
        "candidate_glyph_coverage": decoded.candidate_glyph_coverage,
        "decoded_runs": len(decoded.observations),
        "seconds": seconds,
    }
    cache["_stroked_vector_alphabet"] = decoded.alphabet


def internal_prepare_document_stroked_mappings(
    pages: tuple[Any, ...],
    captures: tuple[CapturedPage, ...],
    context: TaskScope,
) -> tuple[int, int]:
    """OCR the richest flattened-font page, then decode compatible pages structurally."""
    indexes = tuple(
        index
        for index, capture in enumerate(captures)
        if capture.evidence.stroked_vector_text.trusted
    )
    if len(indexes) < 2:
        return 0, 0
    ordered = tuple(
        sorted(
            indexes,
            key=lambda index: (
                -captures[index].evidence.stroked_vector_text.candidate_paths,
                index,
            ),
        )
    )
    alphabet: dict[GlyphSignature, str] = {}
    ambiguous: set[GlyphSignature] = set()
    seed_indexes: list[int] = []
    reused_pages = 0
    for page_index in ordered:
        page = pages[page_index]
        extraction = page_extraction(page)
        capture = extraction.capture()
        if extraction.internal_ocr is None and alphabet:
            with page.internal_page_lock:
                extraction.plan()
            started = time.perf_counter()
            decoded = decode_stroked_text_profile_with_alphabet(
                internal_stroked_text_profile(capture),
                alphabet,
            )
            seconds = time.perf_counter() - started
            if internal_document_stroked_decode_is_sufficient(
                decoded
            ) and internal_install_document_stroked_decode(
                page,
                decoded,
                seconds=seconds,
                seed_pages=tuple(int(pages[index].page_number) for index in seed_indexes),
                alphabet_size=len(alphabet),
            ):
                reused_pages += 1
                continue

        if extraction.internal_ocr is None:
            with page.internal_page_lock:
                if extraction.internal_ocr is None:
                    extraction.ocr(context)
        learned = page.extraction_cache.get("_stroked_vector_alphabet", ())
        if isinstance(learned, tuple):
            internal_merge_document_stroked_alphabet(
                alphabet,
                ambiguous,
                cast(tuple[tuple[GlyphSignature, str], ...], learned),
            )
        seed_indexes.append(page_index)
        page.extraction_cache["document_stroked_glyphs"] = {
            "role": "seed",
            "seed_pages": tuple(int(pages[index].page_number) for index in seed_indexes),
            "alphabet_size": len(alphabet),
            "ambiguous_signatures": len(ambiguous),
        }
    return len(seed_indexes), reused_pages


def parse_document(
    document: Any,
    context: TaskScope,
    pages: Sequence[Any],
) -> Document:
    pages = tuple(pages)
    parsed_pages: tuple[ParsedPage, ...]
    if len(pages) == 1:
        parsed_pages = (parse_page(pages[0], context),)
    else:
        captures_by_index: list[CapturedPage | None] = [None] * len(pages)
        for capture_completed in context.map_completed(
            lambda page: page_extraction(page).capture(),
            pages,
            stage=WorkStage.PAGE,
        ):
            captures_by_index[capture_completed.index] = capture_completed.value
        captures = tuple(capture for capture in captures_by_index if capture is not None)
        if len(captures) == len(pages):
            internal_prepare_document_font_mappings(pages, captures, context)
            internal_prepare_document_stroked_mappings(pages, captures, context)
        parsed_by_index: list[ParsedPage | None] = [None] * len(pages)
        for parse_completed in context.map_completed(
            lambda page: parse_page(page, context),
            pages,
            stage=WorkStage.PAGE,
        ):
            parsed_by_index[parse_completed.index] = parse_completed.value
        parsed_pages = tuple(page for page in parsed_by_index if page is not None)
    diagnostics = tuple(
        Diagnostic("parse", message, page_number=page.page_number)
        for page in parsed_pages
        for message in page.diagnostics
    )
    metadata = document.get_metadata()
    return Document(
        pages=tuple(page_extraction(source_page).emitted_page(context) for source_page in pages),
        metadata=metadata,
        diagnostics=diagnostics,
        schema_version="3.0",
    )


__all__ = ("extract_page", "page_extraction", "parse_document", "parse_page")

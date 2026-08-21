# SPDX-License-Identifier: AGPL-3.0-only
"""Core dataclasses shared by every extraction stage."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, cast

import numpy

from core_pdf.impl.engine.layout.models import TextRun
from core_pdf.impl.engine.spec.s_07_content.capture import CapturedDrawing, CapturedInlineImage
from core_pdf.impl.engine.spec.s_07_content.operations import (
    ContentOperatorCounts,
)
from core_pdf.impl.engine.spec.s_07_content.page_program import PageProgram
from core_pdf.impl.engine.structured import (
    TextSpan,
)

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
internal_OCR_RESCUE_DENSE_MIN_CHARACTERS = 2_000
internal_OCR_RESCUE_DENSE_MIN_CONFIDENCE = 92.0
OCR_RESCUE_LARGE_TEXT_HEIGHT = 32.0
OCR_PARALLEL_TILE_MIN_VECTOR_COMPLEXITY = 100_000
HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE = 80.0
HIDDEN_TEXT_VERIFY_MIN_MATCHED_TOKENS = 24
# Promoting a hidden layer replaces recognition entirely, so a borderline
# match is worse than re-recognizing: an archive-era OCR layer that agrees
# with the preview on only two words in three reads as its own document.
# 0.72 keeps genuinely faithful layers and sends the mediocre ones to OCR.
HIDDEN_TEXT_VERIFY_MIN_TOKEN_OVERLAP = 0.72
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


@dataclass(frozen=True, slots=True)
class TextAnalysis:
    quality: TextQualityStats = field(default_factory=TextQualityStats)
    characters: int = 0
    suspicious_characters: int = 0


def internal_analyze_text(text: str) -> TextAnalysis:
    tokens = text.split()
    if not tokens:
        return TextAnalysis()
    wordlike = 0
    short_tokens = 0
    digit_tokens = 0
    nonspace = 0
    symbols = 0
    non_ascii = 0
    suspicious = 0
    for token in tokens:
        # `tokens` comes from a `\S+` regex match, so every character in `token` already
        # satisfies `not character.isspace()` (CPython's Unicode `\s`/`str.isspace()` share
        # the same whitespace classification) -- iterate `token` directly instead of
        # rebuilding an always-identical filtered copy.
        if len(token) <= 2:
            short_tokens += 1
        has_digit = False
        letter_count = 0
        has_vowel = False
        for character in token:
            codepoint = ord(character)
            nonspace += 1
            if character.isdigit():
                has_digit = True
            if character.isalpha():
                letter_count += 1
                if not has_vowel and character.casefold() in "aeiou":
                    has_vowel = True
            if not character.isalnum():
                symbols += 1
            if codepoint > 127:
                non_ascii += 1
            if (
                character == "\ufffd"
                or 0xE000 <= codepoint <= 0xF8FF
                or (not character.isprintable() and not character.isspace())
            ):
                suspicious += 1
        if has_digit:
            digit_tokens += 1
        if letter_count >= 3 and has_vowel:
            wordlike += 1
    if not nonspace:
        return TextAnalysis(TextQualityStats(token_count=len(tokens)))
    return TextAnalysis(
        quality=TextQualityStats(
            token_count=len(tokens),
            wordlike_ratio=wordlike / len(tokens),
            short_token_ratio=short_tokens / len(tokens),
            symbol_ratio=symbols / nonspace,
            non_ascii_ratio=non_ascii / nonspace,
            digit_token_ratio=digit_tokens / len(tokens),
        ),
        characters=nonspace,
        suspicious_characters=suspicious,
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
class ReadingOrderEvidence:
    """Explain how geometry changed authored line order on one page."""

    line_count: int
    source_inversions: int
    source_inversion_ratio: float
    column_count: int
    rotation_count: int
    repaired: bool
    ambiguous: bool
    confidence: float
    strategy: str


@dataclass(frozen=True, slots=True)
class ParsedPage:
    page_number: int
    width: float
    height: float
    rotation: int
    route: PageRoute
    blocks: tuple[ParsedBlock, ...]
    tables: tuple[Any, ...] = ()
    structured_tables: tuple[Any, ...] = ()
    figures: tuple[Any, ...] = ()
    diagnostics: tuple[str, ...] = ()
    metrics: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def lines(self) -> tuple[ParsedLine, ...]:
        return tuple(line for block in self.blocks for line in block.lines)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


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

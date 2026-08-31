# SPDX-License-Identifier: AGPL-3.0-only
"""Core dataclasses shared by every extraction stage."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import IntEnum, StrEnum
from typing import Any, NamedTuple, cast

import numpy

from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing, CapturedInlineImage
from core_pdf.impl.spec.s_07_content.page_program import PageProgram
from core_pdf.impl.structured.model import (
    TextSpan,
)

# Tesseract page-segmentation modes. Shared stage vocabulary: route.py chooses a
# mode, ocr.py applies it, so neither owns the constants.
PSM_AUTO = 3
PSM_SPARSE_TEXT = 11
PSM_SPARSE_TEXT_OSD = 12

# Drawing paint kinds that put ink on the page. capture.py classifies with these,
# ocr.py filters with them.
VECTOR_PAINT_KINDS = frozenset({"fill", "fillstroke", "shading", "stroke"})


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
internal_OCR_RESCUE_DENSE_MIN_CHARACTERS: int = 2_000
internal_OCR_RESCUE_DENSE_MIN_CONFIDENCE: float = 92.0


def internal_bbox_tuple(row: object) -> tuple[float, float, float, float]:
    """Narrow one four-column NumPy bbox row without allocating an intermediate list."""
    values = cast(numpy.ndarray[Any, Any], row)
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


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
        texts = tuple(text)
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
            isinstance(indexes, (list, tuple, range))
            and len(indexes) == len(self)
            and all(int(cast(Any, index)) == position for position, index in enumerate(indexes))
        ):
            # A no-op selection is already immutable and does not need a
            # second allocation of every column.
            return self
        indexes = numpy.asarray(indexes, dtype=numpy.int64)
        if not len(indexes):
            return ObservationBatch.empty()
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
        selected = int(numpy.count_nonzero(mask))
        if selected == len(self):
            return self
        if selected == 0:
            return ObservationBatch.empty()
        return self.take(numpy.flatnonzero(mask))

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


@dataclass(frozen=True, slots=True)
class TextAnalysis:
    quality: TextQualityStats = field(default_factory=TextQualityStats)
    characters: int = 0
    suspicious_characters: int = 0


internal_ASCII_VOWELS = frozenset("aeiouAEIOU")


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
        if token.isascii() and token.isprintable():
            # Printable ASCII contributes no non-ASCII or suspicious counts,
            # and the common all-letter / all-digit tokens resolve with
            # whole-string C checks instead of six method calls per character.
            nonspace += len(token)
            if token.isalpha():
                if len(token) >= 3 and not internal_ASCII_VOWELS.isdisjoint(token):
                    wordlike += 1
                continue
            if token.isdigit():
                digit_tokens += 1
                continue
            has_digit = False
            letter_count = 0
            has_vowel = False
            for character in token:
                if character.isalnum():
                    if character.isdigit():
                        has_digit = True
                    else:
                        letter_count += 1
                        if not has_vowel and character in "aeiouAEIOU":
                            has_vowel = True
                else:
                    symbols += 1
            if has_digit:
                digit_tokens += 1
            if letter_count >= 3 and has_vowel:
                wordlike += 1
            continue
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
    semantic_characters: int = 0
    authoritative_glyphs: int = 0
    heuristic_glyphs: int = 0
    unknown_glyphs: int = 0
    unsupported_glyphs: int = 0
    low_confidence_glyphs: int = 0
    actual_text_characters: int = 0

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


@dataclass(frozen=True, slots=True)
class StrokedVectorTextEvidence:
    """Compact path families that are likely to be flattened single-line text."""

    trusted: bool = False
    drawing_indexes: tuple[int, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    candidate_paths: int = 0


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
    image_filters: tuple[str, ...] = ()
    text_coverage: float = 0.0
    full_page_image: bool = False
    uncovered_vector_area: float | None = None
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
    newstroke_report: Mapping[str, object] = field(default_factory=dict)


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

    def as_record(self) -> dict[str, object]:
        return {
            "route": self.route.value,
            "reason": self.reason.value,
            "verify_hidden_text": self.verify_hidden_text,
            "fusion_policy": self.fusion_policy.value,
            "allow_direct_image_ocr": self.allow_direct_image_ocr,
            "augment_page_candidates": self.augment_page_candidates,
            "ocr_passes": tuple(
                {
                    item.name: (value.value if isinstance(value, StrEnum) else value)
                    for item in fields(ocr_pass)
                    for value in (getattr(ocr_pass, item.name),)
                }
                for ocr_pass in self.ocr_passes
            ),
        }


@dataclass(frozen=True, slots=True)
class RecognitionReport:
    """One explicit diagnostic product returned by the recognition stage."""

    passes: tuple[Mapping[str, object], ...] = ()
    candidates: tuple[Mapping[str, object], ...] = ()
    candidate_analysis: tuple[Mapping[str, object], ...] = ()
    hidden_text_verification: Mapping[str, object] = field(default_factory=dict)
    stroked_vector_decode: Mapping[str, object] = field(default_factory=dict)
    stroked_vector_packed: Mapping[str, object] = field(default_factory=dict)
    document_stroked_glyphs: Mapping[str, object] = field(default_factory=dict)
    render_timings: Mapping[str, object] = field(default_factory=dict)
    grid_cell_ocr: Mapping[str, object] = field(default_factory=dict)
    render_error: str | None = None
    stroked_vector_alphabet: tuple[tuple[Any, str], ...] = ()

    def as_record(self) -> dict[str, object]:
        return {
            "passes": self.passes,
            "candidates": self.candidates,
            "candidate_analysis": self.candidate_analysis,
            "hidden_text_verification": dict(self.hidden_text_verification),
            "stroked_vector_decode": dict(self.stroked_vector_decode),
            "stroked_vector_packed": dict(self.stroked_vector_packed),
            "document_stroked_glyphs": dict(self.document_stroked_glyphs),
            "render_timings": dict(self.render_timings),
            "grid_cell_ocr": dict(self.grid_cell_ocr),
            "render_error": self.render_error,
        }


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    observations: ObservationBatch
    report: RecognitionReport = field(default_factory=RecognitionReport)


MetricValue = float | int | str | bool


@dataclass(frozen=True, slots=True)
class ParseReport:
    """Typed analysis product for one completed page pipeline."""

    plan: WorkPlan
    recognition: RecognitionReport = field(default_factory=RecognitionReport)
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)

    def as_record(self) -> dict[str, object]:
        return {
            "plan": self.plan.as_record(),
            "recognition": self.recognition.as_record(),
            "metrics": dict(self.metrics),
        }


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
    figures: tuple[Any, ...] = ()
    diagnostics: tuple[str, ...] = ()
    full_page_image: bool = False
    report: ParseReport | None = None

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


class internal_TextUtility(NamedTuple):
    nonspace: int
    alphanumeric: int
    utility: float


def internal_text_utility_stats(text: str, confidence: float) -> internal_TextUtility:
    """Return non-space count, alphanumeric count, and utility in one character scan."""
    # str.split() removes exactly the isspace() characters, and Counter over
    # a map counts in C; per-character casefold keeps expanding folds (one
    # count under a multi-character key) identical to the previous loop.
    stripped = "".join(text.split())
    nonspace = len(stripped)
    if not nonspace:
        return internal_TextUtility(0, 0, 0.0)
    alphanumeric = sum(map(str.isalnum, stripped))
    counts = Counter(map(str.casefold, stripped))
    symbols = nonspace - alphanumeric
    symbol_credit = min(symbols, max(2.0, alphanumeric * 0.5)) * 0.30
    confidence_factor = 0.25 + 0.75 * min(100.0, max(0.0, confidence)) / 100.0
    repetition_penalty = 1.0
    if nonspace >= 6:
        dominant_ratio = max(counts.values()) / nonspace
        if dominant_ratio > 0.60:
            repetition_penalty = max(0.20, 1.0 - (dominant_ratio - 0.60) * 2.0)
    utility = (alphanumeric + symbol_credit) * confidence_factor * repetition_penalty
    return internal_TextUtility(nonspace, alphanumeric, utility)


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
            tokens=tokens,
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

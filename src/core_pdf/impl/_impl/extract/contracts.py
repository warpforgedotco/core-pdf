# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy

from core_pdf.impl._impl.capture.program import PageProgram
from core_pdf.impl._impl.output.model import (
    TextLine,
)
from core_pdf.impl._impl.runtime.array_views import readonly

if TYPE_CHECKING:
    from core_pdf.impl._impl.document.page import PdfPage
    from core_pdf.impl._impl.document.records import RawAnnotation, RawFormField

FloatArray = numpy.ndarray[Any, numpy.dtype[numpy.float32]]
IntArray = numpy.ndarray[Any, numpy.dtype[numpy.int64]]
ByteArray = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]
BoolArray = numpy.ndarray[Any, numpy.dtype[numpy.bool_]]


def internal_column(
    values: Iterable[Any] | None,
    dtype: Any,
    default: Callable[[], numpy.ndarray[Any, Any]] | None = None,
) -> numpy.ndarray[Any, Any]:
    if values is None:
        if default is None:
            raise ValueError("observation column is required")
        return default()
    return numpy.array(
        values if isinstance(values, (list, tuple, range, numpy.ndarray)) else tuple(values),
        dtype=dtype,
    )


def internal_validate_selection_mask(mask: BoolArray, size: int) -> None:
    if mask.dtype != numpy.bool_:
        raise TypeError("observation selection mask must have boolean dtype")
    if mask.shape != (size,):
        raise ValueError("observation selection mask must have shape (n,)")


def internal_bbox_tuple(row: Any) -> tuple[float, float, float, float]:
    return (float(row[0]), float(row[1]), float(row[2]), float(row[3]))


FULL_PAGE_IMAGE_COVERAGE = 0.90


class ObservationSource(IntEnum):
    NATIVE = 0
    STRUCTURE = 2


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    text: tuple[str, ...]
    bbox: FloatArray
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
        if len(self.references) != size:
            raise ValueError("observation references must match the text column")
        columns = (
            ("bbox", self.bbox, (size, 4), numpy.float32),
            ("source", self.source, (size,), numpy.uint8),
            ("confidence", self.confidence, (size,), numpy.float32),
            ("sequence", self.sequence, (size,), numpy.int64),
            ("visible", self.visible, (size,), numpy.bool_),
            ("rotation", self.rotation, (size,), numpy.int64),
            ("font_size", self.font_size, (size,), numpy.float32),
            ("line_break_before", self.line_break_before, (size,), numpy.bool_),
        )
        for name, column, shape, dtype in columns:
            if column.shape != shape:
                raise ValueError(f"observation {name} must have shape {shape}")
            if column.dtype != dtype:
                raise TypeError(f"observation {name} must have dtype {numpy.dtype(dtype)}")
        for _, column, _, _ in columns:
            readonly(column)

    def __len__(self) -> int:
        return len(self.text)

    @classmethod
    def empty(cls) -> ObservationBatch:
        return cls(
            (),
            numpy.empty((0, 4), dtype=numpy.float32),
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
        source: int,
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
        boxes = internal_column(bbox, numpy.float32)
        if size == 0 and boxes.shape == (0,):
            boxes = boxes.reshape((0, 4))
        conf_arr = internal_column(
            confidence, numpy.float32, lambda: numpy.full(size, numpy.nan, dtype=numpy.float32)
        )
        seq_arr = internal_column(
            sequence, numpy.int64, lambda: numpy.arange(size, dtype=numpy.int64)
        )
        vis_arr = internal_column(visible, numpy.bool_, lambda: numpy.ones(size, dtype=numpy.bool_))
        rot_arr = internal_column(
            rotation, numpy.int64, lambda: numpy.zeros(size, dtype=numpy.int64)
        )
        font_arr = internal_column(
            font_size, numpy.float32, lambda: numpy.full(size, numpy.nan, dtype=numpy.float32)
        )
        line_arr = internal_column(
            line_break_before, numpy.bool_, lambda: numpy.zeros(size, dtype=numpy.bool_)
        )
        ref_tuple = (
            (references if isinstance(references, tuple) else tuple(references))
            if references is not None
            else (None,) * size
        )
        return cls(
            texts,
            boxes,
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
        if isinstance(indexes, (list, tuple, range)) and not len(indexes):
            return self if not len(self) else ObservationBatch.empty()
        indexes = numpy.asarray(indexes)
        if indexes.ndim != 1:
            raise ValueError("observation indexes must be one-dimensional")
        if indexes.dtype.kind not in "iu":
            raise TypeError("observation indexes must be integers")
        if numpy.any(indexes >= len(self)) or numpy.any(indexes < -len(self)):
            raise IndexError("observation index out of range")
        if not len(indexes):
            return self if not len(self) else ObservationBatch.empty()
        if numpy.array_equal(indexes, numpy.arange(len(self))):
            return self
        return ObservationBatch(
            tuple(self.text[int(index)] for index in indexes),
            self.bbox[indexes],
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
        internal_validate_selection_mask(mask, len(self))
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
        internal_validate_selection_mask(secondary_mask, len(secondary))
        if not len(primary) and bool(numpy.all(secondary_mask)):
            return secondary
        indexes = numpy.flatnonzero(secondary_mask)
        if not len(primary):
            return secondary.take(indexes)
        if not len(indexes):
            return primary
        size = len(primary) + len(indexes)
        split = len(primary)

        def combine(
            primary_column: numpy.ndarray[Any, Any],
            secondary_column: numpy.ndarray[Any, Any],
        ) -> numpy.ndarray[Any, Any]:
            shape = (size, *secondary_column.shape[1:])
            result = numpy.empty(shape, dtype=secondary_column.dtype)
            result[:split] = primary_column
            result[split:] = secondary_column[indexes]
            return result

        return cls(
            (*primary.text, *(secondary.text[int(index)] for index in indexes)),
            combine(primary.bbox, secondary.bbox),
            combine(primary.source, secondary.source),
            combine(primary.confidence, secondary.confidence),
            combine(primary.sequence, secondary.sequence),
            combine(primary.visible, secondary.visible),
            combine(primary.rotation, secondary.rotation),
            combine(primary.font_size, secondary.font_size),
            combine(primary.line_break_before, secondary.line_break_before),
            (*primary.references, *(secondary.references[int(index)] for index in indexes)),
        )


@dataclass(frozen=True, slots=True)
class TextQualityStats:
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
class GlyphEvidence:
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
class PageEvidence:
    page_area: float
    native_characters: int
    visible_native_characters: int
    suspicious_characters: int
    image_count: int
    image_area_ratio: float
    image_boxes: tuple[tuple[float, float, float, float], ...] = ()
    text_coverage: float = 0.0
    full_page_image: bool = False
    text_quality: TextQualityStats = field(default_factory=TextQualityStats)
    all_text_quality: TextQualityStats = field(default_factory=TextQualityStats)
    glyphs: GlyphEvidence = field(default_factory=GlyphEvidence)
    painted_native_characters: int | None = None
    trusted_hidden_text: bool = False

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


@dataclass(frozen=True, slots=True)
class PageAnalysis:
    page: PdfPage
    width: float
    height: float
    rotation: int
    fields: tuple[RawFormField, ...]
    annotations: tuple[RawAnnotation, ...]
    program: PageProgram
    observations: ObservationBatch
    evidence: PageEvidence


@dataclass(frozen=True, slots=True)
class ParsedLine:
    line: TextLine
    sequence: int = 0
    rotation: int = 0
    font_size: float | None = None

    def __post_init__(self) -> None:
        if self.line.bbox is None:
            raise ValueError("ParsedLine requires a positioned line")


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    lines: tuple[ParsedLine, ...]
    bbox: tuple[float, float, float, float]
    column_index: int | None = None
    kind: str = "paragraph"
    level: int | None = None


@dataclass(frozen=True, slots=True)
class ReadingOrderEvidence:
    line_count: int
    source_inversions: int
    source_inversion_ratio: float
    column_count: int
    rotation_count: int
    repaired: bool
    ambiguous: bool
    confidence: float
    strategy: str

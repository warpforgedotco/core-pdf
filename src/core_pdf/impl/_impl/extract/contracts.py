# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Self

import numpy

from core_pdf.impl._impl.capture.program import PageProgram
from core_pdf.impl._impl.output.model import (
    TextLine,
)
from core_pdf.impl._impl.runtime.array_views import readonly

if TYPE_CHECKING:
    from core_pdf.impl._impl.document.page import PdfPage
    from core_pdf.impl._impl.document.records import RawAnnotation, RawFormField

internal_frozen_setattr = object.__setattr__


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


class ObservationBatch:
    __slots__ = (
        "text",
        "bbox",
        "source",
        "confidence",
        "sequence",
        "visible",
        "rotation",
        "font_size",
        "line_break_before",
        "references",
    )

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

    __fields__: ClassVar[tuple[str, ...]] = (
        "text",
        "bbox",
        "source",
        "confidence",
        "sequence",
        "visible",
        "rotation",
        "font_size",
        "line_break_before",
        "references",
    )
    __match_args__ = (
        "text",
        "bbox",
        "source",
        "confidence",
        "sequence",
        "visible",
        "rotation",
        "font_size",
        "line_break_before",
        "references",
    )

    def __init__(
        self,
        text: tuple[str, ...],
        bbox: FloatArray,
        source: ByteArray,
        confidence: FloatArray,
        sequence: IntArray,
        visible: BoolArray,
        rotation: IntArray,
        font_size: FloatArray,
        line_break_before: BoolArray,
        references: tuple[Any | None, ...],
    ) -> None:
        internal_frozen_setattr(self, "text", text)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "source", source)
        internal_frozen_setattr(self, "confidence", confidence)
        internal_frozen_setattr(self, "sequence", sequence)
        internal_frozen_setattr(self, "visible", visible)
        internal_frozen_setattr(self, "rotation", rotation)
        internal_frozen_setattr(self, "font_size", font_size)
        internal_frozen_setattr(self, "line_break_before", line_break_before)
        internal_frozen_setattr(self, "references", references)
        self._post_init()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"text={self.text!r}, "
            f"bbox={self.bbox!r}, "
            f"source={self.source!r}, "
            f"confidence={self.confidence!r}, "
            f"sequence={self.sequence!r}, "
            f"visible={self.visible!r}, "
            f"rotation={self.rotation!r}, "
            f"font_size={self.font_size!r}, "
            f"line_break_before={self.line_break_before!r}, "
            f"references={self.references!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.text == other.text
            and self.bbox == other.bbox
            and self.source == other.source
            and self.confidence == other.confidence
            and self.sequence == other.sequence
            and self.visible == other.visible
            and self.rotation == other.rotation
            and self.font_size == other.font_size
            and self.line_break_before == other.line_break_before
            and self.references == other.references
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.text,
                self.bbox,
                self.source,
                self.confidence,
                self.sequence,
                self.visible,
                self.rotation,
                self.font_size,
                self.line_break_before,
                self.references,
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
        text = changes.pop("text", self.text)
        bbox = changes.pop("bbox", self.bbox)
        source = changes.pop("source", self.source)
        confidence = changes.pop("confidence", self.confidence)
        sequence = changes.pop("sequence", self.sequence)
        visible = changes.pop("visible", self.visible)
        rotation = changes.pop("rotation", self.rotation)
        font_size = changes.pop("font_size", self.font_size)
        line_break_before = changes.pop("line_break_before", self.line_break_before)
        references = changes.pop("references", self.references)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            text,
            bbox,
            source,
            confidence,
            sequence,
            visible,
            rotation,
            font_size,
            line_break_before,
            references,
        )

    def _post_init(self) -> None:
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


class TextQualityStats:
    __slots__ = (
        "token_count",
        "wordlike_ratio",
        "short_token_ratio",
        "symbol_ratio",
        "non_ascii_ratio",
        "digit_token_ratio",
    )

    token_count: int
    wordlike_ratio: float
    short_token_ratio: float
    symbol_ratio: float
    non_ascii_ratio: float
    digit_token_ratio: float

    __fields__: ClassVar[tuple[str, ...]] = (
        "token_count",
        "wordlike_ratio",
        "short_token_ratio",
        "symbol_ratio",
        "non_ascii_ratio",
        "digit_token_ratio",
    )
    __match_args__ = (
        "token_count",
        "wordlike_ratio",
        "short_token_ratio",
        "symbol_ratio",
        "non_ascii_ratio",
        "digit_token_ratio",
    )

    def __init__(
        self,
        token_count: int = 0,
        wordlike_ratio: float = 0.0,
        short_token_ratio: float = 0.0,
        symbol_ratio: float = 0.0,
        non_ascii_ratio: float = 0.0,
        digit_token_ratio: float = 0.0,
    ) -> None:
        internal_frozen_setattr(self, "token_count", token_count)
        internal_frozen_setattr(self, "wordlike_ratio", wordlike_ratio)
        internal_frozen_setattr(self, "short_token_ratio", short_token_ratio)
        internal_frozen_setattr(self, "symbol_ratio", symbol_ratio)
        internal_frozen_setattr(self, "non_ascii_ratio", non_ascii_ratio)
        internal_frozen_setattr(self, "digit_token_ratio", digit_token_ratio)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"token_count={self.token_count!r}, "
            f"wordlike_ratio={self.wordlike_ratio!r}, "
            f"short_token_ratio={self.short_token_ratio!r}, "
            f"symbol_ratio={self.symbol_ratio!r}, "
            f"non_ascii_ratio={self.non_ascii_ratio!r}, "
            f"digit_token_ratio={self.digit_token_ratio!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.token_count == other.token_count
            and self.wordlike_ratio == other.wordlike_ratio
            and self.short_token_ratio == other.short_token_ratio
            and self.symbol_ratio == other.symbol_ratio
            and self.non_ascii_ratio == other.non_ascii_ratio
            and self.digit_token_ratio == other.digit_token_ratio
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.token_count,
                self.wordlike_ratio,
                self.short_token_ratio,
                self.symbol_ratio,
                self.non_ascii_ratio,
                self.digit_token_ratio,
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
        token_count = changes.pop("token_count", self.token_count)
        wordlike_ratio = changes.pop("wordlike_ratio", self.wordlike_ratio)
        short_token_ratio = changes.pop("short_token_ratio", self.short_token_ratio)
        symbol_ratio = changes.pop("symbol_ratio", self.symbol_ratio)
        non_ascii_ratio = changes.pop("non_ascii_ratio", self.non_ascii_ratio)
        digit_token_ratio = changes.pop("digit_token_ratio", self.digit_token_ratio)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            token_count,
            wordlike_ratio,
            short_token_ratio,
            symbol_ratio,
            non_ascii_ratio,
            digit_token_ratio,
        )

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


class GlyphEvidence:
    __slots__ = (
        "glyph_count",
        "semantic_characters",
        "authoritative_glyphs",
        "heuristic_glyphs",
        "unknown_glyphs",
        "unsupported_glyphs",
        "low_confidence_glyphs",
        "actual_text_characters",
    )

    glyph_count: int
    semantic_characters: int
    authoritative_glyphs: int
    heuristic_glyphs: int
    unknown_glyphs: int
    unsupported_glyphs: int
    low_confidence_glyphs: int
    actual_text_characters: int

    __fields__: ClassVar[tuple[str, ...]] = (
        "glyph_count",
        "semantic_characters",
        "authoritative_glyphs",
        "heuristic_glyphs",
        "unknown_glyphs",
        "unsupported_glyphs",
        "low_confidence_glyphs",
        "actual_text_characters",
    )
    __match_args__ = (
        "glyph_count",
        "semantic_characters",
        "authoritative_glyphs",
        "heuristic_glyphs",
        "unknown_glyphs",
        "unsupported_glyphs",
        "low_confidence_glyphs",
        "actual_text_characters",
    )

    def __init__(
        self,
        glyph_count: int = 0,
        semantic_characters: int = 0,
        authoritative_glyphs: int = 0,
        heuristic_glyphs: int = 0,
        unknown_glyphs: int = 0,
        unsupported_glyphs: int = 0,
        low_confidence_glyphs: int = 0,
        actual_text_characters: int = 0,
    ) -> None:
        internal_frozen_setattr(self, "glyph_count", glyph_count)
        internal_frozen_setattr(self, "semantic_characters", semantic_characters)
        internal_frozen_setattr(self, "authoritative_glyphs", authoritative_glyphs)
        internal_frozen_setattr(self, "heuristic_glyphs", heuristic_glyphs)
        internal_frozen_setattr(self, "unknown_glyphs", unknown_glyphs)
        internal_frozen_setattr(self, "unsupported_glyphs", unsupported_glyphs)
        internal_frozen_setattr(self, "low_confidence_glyphs", low_confidence_glyphs)
        internal_frozen_setattr(self, "actual_text_characters", actual_text_characters)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"glyph_count={self.glyph_count!r}, "
            f"semantic_characters={self.semantic_characters!r}, "
            f"authoritative_glyphs={self.authoritative_glyphs!r}, "
            f"heuristic_glyphs={self.heuristic_glyphs!r}, "
            f"unknown_glyphs={self.unknown_glyphs!r}, "
            f"unsupported_glyphs={self.unsupported_glyphs!r}, "
            f"low_confidence_glyphs={self.low_confidence_glyphs!r}, "
            f"actual_text_characters={self.actual_text_characters!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.glyph_count == other.glyph_count
            and self.semantic_characters == other.semantic_characters
            and self.authoritative_glyphs == other.authoritative_glyphs
            and self.heuristic_glyphs == other.heuristic_glyphs
            and self.unknown_glyphs == other.unknown_glyphs
            and self.unsupported_glyphs == other.unsupported_glyphs
            and self.low_confidence_glyphs == other.low_confidence_glyphs
            and self.actual_text_characters == other.actual_text_characters
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.glyph_count,
                self.semantic_characters,
                self.authoritative_glyphs,
                self.heuristic_glyphs,
                self.unknown_glyphs,
                self.unsupported_glyphs,
                self.low_confidence_glyphs,
                self.actual_text_characters,
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
        glyph_count = changes.pop("glyph_count", self.glyph_count)
        semantic_characters = changes.pop("semantic_characters", self.semantic_characters)
        authoritative_glyphs = changes.pop("authoritative_glyphs", self.authoritative_glyphs)
        heuristic_glyphs = changes.pop("heuristic_glyphs", self.heuristic_glyphs)
        unknown_glyphs = changes.pop("unknown_glyphs", self.unknown_glyphs)
        unsupported_glyphs = changes.pop("unsupported_glyphs", self.unsupported_glyphs)
        low_confidence_glyphs = changes.pop("low_confidence_glyphs", self.low_confidence_glyphs)
        actual_text_characters = changes.pop("actual_text_characters", self.actual_text_characters)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            glyph_count,
            semantic_characters,
            authoritative_glyphs,
            heuristic_glyphs,
            unknown_glyphs,
            unsupported_glyphs,
            low_confidence_glyphs,
            actual_text_characters,
        )

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


class PageEvidence:
    __slots__ = (
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
    )

    page_area: float
    native_characters: int
    visible_native_characters: int
    suspicious_characters: int
    image_count: int
    image_area_ratio: float
    image_boxes: tuple[tuple[float, float, float, float], ...]
    text_coverage: float
    full_page_image: bool
    text_quality: TextQualityStats
    all_text_quality: TextQualityStats
    glyphs: GlyphEvidence
    painted_native_characters: int | None
    trusted_hidden_text: bool

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
            f"trusted_hidden_text={self.trusted_hidden_text!r}"
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


class PageAnalysis:
    __slots__ = (
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

    page: PdfPage
    width: float
    height: float
    rotation: int
    fields: tuple[RawFormField, ...]
    annotations: tuple[RawAnnotation, ...]
    program: PageProgram
    observations: ObservationBatch
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


class ParsedLine:
    __slots__ = ("line", "sequence", "rotation", "font_size")

    line: TextLine
    sequence: int
    rotation: int
    font_size: float | None

    __fields__: ClassVar[tuple[str, ...]] = ("line", "sequence", "rotation", "font_size")
    __match_args__ = ("line", "sequence", "rotation", "font_size")

    def __init__(
        self,
        line: TextLine,
        sequence: int = 0,
        rotation: int = 0,
        font_size: float | None = None,
    ) -> None:
        internal_frozen_setattr(self, "line", line)
        internal_frozen_setattr(self, "sequence", sequence)
        internal_frozen_setattr(self, "rotation", rotation)
        internal_frozen_setattr(self, "font_size", font_size)
        self._post_init()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"line={self.line!r}, "
            f"sequence={self.sequence!r}, "
            f"rotation={self.rotation!r}, "
            f"font_size={self.font_size!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.line == other.line
            and self.sequence == other.sequence
            and self.rotation == other.rotation
            and self.font_size == other.font_size
        )

    def __hash__(self) -> int:
        return hash((self.line, self.sequence, self.rotation, self.font_size))

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
        line = changes.pop("line", self.line)
        sequence = changes.pop("sequence", self.sequence)
        rotation = changes.pop("rotation", self.rotation)
        font_size = changes.pop("font_size", self.font_size)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(line, sequence, rotation, font_size)

    def _post_init(self) -> None:
        if self.line.bbox is None:
            raise ValueError("ParsedLine requires a positioned line")


class ParsedBlock:
    __slots__ = ("lines", "bbox", "column_index", "kind", "level")

    lines: tuple[ParsedLine, ...]
    bbox: tuple[float, float, float, float]
    column_index: int | None
    kind: str
    level: int | None

    __fields__: ClassVar[tuple[str, ...]] = ("lines", "bbox", "column_index", "kind", "level")
    __match_args__ = ("lines", "bbox", "column_index", "kind", "level")

    def __init__(
        self,
        lines: tuple[ParsedLine, ...],
        bbox: tuple[float, float, float, float],
        column_index: int | None = None,
        kind: str = "paragraph",
        level: int | None = None,
    ) -> None:
        internal_frozen_setattr(self, "lines", lines)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "column_index", column_index)
        internal_frozen_setattr(self, "kind", kind)
        internal_frozen_setattr(self, "level", level)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"lines={self.lines!r}, "
            f"bbox={self.bbox!r}, "
            f"column_index={self.column_index!r}, "
            f"kind={self.kind!r}, "
            f"level={self.level!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.lines == other.lines
            and self.bbox == other.bbox
            and self.column_index == other.column_index
            and self.kind == other.kind
            and self.level == other.level
        )

    def __hash__(self) -> int:
        return hash((self.lines, self.bbox, self.column_index, self.kind, self.level))

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
        lines = changes.pop("lines", self.lines)
        bbox = changes.pop("bbox", self.bbox)
        column_index = changes.pop("column_index", self.column_index)
        kind = changes.pop("kind", self.kind)
        level = changes.pop("level", self.level)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(lines, bbox, column_index, kind, level)


class ReadingOrderEvidence:
    __slots__ = (
        "line_count",
        "source_inversions",
        "source_inversion_ratio",
        "column_count",
        "rotation_count",
        "repaired",
        "ambiguous",
        "confidence",
        "strategy",
    )

    line_count: int
    source_inversions: int
    source_inversion_ratio: float
    column_count: int
    rotation_count: int
    repaired: bool
    ambiguous: bool
    confidence: float
    strategy: str

    __fields__: ClassVar[tuple[str, ...]] = (
        "line_count",
        "source_inversions",
        "source_inversion_ratio",
        "column_count",
        "rotation_count",
        "repaired",
        "ambiguous",
        "confidence",
        "strategy",
    )
    __match_args__ = (
        "line_count",
        "source_inversions",
        "source_inversion_ratio",
        "column_count",
        "rotation_count",
        "repaired",
        "ambiguous",
        "confidence",
        "strategy",
    )

    def __init__(
        self,
        line_count: int,
        source_inversions: int,
        source_inversion_ratio: float,
        column_count: int,
        rotation_count: int,
        repaired: bool,
        ambiguous: bool,
        confidence: float,
        strategy: str,
    ) -> None:
        internal_frozen_setattr(self, "line_count", line_count)
        internal_frozen_setattr(self, "source_inversions", source_inversions)
        internal_frozen_setattr(self, "source_inversion_ratio", source_inversion_ratio)
        internal_frozen_setattr(self, "column_count", column_count)
        internal_frozen_setattr(self, "rotation_count", rotation_count)
        internal_frozen_setattr(self, "repaired", repaired)
        internal_frozen_setattr(self, "ambiguous", ambiguous)
        internal_frozen_setattr(self, "confidence", confidence)
        internal_frozen_setattr(self, "strategy", strategy)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"line_count={self.line_count!r}, "
            f"source_inversions={self.source_inversions!r}, "
            f"source_inversion_ratio={self.source_inversion_ratio!r}, "
            f"column_count={self.column_count!r}, "
            f"rotation_count={self.rotation_count!r}, "
            f"repaired={self.repaired!r}, "
            f"ambiguous={self.ambiguous!r}, "
            f"confidence={self.confidence!r}, "
            f"strategy={self.strategy!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.line_count == other.line_count
            and self.source_inversions == other.source_inversions
            and self.source_inversion_ratio == other.source_inversion_ratio
            and self.column_count == other.column_count
            and self.rotation_count == other.rotation_count
            and self.repaired == other.repaired
            and self.ambiguous == other.ambiguous
            and self.confidence == other.confidence
            and self.strategy == other.strategy
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.line_count,
                self.source_inversions,
                self.source_inversion_ratio,
                self.column_count,
                self.rotation_count,
                self.repaired,
                self.ambiguous,
                self.confidence,
                self.strategy,
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
        line_count = changes.pop("line_count", self.line_count)
        source_inversions = changes.pop("source_inversions", self.source_inversions)
        source_inversion_ratio = changes.pop("source_inversion_ratio", self.source_inversion_ratio)
        column_count = changes.pop("column_count", self.column_count)
        rotation_count = changes.pop("rotation_count", self.rotation_count)
        repaired = changes.pop("repaired", self.repaired)
        ambiguous = changes.pop("ambiguous", self.ambiguous)
        confidence = changes.pop("confidence", self.confidence)
        strategy = changes.pop("strategy", self.strategy)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            line_count,
            source_inversions,
            source_inversion_ratio,
            column_count,
            rotation_count,
            repaired,
            ambiguous,
            confidence,
            strategy,
        )

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy

from core_pdf.impl.array_views import make_column, readonly, validate_selection_mask
from core_pdf.impl.capture_program import PageProgram
from core_pdf.impl.geometry import interval_overlap, rect_tuple
from core_pdf.impl.output_model import Table, TextLine
from core_pdf.impl.runs import TextRun
from core_pdf.impl.types import GeneratedRecord, frozen_setattr

if TYPE_CHECKING:
    from core_pdf.impl.document_page import PdfPage
    from core_pdf.impl.document_records import RawAnnotation, RawFormField

FloatArray = numpy.ndarray[Any, numpy.dtype[numpy.float32]]
IntArray = numpy.ndarray[Any, numpy.dtype[numpy.int64]]
ByteArray = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]
BoolArray = numpy.ndarray[Any, numpy.dtype[numpy.bool_]]


FULL_PAGE_IMAGE_COVERAGE = 0.90


class ObservationSource(IntEnum):
    NATIVE = 0
    OCR = 1
    STRUCTURE = 2


class TextReference(Protocol):
    @property
    def text(self) -> str: ...

    @property
    def font_name(self) -> str | None: ...

    @property
    def fill_color(self) -> tuple[float, ...] | None: ...

    def is_bold(self) -> bool: ...

    def is_italic(self) -> bool: ...


class ObservationBatch(GeneratedRecord):
    text: tuple[str, ...]
    bbox: FloatArray
    source: ByteArray
    confidence: FloatArray
    sequence: IntArray
    visible: BoolArray
    rotation: IntArray
    font_size: FloatArray
    line_break_before: BoolArray
    references: tuple[TextReference | None, ...]

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
        references: Iterable[TextReference | None] | None = None,
    ) -> ObservationBatch:
        texts = tuple(text)
        size = len(texts)
        boxes = make_column(bbox, numpy.float32)
        if size == 0 and boxes.shape == (0,):
            boxes = boxes.reshape((0, 4))
        conf_arr = make_column(
            confidence, numpy.float32, lambda: numpy.full(size, numpy.nan, dtype=numpy.float32)
        )
        seq_arr = make_column(sequence, numpy.int64, lambda: numpy.arange(size, dtype=numpy.int64))
        vis_arr = make_column(visible, numpy.bool_, lambda: numpy.ones(size, dtype=numpy.bool_))
        rot_arr = make_column(rotation, numpy.int64, lambda: numpy.zeros(size, dtype=numpy.int64))
        font_arr = make_column(
            font_size, numpy.float32, lambda: numpy.full(size, numpy.nan, dtype=numpy.float32)
        )
        line_arr = make_column(
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
        validate_selection_mask(mask, len(self))
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
        validate_selection_mask(secondary_mask, len(secondary))
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


class TextQualityStats(GeneratedRecord):
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


class GlyphEvidence(GeneratedRecord):
    glyph_count: int = 0
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


class RunSelection(GeneratedRecord):
    runs: tuple[TextRun, ...]
    native_characters: int
    visible_native_characters: int
    suspicious_characters: int
    text_quality: TextQualityStats
    all_text_quality: TextQualityStats
    glyphs: GlyphEvidence
    painted_native_characters: int
    trusted_hidden_text: bool

    def __init__(
        self,
        runs: tuple[TextRun, ...],
        *,
        native_characters: int,
        visible_native_characters: int,
        suspicious_characters: int,
        text_quality: TextQualityStats,
        all_text_quality: TextQualityStats,
        glyphs: GlyphEvidence,
        painted_native_characters: int,
        trusted_hidden_text: bool,
    ) -> None:
        frozen_setattr(self, "runs", runs)
        frozen_setattr(self, "native_characters", native_characters)
        frozen_setattr(self, "visible_native_characters", visible_native_characters)
        frozen_setattr(self, "suspicious_characters", suspicious_characters)
        frozen_setattr(self, "text_quality", text_quality)
        frozen_setattr(self, "all_text_quality", all_text_quality)
        frozen_setattr(self, "glyphs", glyphs)
        frozen_setattr(self, "painted_native_characters", painted_native_characters)
        frozen_setattr(self, "trusted_hidden_text", trusted_hidden_text)


class PageEvidence(GeneratedRecord):
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

    @classmethod
    def measure(
        cls,
        page_width: float,
        page_height: float,
        observations: ObservationBatch,
        program: PageProgram,
        selection: RunSelection,
    ) -> Self:
        drawings = program.drawings
        inline_images = program.inline_images
        page_area = max(1.0, page_width * page_height)
        visible = observations.visible
        boxes = observations.bbox
        box_areas = numpy.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        box_heights = numpy.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        numpy.multiply(box_areas, box_heights, out=box_areas)
        coverage_areas = numpy.multiply(box_areas, visible)
        text_coverage = min(
            1.0,
            float(numpy.sum(coverage_areas, dtype=numpy.float64)) / page_area,
        )
        visible_image_areas: list[float] = []
        visible_image_boxes: list[tuple[float, float, float, float]] = []
        for drawing in drawings:
            if drawing.kind != "image":
                continue
            box = rect_tuple(drawing.rect)
            if box is None:
                continue
            width = interval_overlap(0.0, page_width, box[0], box[2])
            height = interval_overlap(0.0, page_height, box[1], box[3])
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
        full_page_image = any(
            width >= page_width * FULL_PAGE_IMAGE_COVERAGE
            and height >= page_height * FULL_PAGE_IMAGE_COVERAGE
            for width, height in (
                (
                    max(0.0, box[2] - box[0]),
                    max(0.0, box[3] - box[1]),
                )
                for box in visible_image_boxes
            )
        )
        return cls(
            page_area=page_area,
            native_characters=selection.native_characters,
            visible_native_characters=selection.visible_native_characters,
            suspicious_characters=selection.suspicious_characters,
            image_count=image_count,
            image_area_ratio=min(1.0, sum(visible_image_areas) / page_area),
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
            text_quality=selection.text_quality,
            all_text_quality=selection.all_text_quality,
            glyphs=selection.glyphs,
            painted_native_characters=selection.painted_native_characters,
            trusted_hidden_text=selection.trusted_hidden_text,
        )

    def extended[Evidence: PageEvidence](self, cls: type[Evidence], **extra: Any) -> Evidence:
        values = {name: getattr(self, name) for name in self.__fields__}
        values.update(extra)
        return cls(**values)

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


TILED_PAGE_MIN_IMAGES = 8
TILED_PAGE_MIN_IMAGE_AREA = 0.05
IMAGE_OBSTACLE_MIN_AREA = 0.01
IMAGE_MAX_AREA = 0.65


class ReadingOrderPolicy(GeneratedRecord):
    use_xy_cut: bool = True
    image_obstacles: tuple[tuple[float, float, float, float], ...] = ()

    @classmethod
    def from_evidence(cls, evidence: PageEvidence) -> ReadingOrderPolicy:
        return cls(
            use_xy_cut=not (
                evidence.image_count >= TILED_PAGE_MIN_IMAGES
                and TILED_PAGE_MIN_IMAGE_AREA <= evidence.image_area_ratio < IMAGE_MAX_AREA
            ),
            image_obstacles=tuple(
                box
                for box in evidence.image_boxes
                if IMAGE_OBSTACLE_MIN_AREA
                <= ((box[2] - box[0]) * (box[3] - box[1])) / evidence.page_area
                < IMAGE_MAX_AREA
            ),
        )


class PageAnalysis(GeneratedRecord):
    page: PdfPage
    width: float
    height: float
    rotation: int
    fields: tuple[RawFormField, ...]
    annotations: tuple[RawAnnotation, ...]
    program: PageProgram
    observations: ObservationBatch
    evidence: PageEvidence

    def extended[Analysis: PageAnalysis](self, cls: type[Analysis], **changes: Any) -> Analysis:
        values = {name: getattr(self, name) for name in self.__fields__}
        values.update(changes)
        return cls(**values)


class PageFrame(GeneratedRecord):
    width: float
    height: float
    rotation: int = 0
    page_number: int = 0

    def display_boxes(self, boxes: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
        rotation = self.rotation % 360
        if rotation == 0 or not len(boxes):
            return boxes
        width = self.width
        height = self.height
        x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        if rotation == 90:
            corners = (y0, width - x1, y1, width - x0)
        elif rotation == 180:
            corners = (width - x1, height - y1, width - x0, height - y0)
        elif rotation == 270:
            corners = (height - y1, x0, height - y0, x1)
        else:
            return boxes
        rotated = numpy.column_stack(corners).astype(boxes.dtype, copy=False)
        return numpy.column_stack(
            (
                numpy.minimum(rotated[:, 0], rotated[:, 2]),
                numpy.minimum(rotated[:, 1], rotated[:, 3]),
                numpy.maximum(rotated[:, 0], rotated[:, 2]),
                numpy.maximum(rotated[:, 1], rotated[:, 3]),
            )
        ).astype(boxes.dtype, copy=False)

    @staticmethod
    def reading_axis_positions(
        boxes: numpy.ndarray[Any, Any], rotation: int
    ) -> numpy.ndarray[Any, Any]:
        rotation %= 360
        if rotation == 90:
            return (boxes[:, 1] + boxes[:, 3]) * 0.5
        if rotation == 180:
            return -(boxes[:, 0] + boxes[:, 2]) * 0.5
        if rotation == 270:
            return -(boxes[:, 1] + boxes[:, 3]) * 0.5
        return (boxes[:, 0] + boxes[:, 2]) * 0.5


class ParsedLine(GeneratedRecord):
    line: TextLine
    sequence: int = 0
    rotation: int = 0
    font_size: float | None = None

    def __post_init__(self) -> None:
        if self.line.bbox is None:
            raise ValueError("ParsedLine requires a positioned line")


class ParsedBlock(GeneratedRecord):
    lines: tuple[ParsedLine, ...]
    bbox: tuple[float, float, float, float]
    column_index: int | None = None
    kind: str = "paragraph"
    level: int | None = None


class PageState(GeneratedRecord):
    observations: ObservationBatch
    tables: tuple[Table, ...] = ()
    blocks: tuple[ParsedBlock, ...] = ()
    order_ambiguous: bool = False

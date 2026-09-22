# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from copy import replace
from typing import Any, ClassVar, NoReturn, Self

import numpy

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf_ocr.impl.extract.contracts import (
    OCR_RESCUE_LARGE_TEXT_HEIGHT,
    OCR_RESCUE_MIN_CONFIDENCE,
    OCR_RESCUE_MIN_WEAK_INK_RATIO,
    OCR_RESCUE_SATURATED_MEAN_INK,
    OcrPass,
    OcrPassScope,
    internal_OCR_RESCUE_DENSE_MIN_CHARACTERS,
    internal_OCR_RESCUE_DENSE_MIN_CONFIDENCE,
)
from core_pdf_ocr.impl.extract.ocr.raster import internal_raster_ink_grid
from core_pdf_ocr.impl.extract.ocr.region_tasks import internal_weak_region_grid_shape
from core_pdf_ocr.impl.extract.ocr.types import internal_OcrTask, internal_Raster
from core_pdf_ocr.impl.extract.quality import internal_Candidate, internal_text_utility_stats

internal_frozen_setattr = object.__setattr__


def internal_observation_coverage_grid(
    observations: ObservationBatch,
    page_box: tuple[float, float, float, float],
    rows: int,
    columns: int,
) -> numpy.ndarray[Any, Any]:
    if not len(observations):
        return numpy.zeros(rows * columns, dtype=numpy.float32)
    x0, y0, x1, y1 = page_box
    boxes = numpy.asarray(observations.bbox, dtype=numpy.float64)
    clipped = numpy.column_stack(
        (
            numpy.maximum(boxes[:, 0], x0),
            numpy.maximum(boxes[:, 1], y0),
            numpy.minimum(boxes[:, 2], x1),
            numpy.minimum(boxes[:, 3], y1),
        )
    )
    utilities = numpy.asarray(
        [
            internal_text_utility_stats(text, float(confidence)).utility
            for text, confidence in zip(observations.text, observations.confidence, strict=True)
        ],
        dtype=numpy.float64,
    )
    box_widths = clipped[:, 2] - clipped[:, 0]
    box_heights = clipped[:, 3] - clipped[:, 1]
    valid = (box_widths > 0.0) & (box_heights > 0.0) & (utilities > 0.0)
    if not bool(valid.any()):
        return numpy.zeros(rows * columns, dtype=numpy.float32)

    clipped = clipped[valid]
    utilities = utilities[valid]
    areas = box_widths[valid] * box_heights[valid]
    x_edges = numpy.linspace(x0, x1, columns + 1, dtype=numpy.float64)
    y_edges = numpy.linspace(y1, y0, rows + 1, dtype=numpy.float64)
    overlap_x = numpy.maximum(
        0.0,
        numpy.minimum(clipped[:, None, 2], x_edges[None, 1:])
        - numpy.maximum(clipped[:, None, 0], x_edges[None, :-1]),
    )
    overlap_y = numpy.maximum(
        0.0,
        numpy.minimum(clipped[:, None, 3], y_edges[None, :-1])
        - numpy.maximum(clipped[:, None, 1], y_edges[None, 1:]),
    )
    weighted = utilities / areas
    output = numpy.einsum("n,nr,nc->rc", weighted, overlap_y, overlap_x, optimize=True)
    return output.astype(numpy.float32, copy=False).reshape(-1)


class internal_RescueCoverage:
    __slots__ = ("raster_count", "cell_count", "ink", "weak_ink")

    raster_count: int
    cell_count: int
    ink: float
    weak_ink: float

    __fields__: ClassVar[tuple[str, ...]] = ("raster_count", "cell_count", "ink", "weak_ink")
    __match_args__ = ("raster_count", "cell_count", "ink", "weak_ink")

    def __init__(
        self,
        raster_count: int = 0,
        cell_count: int = 0,
        ink: float = 0.0,
        weak_ink: float = 0.0,
    ) -> None:
        internal_frozen_setattr(self, "raster_count", raster_count)
        internal_frozen_setattr(self, "cell_count", cell_count)
        internal_frozen_setattr(self, "ink", ink)
        internal_frozen_setattr(self, "weak_ink", weak_ink)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"raster_count={self.raster_count!r}, "
            f"cell_count={self.cell_count!r}, "
            f"ink={self.ink!r}, "
            f"weak_ink={self.weak_ink!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.raster_count == other.raster_count
            and self.cell_count == other.cell_count
            and self.ink == other.ink
            and self.weak_ink == other.weak_ink
        )

    def __hash__(self) -> int:
        return hash((self.raster_count, self.cell_count, self.ink, self.weak_ink))

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
        raster_count = changes.pop("raster_count", self.raster_count)
        cell_count = changes.pop("cell_count", self.cell_count)
        ink = changes.pop("ink", self.ink)
        weak_ink = changes.pop("weak_ink", self.weak_ink)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(raster_count, cell_count, ink, weak_ink)

    @property
    def mean_ink(self) -> float:
        return self.ink / max(1, self.cell_count)

    @property
    def weak_ink_ratio(self) -> float:
        return self.weak_ink / max(1e-9, self.ink)


def internal_adaptive_rescue_coverage(
    source_tasks: tuple[internal_OcrTask, ...],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
) -> internal_RescueCoverage:
    raster_count = 0
    cell_count = 0
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
        total_ink += float(numpy.sum(ink, dtype=numpy.float64))
        weak_ink += float(numpy.sum(ink[weak], dtype=numpy.float64))
    return internal_RescueCoverage(
        raster_count=raster_count,
        cell_count=cell_count,
        ink=total_ink,
        weak_ink=weak_ink,
    )


def internal_primary_text_is_sufficient(candidate: internal_Candidate) -> bool:
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
) -> bool:
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
    if internal_primary_text_is_sufficient(candidate):
        return False
    if coverage.mean_ink >= OCR_RESCUE_SATURATED_MEAN_INK and (
        (metrics.characters >= 1_000 and metrics.mean_confidence >= OCR_RESCUE_MIN_CONFIDENCE)
        or (
            metrics.characters >= internal_OCR_RESCUE_DENSE_MIN_CHARACTERS
            and metrics.mean_confidence >= internal_OCR_RESCUE_DENSE_MIN_CONFIDENCE
        )
    ):
        return False
    return not (
        metrics.characters >= 300
        and metrics.mean_confidence >= OCR_RESCUE_MIN_CONFIDENCE
        and coverage.raster_count
        and coverage.weak_ink_ratio < OCR_RESCUE_MIN_WEAK_INK_RATIO
        and (metrics.characters >= 600 or coverage.weak_ink_ratio == 0.0)
    )

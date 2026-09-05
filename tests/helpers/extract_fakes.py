# SPDX-License-Identifier: AGPL-3.0-only
"""Production-shaped inputs for the parse stages: evidence, batches, captures, runs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from core_pdf.impl.extract.contracts import (
    ObservationBatch,
    ObservationSource,
    PageAnalysis,
    PageEvidence,
)
from core_pdf.impl.model.geometry import RectBox
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing, CapturedPath
from core_pdf.impl.spec.s_07_content.page_program import PageProgram

Box = tuple[float, float, float, float]


@dataclass(slots=True)
class FakePage:
    """The few page attributes the parse stages read; ``PageAnalysis.page`` is ``Any``."""

    width: float = 600.0
    height: float = 800.0
    page_number: int = 1
    media_box: tuple[float, float, float, float] | None = None


def page_evidence(**overrides: Any) -> PageEvidence:
    """``PageEvidence`` for an empty page; keyword overrides set any field."""
    base = PageEvidence(
        page_area=480_000.0,
        native_characters=0,
        visible_native_characters=0,
        suspicious_characters=0,
        image_count=0,
        image_area_ratio=0.0,
        vector_complexity=0,
    )
    return replace(base, **overrides) if overrides else base


def observations(
    items: Iterable[tuple[str, Box]],
    *,
    source: ObservationSource = ObservationSource.NATIVE,
    confidence: float | Sequence[float] | None = None,
    **columns: Any,
) -> ObservationBatch:
    """An ``ObservationBatch`` from ``(text, bbox)`` pairs plus optional columns."""
    pairs = tuple(items)
    texts = tuple(text for text, _ in pairs)
    boxes = tuple(box for _, box in pairs)
    if isinstance(confidence, (int, float)):
        confidence = tuple(float(confidence) for _ in pairs)
    return ObservationBatch.from_columns(
        texts, boxes, source=source, confidence=confidence, **columns
    )


def capture(
    evidence: PageEvidence | None = None,
    *,
    page: Any = None,
    program: Any = None,
    batch: ObservationBatch | None = None,
    runs: tuple[TextRun, ...] = (),
    drawings: tuple[CapturedDrawing, ...] = (),
    grid_lines: Any = (),
    inline_images: tuple[Any, ...] = (),
    width: float = 600.0,
    height: float = 800.0,
    rotation: int = 0,
) -> PageAnalysis:
    """A real ``PageAnalysis``; ``rotation`` seeds one observation so routing sees it."""
    if batch is None:
        batch = ObservationBatch.from_columns(
            ("x",),
            ((0.0, 0.0, 1.0, 1.0),),
            source=ObservationSource.NATIVE,
            rotation=(rotation,),
        )
    if program is None:
        program = PageProgram(
            runs=runs,
            drawings=drawings,
            lines=grid_lines,
            inline_images=inline_images,
        )
    return PageAnalysis(
        page=page if page is not None else FakePage(width=width, height=height),
        width=width,
        height=height,
        rotation=rotation,
        fields=(),
        annotations=(),
        program=program,
        observations=batch,
        evidence=evidence if evidence is not None else page_evidence(),
    )


def drawing(kind: str, box: Box, *, seqno: int = 0, **fields: Any) -> CapturedDrawing:
    """A ``CapturedDrawing`` whose ``rect`` is ``box``; extra fields pass through."""
    x0, y0, x1, y1 = box
    fields.setdefault("path", CapturedPath())
    return CapturedDrawing(
        seqno=seqno,
        fill=None,
        fill_opacity=None,
        kind=kind,
        bbox=RectBox(x0, y0, x1, y1),
        **fields,
    )


def text_run(
    text: str,
    x0: float = 0.0,
    y0: float = 0.0,
    x1: float = 10.0,
    y1: float = 10.0,
    *,
    tx: float | None = None,
    ty: float | None = None,
    font_size: float = 10.0,
    space_width: float = 4.0,
    order: int = 0,
    stream_order: int | None = None,
    xobject_depth: int = 0,
    **fields: Any,
) -> TextRun:
    """A ``TextRun`` with the twelve positional constructor arguments named."""
    return TextRun(
        text,
        x0,
        y0,
        x1,
        y1,
        x0 if tx is None else tx,
        y0 if ty is None else ty,
        font_size,
        space_width,
        order,
        order if stream_order is None else stream_order,
        xobject_depth,
        **fields,
    )


__all__ = (
    "Box",
    "FakePage",
    "capture",
    "drawing",
    "observations",
    "page_evidence",
    "text_run",
)

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import ClassVar, cast

from core_pdf.impl.model.geometry import union_bbox
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.records import ReplaceFields, ReprFields
from core_pdf.impl.types import Rectangle


def extend_baseline(left: Rectangle | None, right: Rectangle | None) -> Rectangle | None:
    if left is None:
        return right
    if right is None:
        return left
    return (left[0], left[1], right[2], right[3])


def min_optional_confidence(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


class MarkedContentEntry(ReplaceFields, ReprFields):
    __slots__ = ("layer", "actual_text", "mcid", "run", "font_decoder", "effective_font_height")

    layer: str | None
    actual_text: str | None
    mcid: int | None
    run: TextRun | None
    font_decoder: object | None
    effective_font_height: float

    __fields__: ClassVar[tuple[str, ...]] = (
        "layer",
        "actual_text",
        "mcid",
        "run",
        "font_decoder",
        "effective_font_height",
    )
    __match_args__ = (
        "layer",
        "actual_text",
        "mcid",
        "run",
        "font_decoder",
        "effective_font_height",
    )

    def __init__(
        self,
        layer: str | None = None,
        actual_text: str | None = None,
        mcid: int | None = None,
        run: TextRun | None = None,
        font_decoder: object | None = None,
        effective_font_height: float = 0.0,
    ) -> None:
        self.layer = layer
        self.actual_text = actual_text
        self.mcid = mcid
        self.run = run
        self.font_decoder = font_decoder
        self.effective_font_height = effective_font_height

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.layer == other.layer
            and self.actual_text == other.actual_text
            and self.mcid == other.mcid
            and self.run == other.run
            and self.font_decoder == other.font_decoder
            and self.effective_font_height == other.effective_font_height
        )

    __hash__ = None  # type: ignore[assignment]

    def add_run(
        self,
        run: TextRun,
        *,
        font_decoder: object | None = None,
        effective_font_height: float = 0.0,
    ) -> None:
        captured = self.run
        if captured is None:
            self.run = run
            self.font_decoder = font_decoder
            self.effective_font_height = effective_font_height
            return
        captured.x0 = min(captured.x0, run.x0)
        captured.y0 = min(captured.y0, run.y0)
        captured.x1 = max(captured.x1, run.x1)
        captured.y1 = max(captured.y1, run.y1)
        captured.advance_bbox = cast(Rectangle, union_bbox(captured.advance_bbox, run.advance_bbox))
        captured.baseline = extend_baseline(captured.baseline, run.baseline)
        captured.confidence = min_optional_confidence(captured.confidence, run.confidence)

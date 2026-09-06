# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from core_pdf.impl._impl.model.geometry import union_bbox
from core_pdf.impl._impl.model.runs import TextRun
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


@dataclass(slots=True)
class MarkedContentEntry:
    layer: str | None = None
    actual_text: str | None = None
    mcid: int | None = None
    run: TextRun | None = None
    font_decoder: object | None = None
    effective_font_height: float = 0.0

    def add_run(
        self,
        run: TextRun,
        *,
        font_decoder: object | None = None,
        effective_font_height: float = 0.0,
    ) -> None:
        """Collect extents while retaining the first run's style and provenance."""
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

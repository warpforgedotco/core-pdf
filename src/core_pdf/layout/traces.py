# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_pdf.layout.geometry import RectBox


class CapturedLine:
    __slots__ = ("x0", "y0", "x1", "y1", "line_width")

    def __init__(self, x0: float, y0: float, x1: float, y1: float, line_width: float = 1.0) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.line_width = line_width

    def replace(self, **kwargs: Any) -> CapturedLine:
        """Create a new CapturedLine with modified fields."""
        return CapturedLine(
            x0=kwargs.get("x0", self.x0),
            y0=kwargs.get("y0", self.y0),
            x1=kwargs.get("x1", self.x1),
            y1=kwargs.get("y1", self.y1),
            line_width=kwargs.get("line_width", self.line_width),
        )


class GlyphTrace:
    __slots__ = ("rect", "c", "seqno", "fill", "visible")

    def __init__(
        self,
        rect: RectBox,
        c: str,
        seqno: int,
        fill: tuple[float, ...] | None = None,
        visible: bool = True,
    ) -> None:
        self.rect = rect
        self.c = c
        self.seqno = seqno
        self.fill = fill
        self.visible = visible

    def replace(self, **kwargs: Any) -> GlyphTrace:
        """Create a new GlyphTrace with modified fields."""
        return GlyphTrace(
            rect=kwargs.get("rect", self.rect),
            c=kwargs.get("c", self.c),
            seqno=kwargs.get("seqno", self.seqno),
            fill=kwargs.get("fill", self.fill),
            visible=kwargs.get("visible", self.visible),
        )


class DrawingTrace:
    __slots__ = (
        "seqno",
        "fill",
        "fill_opacity",
        "stroke_color",
        "stroke_opacity",
        "line_width",
        "items",
        "kind",
    )

    def __init__(
        self,
        seqno: int,
        fill: tuple[float, ...] | None,
        fill_opacity: float | None,
        stroke_color: tuple[float, ...] | None = None,
        stroke_opacity: float | None = None,
        line_width: float = 1.0,
        kind: str = "fill",
        items: list[tuple[str, RectBox]] | None = None,
    ) -> None:
        self.seqno = seqno
        self.fill = fill
        self.fill_opacity = fill_opacity
        self.stroke_color = stroke_color
        self.stroke_opacity = stroke_opacity
        self.line_width = line_width
        self.kind = kind
        self.items = items if items is not None else []

    def replace(self, **kwargs: Any) -> DrawingTrace:
        """Create a new DrawingTrace with modified fields."""
        return DrawingTrace(
            seqno=kwargs.get("seqno", self.seqno),
            fill=kwargs.get("fill", self.fill),
            fill_opacity=kwargs.get("fill_opacity", self.fill_opacity),
            stroke_color=kwargs.get("stroke_color", self.stroke_color),
            stroke_opacity=kwargs.get("stroke_opacity", self.stroke_opacity),
            line_width=kwargs.get("line_width", self.line_width),
            kind=kwargs.get("kind", self.kind),
            items=kwargs.get("items", self.items),
        )

    @property
    def rect(self) -> RectBox | None:
        if not self.items:
            return None
        rects = [item[1] for item in self.items if item[0] == "re"]
        if not rects:
            return None
        x0 = min(rect.x0 for rect in rects)
        y0 = min(rect.y0 for rect in rects)
        x1 = max(rect.x1 for rect in rects)
        y1 = max(rect.y1 for rect in rects)

        return RectBox(
            x0, y0, x1, y1, seqno=self.seqno, fill=self.fill, fill_opacity=self.fill_opacity
        )

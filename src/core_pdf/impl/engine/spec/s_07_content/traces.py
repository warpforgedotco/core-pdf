# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TypeVar

from core_pdf.impl.engine.spec.s_08_graphics.geometry import RectBox

_T = TypeVar("_T")


class _UnsetType:
    pass


_UNSET = _UnsetType()


def _value_or(value: _T | _UnsetType, default: _T) -> _T:
    if isinstance(value, _UnsetType):
        return default
    return value


class CapturedLine:
    __slots__ = ("x0", "y0", "x1", "y1", "line_width")

    def __init__(self, x0: float, y0: float, x1: float, y1: float, line_width: float = 1.0) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.line_width = line_width

    def replace(
        self,
        *,
        x0: float | _UnsetType = _UNSET,
        y0: float | _UnsetType = _UNSET,
        x1: float | _UnsetType = _UNSET,
        y1: float | _UnsetType = _UNSET,
        line_width: float | _UnsetType = _UNSET,
    ) -> CapturedLine:
        """Create a new CapturedLine with modified fields."""
        return CapturedLine(
            x0=_value_or(x0, self.x0),
            y0=_value_or(y0, self.y0),
            x1=_value_or(x1, self.x1),
            y1=_value_or(y1, self.y1),
            line_width=_value_or(line_width, self.line_width),
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

    def replace(
        self,
        *,
        rect: RectBox | _UnsetType = _UNSET,
        c: str | _UnsetType = _UNSET,
        seqno: int | _UnsetType = _UNSET,
        fill: tuple[float, ...] | None | _UnsetType = _UNSET,
        visible: bool | _UnsetType = _UNSET,
    ) -> GlyphTrace:
        """Create a new GlyphTrace with modified fields."""
        return GlyphTrace(
            rect=_value_or(rect, self.rect),
            c=_value_or(c, self.c),
            seqno=_value_or(seqno, self.seqno),
            fill=_value_or(fill, self.fill),
            visible=_value_or(visible, self.visible),
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

    def replace(
        self,
        *,
        seqno: int | _UnsetType = _UNSET,
        fill: tuple[float, ...] | None | _UnsetType = _UNSET,
        fill_opacity: float | None | _UnsetType = _UNSET,
        stroke_color: tuple[float, ...] | None | _UnsetType = _UNSET,
        stroke_opacity: float | None | _UnsetType = _UNSET,
        line_width: float | _UnsetType = _UNSET,
        kind: str | _UnsetType = _UNSET,
        items: list[tuple[str, RectBox]] | None | _UnsetType = _UNSET,
    ) -> DrawingTrace:
        """Create a new DrawingTrace with modified fields."""
        return DrawingTrace(
            seqno=_value_or(seqno, self.seqno),
            fill=_value_or(fill, self.fill),
            fill_opacity=_value_or(fill_opacity, self.fill_opacity),
            stroke_color=_value_or(stroke_color, self.stroke_color),
            stroke_opacity=_value_or(stroke_opacity, self.stroke_opacity),
            line_width=_value_or(line_width, self.line_width),
            kind=_value_or(kind, self.kind),
            items=_value_or(items, self.items),
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

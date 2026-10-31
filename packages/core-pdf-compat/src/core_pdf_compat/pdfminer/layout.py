from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from ._layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTComponent,
    LTFigure,
    LTImage,
    LTItem,
    LTPage,
    LTText,
    LTTextBox,
    LTTextBoxHorizontal,
    LTTextBoxVertical,
    LTTextContainer,
    LTTextLine,
    LTTextLineHorizontal,
    LTTextLineVertical,
)

Point = tuple[float, float]


def _bound(points: Iterable[Point]) -> tuple[float, float, float, float]:
    xs, ys = zip(*points, strict=True)
    return min(xs), min(ys), max(xs), max(ys)


class LTCurve(LTComponent):
    def __init__(
        self,
        linewidth: float,
        pts: list[Point],
        stroke: bool = False,
        fill: bool = False,
        evenodd: bool = False,
        stroking_color: Any = None,
        non_stroking_color: Any = None,
        original_path: list[Any] | None = None,
        dashing_style: Any = None,
    ) -> None:
        super().__init__(_bound(pts))
        self.pts = pts
        self.linewidth = linewidth
        self.stroke = stroke
        self.fill = fill
        self.evenodd = evenodd
        self.stroking_color = stroking_color
        self.non_stroking_color = non_stroking_color
        self.original_path = original_path
        self.dashing_style = dashing_style

    def get_pts(self) -> str:
        return ",".join(f"{x:.3f},{y:.3f}" for x, y in self.pts)


class LTLine(LTCurve):
    def __init__(self, linewidth: float, p0: Point, p1: Point, *rest: Any) -> None:
        super().__init__(linewidth, [p0, p1], *rest)


class LTRect(LTCurve):
    def __init__(
        self, linewidth: float, bbox: tuple[float, float, float, float], *rest: Any
    ) -> None:
        x0, y0, x1, y1 = bbox
        super().__init__(linewidth, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], *rest)


class LTContainer(LTComponent):
    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        super().__init__(bbox)
        self._objs: list[Any] = []

    def __iter__(self) -> Iterator[Any]:
        return iter(self._objs)

    def __len__(self) -> int:
        return len(self._objs)

    def add(self, obj: Any) -> None:
        self._objs.append(obj)

    def extend(self, objs: Iterable[Any]) -> None:
        for obj in objs:
            self.add(obj)

    def analyze(self, laparams: LAParams) -> None:
        for obj in self._objs:
            obj.analyze(laparams)


__all__ = (
    "LAParams",
    "LTAnno",
    "LTChar",
    "LTComponent",
    "LTContainer",
    "LTCurve",
    "LTFigure",
    "LTImage",
    "LTItem",
    "LTLine",
    "LTPage",
    "LTRect",
    "LTText",
    "LTTextBox",
    "LTTextBoxHorizontal",
    "LTTextBoxVertical",
    "LTTextContainer",
    "LTTextLine",
    "LTTextLineHorizontal",
    "LTTextLineVertical",
)

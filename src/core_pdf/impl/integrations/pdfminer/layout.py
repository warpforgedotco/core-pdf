# SPDX-License-Identifier: AGPL-3.0-only
"""The pdfminer.six layout object model used by compatibility integrations.

The classes in this module intentionally preserve pdfminer.six's public object
contracts.  They are implemented locally so applications can switch imports
without retaining pdfminer.six as a runtime dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from math import inf
from typing import Any, Generic, TypeVar, cast

Rect = tuple[float, float, float, float]
Matrix = tuple[float, float, float, float, float, float]


def _apply_matrix_point(matrix: Matrix, point: tuple[float, float]) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    x, y = point
    return a * x + c * y + e, b * x + d * y + f


def _apply_matrix_rect(matrix: Matrix, rect: Rect) -> Rect:
    x0, y0, x1, y1 = rect
    points = (
        _apply_matrix_point(matrix, (x0, y0)),
        _apply_matrix_point(matrix, (x0, y1)),
        _apply_matrix_point(matrix, (x1, y0)),
        _apply_matrix_point(matrix, (x1, y1)),
    )
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


class LAParams:
    def __init__(
        self,
        line_overlap: float = 0.5,
        char_margin: float = 2.0,
        line_margin: float = 0.5,
        word_margin: float = 0.1,
        boxes_flow: float | None = 0.5,
        detect_vertical: bool = False,
        all_texts: bool = False,
    ) -> None:
        self.line_overlap = line_overlap
        self.char_margin = char_margin
        self.line_margin = line_margin
        self.word_margin = word_margin
        self.boxes_flow = boxes_flow
        self.detect_vertical = detect_vertical
        self.all_texts = all_texts
        self._validate()

    def _validate(self) -> None:
        if self.boxes_flow is None:
            return
        message = "LAParam boxes_flow should be None, or a number between -1 and +1"
        if not isinstance(self.boxes_flow, (int, float)):
            raise TypeError(message)
        if not -1 <= self.boxes_flow <= 1:
            raise ValueError(message)

    def __repr__(self) -> str:
        return (
            f"<LAParams: char_margin={self.char_margin:.1f}, "
            f"line_margin={self.line_margin:.1f}, "
            f"word_margin={self.word_margin:.1f} all_texts={self.all_texts!r}>"
        )


class LTItem:
    def analyze(self, laparams: LAParams) -> None:
        del laparams


class LTText:
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.get_text()!r}>"

    def get_text(self) -> str:
        raise NotImplementedError


class LTComponent(LTItem):
    def __init__(self, bbox: Rect) -> None:
        self.set_bbox(bbox)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.bbox!r}>"

    def set_bbox(self, bbox: Rect) -> None:
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
        self.bbox = bbox

    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def is_hoverlap(self, obj: LTComponent) -> bool:
        return obj.x0 <= self.x1 and self.x0 <= obj.x1

    def hdistance(self, obj: LTComponent) -> float:
        return 0.0 if self.is_hoverlap(obj) else min(abs(self.x0 - obj.x1), abs(self.x1 - obj.x0))

    def hoverlap(self, obj: LTComponent) -> float:
        return min(abs(self.x0 - obj.x1), abs(self.x1 - obj.x0)) if self.is_hoverlap(obj) else 0.0

    def is_voverlap(self, obj: LTComponent) -> bool:
        return obj.y0 <= self.y1 and self.y0 <= obj.y1

    def vdistance(self, obj: LTComponent) -> float:
        return 0.0 if self.is_voverlap(obj) else min(abs(self.y0 - obj.y1), abs(self.y1 - obj.y0))

    def voverlap(self, obj: LTComponent) -> float:
        return min(abs(self.y0 - obj.y1), abs(self.y1 - obj.y0)) if self.is_voverlap(obj) else 0.0


class LTAnno(LTItem, LTText):
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self) -> str:
        return self._text


class LTChar(LTComponent, LTText):
    rendermode: int

    def __init__(
        self,
        matrix: Matrix,
        font: Any,
        fontsize: float,
        scaling: float,
        rise: float,
        text: str,
        textwidth: float,
        textdisp: float | tuple[float | None, float],
        ncs: Any,
        graphicstate: Any,
    ) -> None:
        self._text = text
        self.matrix = matrix
        self.fontname = font.fontname
        self.ncs = ncs
        self.graphicstate = graphicstate
        self.adv = textwidth * fontsize * scaling
        vertical = bool(font.is_vertical())
        if vertical:
            if not isinstance(textdisp, tuple):
                raise TypeError("vertical text displacement must be a tuple")
            vx, vy = textdisp
            vx = fontsize * 0.5 if vx is None else vx * fontsize * 0.001
            vertical_y = (1000 - vy) * fontsize * 0.001
            bbox = (-vx, vertical_y + rise + self.adv, -vx + fontsize, vertical_y + rise)
        else:
            descent = font.get_descent() * fontsize
            bbox = (0.0, descent + rise, self.adv, descent + rise + fontsize)
        a, b, c, d, _, _ = matrix
        self.upright = a * d * scaling > 0 and b * c <= 0
        super().__init__(_apply_matrix_rect(matrix, bbox))
        self.size = self.width if vertical else self.height

    @classmethod
    def from_core(
        cls,
        *,
        text: str,
        bbox: Rect,
        matrix: Matrix,
        fontname: str,
        fontsize: float,
        adv: float,
        upright: bool,
        graphicstate: Any = None,
        ncs: Any = None,
        rendermode: int = 0,
    ) -> LTChar:
        char = cls.__new__(cls)
        char._text = text
        char.matrix = matrix
        char.fontname = fontname
        char.ncs = ncs
        char.graphicstate = graphicstate
        char.adv = adv
        char.upright = upright
        LTComponent.__init__(char, bbox)
        char.size = char.height if upright else char.width
        char.rendermode = rendermode
        return char

    def get_text(self) -> str:
        return self._text


LTItemT = TypeVar("LTItemT", bound=LTItem)


class LTContainer(LTComponent, Generic[LTItemT]):
    def __init__(self, bbox: Rect) -> None:
        super().__init__(bbox)
        self._objs: list[LTItemT] = []

    def __iter__(self) -> Iterator[LTItemT]:
        return iter(self._objs)

    def __len__(self) -> int:
        return len(self._objs)

    def add(self, obj: LTItemT) -> None:
        self._objs.append(obj)

    def extend(self, objs: Iterable[LTItemT]) -> None:
        for obj in objs:
            self.add(obj)

    def analyze(self, laparams: LAParams) -> None:
        for obj in self._objs:
            obj.analyze(laparams)


class LTExpandableContainer(LTContainer[LTItemT]):
    def __init__(self) -> None:
        super().__init__((inf, inf, -inf, -inf))

    def add(self, obj: LTItemT) -> None:
        super().add(obj)
        component = cast(LTComponent, obj)
        self.set_bbox(
            (
                min(self.x0, component.x0),
                min(self.y0, component.y0),
                max(self.x1, component.x1),
                max(self.y1, component.y1),
            )
        )


class LTTextContainer(LTExpandableContainer[LTItemT], LTText):
    def get_text(self) -> str:
        return "".join(cast(LTText, obj).get_text() for obj in self if isinstance(obj, LTText))


class LTTextLine(LTTextContainer[LTItem]):
    def __init__(self, word_margin: float) -> None:
        super().__init__()
        self.word_margin = word_margin

    def analyze(self, laparams: LAParams) -> None:
        super().analyze(laparams)
        LTContainer.add(self, LTAnno("\n"))

    def is_empty(self) -> bool:
        return super().is_empty() or self.get_text().isspace()


class LTTextLineHorizontal(LTTextLine):
    def __init__(self, word_margin: float) -> None:
        super().__init__(word_margin)
        self._x1 = inf

    def add(self, obj: LTItem) -> None:
        if isinstance(obj, LTChar) and self.word_margin:
            margin = self.word_margin * max(obj.width, obj.height)
            if self._x1 < obj.x0 - margin:
                LTContainer.add(self, LTAnno(" "))
        if isinstance(obj, LTComponent):
            self._x1 = obj.x1
        super().add(obj)


class LTTextLineVertical(LTTextLine):
    def __init__(self, word_margin: float) -> None:
        super().__init__(word_margin)
        self._y0 = -inf

    def add(self, obj: LTItem) -> None:
        if isinstance(obj, LTChar) and self.word_margin:
            margin = self.word_margin * max(obj.width, obj.height)
            if obj.y1 + margin < self._y0:
                LTContainer.add(self, LTAnno(" "))
        if isinstance(obj, LTComponent):
            self._y0 = obj.y0
        super().add(obj)


class LTTextBox(LTTextContainer[LTTextLine]):
    def __init__(self) -> None:
        super().__init__()
        self.index = -1

    def get_writing_mode(self) -> str:
        raise NotImplementedError


class LTTextBoxHorizontal(LTTextBox):
    def get_writing_mode(self) -> str:
        return "lr-tb"


class LTTextBoxVertical(LTTextBox):
    def get_writing_mode(self) -> str:
        return "tb-rl"


class LTLayoutContainer(LTContainer[LTComponent]):
    def __init__(self, bbox: Rect) -> None:
        super().__init__(bbox)
        self.groups: list[Any] | None = None


class LTFigure(LTLayoutContainer):
    def __init__(self, name: str, bbox: Rect, matrix: Matrix) -> None:
        self.name = name
        self.matrix = matrix
        x, y, width, height = bbox
        super().__init__(_apply_matrix_rect(matrix, (x, y, x + width, y + height)))

    def analyze(self, laparams: LAParams) -> None:
        if laparams.all_texts:
            super().analyze(laparams)


class LTImage(LTComponent):
    def __init__(self, name: str, stream: Any, bbox: Rect) -> None:
        super().__init__(bbox)
        self.name = name
        self.stream = stream

        def get_any(keys: tuple[str, ...], default: Any = None) -> Any:
            if hasattr(stream, "get_any"):
                return stream.get_any(keys, default)
            attrs = getattr(stream, "attrs", stream if isinstance(stream, dict) else {})
            for key in keys:
                if key in attrs:
                    return attrs[key]
            return default

        self.srcsize = (get_any(("W", "Width")), get_any(("H", "Height")))
        self.imagemask = get_any(("IM", "ImageMask"))
        self.bits = get_any(("BPC", "BitsPerComponent"), 1)
        self.colorspace = get_any(("CS", "ColorSpace"))
        if not isinstance(self.colorspace, list):
            self.colorspace = [self.colorspace]


class LTPage(LTLayoutContainer):
    def __init__(self, pageid: int, bbox: Rect, rotate: float = 0) -> None:
        super().__init__(bbox)
        self.pageid = pageid
        self.rotate = rotate


__all__ = (
    "LAParams",
    "LTAnno",
    "LTChar",
    "LTComponent",
    "LTContainer",
    "LTExpandableContainer",
    "LTFigure",
    "LTImage",
    "LTItem",
    "LTLayoutContainer",
    "LTPage",
    "LTText",
    "LTTextBox",
    "LTTextBoxHorizontal",
    "LTTextBoxVertical",
    "LTTextContainer",
    "LTTextLine",
    "LTTextLineHorizontal",
    "LTTextLineVertical",
)

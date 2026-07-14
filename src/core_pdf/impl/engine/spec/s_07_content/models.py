# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, TypeVar

_COORDS_TEMPLATE = array("d", [0.0] * 8)
_T = TypeVar("_T")


class _UnsetType:
    pass


_UNSET = _UnsetType()


def _value_or(value: _T | _UnsetType, default: _T) -> _T:
    if isinstance(value, _UnsetType):
        return default
    return value


if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_08_graphics.geometry import RectBox


class TextRun:
    __slots__ = (
        "text",
        "coords",
        "font_name",
        "order",
        "stream_order",
        "xobject_depth",
        "is_vertical",
        "rotation_angle",
        "visible",
        "line_break_before",
        "seqno",
        "fill_color",
    )

    X0 = 0
    Y0 = 1
    X1 = 2
    Y1 = 3
    TX = 4
    TY = 5
    FONT_SIZE = 6
    SPACE_WIDTH = 7

    text: str
    font_name: str | None
    order: int
    stream_order: int
    xobject_depth: int
    is_vertical: bool
    rotation_angle: int
    visible: bool
    line_break_before: bool
    seqno: int
    fill_color: tuple[float, ...] | None

    @property
    def x0(self) -> float:
        return self.coords[self.X0]

    @x0.setter
    def x0(self, v: float) -> None:
        self.coords[self.X0] = v

    @property
    def y0(self) -> float:
        return self.coords[self.Y0]

    @y0.setter
    def y0(self, v: float) -> None:
        self.coords[self.Y0] = v

    @property
    def x1(self) -> float:
        return self.coords[self.X1]

    @x1.setter
    def x1(self, v: float) -> None:
        self.coords[self.X1] = v

    @property
    def y1(self) -> float:
        return self.coords[self.Y1]

    @y1.setter
    def y1(self, v: float) -> None:
        self.coords[self.Y1] = v

    @property
    def tx(self) -> float:
        return self.coords[self.TX]

    @tx.setter
    def tx(self, v: float) -> None:
        self.coords[self.TX] = v

    @property
    def ty(self) -> float:
        return self.coords[self.TY]

    @ty.setter
    def ty(self, v: float) -> None:
        self.coords[self.TY] = v

    @property
    def font_size(self) -> float:
        return self.coords[self.FONT_SIZE]

    @font_size.setter
    def font_size(self, v: float) -> None:
        self.coords[self.FONT_SIZE] = v

    @property
    def space_width(self) -> float:
        return self.coords[self.SPACE_WIDTH]

    @space_width.setter
    def space_width(self, v: float) -> None:
        self.coords[self.SPACE_WIDTH] = v

    @property
    def mid_x(self) -> float:
        c = self.coords
        return (c[self.X0] + c[self.X1]) * 0.5

    @property
    def mid_y(self) -> float:
        c = self.coords
        return (c[self.Y0] + c[self.Y1]) * 0.5

    @property
    def height(self) -> float:
        return self.coords[self.Y1] - self.coords[self.Y0]

    def __init__(
        self,
        text: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        tx: float,
        ty: float,
        font_size: float,
        space_width: float,
        order: int,
        stream_order: int,
        xobject_depth: int,
        font_name: str | None = None,
        is_vertical: bool = False,
        rotation_angle: int = 0,
        visible: bool = True,
        line_break_before: bool = False,
        seqno: int = -1,
        fill_color: tuple[float, ...] | None = None,
    ) -> None:
        self.coords = array(
            "d",
            [x0, y0, x1, y1, tx, ty, font_size, space_width],
        )
        self.text = text
        self.font_name = font_name
        self.order = order
        self.stream_order = stream_order
        self.xobject_depth = xobject_depth
        self.is_vertical = is_vertical
        self.rotation_angle = rotation_angle
        self.visible = visible
        self.line_break_before = line_break_before
        self.seqno = seqno
        self.fill_color = fill_color

    def is_bold(self) -> bool:
        if not self.font_name:
            return False
        fn = self.font_name.lower()
        return "bold" in fn or "black" in fn or "heavy" in fn

    def is_italic(self) -> bool:
        if not self.font_name:
            return False
        fn = self.font_name.lower()
        return "italic" in fn or "oblique" in fn or "slanted" in fn

    @classmethod
    def reinit(
        cls,
        existing: TextRun | None,
        text: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        tx: float,
        ty: float,
        font_size: float,
        space_width: float,
        order: int,
        stream_order: int,
        xobject_depth: int,
        font_name: str | None,
        is_vertical: bool,
        rotation_angle: int,
        visible: bool,
        line_break_before: bool,
        seqno: int,
        fill_color: tuple[float, ...] | None,
    ) -> TextRun:
        if existing is not None:
            c = existing.coords
            c[cls.X0] = x0
            c[cls.Y0] = y0
            c[cls.X1] = x1
            c[cls.Y1] = y1
            c[cls.TX] = tx
            c[cls.TY] = ty
            c[cls.FONT_SIZE] = font_size
            c[cls.SPACE_WIDTH] = space_width
            existing.text = text
            existing.font_name = font_name
            existing.order = order
            existing.stream_order = stream_order
            existing.xobject_depth = xobject_depth
            existing.is_vertical = is_vertical
            existing.rotation_angle = rotation_angle
            existing.visible = visible
            existing.line_break_before = line_break_before
            existing.seqno = seqno
            existing.fill_color = fill_color
            return existing
        r = object.__new__(cls)
        c = array("d", _COORDS_TEMPLATE)
        c[cls.X0] = x0
        c[cls.Y0] = y0
        c[cls.X1] = x1
        c[cls.Y1] = y1
        c[cls.TX] = tx
        c[cls.TY] = ty
        c[cls.FONT_SIZE] = font_size
        c[cls.SPACE_WIDTH] = space_width
        r.coords = c
        r.text = text
        r.font_name = font_name
        r.order = order
        r.stream_order = stream_order
        r.xobject_depth = xobject_depth
        r.is_vertical = is_vertical
        r.rotation_angle = rotation_angle
        r.visible = visible
        r.line_break_before = line_break_before
        r.seqno = seqno
        r.fill_color = fill_color
        return r

    def replace(
        self,
        *,
        text: str | _UnsetType = _UNSET,
        x0: float | _UnsetType = _UNSET,
        y0: float | _UnsetType = _UNSET,
        x1: float | _UnsetType = _UNSET,
        y1: float | _UnsetType = _UNSET,
        tx: float | _UnsetType = _UNSET,
        ty: float | _UnsetType = _UNSET,
        font_size: float | _UnsetType = _UNSET,
        space_width: float | _UnsetType = _UNSET,
        order: int | _UnsetType = _UNSET,
        stream_order: int | _UnsetType = _UNSET,
        xobject_depth: int | _UnsetType = _UNSET,
        font_name: str | None | _UnsetType = _UNSET,
        is_vertical: bool | _UnsetType = _UNSET,
        rotation_angle: int | _UnsetType = _UNSET,
        visible: bool | _UnsetType = _UNSET,
        line_break_before: bool | _UnsetType = _UNSET,
        seqno: int | _UnsetType = _UNSET,
        fill_color: tuple[float, ...] | None | _UnsetType = _UNSET,
    ) -> TextRun:
        return TextRun(
            text=_value_or(text, self.text),
            x0=_value_or(x0, self.x0),
            y0=_value_or(y0, self.y0),
            x1=_value_or(x1, self.x1),
            y1=_value_or(y1, self.y1),
            tx=_value_or(tx, self.tx),
            ty=_value_or(ty, self.ty),
            font_size=_value_or(font_size, self.font_size),
            space_width=_value_or(space_width, self.space_width),
            order=_value_or(order, self.order),
            stream_order=_value_or(stream_order, self.stream_order),
            xobject_depth=_value_or(xobject_depth, self.xobject_depth),
            font_name=_value_or(font_name, self.font_name),
            is_vertical=_value_or(is_vertical, self.is_vertical),
            rotation_angle=_value_or(rotation_angle, self.rotation_angle),
            visible=_value_or(visible, self.visible),
            line_break_before=_value_or(line_break_before, self.line_break_before),
            seqno=_value_or(seqno, self.seqno),
            fill_color=_value_or(fill_color, self.fill_color),
        )


class LayoutLine:
    __slots__ = (
        "runs",
        "x0",
        "y0",
        "x1",
        "y1",
        "is_vertical",
        "rotation_angle",
        "max_order",
        "max_depth",
        "min_order",
        "mid_y",
        "height",
    )

    runs: list[TextRun]
    x0: float
    y0: float
    x1: float
    y1: float
    is_vertical: bool
    rotation_angle: int
    max_order: int
    max_depth: int
    min_order: int
    mid_y: float

    def __init__(
        self,
        runs: list[TextRun] | None = None,
        x0: float = 0.0,
        y0: float = 0.0,
        x1: float = 0.0,
        y1: float = 0.0,
        is_vertical: bool = False,
        rotation_angle: int = 0,
        max_order: int = -1,
        max_depth: int = -1,
        min_order: int = 999999,
        mid_y: float = 0.0,
        height: float = 0.0,
    ) -> None:
        self.runs = runs if runs is not None else []
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.is_vertical = is_vertical
        self.rotation_angle = rotation_angle
        self.max_order = max_order
        self.max_depth = max_depth
        self.min_order = min_order
        self.mid_y = mid_y
        self.height = height

    def replace(
        self,
        *,
        runs: list[TextRun] | _UnsetType = _UNSET,
        x0: float | _UnsetType = _UNSET,
        y0: float | _UnsetType = _UNSET,
        x1: float | _UnsetType = _UNSET,
        y1: float | _UnsetType = _UNSET,
        is_vertical: bool | _UnsetType = _UNSET,
        rotation_angle: int | _UnsetType = _UNSET,
        max_order: int | _UnsetType = _UNSET,
        max_depth: int | _UnsetType = _UNSET,
        min_order: int | _UnsetType = _UNSET,
        mid_y: float | _UnsetType = _UNSET,
        height: float | _UnsetType = _UNSET,
    ) -> LayoutLine:
        """Create a new LayoutLine with modified fields."""
        return LayoutLine(
            runs=_value_or(runs, self.runs),
            x0=_value_or(x0, self.x0),
            y0=_value_or(y0, self.y0),
            x1=_value_or(x1, self.x1),
            y1=_value_or(y1, self.y1),
            is_vertical=_value_or(is_vertical, self.is_vertical),
            rotation_angle=_value_or(rotation_angle, self.rotation_angle),
            max_order=_value_or(max_order, self.max_order),
            max_depth=_value_or(max_depth, self.max_depth),
            min_order=_value_or(min_order, self.min_order),
            mid_y=_value_or(mid_y, self.mid_y),
            height=_value_or(height, self.height),
        )

    def add(self, run: TextRun) -> None:
        if not self.runs:
            self.x0, self.y0, self.x1, self.y1 = run.x0, run.y0, run.x1, run.y1
            self.is_vertical = run.is_vertical
            self.rotation_angle = run.rotation_angle
            self.max_order = run.order
            self.min_order = run.order
            self.max_depth = run.xobject_depth
            self.mid_y = run.mid_y
            self.height = self.y1 - self.y0
        else:
            if run.x0 < self.x0:
                self.x0 = run.x0
            if run.y0 < self.y0:
                self.y0 = run.y0
            if run.x1 > self.x1:
                self.x1 = run.x1
            if run.y1 > self.y1:
                self.y1 = run.y1
            if run.order > self.max_order:
                self.max_order = run.order
            if run.order < self.min_order:
                self.min_order = run.order
            if run.xobject_depth > self.max_depth:
                self.max_depth = run.xobject_depth
            self.mid_y = (self.y0 + self.y1) * 0.5
            self.height = self.y1 - self.y0
        self.runs.append(run)


class LayoutBox:
    __slots__ = ("lines", "x0", "y0", "x1", "y1", "max_depth", "mid_y")

    lines: list[LayoutLine]
    x0: float
    y0: float
    x1: float
    y1: float
    max_depth: int
    mid_y: float

    def __init__(
        self,
        lines: list[LayoutLine] | None = None,
        x0: float = 0.0,
        y0: float = 0.0,
        x1: float = 0.0,
        y1: float = 0.0,
        max_depth: int = -1,
        mid_y: float = 0.0,
    ) -> None:
        self.lines = lines if lines is not None else []
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.max_depth = max_depth
        self.mid_y = mid_y

    def replace(
        self,
        *,
        lines: list[LayoutLine] | _UnsetType = _UNSET,
        x0: float | _UnsetType = _UNSET,
        y0: float | _UnsetType = _UNSET,
        x1: float | _UnsetType = _UNSET,
        y1: float | _UnsetType = _UNSET,
        max_depth: int | _UnsetType = _UNSET,
        mid_y: float | _UnsetType = _UNSET,
    ) -> LayoutBox:
        """Create a new LayoutBox with modified fields."""
        return LayoutBox(
            lines=_value_or(lines, self.lines),
            x0=_value_or(x0, self.x0),
            y0=_value_or(y0, self.y0),
            x1=_value_or(x1, self.x1),
            y1=_value_or(y1, self.y1),
            max_depth=_value_or(max_depth, self.max_depth),
            mid_y=_value_or(mid_y, self.mid_y),
        )

    @property
    def bbox_rect(self) -> RectBox:
        from core_pdf.impl.engine.spec.s_08_graphics.geometry import RectBox

        return RectBox(self.x0, self.y0, self.x1, self.y1)

    def add(self, line: LayoutLine) -> None:
        if not self.lines:
            self.x0, self.y0, self.x1, self.y1 = (line.x0, line.y0, line.x1, line.y1)
            self.max_depth = line.max_depth
            self.mid_y = line.mid_y
        else:
            if line.x0 < self.x0:
                self.x0 = line.x0
            if line.y0 < self.y0:
                self.y0 = line.y0
            if line.x1 > self.x1:
                self.x1 = line.x1
            if line.y1 > self.y1:
                self.y1 = line.y1
            if line.max_depth > self.max_depth:
                self.max_depth = line.max_depth
            self.mid_y = (self.y0 + self.y1) * 0.5
        self.lines.append(line)


class TableGrid:
    __slots__ = ("cols", "rows")

    cols: list[float]
    rows: list[float]

    def __init__(self, cols: list[float], rows: list[float]) -> None:
        if len(cols) < 2 or len(rows) < 2:
            raise ValueError("invalid table grid")
        if not all(isinstance(v, (int, float)) for v in cols):
            raise ValueError("invalid table grid")
        if not all(isinstance(v, (int, float)) for v in rows):
            raise ValueError("invalid table grid")
        if any(cols[i] > cols[i + 1] for i in range(len(cols) - 1)):
            raise ValueError("invalid table grid")
        if any(rows[i] < rows[i + 1] for i in range(len(rows) - 1)):
            raise ValueError("invalid table grid")
        self.cols = cols
        self.rows = rows

    def is_valid(self) -> bool:
        return len(self.cols) >= 2 and len(self.rows) >= 2

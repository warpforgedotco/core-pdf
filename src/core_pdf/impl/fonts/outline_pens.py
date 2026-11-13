# SPDX-License-Identifier: AGPL-3.0-only

"""Glyph outlines as pen operations, for programs whose outlines the compiled kernels
do not draw: OpenType CFF and CFF2 glyphs, and the fallbacks of TrueType and Type 1.

A recording is a list of ``(operator, points)``: ``moveTo``, ``lineTo``, ``curveTo``,
``qCurveTo``, ``closePath``, and ``addComponent`` with ``(glyph name, transform)``.
draw_decomposed() replaces components by the glyphs they name, the way fontTools'
decomposing pens do: a component whose glyph is missing is skipped, transforms are
composed in fontTools' order, and points are transformed with the same arithmetic, so
the coordinates match fontTools' to the last bit. outline_bounds() is fontTools'
BoundsPen: control points, plus each curve's exact extremes where a control point lies
outside the box so far.

The Type 2 interpreter follows fontTools' outline extractor, including where it is
lenient: an unknown operator ends its charstring, ``return`` and ``endchar`` do not stop
it, and ``blend`` keeps each value's default and drops its deltas.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from math import sqrt
from typing import Any

from core_pdf.impl.adobe_fonts._vendor.font_data.encoding_names import STANDARD_ENCODING_GLYPH_NAMES

Point = tuple[float, float]
Transform = tuple[float, float, float, float, float, float]
Operation = tuple[str, tuple[Any, ...]]
Recording = list[Operation]

IDENTITY = (1, 0, 0, 1, 0, 0)


def compose(outer: Sequence[float], inner: Sequence[float]) -> Transform:
    """outer.transform(inner) in fontTools' Transform terms."""
    xx1, xy1, yx1, yy1, dx1, dy1 = inner
    xx2, xy2, yx2, yy2, dx2, dy2 = outer
    return (
        xx1 * xx2 + xy1 * yx2,
        xx1 * xy2 + xy1 * yy2,
        yx1 * xx2 + yy1 * yx2,
        yx1 * xy2 + yy1 * yy2,
        xx2 * dx1 + yx2 * dy1 + dx2,
        xy2 * dx1 + yy2 * dy1 + dy2,
    )


def transform_point(transform: Sequence[float], point: Point) -> Point:
    x, y = point
    xx, xy, yx, yy, dx, dy = transform
    return (xx * x + yx * y + dx, xy * x + yy * y + dy)


def transform_operation(operation: Operation, transform: Sequence[float]) -> Operation:
    operator, points = operation
    return (
        operator,
        tuple(point if point is None else transform_point(transform, point) for point in points),
    )


def draw_decomposed(
    record: Callable[[str], Recording],
    name: str,
    transform: Sequence[float] | None,
    emit: Callable[[Operation], None],
    has_glyph: Callable[[str], bool],
) -> None:
    """Draw glyph ``name`` through ``transform`` into ``emit``, components in place.

    ``record(name)`` returns a glyph's own operations; ``has_glyph(name)`` is False for a
    component to skip. A transform of None leaves points as they are; the top level
    passes the transform its pen applies (fontTools applies it even when it is the
    identity), components pass none when theirs is the identity.
    """
    for operation in record(name):
        if operation[0] == "addComponent":
            component, component_transform = operation[1]
            combined = (
                tuple(component_transform)
                if transform is None
                else compose(transform, component_transform)
            )
            if not has_glyph(component):
                continue
            draw_decomposed(
                record,
                component,
                None if tuple(combined) == IDENTITY else combined,
                emit,
                has_glyph,
            )
        elif transform is None:
            emit(operation)
        else:
            emit(transform_operation(operation, transform))


# Bounds, as fontTools' BoundsPen computes them.

EPSILON = 1e-10


def solve_quadratic(a: float, b: float, c: float) -> list[float]:
    if abs(a) < EPSILON:
        return [] if abs(b) < EPSILON else [-c / b]
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return []
    root = sqrt(discriminant)
    return [(-b + root) / 2.0 / a, (-b - root) / 2.0 / a]


def points_bounds(points: list[Point]) -> tuple[float, float, float, float]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


def cubic_bounds(p1: Point, p2: Point, p3: Point, p4: Point) -> tuple[float, float, float, float]:
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    dx, dy = p1
    cx = (x2 - dx) * 3.0
    cy = (y2 - dy) * 3.0
    bx = (x3 - x2) * 3.0 - cx
    by = (y3 - y2) * 3.0 - cy
    ax = x4 - dx - cx - bx
    ay = y4 - dy - cy - by
    roots = [t for t in solve_quadratic(ax * 3.0, bx * 2.0, cx) if 0 <= t < 1]
    roots += [t for t in solve_quadratic(ay * 3.0, by * 2.0, cy) if 0 <= t < 1]
    points = [
        (
            ax * t * t * t + bx * t * t + cx * t + dx,
            ay * t * t * t + by * t * t + cy * t + dy,
        )
        for t in roots
    ]
    return points_bounds([*points, p1, p4])


def quadratic_bounds(p1: Point, p2: Point, p3: Point) -> tuple[float, float, float, float]:
    x2, y2 = p2
    x3, y3 = p3
    cx, cy = p1
    bx = (x2 - cx) * 2.0
    by = (y2 - cy) * 2.0
    ax = x3 - cx - bx
    ay = y3 - cy - by
    roots = []
    if ax * 2.0 != 0:
        roots.append(-bx / (ax * 2.0))
    if ay * 2.0 != 0:
        roots.append(-by / (ay * 2.0))
    points = [(ax * t * t + bx * t + cx, ay * t * t + by * t + cy) for t in roots if 0 <= t < 1]
    return points_bounds([*points, p1, p3])


def update_bounds(
    bounds: tuple[float, float, float, float] | None, point: Point
) -> tuple[float, float, float, float]:
    x, y = point
    if bounds is None:
        return (x, y, x, y)
    x0, y0, x1, y1 = bounds
    return (min(x0, x), min(y0, y), max(x1, x), max(y1, y))


def in_bounds(point: Point, bounds: tuple[float, float, float, float]) -> bool:
    x, y = point
    x0, y0, x1, y1 = bounds
    return x0 <= x <= x1 and y0 <= y <= y1


def union(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


class BoundsSink:
    """Receives drawing operations and keeps fontTools' BoundsPen box."""

    __slots__ = ("bounds", "current", "start")

    def __init__(self) -> None:
        self.bounds: tuple[float, float, float, float] | None = None
        self.current: Point | None = None

    def __call__(self, operation: Operation) -> None:
        operator, points = operation
        match operator:
            case "moveTo":
                self.bounds = update_bounds(self.bounds, points[0])
                self.current = points[0]
            case "lineTo":
                self.bounds = update_bounds(self.bounds, points[0])
                self.current = points[0]
            case "curveTo":
                self.curve(points)
            case "qCurveTo":
                self.quadratic(points)
            case "closePath" | "endPath":
                self.current = None

    def curve(self, points: tuple[Any, ...]) -> None:
        count = len(points) - 1
        if count == 2:
            self.cubic(*points)
        elif count == 1:
            self.quadratic((points[0], points[1]))
        elif count == 0:
            self.bounds = update_bounds(self.bounds, points[0])
            self.current = points[0]
        else:
            raise ValueError("unsupported curve: more than two off-curve points")

    def cubic(self, c1: Point, c2: Point, end: Point) -> None:
        bounds = update_bounds(self.bounds, end)
        if not in_bounds(c1, bounds) or not in_bounds(c2, bounds):
            if self.current is None:
                raise ValueError("curve without a current point")
            bounds = union(bounds, cubic_bounds(self.current, c1, c2, end))
        self.bounds = bounds
        self.current = end

    def quadratic(self, points: tuple[Any, ...]) -> None:
        if len(points) != 2 or points[-1] is None:
            raise ValueError("unsupported quadratic curve")
        control, end = points
        bounds = update_bounds(self.bounds, end)
        if not in_bounds(control, bounds):
            if self.current is None:
                raise ValueError("curve without a current point")
            bounds = union(bounds, quadratic_bounds(self.current, control, end))
        self.bounds = bounds
        self.current = end


# Type 2 charstrings.

STANDARD_ENCODING = STANDARD_ENCODING_GLYPH_NAMES


def operand_index(value: float) -> int:
    """An operand used as an index; a fraction fails, as indexing with it does."""
    if not isinstance(value, int):
        raise TypeError("charstring index is not an integer")
    return value


def subroutine_bias(subroutines: Sequence[bytes]) -> int:
    count = len(subroutines)
    if count < 1240:
        return 107
    if count < 33900:
        return 1131
    return 32768


TYPE2_OPERATORS: dict[int | tuple[int, int], str] = {
    1: "hstem",
    3: "vstem",
    4: "vmoveto",
    5: "rlineto",
    6: "hlineto",
    7: "vlineto",
    8: "rrcurveto",
    10: "callsubr",
    11: "return",
    14: "endchar",
    15: "vsindex",
    16: "blend",
    18: "hstemhm",
    19: "hintmask",
    20: "cntrmask",
    21: "rmoveto",
    22: "hmoveto",
    23: "vstemhm",
    24: "rcurveline",
    25: "rlinecurve",
    26: "vvcurveto",
    27: "hhcurveto",
    29: "callgsubr",
    30: "vhcurveto",
    31: "hvcurveto",
    (12, 0): "ignore",
    (12, 12): "div",
    (12, 34): "hflex",
    (12, 35): "flex",
    (12, 36): "hflex1",
    (12, 37): "flex1",
}
# The other arithmetic and storage operators: fontTools does not implement them, so a
# charstring using one fails to draw.
UNSUPPORTED_OPERATORS = frozenset(
    (12, code)
    for code in (3, 4, 5, 8, 9, 10, 11, 13, 14, 15, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30)
)


class Type2Outline:
    """Draws one Type 2 charstring, with fresh state, into a recording."""

    __slots__ = (
        "local_subrs",
        "local_bias",
        "global_subrs",
        "global_bias",
        "region_count",
        "default_vsindex",
        "default_width",
        "stack",
        "calls",
        "hint_count",
        "hint_mask_bytes",
        "regions",
        "vsindex",
        "got_width",
        "current",
        "saw_move",
        "depth",
        "recording",
    )

    def __init__(
        self,
        local_subrs: Sequence[bytes],
        global_subrs: Sequence[bytes],
        *,
        region_count: Callable[[int], int] | None = None,
        default_vsindex: int = 0,
        cff2: bool = False,
    ) -> None:
        self.local_subrs = local_subrs
        self.local_bias = subroutine_bias(local_subrs)
        self.global_subrs = global_subrs
        self.global_bias = subroutine_bias(global_subrs)
        self.region_count = region_count
        self.default_vsindex = default_vsindex
        # CFF2 charstrings have no width; an odd operand count where one could be fails.
        self.default_width: int | None = None if cff2 else 0

    def record(self, charstring: bytes) -> Recording:
        self.stack: list[int | float] = []
        self.calls: list[bytes] = []
        self.hint_count = 0
        self.hint_mask_bytes = 0
        self.regions = 0
        self.vsindex = 0
        self.got_width = False
        self.current: Point = (0, 0)
        self.saw_move = False
        self.depth = 0
        self.recording: Recording = []
        self.execute(charstring)
        return self.recording

    # Execution.

    def execute(self, charstring: bytes) -> None:
        self.depth += 1
        self.calls.append(charstring)
        index = 0
        length = len(charstring)
        while index < length:
            b0 = charstring[index]
            index += 1
            if b0 < 32 and b0 != 28:
                if b0 == 12:
                    operator: int | tuple[int, int] = (12, charstring[index])
                    index += 1
                else:
                    operator = b0
                name = TYPE2_OPERATORS.get(operator)
                if name is None:
                    if operator in UNSUPPORTED_OPERATORS:
                        raise NotImplementedError(f"Type 2 operator {operator}")
                    break
                index = self.operate(name, index)
            elif b0 == 28:
                (value,) = struct.unpack(">h", charstring[index : index + 2])
                index += 2
                self.stack.append(value)
            elif b0 < 247:
                self.stack.append(b0 - 139)
            elif b0 < 251:
                self.stack.append((b0 - 247) * 256 + charstring[index] + 108)
                index += 1
            elif b0 < 255:
                self.stack.append(-(b0 - 251) * 256 - charstring[index] - 108)
                index += 1
            else:
                (value,) = struct.unpack(">l", charstring[index : index + 4])
                index += 4
                self.stack.append(value / 65536)
        del self.calls[-1]
        self.depth -= 1
        if self.depth == 0:
            self.end_path()

    def popall(self) -> list[int | float]:
        stack = self.stack[:]
        self.stack[:] = []
        return stack

    def pop_with_width(self, even_odd: int = 0) -> list[int | float]:
        args = self.popall()
        if not self.got_width:
            if even_odd ^ (len(args) % 2):
                if self.default_width is None:
                    raise ValueError("CFF2 charstrings have no width")
                args = args[1:]
            self.got_width = True
        return args

    def operate(self, name: str, index: int) -> int:  # noqa: C901, PLR0912
        match name:
            case "callsubr":
                subr_index = self.stack.pop()
                self.execute(self.local_subrs[operand_index(subr_index) + self.local_bias])
            case "callgsubr":
                subr_index = self.stack.pop()
                self.execute(self.global_subrs[operand_index(subr_index) + self.global_bias])
            case "return" | "ignore":
                pass
            case "hstem" | "vstem" | "hstemhm" | "vstemhm":
                self.count_hints()
            case "hintmask" | "cntrmask":
                if not self.hint_mask_bytes:
                    self.count_hints()
                    self.hint_mask_bytes = (self.hint_count + 7) // 8
                end = index + self.hint_mask_bytes
                if len(self.calls[-1][index:end]) != self.hint_mask_bytes:
                    raise ValueError("truncated hint mask")
                index = end
            case "blend":
                self.blend()
            case "vsindex":
                self.vsindex = operand_index(self.stack.pop())
                self.regions = self.regions_for(self.vsindex)
            case "div":
                divisor = self.stack.pop()
                dividend = self.stack.pop()
                whole = dividend // divisor
                exact = dividend / divisor
                self.stack.append(whole if whole == exact else exact)
            case "rmoveto":
                self.end_path()
                self.move(self.pop_with_width())
            case "hmoveto":
                self.end_path()
                self.move((self.pop_with_width(1)[0], 0))
            case "vmoveto":
                self.end_path()
                self.move((0, self.pop_with_width(1)[0]))
            case "endchar":
                self.end_path()
                args = self.pop_with_width()
                if args:
                    adx, ady, base, accent = args
                    self.recording.append(
                        ("addComponent", (STANDARD_ENCODING[operand_index(base)], IDENTITY))
                    )
                    self.recording.append(
                        (
                            "addComponent",
                            (STANDARD_ENCODING[operand_index(accent)], (1, 0, 0, 1, adx, ady)),
                        )
                    )
            case "rlineto":
                args = self.popall()
                for i in range(0, len(args), 2):
                    self.line(args[i : i + 2])
            case "hlineto":
                self.alternating_lines(horizontal=True)
            case "vlineto":
                self.alternating_lines(horizontal=False)
            case "rrcurveto":
                args = self.popall()
                for i in range(0, len(args), 6):
                    dxa, dya, dxb, dyb, dxc, dyc = args[i : i + 6]
                    self.curve((dxa, dya), (dxb, dyb), (dxc, dyc))
            case "rcurveline":
                args = self.popall()
                for i in range(0, len(args) - 2, 6):
                    dxb, dyb, dxc, dyc, dxd, dyd = args[i : i + 6]
                    self.curve((dxb, dyb), (dxc, dyc), (dxd, dyd))
                self.line(args[-2:])
            case "rlinecurve":
                args = self.popall()
                lines = args[:-6]
                for i in range(0, len(lines), 2):
                    self.line(lines[i : i + 2])
                dxb, dyb, dxc, dyc, dxd, dyd = args[-6:]
                self.curve((dxb, dyb), (dxc, dyc), (dxd, dyd))
            case "vvcurveto":
                args = self.popall()
                dx1: int | float = 0
                if len(args) % 2:
                    dx1 = args[0]
                    args = args[1:]
                for i in range(0, len(args), 4):
                    dya, dxb, dyb, dyc = args[i : i + 4]
                    self.curve((dx1, dya), (dxb, dyb), (0, dyc))
                    dx1 = 0
            case "hhcurveto":
                args = self.popall()
                dy1: int | float = 0
                if len(args) % 2:
                    dy1 = args[0]
                    args = args[1:]
                for i in range(0, len(args), 4):
                    dxa, dxb, dyb, dxc = args[i : i + 4]
                    self.curve((dxa, dy1), (dxb, dyb), (dxc, 0))
                    dy1 = 0
            case "vhcurveto":
                args = self.popall()
                while args:
                    args = self.vertical_curve(args)
                    if args:
                        args = self.horizontal_curve(args)
            case "hvcurveto":
                args = self.popall()
                while args:
                    args = self.horizontal_curve(args)
                    if args:
                        args = self.vertical_curve(args)
            case "hflex":
                dx1, dx2, dy2, dx3, dx4, dx5, dx6 = self.popall()
                self.curve((dx1, 0), (dx2, dy2), (dx3, 0))
                self.curve((dx4, 0), (dx5, -dy2), (dx6, 0))
            case "flex":
                dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5, dx6, dy6, _fd = self.popall()
                self.curve((dx1, dy1), (dx2, dy2), (dx3, dy3))
                self.curve((dx4, dy4), (dx5, dy5), (dx6, dy6))
            case "hflex1":
                dx1, dy1, dx2, dy2, dx3, dx4, dx5, dy5, dx6 = self.popall()
                dy6 = -(dy1 + dy2 + 0 + 0 + dy5)
                self.curve((dx1, dy1), (dx2, dy2), (dx3, 0))
                self.curve((dx4, 0), (dx5, dy5), (dx6, dy6))
            case "flex1":
                dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5, d6 = self.popall()
                dx = dx1 + dx2 + dx3 + dx4 + dx5
                dy = dy1 + dy2 + dy3 + dy4 + dy5
                if abs(dx) > abs(dy):
                    dx6, dy6 = d6, -dy
                else:
                    dx6, dy6 = -dx, d6
                self.curve((dx1, dy1), (dx2, dy2), (dx3, dy3))
                self.curve((dx4, dy4), (dx5, dy5), (dx6, dy6))
        return index

    def count_hints(self) -> None:
        self.hint_count += len(self.pop_with_width()) // 2

    def regions_for(self, vsindex: int) -> int:
        if self.region_count is None:
            raise ValueError("blend outside a CFF2 font")
        return self.region_count(vsindex)

    def blend(self) -> None:
        if self.regions == 0:
            self.regions = self.regions_for(self.default_vsindex)
        count = operand_index(self.stack.pop())
        operands = count * (self.regions + 1)
        del self.stack[-(operands - count) :]

    # Drawing.

    def next_point(self, delta: Sequence[int | float]) -> Point:
        x, y = self.current
        point = (x + delta[0], y + delta[1])
        self.current = point
        return point

    def move(self, delta: Sequence[int | float]) -> None:
        self.recording.append(("moveTo", (self.next_point(delta),)))
        self.saw_move = True

    def line(self, delta: Sequence[int | float]) -> None:
        if not self.saw_move:
            self.move((0, 0))
        self.recording.append(("lineTo", (self.next_point(delta),)))

    def curve(
        self,
        d1: Sequence[int | float],
        d2: Sequence[int | float],
        d3: Sequence[int | float],
    ) -> None:
        if not self.saw_move:
            self.move((0, 0))
        p1 = self.next_point(d1)
        p2 = self.next_point(d2)
        p3 = self.next_point(d3)
        self.recording.append(("curveTo", (p1, p2, p3)))

    def end_path(self) -> None:
        if self.saw_move:
            self.recording.append(("closePath", ()))
        self.saw_move = False

    def alternating_lines(self, *, horizontal: bool) -> None:
        for arg in self.popall():
            self.line((arg, 0) if horizontal else (0, arg))
            horizontal = not horizontal

    def vertical_curve(self, args: list[int | float]) -> list[int | float]:
        dya, dxb, dyb, dxc = args[:4]
        args = args[4:]
        dyc: int | float = 0
        if len(args) == 1:
            dyc = args[0]
            args = []
        self.curve((0, dya), (dxb, dyb), (dxc, dyc))
        return args

    def horizontal_curve(self, args: list[int | float]) -> list[int | float]:
        dxa, dxb, dyb, dyc = args[:4]
        args = args[4:]
        dxc: int | float = 0
        if len(args) == 1:
            dxc = args[0]
            args = []
        self.curve((dxa, 0), (dxb, dyb), (dxc, dyc))
        return args


TYPE1_OPERATORS: dict[int | tuple[int, int], str] = {
    1: "hstem",
    3: "vstem",
    4: "vmoveto",
    5: "rlineto",
    6: "hlineto",
    7: "vlineto",
    8: "rrcurveto",
    9: "closepath",
    10: "callsubr",
    11: "return",
    13: "hsbw",
    14: "endchar",
    21: "rmoveto",
    22: "hmoveto",
    30: "vhcurveto",
    31: "hvcurveto",
    (12, 0): "dotsection",
    (12, 1): "vstem3",
    (12, 2): "hstem3",
    (12, 6): "seac",
    (12, 7): "sbw",
    (12, 12): "div",
    (12, 16): "callothersubr",
    (12, 17): "pop",
    (12, 33): "setcurrentpoint",
}


class Type1Outline(Type2Outline):
    """Draws one Type 1 charstring (decrypted) into a recording, the way fontTools'
    Type 1 outline extractor does: subroutines by plain index, flex through othersubrs
    0 and 1, the start point set by hsbw, and seac as two components whose accent is
    offset by its adx plus the glyph's side bearing minus asb."""

    __slots__ = ("subrs", "flexing", "sbx")

    def __init__(self, subrs: Sequence[bytes]) -> None:
        self.subrs = subrs

    def record(self, charstring: bytes) -> Recording:
        self.flexing = False
        self.sbx: int | float = 0
        return super().record(charstring)

    def execute(self, charstring: bytes) -> None:
        self.depth += 1
        index = 0
        length = len(charstring)
        while index < length:
            b0 = charstring[index]
            index += 1
            if b0 < 32:
                if b0 == 12:
                    operator: int | tuple[int, int] = (12, charstring[index])
                    index += 1
                else:
                    operator = b0
                name = TYPE1_OPERATORS.get(operator)
                if name is None:
                    break
                self.operate_type1(name)
            elif b0 < 247:
                self.stack.append(b0 - 139)
            elif b0 < 251:
                self.stack.append((b0 - 247) * 256 + charstring[index] + 108)
                index += 1
            elif b0 < 255:
                self.stack.append(-(b0 - 251) * 256 - charstring[index] - 108)
                index += 1
            else:
                (value,) = struct.unpack(">l", charstring[index : index + 4])
                index += 4
                self.stack.append(value)
        self.depth -= 1
        if self.depth == 0:
            self.end_type1_path()

    def end_type1_path(self) -> None:
        if self.saw_move:
            self.recording.append(("endPath", ()))
        self.saw_move = False

    def operate_type1(self, name: str) -> None:  # noqa: C901, PLR0912
        stack = self.stack
        match name:
            case "rmoveto":
                if self.flexing:
                    return
                self.end_type1_path()
                self.move(self.popall())
            case "hmoveto":
                if self.flexing:
                    stack.append(0)
                    return
                self.end_type1_path()
                self.move((self.popall()[0], 0))
            case "vmoveto":
                if self.flexing:
                    stack.append(0)
                    stack[-1], stack[-2] = stack[-2], stack[-1]
                    return
                self.end_type1_path()
                self.move((0, self.popall()[0]))
            case "closepath":
                self.end_path()
            case "setcurrentpoint":
                x, y = self.popall()
                self.current = (x, y)
            case "endchar":
                self.end_type1_path()
            case "hsbw":
                sbx, _width = self.popall()
                self.sbx = sbx
                self.current = (sbx, self.current[1])
            case "sbw" | "dotsection" | "hstem3" | "vstem3":
                self.popall()
            case "hstem" | "vstem":
                self.hint_count += len(self.popall()) // 2
            case "callsubr":
                subr_index = stack.pop()
                self.execute(self.subrs[operand_index(subr_index)])
            case "callothersubr":
                subr_index = stack.pop()
                count = stack.pop()
                if subr_index == 0 and count == 3:
                    self.flex()
                    self.flexing = False
                elif subr_index == 1 and count == 0:
                    self.flexing = True
            case "return" | "pop":
                pass
            case "div":
                self.operate("div", 0)
            case "seac":
                asb, adx, ady, base, accent = self.popall()
                self.recording.append(
                    ("addComponent", (STANDARD_ENCODING[operand_index(base)], IDENTITY))
                )
                adx = adx + self.sbx - asb
                self.recording.append(
                    (
                        "addComponent",
                        (STANDARD_ENCODING[operand_index(accent)], (1, 0, 0, 1, adx, ady)),
                    )
                )
            case _:
                self.operate(name, 0)

    def flex(self) -> None:
        stack = self.stack
        final_y = stack.pop()
        final_x = stack.pop()
        stack.pop()
        p3y, p3x = stack.pop(), stack.pop()
        c4y, c4x = stack.pop(), stack.pop()
        c3y, c3x = stack.pop(), stack.pop()
        p2y, p2x = stack.pop(), stack.pop()
        c2y, c2x = stack.pop(), stack.pop()
        c1y, c1x = stack.pop(), stack.pop()
        ry, rx = stack.pop(), stack.pop()
        stack.extend((c1x + rx, c1y + ry, c2x, c2y, p2x, p2y))
        self.operate("rrcurveto", 0)
        stack.extend((c3x, c3y, c4x, c4y, p3x, p3y))
        self.operate("rrcurveto", 0)
        stack.extend((final_x, final_y))


__all__ = (
    "IDENTITY",
    "BoundsSink",
    "Recording",
    "Type1Outline",
    "Type2Outline",
    "compose",
    "draw_decomposed",
    "transform_point",
)

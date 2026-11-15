# SPDX-License-Identifier: AGPL-3.0-only

"""Check rendered strokes against a Python reference of their outlines.

A stroke covers, for each segment of a subpath, the rectangle a line width wide around it
(ISO 32000-2 8.5.3.2); where two segments meet at an angle, the line join adds a miter, a
round arc or a bevel triangle on the outer side (8.4.3.4), the miter becoming a bevel when
the miter length over the line width, 1 / sin(angle / 2), passes the miter limit (8.4.3.5);
an open subpath, and every dash, ends in butt, round or projecting square caps (8.4.3.3); a
dash pattern turns the path into dashes measured along it, a dash wrapping a closed
subpath's start joined into one (8.4.3.6); a subpath of one point paints a dot with round
caps and nothing otherwise (8.5.3.2). A gs operator's LW, LC, LJ, ML and D set the same
parameters as w, J, j, M and d (8.4.5).

This writes PDFs that stroke polylines with every join and cap, at acute, right and obtuse
angles, with miter limits just above and just below a join's ratio, through zero-length
segments, closed and open, dashed, scaled and rotated by the CTM, under a blend mode, and
with the parameters set by gs, renders each with core, and works out in Python whether each
pixel centre lies inside the stroke's outline. Pixels within a device pixel and a half of
the outline, where coverage blends, are skipped; every other pixel must be the stroke's
opaque black or untouched.

uv run --all-packages python scripts/check_strokes.py
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence

import numpy

from core_pdf import PdfDocument

type Point = tuple[float, float]
type Affine = tuple[float, float, float, float, float, float]
# A convex polygon, its vertices in order, or a disc: (centre, radius).
type Shape = tuple[str, tuple[Point, ...], float]

IDENTITY: Affine = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
SIZE = 300
MARGIN = 1.5
BUTT, ROUND, SQUARE = 0, 1, 2
MITER, ROUND_JOIN, BEVEL = 0, 1, 2


def numbers(values: Sequence[float]) -> bytes:
    return " ".join(f"{value:.6g}" for value in values).encode()


def apply(matrix: Affine, point: Point) -> Point:
    a, b, c, d, e, f = matrix
    x, y = point
    return a * x + c * y + e, b * x + d * y + f


def rotation(degrees: float, scale: float = 1.0, e: float = 0.0, f: float = 0.0) -> Affine:
    radians = math.radians(degrees)
    c, s = math.cos(radians) * scale, math.sin(radians) * scale
    return (c, s, -s, c, e, f)


def polar(origin: Point, length: float, degrees: float) -> Point:
    radians = math.radians(degrees)
    return origin[0] + length * math.cos(radians), origin[1] + length * math.sin(radians)


class Stroke:
    """One stroked subpath and its parameters, in user space."""

    def __init__(
        self,
        points: Sequence[Point],
        *,
        closed: bool = False,
        width: float = 16.0,
        cap: int = BUTT,
        join: int = MITER,
        miter_limit: float = 10.0,
        dash: tuple[Sequence[float], float] | None = None,
        explicit_close: bool = False,
    ) -> None:
        # As the content stream writes them, so that the reference strokes the same path.
        self.points = tuple((float(f"{x:.6g}"), float(f"{y:.6g}")) for x, y in points)
        self.closed = closed
        self.width = width
        self.cap = cap
        self.join = join
        self.miter_limit = miter_limit
        self.dash = dash
        # Repeat the first point before h, as some producers do.
        self.explicit_close = explicit_close

    def path(self) -> bytes:
        first, *rest = self.points
        if self.explicit_close:
            rest = [*rest, first]
        out = b"%s m " % numbers(first) + b"".join(b"%s l " % numbers(point) for point in rest)
        return out + (b"h S" if self.closed else b"S")

    def operators(self) -> bytes:
        dash = b"[] 0 d"
        if self.dash is not None:
            dash = b"[%s] %s d" % (numbers(self.dash[0]), numbers([self.dash[1]]))
        return b"%s w %d J %d j %s M %s " % (
            numbers([self.width]),
            self.cap,
            self.join,
            numbers([self.miter_limit]),
            dash,
        )


def unit(a: Point, b: Point) -> Point:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def segment_box(a: Point, b: Point, half: float) -> Shape:
    ux, uy = unit(a, b)
    nx, ny = -uy * half, ux * half
    return (
        "polygon",
        (
            (a[0] + nx, a[1] + ny),
            (b[0] + nx, b[1] + ny),
            (b[0] - nx, b[1] - ny),
            (a[0] - nx, a[1] - ny),
        ),
        0.0,
    )


def cap_shapes(point: Point, direction: Point, half: float, cap: int) -> list[Shape]:
    """The cap at point of a stroke leaving it along direction, outward."""
    if cap == ROUND:
        return [("disc", (point,), half)]
    if cap == SQUARE:
        ahead = (point[0] + direction[0] * half, point[1] + direction[1] * half)
        return [segment_box(point, ahead, half)]
    return []


def join_shapes(
    vertex: Point, incoming: Point, outgoing: Point, half: float, join: int, limit: float
) -> list[Shape]:
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    if join == ROUND_JOIN:
        return [("disc", (vertex,), half)]
    if abs(cross) < 1e-12:
        return []
    # The outer side: right of the path on a left turn, left on a right turn.
    side = -1.0 if cross > 0 else 1.0
    a = (vertex[0] - side * incoming[1] * half, vertex[1] + side * incoming[0] * half)
    b = (vertex[0] - side * outgoing[1] * half, vertex[1] + side * outgoing[0] * half)
    angle = math.acos(max(-1.0, min(1.0, -dot)))
    if join == MITER and 1.0 / math.sin(angle / 2.0) <= limit:
        reach = half / math.sin(angle / 2.0)
        mx, my = a[0] + b[0] - 2 * vertex[0], a[1] + b[1] - 2 * vertex[1]
        length = math.hypot(mx, my)
        tip = (vertex[0] + mx / length * reach, vertex[1] + my / length * reach)
        return [("polygon", (vertex, a, tip, b), 0.0)]
    return [("polygon", (vertex, a, b), 0.0)]


def polyline_shapes(
    points: Sequence[Point], closed: bool, stroke: Stroke, half: float
) -> list[Shape]:
    distinct = [points[0]]
    for point in points[1:]:
        if point != distinct[-1]:
            distinct.append(point)
    if closed and len(distinct) > 1 and distinct[-1] == distinct[0]:
        distinct.pop()
    if len(distinct) == 1:
        return [("disc", (distinct[0],), half)] if stroke.cap == ROUND else []
    vertices = [*distinct, distinct[0]] if closed else distinct
    segments = list(zip(vertices, vertices[1:]))
    shapes = [segment_box(a, b, half) for a, b in segments]
    directions = [unit(a, b) for a, b in segments]
    pairs = list(range(len(segments) - 1))
    for index in pairs:
        shapes += join_shapes(
            segments[index][1],
            directions[index],
            directions[index + 1],
            half,
            stroke.join,
            stroke.miter_limit,
        )
    if closed:
        shapes += join_shapes(
            vertices[0], directions[-1], directions[0], half, stroke.join, stroke.miter_limit
        )
    else:
        start = directions[0]
        shapes += cap_shapes(distinct[0], (-start[0], -start[1]), half, stroke.cap)
        shapes += cap_shapes(distinct[-1], directions[-1], half, stroke.cap)
    return shapes


def dashes(
    points: Sequence[Point], closed: bool, pattern: Sequence[float], phase: float
) -> list[tuple[list[Point], bool, Point | None]]:
    """The dashes of a subpath: (points, closed, the direction of a zero-length dash)."""
    lengths = list(pattern) * (2 if len(pattern) % 2 else 1)
    total = sum(lengths)
    vertices = [*points, points[0]] if closed else list(points)
    position = phase % total
    index = 0
    while position > 0.0 and position >= lengths[index]:
        position -= lengths[index]
        index = (index + 1) % len(lengths)
    left = lengths[index] - position
    found: list[tuple[list[Point], bool, Point | None]] = []
    current: list[Point] = []
    starts_on = index % 2 == 0
    for a, b in zip(vertices, vertices[1:]):
        length = math.dist(a, b)
        if length == 0.0:
            continue
        direction = unit(a, b)
        done = 0.0
        while True:
            on = index % 2 == 0
            if on and not current:
                current.append(polar_along(a, direction, done))
            if left > length - done:
                left -= length - done
                if on:
                    current.append(b)
                break
            done += left
            point = polar_along(a, direction, done)
            if on:
                if lengths[index] == 0.0:
                    found.append(([point], False, direction))
                else:
                    current.append(point)
                    found.append((current, False, None))
                current = []
            index = (index + 1) % len(lengths)
            left = lengths[index]
    if current:
        found.append((current, False, None))
    if closed and starts_on and found and current and len(found) > 1:
        last = found.pop()
        found[0] = ([*last[0], *found[0][0][1:]], False, None)
    elif closed and starts_on and len(found) == 1 and current:
        found[0] = (found[0][0], True, None)
    return found


def polar_along(origin: Point, direction: Point, distance: float) -> Point:
    return origin[0] + direction[0] * distance, origin[1] + direction[1] * distance


def stroke_shapes(stroke: Stroke, ctm: Affine) -> list[Shape]:
    """The outline of a stroke in device space, the CTM a uniform scale and rotation."""
    scale = math.sqrt(abs(ctm[0] * ctm[3] - ctm[1] * ctm[2]))
    half = stroke.width * scale / 2.0
    points = [apply(ctm, point) for point in stroke.points]
    if stroke.dash is None:
        return polyline_shapes(points, stroke.closed, stroke, half)
    pattern = [value * scale for value in stroke.dash[0]]
    shapes: list[Shape] = []
    for piece, closed, direction in dashes(points, stroke.closed, pattern, stroke.dash[1] * scale):
        if direction is not None:
            point = piece[0]
            if stroke.cap == ROUND:
                shapes.append(("disc", (point,), half))
            elif stroke.cap == SQUARE:
                back = (point[0] - direction[0] * half, point[1] - direction[1] * half)
                ahead = (point[0] + direction[0] * half, point[1] + direction[1] * half)
                shapes.append(segment_box(back, ahead, half))
            continue
        shapes += polyline_shapes(piece, closed, stroke, half)
    return shapes


type Grid = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]


def segment_distance(x: Grid, y: Grid, a: Point, b: Point) -> Grid:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = numpy.clip(((x - a[0]) * dx + (y - a[1]) * dy) / length2, 0.0, 1.0)
    return numpy.hypot(x - a[0] - t * dx, y - a[1] - t * dy)


def signed_distance(shape: Shape, x: Grid, y: Grid) -> Grid:
    """Negative inside the shape, positive outside: the distance to its outline."""
    kind, vertices, radius = shape
    if kind == "disc":
        cx, cy = vertices[0]
        return numpy.hypot(x - cx, y - cy) - radius
    edges = list(zip(vertices, (*vertices[1:], vertices[0])))
    distance = numpy.minimum.reduce([segment_distance(x, y, a, b) for a, b in edges])
    signs = [(b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) for a, b in edges]
    inside = numpy.logical_and.reduce([sign >= 0 for sign in signs]) | numpy.logical_and.reduce(
        [sign <= 0 for sign in signs]
    )
    return numpy.where(inside, -distance, distance)


def stream(dictionary: bytes, data: bytes) -> bytes:
    return b"<< " + dictionary + b" /Length %d >>\nstream\n" % len(data) + data + b"\nendstream"


def document(content: bytes, resources: bytes) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Contents 4 0 R /Resources %s >>"
        % (SIZE, SIZE, resources),
        stream(b"", content),
    ]
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


class Case:
    """A page stroking `strokes` in black under `ctm`. With `extgstate`, the page sets the
    parameters with gs instead of the operators; `blend` strokes under that blend mode."""

    def __init__(
        self,
        name: str,
        strokes: Sequence[Stroke],
        *,
        ctm: Affine = IDENTITY,
        extgstate: bool = False,
        blend: str | None = None,
    ) -> None:
        self.name = name
        self.strokes = tuple(strokes)
        self.ctm = ctm
        self.extgstate = extgstate
        self.blend = blend

    def pdf(self) -> bytes:
        states: list[bytes] = []
        body = b"0 G q %s cm " % numbers(self.ctm)
        if self.blend is not None:
            states.append(b"/BM /%s" % self.blend.encode())
            body += b"/G0 gs "
        for stroke in self.strokes:
            if self.extgstate:
                name = len(states)
                dash = b" /D [[] 0]"
                if stroke.dash is not None:
                    dash = b" /D [[%s] %s]" % (numbers(stroke.dash[0]), numbers([stroke.dash[1]]))
                states.append(
                    b"/LW %s /LC %d /LJ %d /ML %s%s"
                    % (
                        numbers([stroke.width]),
                        stroke.cap,
                        stroke.join,
                        numbers([stroke.miter_limit]),
                        dash,
                    )
                )
                # Operators that gs must override: the defaults' opposites.
                body += b"40 w 1 J 1 j 1 M [1 1] 0 d /G%d gs %s " % (name, stroke.path())
            else:
                body += stroke.operators() + stroke.path() + b" "
        body += b"Q"
        resources = b"<< /ExtGState << %s >> >>" % b" ".join(
            b"/G%d << /Type /ExtGState %s >>" % (index, state) for index, state in enumerate(states)
        )
        return document(body, resources)

    def shapes(self) -> list[Shape]:
        return [shape for stroke in self.strokes for shape in stroke_shapes(stroke, self.ctm)]


def zigzag(
    origin: Point, turns: Sequence[float], length: float = 90.0, heading: float = 20.0
) -> list[Point]:
    """A polyline from origin, each segment `length` long, turning by each angle in turn."""
    points = [origin]
    for turn in (0.0, *turns):
        heading += turn
        points.append(polar(points[-1], length, heading))
    return points


# Turns of 150 and 60 degrees: corners of 30 (acute) and 120 (obtuse) degrees, at headings
# no axis lines up with.
ZIGZAG = zigzag((110.0, 50.0), (150.0, -60.0), 100.0, 25.0)
RIGHT_ANGLE = [(30.0, 250.0), (130.0, 250.0), (130.0, 200.0)]
# A corner of 40 degrees has a miter ratio of 1 / sin(20 degrees), 2.9238.
NARROW = zigzag((60.0, 80.0), (140.0,), 150.0, 30.0)
NARROW_RATIO = 1.0 / math.sin(math.radians(20.0))
TRIANGLE = [(50.0, 50.0), (250.0, 80.0), (120.0, 240.0)]


def cases() -> list[Case]:
    found = [
        Case(
            f"join {join}, cap {cap}",
            [Stroke(ZIGZAG, join=join, cap=cap), Stroke(RIGHT_ANGLE, join=join, cap=cap)],
        )
        for join in (MITER, ROUND_JOIN, BEVEL)
        for cap in (BUTT, ROUND, SQUARE)
    ]
    found += [
        Case("miter limit just above the ratio", [Stroke(NARROW, miter_limit=NARROW_RATIO + 0.05)]),
        Case("miter limit just below the ratio", [Stroke(NARROW, miter_limit=NARROW_RATIO - 0.05)]),
        Case(
            "right angle, limit above the square root of 2", [Stroke(RIGHT_ANGLE, miter_limit=1.45)]
        ),
        Case(
            "right angle, limit below the square root of 2", [Stroke(RIGHT_ANGLE, miter_limit=1.38)]
        ),
        Case(
            "zero-length subpaths",
            [
                Stroke([(60.0, 60.0), (60.0, 60.0)], cap=ROUND, width=20.0),
                Stroke([(150.0, 60.0), (150.0, 60.0)], cap=SQUARE, width=20.0),
                Stroke([(240.0, 60.0), (240.0, 60.0)], cap=BUTT, width=20.0),
                Stroke([(60.0, 200.0), (60.0, 200.0), (60.0, 200.0)], closed=True, cap=ROUND),
                Stroke([(150.0, 200.0), (150.0, 200.0), (150.0, 200.0)], cap=SQUARE),
            ],
        ),
        Case(
            "zero-length segments inside a path",
            [
                Stroke(
                    [(40.0, 40.0), (40.0, 40.0), (150.0, 220.0), (150.0, 220.0), (260.0, 40.0)],
                    cap=SQUARE,
                )
            ],
        ),
        Case("closed triangle", [Stroke(TRIANGLE, closed=True)]),
        Case(
            "closed triangle, first point repeated",
            [Stroke(TRIANGLE, closed=True, explicit_close=True)],
        ),
        Case("closed triangle, bevel", [Stroke(TRIANGLE, closed=True, join=BEVEL)]),
        Case("open triangle", [Stroke([*TRIANGLE, TRIANGLE[0]], cap=BUTT)]),
        Case("dashes, square caps", [Stroke(ZIGZAG, cap=SQUARE, dash=([30.0, 22.0], 9.0))]),
        Case("dashes, round caps", [Stroke(ZIGZAG, cap=ROUND, dash=([30.0, 22.0], 0.0))]),
        Case(
            "dashes around a closed triangle's corners",
            [Stroke(TRIANGLE, closed=True, dash=([140.0, 30.0], 40.0))],
        ),
        Case(
            "zero-length dashes",
            [
                Stroke(zigzag((40.0, 40.0), (), 230.0, 35.0), cap=ROUND, dash=([0.0, 40.0], 0.0)),
                Stroke(zigzag((30.0, 150.0), (), 230.0, 20.0), cap=SQUARE, dash=([0.0, 40.0], 0.0)),
            ],
        ),
        Case(
            "scaled and rotated",
            [Stroke(ZIGZAG, cap=SQUARE, width=10.0), Stroke(NARROW, width=8.0)],
            ctm=rotation(17.0, 0.8, 60.0, -10.0),
        ),
        Case(
            "blend mode",
            [Stroke(ZIGZAG, cap=SQUARE), Stroke(TRIANGLE, closed=True, join=BEVEL, width=10.0)],
            blend="Multiply",
        ),
        Case(
            "graphics state parameters",
            [
                Stroke(NARROW, width=18.0, cap=SQUARE, miter_limit=NARROW_RATIO + 0.05),
                Stroke(RIGHT_ANGLE, width=12.0, cap=BUTT, join=BEVEL, dash=([40.0, 20.0], 10.0)),
            ],
            extgstate=True,
        ),
    ]
    return found


def check(case: Case) -> tuple[int, int]:
    with PdfDocument(case.pdf()) as pdf:
        image = pdf.pages[0].render().rasterize()
    pixels = numpy.frombuffer(bytes(image.pixels), numpy.uint8).reshape(
        image.height, image.width, 4
    )
    # Pixel centres in page space.
    rows, columns = numpy.mgrid[0 : image.height, 0 : image.width]
    x = columns + 0.5
    y = SIZE - (rows + 0.5)
    distance = numpy.minimum.reduce([signed_distance(shape, x, y) for shape in case.shapes()])
    checked = numpy.abs(distance) >= MARGIN
    inside = distance < 0
    painted = (pixels[..., 3] == 255) & ~pixels[..., :3].any(axis=2)
    ok = numpy.where(inside, painted, pixels[..., 3] == 0)
    wrong = numpy.argwhere(checked & ~ok)
    for row, column in wrong[:3]:
        expected = "stroke" if inside[row, column] else "nothing"
        pixel = tuple(int(value) for value in pixels[row, column])
        print(f"  {case.name}: pixel {column},{row}: {pixel}, not {expected}")
    return int(checked.sum()), len(wrong)


def main() -> int:
    total = failures = failed_pages = 0
    selected = cases()
    for case in selected:
        checked, failed = check(case)
        total += checked
        failures += failed
        if failed:
            failed_pages += 1
            print(f"{case.name}: {failed} of {checked} pixels differ")
    print(
        f"{len(selected)} stroke pages, {total} pixels: {failures} failures on {failed_pages} pages"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

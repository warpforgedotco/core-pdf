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

A stroke is one shape, painted once (8.5.3.2, 11.7): where its segments, joins, caps and
dashes overlap or meet, a constant alpha, a soft mask or a blend mode applies once, and no
seam shows. Its line width is a diameter in user space (8.4.3.2): under a CTM that skews
or scales unevenly the pen is an ellipse on the page, and joins, caps, the miter limit and
dashes follow user space.

This writes PDFs that stroke polylines with every join and cap, at acute, right and obtuse
angles, with miter limits just above and just below a join's ratio, through zero-length
segments, closed and open, dashed, scaled and rotated by the CTM, under a blend mode, and
with the parameters set by gs; then the same at a constant alpha of 0.5, crossing
themselves, dashed, under a soft mask and in Multiply over a backdrop; opaque rings of short
segments and circles of curves; and every join and cap under skewed, stretched and
stretched and rotated CTMs. It renders each with core and works out in Python, in user
space, which pixels each stroke's pieces cover: a pixel is inside if any piece holds it, so
that where pieces meet inside a stroke is no edge. Pixels within a device pixel and a half
of a stroke's edge, or with any of the points an eighth of a pixel apart within them on
both sides of it, are skipped; every other pixel must be the stroke painted once over the
backdrop, within a byte, or the backdrop.

A line width of 0 is 1 device pixel wide (8.4.3.2), and a thinner stroke of non-zero width
is floored there too, as Acrobat draws it, so that it never disappears (10.7.4): under a
CTM that scales unevenly, the pen is at least a device pixel across its narrower axis. This
also writes pages of a dozen parallel lines of widths from 0 to 1.5 device pixels, at
several angles, opaque, at a constant alpha of 0.5, in Multiply, in a knockout group,
dashed, and under CTMs that stretch, skew, stretch and rotate, and scale down, and checks
the area they cover, the sum of the pixels' alpha, against the lines' length times their
width, no thinner than one device pixel. It checks that a line 0.1 units wide is a pixel
wide at scales 1 and 4, that a line width of 0 paints the same pixels as a tiny one, and
that a thin line under a rotating, unevenly scaling CTM is a pixel wide along the pen's
narrower axis. And it writes dotted hairlines, dashes of no length with round and projecting
square caps along lines at 0 and 45 degrees, and subpaths of one point with round caps,
widths 0.05 to 0.2, centred on whole pixels, half pixels and in between, opaque, at alpha
0.5, in Multiply and under a stretching CTM, and the same with segments a tenth of a pixel
long, subpaths and dashes, with each cap: each dot must cover the area of its pen, a pixel
across, and never disappear between the samples.

uv run --all-packages python scripts/check_strokes.py
"""

from __future__ import annotations

import copy
import math
import sys
from collections.abc import Sequence
from typing import cast

import numpy

from core_pdf import PdfDocument

type Point = tuple[float, float]
type Affine = tuple[float, float, float, float, float, float]
# A convex polygon, its vertices in order; a disc: (centre, radius); or a ring:
# (centre, (radius, half its width)).
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
        circle: tuple[Point, float] | None = None,
        repeat: int = 1,
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
        # A circle (centre, radius) drawn with four Bezier curves instead of the points.
        self.circle = circle
        # How many times the path holds the subpath: the stroke, one shape, is the same.
        self.repeat = repeat

    def path(self) -> bytes:
        return self.subpath() * self.repeat + b"S"

    def subpath(self) -> bytes:
        if self.circle is not None:
            (cx, cy), r = self.circle
            k = r * 0.5522847498
            out = b"%s m " % numbers([cx + r, cy])
            for (x0, y0), (x1, y1), (x2, y2) in (
                ((cx + r, cy + k), (cx + k, cy + r), (cx, cy + r)),
                ((cx - k, cy + r), (cx - r, cy + k), (cx - r, cy)),
                ((cx - r, cy - k), (cx - k, cy - r), (cx, cy - r)),
                ((cx + k, cy - r), (cx + r, cy - k), (cx + r, cy)),
            ):
                out += b"%s c " % numbers([x0, y0, x1, y1, x2, y2])
            return out + b"h "
        first, *rest = self.points
        if self.explicit_close:
            rest = [*rest, first]
        out = b"%s m " % numbers(first) + b"".join(b"%s l " % numbers(point) for point in rest)
        return out + (b"h " if self.closed else b"")

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


def stroke_shapes(stroke: Stroke) -> list[Shape]:
    """The outline of a stroke in user space, where the pen is a circle a line width
    across and dashes are measured (8.4.3.2, 8.4.3.6)."""
    half = stroke.width / 2.0
    if stroke.circle is not None:
        # The curves, flattened, stay well within a pixel of the circle.
        centre, radius = stroke.circle
        return [("ring", (centre, (radius, half)), 0.0)]
    points = list(stroke.points)
    if stroke.dash is None:
        return polyline_shapes(points, stroke.closed, stroke, half)
    shapes: list[Shape] = []
    for piece, closed, direction in dashes(points, stroke.closed, *stroke.dash):
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
    if kind == "ring":
        (cx, cy), (middle, half) = vertices
        return numpy.abs(numpy.hypot(x - cx, y - cy) - middle) - half
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


def document(content: bytes, resources: bytes, extra: Sequence[bytes] = ()) -> bytes:
    """A one-page PDF; extra are its objects from number 5 on."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Contents 4 0 R /Resources %s >>"
        % (SIZE, SIZE, resources),
        stream(b"", content),
        *extra,
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


type Colour = tuple[float, float, float]
BLACK: Colour = (0.0, 0.0, 0.0)
# The grey a soft mask's group paints: its luminosity is the mask's value (11.6.5.2).
MASK_GREY = 0.6


def blend(mode: str, backdrop: float, source: float) -> float:
    if mode == "Multiply":
        return backdrop * source
    if mode == "Screen":
        return backdrop + source - backdrop * source
    return source


class Case:
    """A page stroking `strokes` in `colour` under `ctm`. With `extgstate`, the page sets
    the parameters with gs instead of the operators. A graphics state sets the blend mode
    `blend`, the constant alpha `alpha` and, with `soft_mask`, a luminosity soft mask of
    MASK_GREY; `backdrop` fills the page first. Each stroke on a page is apart from the
    others, so that each pixel is painted by at most one."""

    def __init__(
        self,
        name: str,
        strokes: Sequence[Stroke],
        *,
        ctm: Affine = IDENTITY,
        extgstate: bool = False,
        blend: str | None = None,
        alpha: float = 1.0,
        soft_mask: bool = False,
        backdrop: Colour | None = None,
        colour: Colour = BLACK,
    ) -> None:
        self.name = name
        self.strokes = tuple(strokes)
        self.ctm = ctm
        self.extgstate = extgstate
        self.blend = blend
        self.alpha = alpha
        self.soft_mask = soft_mask
        self.backdrop = backdrop
        self.colour = colour

    def pdf(self) -> bytes:
        states: list[bytes] = []
        extra: list[bytes] = []
        body = b""
        if self.backdrop is not None:
            body += b"%s rg 0 0 %d %d re f " % (numbers(self.backdrop), SIZE, SIZE)
        body += b"%s RG q %s cm " % (numbers(self.colour), numbers(self.ctm))
        paint: list[bytes] = []
        if self.blend is not None:
            paint.append(b"/BM /%s" % self.blend.encode())
        if self.alpha != 1.0:
            paint.append(b"/CA %s /ca %s" % (numbers([self.alpha]), numbers([self.alpha])))
        if self.soft_mask:
            group = b"%s g 0 0 %d %d re f" % (numbers([MASK_GREY]), SIZE, SIZE)
            extra.append(
                stream(
                    b"/Type /XObject /Subtype /Form /BBox [0 0 %d %d]"
                    b" /Group << /S /Transparency /CS /DeviceGray >>" % (SIZE, SIZE),
                    group,
                )
            )
            paint.append(b"/SMask << /Type /Mask /S /Luminosity /G 5 0 R >>")
        if paint:
            states.append(b" ".join(paint))
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
        return document(body, resources, extra)

    def shapes(self) -> list[Shape]:
        return [shape for stroke in self.strokes for shape in stroke_shapes(stroke)]

    def expected(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The RGBA of a pixel the stroke covers, painted once over the backdrop, and of one
        it does not: the backdrop (11.3.6)."""
        source = self.alpha * (MASK_GREY if self.soft_mask else 1.0)
        mode = self.blend or "Normal"
        if self.backdrop is None:
            outside = (0, 0, 0, 0)
            inside = (*(round(255 * value) for value in self.colour), round(255 * source))
            return inside, outside
        outside = (*(round(255 * value) for value in self.backdrop), 255)
        inside = (
            *(
                round(255 * ((1.0 - source) * back + source * blend(mode, back, value)))
                for back, value in zip(self.backdrop, self.colour, strict=True)
            ),
            255,
        )
        return inside, outside


def zigzag(
    origin: Point, turns: Sequence[float], length: float = 90.0, heading: float = 20.0
) -> list[Point]:
    """A polyline from origin, each segment `length` long, turning by each angle in turn."""
    points = [origin]
    for turn in (0.0, *turns):
        heading += turn
        points.append(polar(points[-1], length, heading))
    return points


def scaled(points: Sequence[Point], factor: float, origin: Point = (0.0, 0.0)) -> list[Point]:
    return [(origin[0] + x * factor, origin[1] + y * factor) for x, y in points]


# Turns of 150 and 60 degrees: corners of 30 (acute) and 120 (obtuse) degrees, at headings
# no axis lines up with.
ZIGZAG = zigzag((110.0, 50.0), (150.0, -60.0), 100.0, 25.0)
RIGHT_ANGLE = [(30.0, 250.0), (130.0, 250.0), (130.0, 200.0)]
# A corner of 40 degrees has a miter ratio of 1 / sin(20 degrees), 2.9238.
NARROW = zigzag((60.0, 80.0), (140.0,), 150.0, 30.0)
NARROW_RATIO = 1.0 / math.sin(math.radians(20.0))
TRIANGLE = [(50.0, 50.0), (250.0, 80.0), (120.0, 240.0)]
# A closed five-pointed star, whose segments cross one another.
# A ring of many short segments, as a flattened curve is, and a circle of curves.
POLYGON_RING = [polar((150.0, 150.0), 100.0, 3.0 + 5.0 * index) for index in range(72)]
CIRCLE = ((150.0, 150.0), 90.0)
STAR = [polar((150.0, 150.0), 110.0, 90.0 + 144.0 * index) for index in range(5)]
# An open path that loops back over itself.
LOOP = [(40.0, 60.0), (230.0, 60.0), (230.0, 200.0), (120.0, 200.0), (120.0, 20.0)]
# A stroke's joins and caps under a CTM, small enough for the page under each CTM below.
SMALL_ZIGZAG = zigzag((-60.0, -40.0), (150.0, -60.0), 60.0, 25.0)
SMALL_RIGHT_ANGLE = [(-70.0, 60.0), (-10.0, 60.0), (-10.0, 30.0)]
SKEW: Affine = (1.0, 0.0, 0.6, 1.0, 140.0, 150.0)
STRETCH: Affine = (1.7, 0.0, 0.0, 0.6, 150.0, 150.0)


def rotated_stretch(degrees: float, x_scale: float, y_scale: float) -> Affine:
    """A stretch along the axes, then a rotation, about the page's centre."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return (c * x_scale, s * x_scale, -s * y_scale, c * y_scale, 150.0, 150.0)


UNEVEN: dict[str, Affine] = {
    "skewed": SKEW,
    "stretched": STRETCH,
    "stretched and rotated": rotated_stretch(35.0, 1.5, 0.65),
}
ORANGE: Colour = (0.8, 0.6, 0.2)
BLUE: Colour = (0.2, 0.4, 0.9)


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
    # A stroke is one shape painted once (8.5.3.2, 11.7): where its pieces overlap or meet,
    # a constant alpha, a soft mask or a blend mode applies once.
    found += [
        Case(
            f"alpha 0.5, join {join}, cap {cap}",
            [Stroke(ZIGZAG, join=join, cap=cap), Stroke(RIGHT_ANGLE, join=join, cap=cap)],
            alpha=0.5,
        )
        for join in (MITER, ROUND_JOIN, BEVEL)
        for cap in (BUTT, ROUND, SQUARE)
    ]
    found += [
        Case(
            "alpha 0.5, a star crossing itself",
            [Stroke(STAR, closed=True, width=12.0, join=ROUND_JOIN)],
            alpha=0.5,
            backdrop=ORANGE,
            colour=BLUE,
        ),
        Case(
            "alpha 0.5, a loop crossing itself",
            [Stroke(LOOP, width=20.0, cap=SQUARE, join=BEVEL)],
            alpha=0.5,
        ),
        Case(
            "alpha 0.5, dashes",
            [
                Stroke(ZIGZAG, cap=SQUARE, join=ROUND_JOIN, dash=([30.0, 22.0], 9.0)),
                Stroke(RIGHT_ANGLE, cap=ROUND, dash=([0.0, 25.0], 0.0)),
            ],
            alpha=0.5,
            backdrop=ORANGE,
        ),
        Case(
            "alpha 0.5, dashes around a closed triangle's corners",
            [Stroke(TRIANGLE, closed=True, dash=([140.0, 30.0], 40.0))],
            alpha=0.5,
        ),
        Case(
            "soft mask",
            [Stroke(ZIGZAG, cap=ROUND, join=MITER), Stroke(RIGHT_ANGLE, cap=SQUARE)],
            soft_mask=True,
            backdrop=ORANGE,
        ),
        Case(
            "soft mask and alpha 0.5, dashes",
            [Stroke(STAR, closed=True, width=10.0, dash=([50.0, 10.0], 0.0))],
            soft_mask=True,
            alpha=0.5,
        ),
        Case(
            "Multiply over a backdrop",
            [Stroke(ZIGZAG, cap=SQUARE), Stroke(RIGHT_ANGLE, join=ROUND_JOIN, cap=ROUND)],
            blend="Multiply",
            backdrop=ORANGE,
            colour=BLUE,
        ),
        Case(
            "Multiply and alpha 0.5, a star crossing itself",
            [Stroke(STAR, closed=True, width=12.0, join=MITER)],
            blend="Multiply",
            alpha=0.5,
            backdrop=ORANGE,
            colour=BLUE,
        ),
    ]
    # Where a stroke's pieces meet along an edge, partial coverages painted in turn would
    # leave a lighter seam, even for an opaque stroke.
    found += [
        Case(
            f"opaque ring of segments, join {join}",
            [Stroke(POLYGON_RING, closed=True, join=join, width=14.0)],
            backdrop=ORANGE,
            colour=BLUE,
        )
        for join in (MITER, ROUND_JOIN, BEVEL)
    ]
    found += [
        Case("opaque circle of curves", [Stroke((), circle=CIRCLE, width=3.0)]),
        Case("opaque thick circle of curves", [Stroke((), circle=CIRCLE, width=20.0, join=BEVEL)]),
        Case(
            "alpha 0.5, circle of curves",
            [Stroke((), circle=CIRCLE, width=6.0)],
            alpha=0.5,
            backdrop=ORANGE,
        ),
        Case(
            "skewed circle of curves", [Stroke((), circle=((0.0, 0.0), 60.0), width=10.0)], ctm=SKEW
        ),
    ]
    # A segment along an axis shorter than half the line width: its square cap, and a right
    # angle's miter at its other end, reach no further back than it does.
    short_end = [
        Stroke([(40.0, 120.0), (140.0, 220.0), (140.0, 214.0)], cap=SQUARE, join=BEVEL, width=20.0),
        Stroke([(160.0, 40.0), (260.0, 40.0), (260.0, 46.0)], join=MITER, width=20.0),
        Stroke(
            [(170.0, 130.0), (250.0, 130.0), (250.0, 136.0), (180.0, 210.0)],
            cap=SQUARE,
            join=MITER,
            width=20.0,
        ),
    ]
    found += [
        Case("short segments along the axes, opaque", short_end),
        Case("short segments along the axes, alpha 0.5", short_end, alpha=0.5),
    ]
    # A path holding its subpath twice strokes one shape: no edge pixel blends twice.
    diagonal = [(40.0, 40.0), (260.0, 230.0)]
    found += [
        Case(
            "opaque path held twice, round joins",
            [Stroke(ZIGZAG, join=ROUND_JOIN, cap=ROUND, repeat=2)],
        ),
        Case(
            "opaque dashed diagonal held twice",
            [Stroke(diagonal, width=4.0, dash=([10.0, 10.0], 0.0), repeat=2)],
        ),
        Case(
            "opaque dashed path held twice, square caps",
            [Stroke(ZIGZAG, cap=SQUARE, dash=([30.0, 22.0], 9.0), repeat=2)],
        ),
        Case(
            "alpha 0.5, dashed path held twice",
            [Stroke(ZIGZAG, cap=ROUND, dash=([30.0, 22.0], 0.0), repeat=2)],
            alpha=0.5,
        ),
        Case(
            "opaque circle of curves held twice", [Stroke((), circle=CIRCLE, width=6.0, repeat=2)]
        ),
        Case(
            "Multiply, path held twice",
            [Stroke(TRIANGLE, closed=True, join=BEVEL, width=10.0, repeat=2)],
            blend="Multiply",
            backdrop=ORANGE,
            colour=BLUE,
        ),
    ]
    # The line width is a diameter in user space (8.4.3.2): under a CTM that skews or
    # scales unevenly, the pen is an ellipse on the page, and the joins, caps, miter limit
    # and dashes follow user space.
    found += [
        Case(
            f"{name}, join {join}, cap {cap}",
            [
                Stroke(SMALL_ZIGZAG, join=join, cap=cap, width=12.0),
                Stroke(SMALL_RIGHT_ANGLE, join=join, cap=cap, width=10.0),
            ],
            ctm=ctm,
        )
        for name, ctm in UNEVEN.items()
        for join in (MITER, ROUND_JOIN, BEVEL)
        for cap in (BUTT, ROUND, SQUARE)
    ]
    found += [
        Case(
            "skewed, miter limit just above the ratio",
            [Stroke(scaled(NARROW, 0.6, (-90.0, -60.0)), miter_limit=NARROW_RATIO + 0.05)],
            ctm=SKEW,
        ),
        Case(
            "stretched, miter limit just below the ratio",
            [Stroke(scaled(NARROW, 0.6, (-60.0, -60.0)), miter_limit=NARROW_RATIO - 0.05)],
            ctm=STRETCH,
        ),
        Case(
            "stretched, dashes",
            [Stroke(SMALL_ZIGZAG, cap=SQUARE, width=10.0, dash=([18.0, 12.0], 4.0))],
            ctm=STRETCH,
        ),
        Case(
            "skewed, alpha 0.5 and zero-length dashes",
            [
                Stroke(SMALL_ZIGZAG, cap=ROUND, join=ROUND_JOIN, width=12.0),
                Stroke(SMALL_RIGHT_ANGLE, cap=SQUARE, width=8.0, dash=([0.0, 15.0], 0.0)),
            ],
            ctm=SKEW,
            alpha=0.5,
        ),
    ]
    return found


def contains(
    shape: Shape, x: Grid, y: Grid
) -> numpy.ndarray[tuple[int, int], numpy.dtype[numpy.bool]]:
    """Whether each point lies inside the shape or on its edge, so that a point on the
    edge two pieces share, as a segment's end and its join do, lies in the union."""
    kind, vertices, radius = shape
    if kind == "disc":
        cx, cy = vertices[0]
        return (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius
    if kind == "ring":
        (cx, cy), (middle, half) = vertices
        return numpy.abs(numpy.hypot(x - cx, y - cy) - middle) <= half
    edges = list(zip(vertices, (*vertices[1:], vertices[0])))
    signs = [(b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) for a, b in edges]
    return numpy.logical_and.reduce([sign >= 0 for sign in signs]) | numpy.logical_and.reduce(
        [sign <= 0 for sign in signs]
    )


def shape_box(shape: Shape) -> tuple[float, float, float, float]:
    kind, vertices, radius = shape
    if kind == "disc":
        (cx, cy), reach = vertices[0], radius
    elif kind == "ring":
        (cx, cy), (middle, half) = vertices
        reach = middle + half
    else:
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]
        return min(xs), min(ys), max(xs), max(ys)
    return cx - reach, cy - reach, cx + reach, cy + reach


# The points of the page the reference samples, every STEP of a pixel; pixel centres are
# among them.
STEP = 0.5


def user_grid(ctm: Affine, step: float, count: int) -> tuple[Grid, Grid]:
    """The user-space points of a grid of count by count page points, step apart, from the
    page's top left corner."""
    a, b, c, d, e, f = ctm
    determinant = a * d - b * c
    page = numpy.arange(count) * step
    page_x = page[None, :] - e
    page_y = SIZE - page[:, None] - f
    x = (d * page_x - c * page_y) / determinant
    y = (a * page_y - b * page_x) / determinant
    return cast("Grid", x), cast("Grid", y)


# Core samples each pixel at 4 x 4 points, an eighth of a pixel and then a quarter apart;
# the reference samples every eighth, so that it sees any gap core does.
FINE = 8


def outline_pixels(
    case: Case, height: int, width: int
) -> tuple[numpy.ndarray[tuple[int, int], numpy.dtype[numpy.bool]], ...]:
    """Which pixels the strokes cover, and which lie MARGIN or more from the edge of each
    stroke, the union of its pieces, where coverage does not blend. A point is inside a
    stroke if any of its pieces holds it, so that where pieces meet inside the stroke is no
    edge: the stroke must be painted once there too. Each union is sampled every STEP on
    the page, its points mapped to user space, where the pieces are, to find the pixels
    MARGIN from its edge; and every eighth of a pixel within each pixel, which must lie all
    inside or all outside for the pixel to be checked."""
    count = round(SIZE / STEP) + 1
    x, y = user_grid(case.ctm, STEP, count)
    fine_x, fine_y = user_grid(case.ctm, 1.0 / FINE, SIZE * FINE)
    centres = (slice(1, 2 * height, 2), slice(1, 2 * width, 2))
    covered = numpy.zeros((height, width), dtype=bool)
    clear = numpy.ones((height, width), dtype=bool)
    for stroke in case.strokes:
        all_inside, all_outside = eroded(stroke_union(stroke, case.ctm, x, y, STEP))
        fine = stroke_union(stroke, case.ctm, fine_x, fine_y, 1.0 / FINE)
        blocks = fine.reshape(SIZE, FINE, SIZE, FINE)[:height, :, :width, :]
        whole = blocks.all(axis=(1, 3))
        none = ~blocks.any(axis=(1, 3))
        covered |= whole
        clear &= (all_inside[centres] & whole) | (all_outside[centres] & none)
    return covered, clear


def stroke_union(
    stroke: Stroke, ctm: Affine, x: Grid, y: Grid, step: float
) -> numpy.ndarray[tuple[int, int], numpy.dtype[numpy.bool]]:
    """Which of the sampled points, step apart on the page and at x and y in user space,
    the stroke holds."""
    a, b, c, d, e, f = ctm
    count = x.shape[0]
    union = numpy.zeros((count, count), dtype=bool)
    for shape in stroke_shapes(stroke):
        x0, y0, x1, y1 = shape_box(shape)
        corners = [(a * u + c * v + e, b * u + d * v + f) for u in (x0, x1) for v in (y0, y1)]
        column0 = max(0, math.floor(min(p[0] for p in corners) / step))
        column1 = min(count, math.ceil(max(p[0] for p in corners) / step) + 1)
        row0 = max(0, math.floor((SIZE - max(p[1] for p in corners)) / step))
        row1 = min(count, math.ceil((SIZE - min(p[1] for p in corners)) / step) + 1)
        if column1 <= column0 or row1 <= row0:
            continue
        window = (slice(row0, row1), slice(column0, column1))
        union[window] |= contains(shape, x[window], y[window])
    return union


def eroded(
    union: numpy.ndarray[tuple[int, int], numpy.dtype[numpy.bool]],
) -> tuple[numpy.ndarray[tuple[int, int], numpy.dtype[numpy.bool]], ...]:
    """The sampled points MARGIN or more inside the union, and those as far outside it."""
    count = union.shape[0]
    # MARGIN, and a STEP for the sampling, in steps.
    reach = math.ceil((MARGIN + STEP) / STEP)
    padded = numpy.pad(union, reach)
    all_inside = union.copy()
    all_outside = ~union
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if dx * dx + dy * dy > reach * reach:
                continue
            shifted = padded[reach + dy : reach + dy + count, reach + dx : reach + dx + count]
            all_inside &= shifted
            all_outside &= ~shifted
    return all_inside, all_outside


type Pixels = numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]]


def rendered(case: Case | Hairline | Dots | ThinLine, scale: float = 1.0) -> Pixels:
    with PdfDocument(case.pdf()) as pdf:
        image = pdf.pages[0].render().rasterize(scale=scale)
    return numpy.frombuffer(bytes(image.pixels), numpy.uint8).reshape(image.height, image.width, 4)


# The thinnest pen, in device pixels: a line width of 0 is this wide, and a thinner stroke
# of non-zero width is widened to it, across the pen's narrower axis.
HAIRLINE_FLOOR = 1.0
HAIRLINE_LENGTH = 200.0
HAIRLINE_COUNT = 12
# Apart by more than a pixel and by no multiple of the samples' spacing, so that the lines
# fall at every phase of the sample grid between them.
HAIRLINE_SPACING = 7.37


class Hairline:
    """HAIRLINE_COUNT parallel lines `degrees` from the x axis in user space, about the
    origin, HAIRLINE_LENGTH long and HAIRLINE_SPACING apart on the page, a line width of
    `width`, under `ctm` (by default, centred on the page): at a constant alpha `alpha`, in
    the blend mode `blend`, with caps `cap`, dashed by `dash`, or in a knockout group. A
    stroke covers the points within half the line width of its path in user space (ISO
    32000-2 8.4.3.2); a line width of 0 is 1 device pixel wide (8.4.3.2), and so is a
    thinner one, so that it never disappears (10.7.4)."""

    def __init__(
        self,
        name: str,
        width: float,
        degrees: float,
        *,
        ctm: Affine = (1.0, 0.0, 0.0, 1.0, SIZE / 2.0, SIZE / 2.0),
        alpha: float = 1.0,
        blend: str | None = None,
        cap: int = BUTT,
        dash: tuple[Sequence[float], float] | None = None,
        knockout: bool = False,
    ) -> None:
        self.name = name
        self.width = width
        self.degrees = degrees
        self.ctm = ctm
        self.alpha = alpha
        self.blend = blend
        self.cap = cap
        self.dash = dash
        self.knockout = knockout

    def scale(self) -> float:
        """The factor the CTM scales the line width and dashes by: the mean of its scales
        along the axes."""
        a, b, c, d = self.ctm[:4]
        return (math.hypot(a, b) + math.hypot(c, d)) / 2.0

    def lines(self) -> list[tuple[Point, Point]]:
        """Each line's ends in user space."""
        radians = math.radians(self.degrees)
        ux, uy = math.cos(radians), math.sin(radians)
        unit = 1.0 / self.scale()
        found = []
        for index in range(HAIRLINE_COUNT):
            offset = (index - (HAIRLINE_COUNT - 1) / 2.0) * HAIRLINE_SPACING * unit
            cx, cy = -uy * offset, ux * offset
            half = HAIRLINE_LENGTH / 2.0 * unit
            found.append(((cx - ux * half, cy - uy * half), (cx + ux * half, cy + uy * half)))
        return found

    def pdf(self) -> bytes:
        paint = []
        if self.alpha != 1.0:
            paint.append(b"/CA %s" % numbers([self.alpha]))
        if self.blend is not None:
            paint.append(b"/BM /%s" % self.blend.encode())
        dash = b"[] 0 d"
        if self.dash is not None:
            dash = b"[%s] %s d" % (numbers(self.dash[0]), numbers([self.dash[1]]))
        body = b"%s w %d J %s " % (numbers([self.width]), self.cap, dash)
        if paint:
            body += b"/G0 gs "
        body += b"".join(b"%s m %s l S " % (numbers(a), numbers(b)) for a, b in self.lines())
        resources = b"<< /ExtGState << /G0 << %s >> >> >>" % b" ".join(paint)
        if not self.knockout:
            content = b"q %s cm %sQ" % (numbers(self.ctm), body)
            return document(content, resources)
        form = stream(
            b"/Type /XObject /Subtype /Form /BBox [0 0 %d %d] /Resources %s"
            b" /Group << /S /Transparency /K true >>" % (SIZE, SIZE, resources),
            b"q %s cm %sQ" % (numbers(self.ctm), body),
        )
        return document(b"/X Do", b"<< /XObject << /X 5 0 R >> >>", [form])

    def expected_area(self) -> float:
        """The page area, in square device pixels, the lines cover: each line's length on
        the page, dashes and caps included, times the width of its pen across it there. The
        pen is a circle a line width across in stroke space, user space scaled by the mean
        of the CTM's scales along the axes, widened to HAIRLINE_FLOOR on the page across
        the CTM's least stretch, a line width of 0 included."""
        a, b, c, d = self.ctm[:4]
        scale = self.scale()
        forward = (a / scale, b / scale, c / scale, d / scale)
        smallest, _ = matrix_stretches(forward)
        determinant = abs(forward[0] * forward[3] - forward[1] * forward[2])
        radians = math.radians(self.degrees)
        ux, uy = math.cos(radians), math.sin(radians)
        # The line's direction on the page, and how much the stroke matrix stretches it.
        along = math.hypot(forward[0] * ux + forward[2] * uy, forward[1] * ux + forward[3] * uy)
        width = self.width * scale
        half = max(HAIRLINE_FLOOR / 2.0 / smallest, width / 2.0)
        across = 2.0 * half * determinant / along
        length = HAIRLINE_LENGTH * along
        if self.dash is not None:
            pattern = list(self.dash[0])
            length *= pattern[0] / sum(pattern)
        caps = 0.0
        if self.cap == ROUND:
            dashes = HAIRLINE_LENGTH / scale / sum(self.dash[0]) if self.dash else 1.0
            caps = math.pi * (across / 2.0) ** 2 * dashes
        return HAIRLINE_COUNT * (length * across + caps)


def matrix_stretches(matrix: tuple[float, float, float, float]) -> tuple[float, float]:
    """The least and the most the linear map (a, b, c, d) stretches a length by."""
    a, b, c, d = matrix
    total = a * a + b * b + c * c + d * d
    determinant = a * d - b * c
    root = math.sqrt(max(0.0, total * total - 4.0 * determinant * determinant))
    largest = math.sqrt((total + root) / 2.0)
    return abs(determinant) / largest, largest


def hairlines() -> list[Hairline]:
    widths = (0.0, 0.05, 0.12, 0.2, 0.3, 0.5, 0.8, 1.5)
    found = [
        Hairline(f"opaque, width {width}, {degrees} degrees", width, degrees)
        for width in widths
        for degrees in (0.0, 7.0, 30.0, 45.0, 90.0)
    ]
    found += [
        Hairline(f"alpha 0.5, round caps, width {width}", width, 20.0, alpha=0.5, cap=ROUND)
        for width in widths
    ]
    found += [
        Hairline(f"Multiply, width {width}", width, 33.0, blend="Multiply") for width in widths
    ]
    found += [
        Hairline(f"Multiply, round caps, width {width}", width, 61.0, blend="Multiply", cap=ROUND)
        for width in widths
    ]
    found += [
        Hairline(f"knockout group, width {width}", width, 12.0, knockout=True) for width in widths
    ]
    found += [
        Hairline(f"dashed, width {width}", width, 25.0, dash=([6.0, 4.0], 0.0)) for width in widths
    ]
    # Under a CTM that scales unevenly the pen is an ellipse, widened to the floor across
    # its narrower axis.
    found += [
        Hairline(f"{name}, width {width}", width, 15.0, ctm=ctm)
        for name, ctm in (
            ("stretched", (1.0, 0.0, 0.0, 0.4, 150.0, 150.0)),
            ("skewed", SKEW),
            ("stretched and rotated", stretched_rotation(35.0, 1.6, 0.5)),
        )
        for width in widths
    ]
    # Scaled down by the CTM: the line width is in user space.
    found += [
        Hairline(f"scaled by 0.1, width {width}", width, 40.0, ctm=(0.1, 0.0, 0.0, 0.1, 150, 150))
        for width in (0.0, 0.5, 1.2, 3.0, 8.0)
    ]
    return found


def stretched_rotation(
    degrees: float, x_scale: float, y_scale: float, e: float = SIZE / 2.0, f: float = SIZE / 2.0
) -> Affine:
    """Scale by x_scale and y_scale along the axes, then rotate by degrees, then move by
    (e, f), by default to the page's centre."""
    a, b, c, d, _, _ = rotation(degrees)
    return (a * x_scale, b * x_scale, c * y_scale, d * y_scale, e, f)


# Sampling a quarter of a pixel apart, the samples a band meets vary with where it lies: the
# ink of a dozen lines at as many phases is within this fraction of the band's area.
HAIRLINE_TOLERANCE = 0.1


def check_hairline(hairline: Hairline) -> bool:
    pixels = rendered(hairline)
    ink = float(pixels[..., 3].astype(numpy.float64).sum()) / 255.0 / hairline.alpha
    expected = hairline.expected_area()
    if abs(ink - expected) <= HAIRLINE_TOLERANCE * expected:
        return True
    print(f"  {hairline.name}: covers {ink:.1f} square pixels, not {expected:.1f}")
    return False


DOT_ROWS = 12
DOT_SPACING = 8.0
# A dot or short segment covers the area of its pen, a pixel across or more, to within
# what the samples miss along its edge; and never less than the sample nearest its centre,
# 1/16 of a pixel, so that it never disappears.
DOT_SLACK = 0.2
DOT_LEAST = 0.03


class Dots:
    """DOT_ROWS rows of dots DOT_SPACING apart in user space, from `origin`: the dashes of no
    length of a dash pattern [0 DOT_SPACING] along lines `degrees` from the x axis, or, with
    `points`, subpaths of one point; a line width of `width`, caps `cap`, under `ctm`, at a
    constant alpha `alpha` or in the blend mode `blend`. A dot of round caps is a disc, of
    projecting square caps a square along the path; one of non-zero width never disappears
    (ISO 32000-2 8.4.3.6, 8.5.3.2, 10.7.4), wherever it lies on the pixel grid. With
    `length`, each dot is instead a segment that long, a subpath or a dash of [length,
    DOT_SPACING - length], shorter than a pixel: nor does it. A line width this thin is
    widened to a pen a device pixel across its narrower axis (8.4.3.2)."""

    def __init__(
        self,
        name: str,
        width: float,
        cap: int,
        *,
        degrees: float = 0.0,
        points: bool = False,
        length: float = 0.0,
        origin: Point = (40.0, 40.0),
        ctm: Affine = IDENTITY,
        alpha: float = 1.0,
        blend: str | None = None,
    ) -> None:
        self.name = name
        self.width = width
        self.cap = cap
        self.degrees = degrees
        self.points = points
        self.length = length
        self.origin = origin
        self.ctm = ctm
        self.alpha = alpha
        self.blend = blend

    def centres(self) -> list[Point]:
        """Each dot's centre in user space."""
        radians = math.radians(self.degrees)
        ux, uy = math.cos(radians), math.sin(radians)
        x0, y0 = self.origin
        return [
            (x0 + column * DOT_SPACING * ux, y0 + row * DOT_SPACING + column * DOT_SPACING * uy)
            for row in range(DOT_ROWS)
            for column in range(DOT_ROWS)
        ]

    def expected_area(self) -> float:
        """The page area, in square device pixels, of each dot: its pen, a circle a line
        width across in stroke space (user space scaled by the mean of the CTM's scales
        along the axes) widened to HAIRLINE_FLOOR on the page across the CTM's least
        stretch, swept along its length, with its caps."""
        a, b, c, d = self.ctm[:4]
        scale = (math.hypot(a, b) + math.hypot(c, d)) / 2.0
        forward = (a / scale, b / scale, c / scale, d / scale)
        smallest, _ = matrix_stretches(forward)
        determinant = abs(forward[0] * forward[3] - forward[1] * forward[2])
        half = max(HAIRLINE_FLOOR / 2.0 / smallest, self.width * scale / 2.0)
        length = self.length * scale
        if self.cap == ROUND:
            area = math.pi * half * half + 2.0 * half * length
        elif self.cap == SQUARE:
            area = 2.0 * half * (2.0 * half + length)
        else:
            area = 2.0 * half * length
        return area * determinant

    def pdf(self) -> bytes:
        paint = []
        if self.alpha != 1.0:
            paint.append(b"/CA %s" % numbers([self.alpha]))
        if self.blend is not None:
            paint.append(b"/BM /%s" % self.blend.encode())
        body = b"q %s cm %s w %d J " % (numbers(self.ctm), numbers([self.width]), self.cap)
        if paint:
            body += b"/G0 gs "
        radians = math.radians(self.degrees)
        ux, uy = math.cos(radians), math.sin(radians)
        if self.points:
            body += b"".join(
                b"%s m %s l "
                % (numbers(c), numbers((c[0] + ux * self.length, c[1] + uy * self.length)))
                for c in self.centres()
            )
        else:
            # The last dash falls half a spacing before each line's end.
            length = (DOT_ROWS - 0.5) * DOT_SPACING
            body += b"[%s] 0 d " % numbers([self.length, DOT_SPACING - self.length])
            x0, y0 = self.origin
            for row in range(DOT_ROWS):
                start = (x0, y0 + row * DOT_SPACING)
                end = (start[0] + ux * length, start[1] + uy * length)
                body += b"%s m %s l " % (numbers(start), numbers(end))
        body += b"S Q"
        resources = b"<< /ExtGState << /G0 << %s >> >> >>" % b" ".join(paint)
        return document(body, resources)


def dots() -> list[Dots]:
    found = []
    for width in (0.05, 0.12, 0.2):
        for paint, alpha, blend in (
            ("opaque", 1.0, None),
            ("alpha 0.5", 0.5, None),
            ("Multiply", 1.0, "Multiply"),
        ):
            for origin in ((40.0, 40.0), (40.5, 40.5), (40.37, 40.81)):
                where = f"at {origin[0]:g}, {origin[1]:g}"
                found += [
                    Dots(
                        f"round dashes, {paint}, width {width}, {where}",
                        width,
                        ROUND,
                        origin=origin,
                        alpha=alpha,
                        blend=blend,
                    ),
                    Dots(
                        f"round dashes at 45 degrees, {paint}, width {width}, {where}",
                        width,
                        ROUND,
                        degrees=45.0,
                        origin=origin,
                        alpha=alpha,
                        blend=blend,
                    ),
                    Dots(
                        f"square dashes, {paint}, width {width}, {where}",
                        width,
                        SQUARE,
                        origin=origin,
                        alpha=alpha,
                        blend=blend,
                    ),
                    Dots(
                        f"square dashes at 45 degrees, {paint}, width {width}, {where}",
                        width,
                        SQUARE,
                        degrees=45.0,
                        origin=origin,
                        alpha=alpha,
                        blend=blend,
                    ),
                    Dots(
                        f"round points, {paint}, width {width}, {where}",
                        width,
                        ROUND,
                        points=True,
                        origin=origin,
                        alpha=alpha,
                        blend=blend,
                    ),
                ]
    # Segments a tenth of a pixel long: subpaths and dashes, with each cap.
    for width in (0.05, 0.12, 0.2):
        for paint, alpha, blend in (
            ("opaque", 1.0, None),
            ("alpha 0.5", 0.5, None),
            ("Multiply", 1.0, "Multiply"),
        ):
            for origin in ((40.0, 40.0), (40.5, 40.5), (40.37, 40.81)):
                where = f"at {origin[0]:g}, {origin[1]:g}"
                found += [
                    Dots(
                        f"{kind} 0.1 long, cap {cap}, {degrees:g} degrees, {paint}, width {width},"
                        f" {where}",
                        width,
                        cap,
                        degrees=degrees,
                        points=points,
                        length=0.1,
                        origin=origin,
                        alpha=alpha,
                        blend=blend,
                    )
                    for kind, points in (("segments", True), ("dashes", False))
                    for cap in (BUTT, ROUND, SQUARE)
                    for degrees in (0.0, 45.0)
                ]
    # Under a stroke matrix, where the dot is an ellipse on the page.
    stretch: Affine = (1.0, 0.0, 0.0, 0.5, 0.0, 0.0)
    found += [
        Dots(
            f"stretched, round {kind}, width 0.1",
            0.1,
            ROUND,
            points=points,
            origin=(40.0, 80.0),
            ctm=stretch,
            alpha=0.5,
        )
        for kind, points in (("dashes", False), ("points", True))
    ]
    found.append(
        Dots(
            "stretched, square dashes, width 0.1",
            0.1,
            SQUARE,
            origin=(40.0, 80.0),
            ctm=stretch,
            blend="Multiply",
        )
    )
    return found


def check_dots(case: Dots) -> bool:
    """Each dot's coverage, the alpha in the 3 x 3 pixels around its centre, must be its
    expected area within DOT_SLACK and a tenth of it, and at least DOT_LEAST."""
    pixels = rendered(case)
    alpha = pixels[..., 3].astype(numpy.float64) / 255.0 / case.alpha
    height = alpha.shape[0]
    expected = case.expected_area()
    slack = DOT_SLACK + 0.1 * expected
    wrong = []
    for centre in case.centres():
        x, y = apply(case.ctm, centre)
        column, row = math.floor(x), math.floor(height - y)
        ink = float(alpha[row - 1 : row + 2, column - 1 : column + 2].sum())
        if ink < DOT_LEAST or abs(ink - expected) > slack:
            wrong.append((centre, ink))
    for centre, ink in wrong[:2]:
        print(
            f"  {case.name}: the dot at {centre[0]:g}, {centre[1]:g} covers {ink:.3f} of a"
            f" pixel, not {expected:.3f}"
        )
    if wrong:
        print(f"{case.name}: {len(wrong)} of {DOT_ROWS * DOT_ROWS} dots wrong")
    return not wrong


class ThinLine:
    """One line from `start` to `end` in user space, a line width of `width`, under `ctm`."""

    def __init__(self, width: float, start: Point, end: Point, ctm: Affine = IDENTITY) -> None:
        self.width = width
        self.start = start
        self.end = end
        self.ctm = ctm

    def pdf(self) -> bytes:
        body = b"q %s cm %s w %s m %s l S Q" % (
            numbers(self.ctm),
            numbers([self.width]),
            numbers(self.start),
            numbers(self.end),
        )
        return document(body, b"<< >>")

    def device_length(self, scale: float) -> float:
        (x0, y0), (x1, y1) = apply(self.ctm, self.start), apply(self.ctm, self.end)
        return math.hypot(x1 - x0, y1 - y0) * scale


# The ink of a pixel-wide line, at every phase of the sample grid along it, is within this
# fraction of its length.
THIN_TOLERANCE = 0.05


def check_thin_lines() -> int:
    """A line thinner than a device pixel is a pixel wide, at any scale; a line width of 0
    paints the very pixels a tiny one does; and under a CTM that rotates and scales
    unevenly, a thin line along the pen's narrower axis is a pixel wide (ISO 32000-2
    8.4.3.2, 10.7.4). Returns the failures."""
    turned = stretched_rotation(30.0, 3.0, 0.4)
    wide = [
        (f"width 0.1 at scale {scale:g}, {name}", ThinLine(0.1, start, end), scale)
        for scale in (1.0, 4.0)
        for name, start, end in (
            ("sloping", (50.0, 100.0), (250.0, 130.0)),
            ("horizontal", (50.0, 150.3), (250.0, 150.3)),
        )
    ]
    # The user space x axis runs along the pen's wider axis, so the line is as wide as the
    # narrower one: a pixel.
    wide += [
        (
            f"width {width:g}, rotated and stretched",
            ThinLine(width, (-40.0, 0.0), (40.0, 0.0), turned),
            1.0,
        )
        for width in (0.0, 0.05)
    ]
    failures = 0
    for name, line, scale in wide:
        pixels = rendered(line, scale)
        across = (
            float(pixels[..., 3].astype(numpy.float64).sum()) / 255.0 / line.device_length(scale)
        )
        if abs(across - 1.0) > THIN_TOLERANCE:
            failures += 1
            print(f"  {name}: {across:.3f} device pixels wide, not 1")
    same = [
        (f"scale {scale:g}", (50.0, 100.0), (250.0, 130.0), IDENTITY, scale) for scale in (1.0, 4.0)
    ]
    same.append(("rotated and stretched", (-30.0, -20.0), (40.0, 60.0), turned, 1.0))
    for name, start, end, ctm, scale in same:
        zero = rendered(ThinLine(0.0, start, end, ctm), scale)
        tiny = rendered(ThinLine(1e-4, start, end, ctm), scale)
        if not numpy.array_equal(zero, tiny):
            failures += 1
            print(f"  width 0 and width 0.0001 differ, {name}")
    return failures


def once(case: Case) -> Case:
    """The case with each stroke's subpath held once."""
    twin = copy.copy(case)
    twin.strokes = tuple(copy.copy(stroke) for stroke in case.strokes)
    for stroke in twin.strokes:
        stroke.repeat = 1
    return twin


def check(case: Case) -> tuple[int, int]:
    pixels = rendered(case)
    height, width = pixels.shape[:2]
    inside, checked = outline_pixels(case, height, width)
    painted, untouched = case.expected()
    difference = numpy.abs(pixels.astype(int) - numpy.array(painted)).max(axis=2)
    # A pixel nothing paints is transparent, whatever its colour.
    outside = numpy.abs(pixels.astype(int) - numpy.array(untouched))
    outside_error = outside[..., 3] if untouched[3] == 0 else outside.max(axis=2)
    error = numpy.where(inside, difference, outside_error)
    # Blending rounds each channel once: a byte either way.
    wrong = numpy.argwhere(checked & (error > 2))
    for row, column in wrong[:3]:
        expected = painted if inside[row, column] else untouched
        pixel = tuple(int(value) for value in pixels[row, column])
        print(f"  {case.name}: pixel {column},{row}: {pixel}, not {expected}")
    if all(stroke.repeat == 1 for stroke in case.strokes):
        return int(checked.sum()), len(wrong)
    # A stroke holding its subpath twice is the same shape, painted the same, to the
    # edges' blended pixels: each pixel must match the stroke holding it once.
    twin = rendered(once(case))
    unlike = numpy.argwhere((pixels != twin).any(axis=2))
    for row, column in unlike[:3]:
        pixel = tuple(int(value) for value in pixels[row, column])
        expected = tuple(int(value) for value in twin[row, column])
        print(f"  {case.name}: pixel {column},{row}: {pixel}, not {expected} as held once")
    return int(checked.sum()) + height * width, len(wrong) + len(unlike)


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
    thin = hairlines()
    thin_failures = sum(not check_hairline(hairline) for hairline in thin)
    print(f"{len(thin)} hairline pages: {thin_failures} failures")
    dotted = dots()
    dot_failures = sum(not check_dots(case) for case in dotted)
    print(f"{len(dotted)} dotted hairline pages: {dot_failures} failures")
    line_failures = check_thin_lines()
    print(f"thin lines: {line_failures} failures")
    return 1 if failures or thin_failures or dot_failures or line_failures else 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: AGPL-3.0-only

"""Check that large filled paths are antialiased like small ones, against their exact spans.

A fill paints the pixels of its path's interior under the nonzero or even-odd rule (ISO
32000-2 8.5.3.3), and core gives a pixel the share of it the path covers, whatever the
path's size: the exact signed area for a nonzero fill in the Normal blend mode under a
rectangular clip, and otherwise the share of a 4x4 grid of sample points inside the path.

This writes pages that fill paths a few hundred pixels across: convex and many-edged
polygons, a pentagram and a bow tie that cross themselves, rings of Bezier circles under
both rules, a thin sliver, nearly horizontal edges, a triangle whose corners are far off
the page, and black fills large and small. Some are drawn at opacity 0.5, in Multiply over a
backdrop, through a Luminosity soft mask, in a knockout group, with a shading and a tiling
pattern of one colour, and under a rectangular and a triangular clip. The reference, in
Python, measures the path's spans exactly along 64 rows of each pixel; a pixel may be off it
by what a 4x4 grid can miss, a sixteenth of the pixel for each of its 4x4 cells an edge
passes through, and a row's height along each edge for the reference's own error, through
the colour that coverage gives the pixel, plus SLACK for rounding. Curves are flattened as
core flattens them, and a hard clip's pixels are in by their centres. On a commit where a
fill this large covers a pixel by its centre, every page fails.

It also checks supersampled_coverage_plane, which counts the 4x4 samples, against the
Python below that reads every edge at every sample row, on random paths, for equality.

uv run --all-packages python scripts/check_large_fills.py [--pdfs DIRECTORY]

--pdfs writes each page to DIRECTORY, to compare with another renderer.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pdf import PdfDocument  # noqa: E402
from core_pdf_cythonized import supersampled_coverage_plane  # noqa: E402
from scripts.check_patterns import document, numbers, stream  # noqa: E402

type Point = tuple[float, float]
type Contour = list[Point]
type Colour = tuple[float, float, float]
type Expected = Callable[[numpy.ndarray], numpy.ndarray]

SIZE = 320
# The reference measures the path's spans along this many rows of each pixel.
ROWS = 64
# Core's sample points per pixel along each axis.
SAMPLES = 4
# A pixel may be off the reference by this much in a channel beyond what the 4x4 grid misses:
# the rounding of its alpha and colour.
SLACK = 2.0
# Each page must have at least this many pixels its path covers in part.
MIN_EDGE_PIXELS = 100
COLOUR: Colour = (0.2, 0.4, 0.8)
WHITE: Colour = (1.0, 1.0, 1.0)
BACKDROP: Colour = (0.9, 0.7, 0.3)
GREEN: Colour = (0.1, 0.8, 0.2)
RED: Colour = (0.9, 0.1, 0.1)
# A curve is flattened into lines at even steps of t, one per this many units of its control
# polygon's length (for the default flatness of 1), and at least 4 and at most 128 of them, as
# core flattens it: the reference checks the fill of those lines, not the flattening.
CURVE_LENGTH_PER_LINE = 8.0


def regular(count: int, radius: float, turn: float, centre: Point = (160.3, 159.7)) -> Contour:
    cx, cy = centre
    return [
        (
            cx + radius * math.cos(turn + 2 * math.pi * k / count),
            cy + radius * math.sin(turn + 2 * math.pi * k / count),
        )
        for k in range(count)
    ]


def circle(cx: float, cy: float, radius: float, *, reverse: bool = False) -> list[Point]:
    """Four Bezier curves, as 13 points: the start, then three per curve."""
    k = 0.5522847498 * radius
    points = [
        (cx + radius, cy),
        (cx + radius, cy + k),
        (cx + k, cy + radius),
        (cx, cy + radius),
        (cx - k, cy + radius),
        (cx - radius, cy + k),
        (cx - radius, cy),
        (cx - radius, cy - k),
        (cx - k, cy - radius),
        (cx, cy - radius),
        (cx + k, cy - radius),
        (cx + radius, cy - k),
        (cx + radius, cy),
    ]
    if reverse:
        points.reverse()
    return points


def blob(count: int) -> Contour:
    points = []
    for k in range(count):
        t = 2 * math.pi * k / count
        radius = 130 * (1 + 0.12 * math.sin(7 * t) + 0.05 * math.sin(61 * t))
        points.append((160.2 + radius * math.cos(t), 160.1 + radius * math.sin(t)))
    return points


class Shape:
    """Contours of lines, or of Bezier curves (a start and three points per curve) when
    curved, filled under rule."""

    def __init__(self, name: str, contours: list[list[Point]], rule: str, *, curved=False):
        self.name = name
        self.contours = contours
        self.rule = rule
        self.curved = curved

    def operators(self) -> bytes:
        out = b""
        for contour in self.contours:
            (x, y), *rest = contour
            out += b"%s m " % numbers((x, y))
            if self.curved:
                for index in range(0, len(rest), 3):
                    out += b"%s c " % numbers(
                        [v for point in rest[index : index + 3] for v in point]
                    )
            else:
                for point in rest:
                    out += b"%s l " % numbers(point)
            out += b"h "
        return out + (b"f*" if self.rule == "evenodd" else b"f")

    def polygons(self) -> list[Contour]:
        """The contours as polygons, each curve flattened as core flattens it."""
        if not self.curved:
            return self.contours
        polygons = []
        for contour in self.contours:
            polygon = [contour[0]]
            for index in range(1, len(contour), 3):
                p0 = polygon[-1]
                p1, p2, p3 = contour[index : index + 3]
                length = math.dist(p0, p1) + math.dist(p1, p2) + math.dist(p2, p3)
                steps = min(128, max(4, math.ceil(length / CURVE_LENGTH_PER_LINE)))
                for step in range(1, steps + 1):
                    t = step / steps
                    u = 1 - t
                    polygon.append(
                        (
                            u**3 * p0[0]
                            + 3 * u * u * t * p1[0]
                            + 3 * u * t * t * p2[0]
                            + t**3 * p3[0],
                            u**3 * p0[1]
                            + 3 * u * u * t * p1[1]
                            + 3 * u * t * t * p2[1]
                            + t**3 * p3[1],
                        )
                    )
            polygons.append(polygon)
        return polygons


STAR = regular(5, 150.0, 0.31)
SHAPES = {
    "heptagon": Shape("heptagon", [regular(7, 140.0, 0.17)], "nonzero"),
    "pentagram": Shape("pentagram", [[STAR[k * 2 % 5] for k in range(5)]], "nonzero"),
    "pentagram even-odd": Shape(
        "pentagram even-odd", [[STAR[k * 2 % 5] for k in range(5)]], "evenodd"
    ),
    "bow tie": Shape(
        "bow tie", [[(20.3, 30.7), (300.1, 290.2), (300.6, 40.4), (19.8, 280.9)]], "nonzero"
    ),
    "ring even-odd": Shape(
        "ring even-odd",
        [circle(160.4, 160.2, 140.3), circle(171.1, 150.6, 70.2)],
        "evenodd",
        curved=True,
    ),
    "ring nonzero": Shape(
        "ring nonzero",
        [circle(160.4, 160.2, 140.3), circle(171.1, 150.6, 70.2, reverse=True)],
        "nonzero",
        curved=True,
    ),
    "blob of 600 edges": Shape("blob of 600 edges", [blob(600)], "nonzero"),
    "sliver": Shape(
        "sliver", [[(10.2, 20.1), (300.4, 290.3), (300.4, 290.75), (10.2, 20.45)]], "nonzero"
    ),
    "nearly horizontal edges": Shape(
        "nearly horizontal edges",
        [[(15.2, 40.1), (305.3, 41.6), (300.4, 280.2), (20.1, 278.9)]],
        "nonzero",
    ),
    "far-off corners": Shape(
        "far-off corners", [[(-2.0e5, 160.3), (2.0e5, 140.7), (0.37, 3.0e5)]], "nonzero"
    ),
    "far-off corners even-odd": Shape(
        "far-off corners even-odd",
        [[(-2.0e5, 160.3), (2.0e5, 140.7), (0.37, 3.0e5)], regular(6, 60.0, 0.4)],
        "evenodd",
    ),
}
# A rectangle on pixel boundaries, and a triangle, as hard clips.
CLIP_RECT = (40.0, 50.0, 230.0, 210.0)
CLIP_TRIANGLE: Contour = [(30.2, 20.5), (300.4, 90.3), (100.8, 305.9)]


def over(colour: Colour, base: numpy.ndarray, alpha: numpy.ndarray) -> numpy.ndarray:
    """colour at alpha over base, in bytes."""
    return base + alpha[..., None] * (numpy.array(colour) * 255 - base)


def flat(colour: Colour) -> numpy.ndarray:
    return numpy.broadcast_to(numpy.array(colour) * 255, (SIZE, SIZE, 3))


def plain(colour: Colour = COLOUR) -> Expected:
    return lambda c: over(colour, flat(WHITE), c)


class Case:
    """A page that fills shape with `paint` (a colour operator), inside `before` and
    `after`, in a knockout group when knockout; expected maps each pixel's coverage to the
    colour over white it must show."""

    def __init__(
        self,
        name: str,
        shape: Shape,
        expected: Expected,
        *,
        paint: bytes = b"%s rg" % numbers(COLOUR),
        before: bytes = b"",
        after: bytes = b"",
        knockout: bool = False,
        mask: numpy.ndarray | None = None,
    ) -> None:
        self.name = name
        self.shape = shape
        self.expected = expected
        self.paint = paint
        self.before = before
        self.after = after
        self.knockout = knockout
        self.mask = mask

    def pdf(self) -> bytes:
        body = b"%s %s %s %s" % (self.before, self.paint, self.shape.operators(), self.after)
        resources = (
            b"<< /Pattern << /S 6 0 R /T 7 0 R >> /ExtGState << /Half 8 0 R /Multiply 9 0 R "
            b"/Mask 10 0 R >> /XObject << /K 11 0 R >> >>"
        )
        if self.knockout:
            content = b"%s rg 0 0 %d %d re f /K Do" % (numbers(RED), SIZE, SIZE)
            group = b"%s rg 0 0 %d %d re f /Half gs %s" % (numbers(GREEN), SIZE, SIZE, body)
        else:
            content = body
            group = b""
        shading_function = b"<< /FunctionType 2 /Domain [0 1] /C0 [%s] /C1 [%s] /N 1 >>" % (
            numbers(COLOUR),
            numbers(COLOUR),
        )
        mask_group = stream(
            b"/Type /XObject /Subtype /Form /BBox [0 0 %d %d] "
            b"/Group << /S /Transparency /CS /DeviceGray >>" % (SIZE, SIZE),
            b"0.5 g 0 0 %d %d re f" % (SIZE, SIZE),
        )
        return document(
            [
                b"<< /Type /Catalog /Pages 2 0 R >>",
                b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Resources %s "
                b"/Contents 4 0 R >>" % (SIZE, SIZE, resources),
                stream(b"", content),
                b"<< >>",
                b"<< /PatternType 2 /Shading << /ShadingType 2 /ColorSpace /DeviceRGB "
                b"/Coords [0 0 %d 0] /Extend [true true] /Function %s >> >>"
                % (SIZE, shading_function),
                stream(
                    b"/PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 10 10] /XStep 10 "
                    b"/YStep 10 /Resources << >>",
                    b"%s rg 0 0 10 10 re f" % numbers(COLOUR),
                ),
                b"<< /Type /ExtGState /ca 0.5 >>",
                b"<< /Type /ExtGState /BM /Multiply >>",
                b"<< /Type /ExtGState /SMask << /S /Luminosity /G 12 0 R >> >>",
                stream(
                    b"/Type /XObject /Subtype /Form /BBox [0 0 %d %d] /Resources %s "
                    b"/Group << /S /Transparency /K true >>" % (SIZE, SIZE, resources),
                    group,
                ),
                mask_group,
            ]
        )


def render(pdf: bytes) -> numpy.ndarray:
    """The page over white, as floats in bytes."""
    with PdfDocument(pdf) as opened:
        image = opened.pages[0].render().rasterize()
    pixels = numpy.frombuffer(bytes(image.pixels), numpy.uint8).reshape(SIZE, SIZE, 4)
    alpha = pixels[..., 3:].astype(numpy.float64) / 255
    return pixels[..., :3].astype(numpy.float64) * alpha + 255 * (1 - alpha)


def device_edges(polygons: list[Contour]) -> numpy.ndarray:
    rows = []
    for polygon in polygons:
        for index, (x0, y0) in enumerate(polygon):
            x1, y1 = polygon[(index + 1) % len(polygon)]
            if y0 != y1:
                rows.append((x0, SIZE - y0, x1, SIZE - y1))
    return numpy.array(rows, dtype=numpy.float64)


def clipped_edges(polygons: list[Contour]) -> numpy.ndarray:
    """The polygons' edges in device space, cut to the page (with a pixel's margin)."""
    clipped = []
    for x0, y0, x1, y1 in device_edges(polygons).tolist():
        # Liang-Barsky against the page square.
        t0, t1 = 0.0, 1.0
        dx, dy = x1 - x0, y1 - y0
        for p, q in ((-dx, x0 + 1), (dx, SIZE + 1 - x0), (-dy, y0 + 1), (dy, SIZE + 1 - y0)):
            if p == 0:
                if q < 0:
                    t0, t1 = 1.0, 0.0
                continue
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
        if t0 < t1:
            clipped.append((x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy))
    return numpy.array(clipped, dtype=numpy.float64).reshape(-1, 4)


def reference(shape: Shape) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Each pixel's coverage, and how far off it the 4x4 grid may be.

    The coverage is the path's spans along ROWS rows of each pixel, each span's length
    measured exactly, so it is within a row's height of the area along each edge. The 4x4
    grid can be off by a sixteenth for each of its cells an edge passes through, found by
    walking each edge in steps of a sixty-fourth of a pixel."""
    edges = device_edges(shape.polygons())
    x0, y0, x1, y1 = edges.T
    low = numpy.minimum(y0, y1)
    high = numpy.maximum(y0, y1)
    direction = numpy.where(y1 > y0, 1, -1)
    lefts = numpy.arange(SIZE, dtype=numpy.float64)
    coverage = numpy.zeros((SIZE, SIZE))
    for row in range(SIZE * ROWS):
        y = (row + 0.5) / ROWS
        crossing = (low <= y) & (y < high)
        if not crossing.any():
            continue
        cx = x0[crossing] + (y - y0[crossing]) / (y1[crossing] - y0[crossing]) * (
            x1[crossing] - x0[crossing]
        )
        order = numpy.argsort(cx)
        cx = cx[order]
        if shape.rule == "evenodd":
            inside = numpy.arange(1, len(cx) + 1) % 2 == 1
        else:
            inside = numpy.cumsum(direction[crossing][order]) != 0
        starts = cx[:-1][inside[:-1]]
        ends = cx[1:][inside[:-1]]
        if len(starts) == 0:
            continue
        lengths = numpy.minimum(ends[:, None], lefts + 1) - numpy.maximum(starts[:, None], lefts)
        coverage[row // ROWS] += numpy.clip(lengths, 0, 1).sum(axis=0) / ROWS
    crossed = numpy.zeros((SIZE * SAMPLES, SIZE * SAMPLES), bool)
    for ex0, ey0, ex1, ey1 in clipped_edges(shape.polygons()).tolist():
        steps = int(math.hypot(ex1 - ex0, ey1 - ey0) * 64) + 2
        t = numpy.linspace(0.0, 1.0, steps)
        columns = numpy.floor((ex0 + t * (ex1 - ex0)) * SAMPLES).astype(int)
        rows = numpy.floor((ey0 + t * (ey1 - ey0)) * SAMPLES).astype(int)
        keep = (columns >= 0) & (columns < SIZE * SAMPLES) & (rows >= 0) & (rows < SIZE * SAMPLES)
        crossed[rows[keep], columns[keep]] = True
    cells = crossed.reshape(SIZE, SAMPLES, SIZE, SAMPLES).sum(axis=(1, 3))
    # The reference itself may be off by a row's height along each edge.
    missable = cells / (SAMPLES * SAMPLES) + numpy.where(cells > 0, 2.0 / ROWS, 0.0)
    return numpy.clip(coverage, 0, 1), missable


def centres_inside(polygon: Contour) -> numpy.ndarray:
    edges = device_edges([polygon])
    centres = (numpy.arange(SIZE) + 0.5)[:, None]
    mask = numpy.zeros((SIZE, SIZE), bool)
    xs = numpy.arange(SIZE) + 0.5
    for row in range(SIZE):
        y = centres[row, 0]
        x0, y0, x1, y1 = edges.T
        crossing = (numpy.minimum(y0, y1) <= y) & (y < numpy.maximum(y0, y1))
        cx = numpy.sort(
            x0[crossing]
            + (y - y0[crossing]) / (y1[crossing] - y0[crossing]) * (x1[crossing] - x0[crossing])
        )
        mask[row] = numpy.searchsorted(cx, xs, side="right") % 2 == 1
    return mask


def rect_mask() -> numpy.ndarray:
    x0, y0, x1, y1 = CLIP_RECT
    mask = numpy.zeros((SIZE, SIZE), bool)
    mask[int(SIZE - y1) : int(SIZE - y0), int(x0) : int(x1)] = True
    return mask


def cases() -> list[Case]:
    rect = b"q %s re W n" % numbers(
        (CLIP_RECT[0], CLIP_RECT[1], CLIP_RECT[2] - CLIP_RECT[0], CLIP_RECT[3] - CLIP_RECT[1])
    )
    triangle = b"q %s m %s l %s l h W n" % tuple(numbers(point) for point in CLIP_TRIANGLE)
    backdrop = b"%s rg 0 0 %d %d re f" % (numbers(BACKDROP), SIZE, SIZE)
    found = [Case(name, shape, plain()) for name, shape in SHAPES.items()]
    found += [
        Case("heptagon, black", SHAPES["heptagon"], plain((0.0, 0.0, 0.0)), paint=b"0 g"),
        Case(
            "small triangle, black",
            Shape("small triangle", [[(100.3, 100.2), (160.7, 104.9), (121.6, 151.1)]], "nonzero"),
            plain((0.0, 0.0, 0.0)),
            paint=b"0 g",
        ),
    ]
    for key in ("pentagram", "pentagram even-odd", "ring even-odd", "blob of 600 edges"):
        shape = SHAPES[key]
        found += [
            Case(
                f"{key}, opacity 0.5",
                shape,
                lambda c: over(COLOUR, flat(WHITE), c * 0.5),
                before=b"/Half gs",
            ),
            Case(
                f"{key}, Multiply over a backdrop",
                shape,
                lambda c: over(
                    tuple(numpy.array(BACKDROP) * numpy.array(COLOUR)), flat(BACKDROP), c
                ),
                before=backdrop + b" /Multiply gs",
            ),
            Case(
                f"{key}, Luminosity soft mask of 0.5",
                shape,
                lambda c: over(COLOUR, flat(WHITE), c * 128 / 255),
                before=b"/Mask gs",
            ),
            Case(
                f"{key}, knockout group at opacity 0.5",
                shape,
                lambda c: over(
                    tuple(0.5 * numpy.array(COLOUR) + 0.5 * numpy.array(RED)), flat(GREEN), c
                ),
                knockout=True,
            ),
            Case(f"{key}, shading pattern", shape, plain(), paint=b"/Pattern cs /S scn"),
            Case(f"{key}, tiling pattern", shape, plain(), paint=b"/Pattern cs /T scn"),
            Case(
                f"{key}, rectangular clip",
                shape,
                plain(),
                before=rect,
                after=b"Q",
                mask=rect_mask(),
            ),
            Case(
                f"{key}, triangular clip",
                shape,
                plain(),
                before=triangle,
                after=b"Q",
                mask=centres_inside(CLIP_TRIANGLE),
            ),
        ]
    return found


def check(case: Case, coverage: numpy.ndarray, missable: numpy.ndarray) -> int:
    rendered = render(case.pdf())
    if case.mask is not None:
        coverage = coverage * case.mask
        missable = missable * case.mask
    expected = case.expected(coverage)
    low = case.expected(numpy.clip(coverage - missable, 0, 1))
    high = case.expected(numpy.clip(coverage + missable, 0, 1))
    allowed = numpy.maximum(numpy.abs(low - expected), numpy.abs(high - expected)).max(axis=2)
    error = numpy.abs(rendered - expected).max(axis=2)
    bad = error > allowed + SLACK
    failed = int(bad.sum())
    edges = int(((coverage > 0.05) & (coverage < 0.95)).sum())
    if edges < MIN_EDGE_PIXELS:
        print(f"  {case.name}: only {edges} edge pixels")
        failed += 1
    if failed:
        row, column = numpy.argwhere(bad)[0] if bad.any() else (0, 0)
        print(
            f"  {case.name}: {int(bad.sum())} pixels off the reference, the first at "
            f"{column},{row}: {rendered[row, column].round(1)} for coverage "
            f"{coverage[row, column]:.3f}, expected {expected[row, column].round(1)} "
            f"within {allowed[row, column] + SLACK:.1f}"
        )
    return failed


def reference_counts(
    edges: numpy.ndarray,
    crop_x0: float,
    crop_y1: float,
    scale: float,
    box: tuple[int, int, int, int],
    evenodd: bool,
) -> tuple[numpy.ndarray, int] | None:
    """supersampled_coverage_plane in Python, reading every edge at every sample row."""
    ix0, iy0, ix1, iy1 = box
    width = ix1 - ix0
    height = iy1 - iy0
    kept = [tuple(row) for row in edges.tolist() if row[1] != row[3]]
    if not kept or width <= 0 or height <= 0:
        return None
    counts = numpy.zeros((height, width), numpy.uint8)

    def add_span(start_x: float, end_x: float, deltas: list[int]) -> None:
        span_start = (start_x - crop_x0) * scale
        span_end = (end_x - crop_x0) * scale
        for sx in range(SAMPLES):
            offset = (sx + 0.5) / SAMPLES
            first = math.ceil(span_start - offset)
            last = math.ceil(span_end - offset)
            start = max(ix0, min(ix1, first))
            end = max(ix0, min(ix1, last))
            if end > start:
                deltas[start - ix0] += 1
                deltas[end - ix0] -= 1

    rows: list[int] = []
    for row in range(height):
        deltas = [0] * (width + 1)
        for sample_row in range(SAMPLES):
            page_y = crop_y1 - (float(iy0 + row) + (sample_row + 0.5) / SAMPLES) / scale
            crossings = []
            for x0, y0, x1, y1 in kept:
                if min(y0, y1) <= page_y < max(y0, y1):
                    crossings.append(
                        (x0 + ((page_y - y0) / (y1 - y0) * (x1 - x0)), 1 if y1 > y0 else -1)
                    )
            crossings.sort()
            if evenodd:
                for index in range(0, len(crossings) - 1, 2):
                    if crossings[index + 1][0] > crossings[index][0]:
                        add_span(crossings[index][0], crossings[index + 1][0], deltas)
                continue
            winding = 0
            index = 0
            previous = 0.0
            while index < len(crossings):
                x = crossings[index][0]
                if index and winding and x > previous:
                    add_span(previous, x, deltas)
                while index < len(crossings) and crossings[index][0] == x:
                    winding += crossings[index][1]
                    index += 1
                previous = x
        counts[row] = numpy.cumsum(deltas[:width])
        if counts[row].any():
            rows.append(row)
    if not rows:
        return None
    return counts[rows[0] : rows[-1] + 1], rows[0]


def check_kernel(count: int = 300) -> int:
    rng = numpy.random.default_rng(20261117)
    failed = 0
    for case in range(count):
        corners = int(rng.integers(3, 40))
        size = int(rng.integers(4, 90))
        spread = 1e5 if case % 9 == 0 else 1.0
        points = rng.normal(size / 2, size / 3, (corners, 2)) * [spread, 1.0]
        if case % 4 == 0:
            # Shared coordinates: horizontal edges and crossings that meet.
            points = numpy.round(points * 2) / 2
        edges = numpy.array(
            [(*points[k], *points[(k + 1) % corners]) for k in range(corners)], numpy.float64
        )
        scale = float(rng.choice([1.0, 0.75, 2.0, 1.37, -1.0]))
        crop_x0 = float(rng.normal(0, 3))
        crop_y1 = float(size / scale + rng.normal(0, 3))
        ix0, iy0 = int(rng.integers(0, 5)), int(rng.integers(0, 5))
        box = (ix0, iy0, ix0 + size, iy0 + size)
        for evenodd in (False, True):
            got = supersampled_coverage_plane(edges, crop_x0, crop_y1, scale, *box, evenodd)
            want = reference_counts(edges, crop_x0, crop_y1, scale, box, evenodd)
            same = (got is None and want is None) or (
                got is not None
                and want is not None
                and got[1] == want[1]
                and numpy.array_equal(got[0], want[0])
            )
            if not same:
                failed += 1
                if failed <= 3:
                    print(
                        f"  supersampled_coverage_plane differs on random path {case}, {evenodd=}"
                    )
    print(f"{2 * count} random paths: supersampled_coverage_plane differs on {failed}")
    return failed


def main(argv: Sequence[str]) -> int:
    pdfs = Path(argv[argv.index("--pdfs") + 1]) if "--pdfs" in argv else None
    failures = check_kernel()
    references: dict[int, tuple[numpy.ndarray, numpy.ndarray]] = {}
    selected = cases()
    for case in selected:
        key = id(case.shape)
        if key not in references:
            references[key] = reference(case.shape)
        failures += check(case, *references[key])
        if pdfs is not None:
            pdfs.mkdir(parents=True, exist_ok=True)
            name = "".join(ch if ch.isalnum() else "-" for ch in case.name)
            (pdfs / f"{name}.pdf").write_bytes(case.pdf())
    print(f"{len(selected)} large fill pages: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

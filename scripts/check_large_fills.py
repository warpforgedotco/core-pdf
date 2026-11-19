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

import itertools
import math
import sys
import tracemalloc
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pdf import PdfDocument  # noqa: E402
from core_pdf.impl.fonts.glyph_geometry import outline_arrays  # noqa: E402
from core_pdf_cythonized import (  # noqa: E402
    area_coverage_plane,
    fill_glyph_coverage_at,
    glyph_outline_edges,
    outline_simple,
    supersampled_coverage_plane,
    windings_mix,
)
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
# A pixel of area_coverage_plane may be off span_coverage's measure by this much: a 64th of
# its height for each of the few edges through it.
SPAN_TOLERANCE = 0.05
# How far a row supersampled 4x4, as rows past the sweep's budget are, may be from the span
# measure: two edges in a pixel, each up to an eighth off in x, and the lines' y.
SAMPLED_TOLERANCE = 0.3
# The kernel's TOUCH: a crossing this close outside two edges is taken at their end.
TOUCH = 1e-7
# The kernel's CROSS_SLACK: chains closer than this, relative to x, meet rather than cross.
CROSS_SLACK = 1e-9
# The kernel's EXACT_EDGES and ROW_SAMPLES: a row to measure exactly with more active edges is
# supersampled, ROW_SAMPLES x ROW_SAMPLES per pixel, as are the rows past the sweep's budget.
EXACT_EDGES = 256
ROW_SAMPLES = 4

# A pixel may be off the reference by this much in a channel beyond what the 4x4 grid misses:
# the rounding of its alpha and colour.
SLACK = 2.0
# A page of large fills may hold this much more memory at its peak than an empty page.
MEMORY_ALLOWANCE = 16 * 2**20
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
    "shared edge, opposite orientations": Shape(
        "shared edge, opposite orientations",
        [
            [(10.3, 10.2), (150.4, 12.6), (150.4, 305.1), (12.2, 300.7)],
            [(150.4, 12.6), (150.4, 305.1), (309.5, 298.2), (306.8, 15.3)],
        ],
        "nonzero",
    ),
    "small shared edge, opposite orientations": Shape(
        "small shared edge, opposite orientations",
        [
            [(100.3, 100.2), (140.4, 101.6), (140.4, 149.1), (101.2, 148.7)],
            [(140.4, 101.6), (140.4, 149.1), (180.5, 147.2), (179.8, 102.3)],
        ],
        "nonzero",
    ),
    "doubled heptagon": Shape(
        "doubled heptagon", [regular(7, 140.0, 0.17), regular(7, 140.0, 0.17)], "nonzero"
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
    if bad.any():
        row, column = numpy.argwhere(bad)[0]
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
) -> tuple[numpy.ndarray, int, int] | None:
    """supersampled_coverage_plane in Python, reading every edge at every sample row and
    every column of the box."""
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
    columns = numpy.flatnonzero(counts.any(axis=0))
    return (counts[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1], rows[0], int(columns[0]))


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
                and got[1:] == want[1:]
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


def py_max(a: float, b: float) -> float:
    return b if b > a else a  # noqa: FURB136 - the kernels' py_max, NaN and all


def py_min(a: float, b: float) -> float:
    return b if b < a else a  # noqa: FURB136 - the kernels' py_min, NaN and all


type Edge = tuple[float, float, float, float, float, int, int, int, float]


def edge_table(
    edges: numpy.ndarray,
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: float,
    iy0: float,
    height: int,
) -> list[Edge]:
    """The kernel's edges in device space: top x, top y, bottom y, slope, direction, first
    row, last row and chain."""
    table: list[Edge] = []
    last = (0.0, 0.0, 0.0)
    chain = -1
    for x0, y0, x1, y1 in edges.tolist():
        if y0 == y1:
            continue
        sx, sy = (x0 - crop_x0) * scale - ix0, (crop_y1 - y0) * scale - iy0
        ex, ey = (x1 - crop_x0) * scale - ix0, (crop_y1 - y1) * scale - iy0
        if sy == ey or not all(math.isfinite(v) for v in (sx, sy, ex, ey)):
            continue
        if ey > sy:
            top_x, top_y, bottom_y, slope, direction = sx, sy, ey, (ex - sx) / (ey - sy), 1.0
            bottom_x = ex
        else:
            top_x, top_y, bottom_y, slope, direction = ex, ey, sy, (sx - ex) / (sy - ey), -1.0
            bottom_x = sx
        if not math.isfinite(slope):
            continue
        first = int(py_min(py_max(float(math.floor(top_y)), 0.0), float(height)))
        last_row = int(py_max(py_min(float(math.ceil(bottom_y)), float(height)), 0.0))
        if not (table and (sx, sy, direction) == last):
            chain += 1
        last = (ex, ey, direction)
        table.append((top_x, top_y, bottom_y, slope, direction, first, last_row, chain, bottom_x))
    return table


def mark_span(flags: list[int], height: int, y0: float, y1: float) -> None:
    first = py_max(float(math.floor(y0)), 0.0)
    last = py_min(float(math.ceil(y1)), float(height))
    if not last > first:
        return
    for row in range(int(first), int(last)):
        flags[row] = 1


def mark_rows(table: list[Edge], height: int) -> list[int]:
    """The kernel's _mark_rows, without its budget: which rows to measure exactly."""
    mixed = [0] * (height + 1)
    positive = [0] * (height + 1)
    negative = [0] * (height + 1)
    chains: list[list[float]] = []  # first, last, direction, top, bottom
    for i, edge in enumerate(table):
        if chains and edge[7] == table[i - 1][7]:
            chain = chains[-1]
            chain[1] = i
            chain[3] = py_min(chain[3], edge[1])
            chain[4] = py_max(chain[4], edge[2])
        else:
            chains.append([i, i, 1 if edge[4] > 0.0 else -1, edge[1], edge[2]])
    for chain in chains:
        if chain[2] < 0:
            chain[0], chain[1] = chain[1], chain[0]
    cursors = [int(chain[0]) for chain in chains]
    bottom_cursors = [int(chain[0]) for chain in chains]

    def edge_x(edge: Edge, y: float) -> float:
        if y == edge[2]:
            return edge[8]
        if y == edge[1]:
            return edge[0]
        return edge[0] + (y - edge[1]) * edge[3]

    def chain_x(index: int, cursor: list[int], y: float) -> float:
        chain = chains[index]
        edge = table[cursor[index]]
        while edge[2] < y and cursor[index] != chain[1]:
            cursor[index] += int(chain[2])
            edge = table[cursor[index]]
        return edge_x(edge, y)

    def crossing(left: int, right: int, ya: float, yb: float) -> float | None:
        li, ri = cursors[left], cursors[right]
        y = previous = ya
        while True:
            el, er = table[li], table[ri]
            xl = edge_x(el, y)
            gap = xl - edge_x(er, y)
            if gap > CROSS_SLACK * py_max(1.0, abs(xl)):
                return previous
            if not y < yb:
                return None
            previous = y
            y = py_min(py_min(el[2], er[2]), yb)
            if el[2] <= y and li != chains[left][1]:
                li += int(chains[left][2])
            if er[2] <= y and ri != chains[right][1]:
                ri += int(chains[right][2])

    events = sorted(y for chain in chains for y in (chain[3], chain[4]))
    for low, high in itertools.pairwise(events):
        if not high > low or high <= 0.0 or low >= height:
            continue
        ya = py_max(low, 0.0)
        yb = py_min(high, float(height))
        band = []
        for i, chain in enumerate(chains):
            if chain[3] <= low and chain[4] >= high:
                band.append(
                    (chain_x(i, cursors, ya), chain_x(i, bottom_cursors, (ya + yb) * 0.5), i)
                )
        if not band:
            continue
        band.sort()
        bad = False
        sign = 0
        winding = 0
        for i in range(len(band) - 1):
            winding += int(chains[band[i][2]][2])
            if winding == 0:
                continue
            if abs(winding) > 1 or sign not in {0, winding}:
                bad = True
                break
            sign = winding
        if bad:
            mark_span(mixed, height, ya, yb)
            continue
        if sign > 0:
            mark_span(positive, height, ya, yb)
        elif sign < 0:
            mark_span(negative, height, ya, yb)
        earliest = yb
        for i in range(len(band) - 1):
            y = crossing(band[i][2], band[i + 1][2], ya, yb)
            if y is not None and y < earliest:
                earliest = y
        if earliest < yb:
            mark_span(mixed, height, earliest, yb)
    for row in range(height):
        if positive[row] and negative[row]:
            mixed[row] = 1
    return mixed


def add_piece(
    edge: Edge,
    top: float,
    bottom: float,
    direction: float,
    width: int,
    cells: list[float],
    rights: list[tuple[int, float]],
) -> None:
    top_x, top_y, _, slope = edge[0], edge[1], edge[2], edge[3]
    row_height = bottom - top
    if row_height <= 0.0:
        return
    entry_x = top_x + (top - top_y) * slope
    exit_x = top_x + (bottom - top_y) * slope
    left_x = py_min(exit_x, entry_x)
    right_x = py_max(exit_x, entry_x)
    walk_left = py_min(width + 1.0, py_max(-1.0, left_x))
    walk_right = py_min(width + 1.0, py_max(-1.0, right_x))
    floor_left = float(math.floor(walk_left))
    column_count = int(float(math.floor(walk_right)) - floor_left + 1.0)
    piece_span = right_x - left_x
    vertical = piece_span <= 0.0
    for offset in range(column_count):
        column_start = floor_left + offset
        fragment_left = left_x if offset == 0 else column_start
        fragment_right = right_x if offset == column_count - 1 else column_start + 1.0
        if not (vertical or fragment_right > fragment_left):
            continue
        share = 1.0 if vertical else (fragment_right - fragment_left) / piece_span
        fragment_height = row_height * share
        midpoint = (fragment_left + fragment_right) * 0.5
        column = int(py_min(float(width), py_max(0.0, float(math.floor(midpoint)))))
        in_cell = py_min(1.0, py_max(0.0, midpoint - column))
        signed_height = direction * fragment_height
        cells[column] += signed_height * (1.0 - in_cell)
        rights.append((column + 1, signed_height * in_cell))


def exact_row(active: list[Edge], row: int, width: int) -> list[float]:
    """The kernel's _exact_row: the row's stretches, split into bands at their ends and at
    crossings of edges of different stretches, and the pieces of the outline's edges alone."""
    top, bottom = float(row), row + 1.0
    stretches: list[list[float]] = []  # first, count, top, bottom, direction, chain
    spans: list[tuple[float, float, int, int]] = []  # low x, high x, position, stretch
    for position, edge in enumerate(active):
        ya = py_max(top, edge[1])
        yb = py_min(bottom, edge[2])
        xa = edge[0] + (ya - edge[1]) * edge[3]
        xb = edge[0] + (yb - edge[1]) * edge[3]
        if stretches and stretches[-1][5] == edge[7]:
            stretch = stretches[-1]
            stretch[2] = py_min(stretch[2], ya)
            stretch[3] = py_max(stretch[3], yb)
            stretch[1] += 1
        else:
            stretches.append([position, 1, ya, yb, 1 if edge[4] > 0.0 else -1, edge[7]])
        spans.append((py_min(xa, xb), py_max(xa, xb), position, len(stretches) - 1))
    ys = [top, bottom]
    for stretch in stretches:
        if stretch[2] > top:
            ys.append(stretch[2])
        if stretch[3] < bottom:
            ys.append(stretch[3])
    spans.sort(key=lambda span: (span[0], span[1], span[2]))
    for a in range(len(spans)):
        ei = active[spans[a][2]]
        for j in range(a + 1, len(spans)):
            if spans[j][0] > spans[a][1]:
                break
            if spans[j][3] == spans[a][3]:
                continue
            ej = active[spans[j][2]]
            if ei[3] == ej[3]:
                continue
            y = (ej[0] - ei[0] + ei[1] * ei[3] - ej[1] * ej[3]) / (ei[3] - ej[3])
            low_y, high_y = py_max(ei[1], ej[1]), py_min(ei[2], ej[2])
            if low_y - TOUCH <= y < low_y:
                y = low_y
            elif high_y < y <= high_y + TOUCH:
                y = high_y
            if top < y < bottom and low_y <= y <= high_y:
                ys.append(y)
    ys.sort()
    cells = [0.0] * (width + 2)
    rights: list[tuple[int, float]] = []
    for ya, yb in itertools.pairwise(ys):
        if not yb > ya:
            continue
        middle = (ya + yb) * 0.5
        present = []
        for stretch in stretches:
            if stretch[2] <= ya and stretch[3] >= yb:
                first, count = int(stretch[0]), int(stretch[1])
                x = None
                for edge in active[first : first + count]:
                    if edge[1] <= middle <= edge[2]:
                        x = edge[0] + (middle - edge[1]) * edge[3]
                        break
                if x is None:
                    edge = active[first]
                    x = edge[0] + (middle - edge[1]) * edge[3]
                present.append((x, int(stretch[5]), stretch))
        present.sort(key=lambda item: (item[0], item[1]))
        winding = 0
        for _, _, stretch in present:
            inside = winding != 0
            winding += int(stretch[4])
            if (winding != 0) == inside:
                continue
            first, count = int(stretch[0]), int(stretch[1])
            for edge in active[first : first + count]:
                add_piece(
                    edge,
                    py_max(ya, edge[1]),
                    py_min(yb, edge[2]),
                    -1.0 if inside else 1.0,
                    width,
                    cells,
                    rights,
                )
    return finish_row(cells, rights, width)


def finish_row(cells: list[float], rights: list[tuple[int, float]], width: int) -> list[float]:
    for column, weight in rights:
        cells[column] += weight
    out = []
    running = 0.0
    for column in range(width):
        running += cells[column]
        out.append(py_min(1.0, abs(running)))
    return out


def sampled_row(active: list[Edge], row: int, width: int) -> list[float]:
    """The kernel's _sampled_row: the share of a ROW_SAMPLES x ROW_SAMPLES grid of samples
    per pixel inside the path by nonzero winding, the crossings of each line taken in order
    of x. Sixteenths add exactly, so the order the kernel adds them in does not matter."""
    columns = width * ROW_SAMPLES
    counts = [0] * width
    for line in range(ROW_SAMPLES):
        y = row + (line + 0.5) / ROW_SAMPLES
        crossings = sorted(
            (edge[0] + (y - edge[1]) * edge[3], int(edge[4]))
            for edge in active
            if edge[1] <= y < edge[2]
        )
        winding = 0
        for (x, direction), (following, _) in itertools.pairwise(crossings):
            winding += direction
            if winding == 0:
                continue
            first = int(py_max(py_min(math.ceil(x * ROW_SAMPLES - 0.5), columns), 0.0))
            end = int(py_max(py_min(math.ceil(following * ROW_SAMPLES - 0.5), columns), 0.0))
            for column in range(first, end):
                counts[column // ROW_SAMPLES] += 1
    return [py_min(1.0, count / ROW_SAMPLES**2) for count in counts]


def reference_area(
    edges: numpy.ndarray,
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: float,
    iy0: float,
    width: int,
    height: int,
    unchecked: frozenset[int] = frozenset(),
) -> tuple[numpy.ndarray, int, list[int]] | None:
    """area_coverage_plane in Python, line by line: each row's sums of its edges' pieces, or
    the outline's pieces alone in the rows mark_rows marks, supersampled when such a row has
    more than EXACT_EDGES edges; and the rows in unchecked, which the kernel's sweep ran out
    of budget before (the reference has none), supersampled. Also each row's kind, as the
    kernel numbers them."""
    if not any(row[1] != row[3] for row in edges.tolist()):
        return None
    table = edge_table(edges, crop_x0, crop_y1, scale, ix0, iy0, height)
    plane = numpy.zeros((height, width))
    exact = 0
    kinds = [0] * height
    flags = mark_rows(table, height)
    for row in range(height):
        indices = [i for i, edge in enumerate(table) if edge[5] <= row < edge[6]]
        active = [table[i] for i in indices]
        if not active:
            continue
        if row in unchecked or (flags[row] and len(active) > EXACT_EDGES):
            plane[row] = sampled_row(active, row, width)
            kinds[row] = 2 if row in unchecked else 3
            continue
        if flags[row]:
            plane[row] = exact_row(active, row, width)
            exact += 1
            kinds[row] = 1
            continue
        cells = [0.0] * (width + 2)
        rights: list[tuple[int, float]] = []
        for edge in active:
            add_piece(
                edge,
                py_max(float(row), edge[1]),
                py_min(row + 1.0, edge[2]),
                edge[4],
                width,
                cells,
                rights,
            )
        plane[row] = finish_row(cells, rights, width)
    return plane, exact, kinds


def span_coverage(
    loops: list[list[Point]], crop_y1: float, width: int, height: int, rows: int = ROWS
) -> numpy.ndarray:
    """Nonzero coverage of the loops (page space, scale 1, crop x 0) measured along rows lines
    of each pixel, each span's length exact: independent of the kernel."""
    device = [[(x, crop_y1 - y) for x, y in loop] for loop in loops]
    segments = [
        (x0, y0, x1, y1)
        for loop in device
        for (x0, y0), (x1, y1) in zip(loop, loop[1:] + loop[:1], strict=True)
        if y0 != y1
    ]
    plane = numpy.zeros((height, width))
    lefts = numpy.arange(width, dtype=numpy.float64)
    for line in range(height * rows):
        y = (line + 0.5) / rows
        crossings = sorted(
            (x0 + (y - y0) / (y1 - y0) * (x1 - x0), 1 if y1 > y0 else -1)
            for x0, y0, x1, y1 in segments
            if min(y0, y1) <= y < max(y0, y1)
        )
        winding = 0
        for (x, direction), (following, _) in itertools.pairwise(crossings):
            winding += direction
            if winding:
                plane[line // rows] += (
                    numpy.clip(numpy.minimum(following, lefts + 1) - numpy.maximum(x, lefts), 0, 1)
                    / rows
                )
    return numpy.minimum(plane, 1.0)


def rectangle(x0: float, y0: float, x1: float, y1: float, *, reverse: bool = False) -> list[Point]:
    loop = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return loop[::-1] if reverse else loop


# Paths whose windings meet inside a pixel, with a pixel to check: its row and column, and
# its coverage.
MIXED_CASES: list[tuple[str, list[list[Point]], int, int, float]] = [
    (
        "rectangles of opposite orientation stacked on a shared edge",
        [rectangle(10, 19.5, 20, 30), rectangle(10, 10, 20, 19.5, reverse=True)],
        20,
        15,
        1.0,
    ),
    (
        "slivers of opposite orientation in one pixel",
        [rectangle(10.1, 10, 10.4, 30), rectangle(10.6, 10, 10.9, 30, reverse=True)],
        20,
        10,
        0.6,
    ),
    (
        "a sliver drawn twice",
        [rectangle(10.2, 10, 10.5, 30), rectangle(10.2, 10, 10.5, 30)],
        20,
        10,
        0.3,
    ),
]


def loop_edges(loops: list[list[Point]]) -> numpy.ndarray:
    return numpy.array(
        [(*a, *b) for loop in loops for a, b in zip(loop, loop[1:] + loop[:1], strict=True)],
        numpy.float64,
    )


def check_area_kernel(count: int = 300) -> int:
    """area_coverage_plane, summed densely and row by row, against reference_area for
    equality on random paths: random corners, which cross themselves, and pairs of polygons
    of opposite orientation that share an edge. Then against span_coverage, which measures
    the path independently, within SPAN_TOLERANCE: on those paths and on MIXED_CASES, whose
    named pixel must have its coverage."""
    rng = numpy.random.default_rng(20261119)
    failed = exact_paths = 0
    for case in range(count):
        size = int(rng.integers(3, 48))
        if case % 3 == 0:
            cut = float(rng.uniform(0.2, 0.8)) * size
            left = [(0.3, 0.2), (cut, 0.4), (cut, size - 0.7), (0.6, size - 0.3)]
            right = [(cut, 0.4), (cut, size - 0.7), (size - 0.2, size - 0.6), (size - 0.4, 0.1)]
            loops = [left, right]
        else:
            points = rng.normal(size / 2, size / 3, (int(rng.integers(3, 16)), 2))
            if case % 4 == 1:
                points = numpy.round(points * 2) / 2
            loops = [[(float(x), float(y)) for x, y in points]]
        edges = loop_edges(loops)
        crop_y1 = float(size)
        want = reference_area(edges, 0.0, crop_y1, 1.0, 0.0, 0.0, size, size)
        if want is not None and want[1]:
            exact_paths += 1
        for mode in (1, 2):
            got = area_coverage_plane(edges, 0.0, crop_y1, 1.0, 0.0, 0.0, size, size, mode)
            same = (got is None and want is None) or (
                got is not None
                and want is not None
                and got[1] == want[1]
                and numpy.array_equal(got[0], want[0])
            )
            if not same:
                failed += 1
                if failed <= 3:
                    print(f"  area_coverage_plane differs on random path {case}, {mode=}")
        if want is not None:
            spans = span_coverage(loops, crop_y1, size, size)
            off = numpy.abs(want[0] - spans) > SPAN_TOLERANCE
            if off.any():
                failed += 1
                if failed <= 3:
                    row, column = numpy.argwhere(off)[0]
                    print(
                        f"  random path {case}: {want[0][row, column]:.4f} at {column},{row}, "
                        f"measured {spans[row, column]:.4f}"
                    )
    for name, loops, row, column, expected in MIXED_CASES:
        edges = loop_edges(loops)
        spans = span_coverage(loops, 40.0, 40, 40)
        for mode in (1, 2):
            got = area_coverage_plane(edges, 0.0, 40.0, 1.0, 0.0, 0.0, 40, 40, mode)
            assert got is not None
            off = numpy.abs(got[0] - spans) > SPAN_TOLERANCE
            if abs(got[0][row, column] - expected) > SPAN_TOLERANCE or off.any():
                failed += 1
                print(
                    f"  {name}, {mode=}: {got[0][row, column]:.4f} at {column},{row}, not "
                    f"{expected}; {int(off.sum())} pixels off the measured coverage"
                )
    print(
        f"{count} random paths ({exact_paths} with rows measured exactly) and "
        f"{len(MIXED_CASES)} mixed windings, summed densely and row by row: "
        f"area_coverage_plane fails on {failed}"
    )
    return failed + (exact_paths == 0)


# Glyph outlines whose windings mix somewhere, in font units: a stem with a reversed dot just
# above it, and a composite whose top half is flipped and meets the bottom on an edge. Placed
# rotated, their regions share pixels, so they must not be taken as simple.
MIXED_GLYPHS: list[tuple[str, tuple[tuple[Point, ...], ...]]] = [
    (
        "a stem with a reversed dot",
        (
            ((0.0, 0.0), (10.0, 0.0), (10.0, 30.0), (0.0, 30.0)),
            ((0.0, 30.4), (0.0, 40.0), (10.0, 40.0), (10.0, 30.4)),
        ),
    ),
    (
        "a composite with its top half flipped",
        (
            ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)),
            ((0.0, 20.0), (0.0, 40.0), (20.0, 40.0), (20.0, 20.0)),
        ),
    ),
]
MIXED_GLYPHS.append(
    (
        "two contours whose sides cross twice with no turn in x",
        (
            ((0.0, 0.0), (10.0, 5.0), (11.0, 10.0), (-20.0, 10.0), (-20.0, 0.0)),
            ((5.0, 0.0), (40.0, 0.0), (40.0, 10.0), (16.0, 10.0), (6.0, 5.0)),
        ),
    )
)
# A plain glyph, an O: an outer contour and a reversed hole, which is simple.
SIMPLE_GLYPH: tuple[tuple[Point, ...], ...] = (
    ((0.0, 0.0), (20.0, 0.0), (20.0, 40.0), (0.0, 40.0)),
    ((5.0, 5.0), (5.0, 35.0), (15.0, 35.0), (15.0, 5.0)),
)


def check_glyph_outlines() -> int:
    """A glyph outline counts as simple only when its windings are 0 or one sign of 1
    everywhere; placed rotated, a mixed one's coverage must match span_coverage."""
    failed = 0
    simple_arrays = outline_arrays(SIMPLE_GLYPH)
    if simple_arrays is None or not simple_arrays.simple:
        print("  an O is not taken as simple")
        failed += 1
    angle = math.radians(33.0)
    a, b = math.cos(angle) * 1.37, math.sin(angle) * 1.37
    transform = (a, b, -b, a, 30.3, 12.6)
    size = 80
    for name, contours in MIXED_GLYPHS:
        arrays = outline_arrays(contours)
        assert arrays is not None
        if arrays.simple:
            print(f"  {name} is taken as simple")
            failed += 1
        # Placed as rendering places glyphs: SimpleEdges when the outline counts as simple.
        edges = glyph_outline_edges(arrays.xs, arrays.ys, *transform, arrays.spans, arrays.simple)[
            2
        ]
        assert edges is not None
        target = numpy.zeros((size, size, 4), numpy.uint8)
        fill_glyph_coverage_at(
            edges, 0.0, float(size), 1.0, 0, 0, size, size, (0, 0, 0, 255), target, None, None, 1.0
        )
        loops = [
            [(a * x - b * y + transform[4], b * x + a * y + transform[5]) for x, y in contour]
            for contour in contours
        ]
        expected = span_coverage(loops, float(size), size, size) * 255
        bad = numpy.abs(target[..., 3] - expected) > SPAN_TOLERANCE * 255 + 1
        if bad.any():
            row, column = numpy.argwhere(bad)[0]
            print(
                f"  {name}, placed rotated: {int(bad.sum())} pixels off the measured coverage, "
                f"the first at {column},{row}: {target[row, column, 3]} != "
                f"{expected[row, column]:.1f}"
            )
            failed += int(bad.sum())
    # Columns of different lengths are refused, as glyph_outline_edges refuses them.
    try:
        outline_simple(numpy.zeros(4), numpy.zeros(3), ((0, 3),))
    except ValueError:
        pass
    else:
        print("  outline_simple took columns of different lengths")
        failed += 1
    print(f"{len(MIXED_GLYPHS) + 1} glyph outlines: {failed} failures")
    return failed


def check_sampled(name: str, loops: list[list[Point]], width: int, height: int, kind: int) -> int:
    """Whether area_coverage_plane supersamples at least a third of the rows of the loops
    with kind (2 past the budget, 3 for many edges), and gives reference_area's coverage and
    kinds exactly, taking the rows it left unchecked from the kernel, and the span measure's
    within SAMPLED_TOLERANCE."""
    failed = 0
    edges = loop_edges(loops)
    spans = span_coverage(loops, float(height), width, height, rows=16)
    for mode in (1, 2):
        found = area_coverage_plane(edges, 0.0, float(height), 1.0, 0.0, 0.0, width, height, mode)
        assert found is not None
        kinds = found[2].tolist()
        unchecked = frozenset(row for row, value in enumerate(kinds) if value == 2)
        want = reference_area(edges, 0.0, float(height), 1.0, 0.0, 0.0, width, height, unchecked)
        assert want is not None
        error = numpy.abs(found[0] - spans)
        if kinds.count(kind) < height // 3:
            print(f"  {name}, {mode=}: {kinds.count(kind)} rows of kind {kind}, too few")
            failed += 1
        elif kinds != want[2] or not numpy.array_equal(found[0], want[0]):
            print(f"  {name}, {mode=}: not the reference's coverage")
            failed += 1
        elif error.max() > SAMPLED_TOLERANCE:
            print(f"  {name}, {mode=}: off the span measure by up to {error.max():.4f}")
            failed += 1
    return failed


def check_engine_edges() -> int:
    """Cases the review of the row engine raised: a tall path whose few edges once spent the
    sweep's budget in its first band, edges whose x overflows to infinity among more than 16
    active ones, a path past the budget, whose rows left unchecked are supersampled, rows of
    over 256 edges whose windings meet, supersampled too, an edge
    of infinite slope, and simple outlines filled as SimpleEdges and as plain rows, which
    must give the same bytes."""
    failed = 0
    # 20 edges 3000 rows tall: a tall frame, and in its lower half two rectangles of opposite
    # orientation sharing an edge at x = 60.4, whose seam column must be covered.
    height, width = 3000, 120
    loops = [
        rectangle(2.0, 0.5, 10.0, 2999.5),
        rectangle(110.0, 0.5, 118.0, 2999.5),
        rectangle(30.2, 200.0, 60.4, 1200.0),
        rectangle(60.4, 200.0, 90.7, 1200.0, reverse=True),
        [(20.0, 10.0), (25.0, 2990.0), (24.0, 2990.0), (19.0, 10.0)],
    ]
    edges = loop_edges(loops)
    if not windings_mix(edges):
        print("  a tall path with opposite windings in its lower half is taken as simple")
        failed += 1
    for mode in (1, 2):
        found = area_coverage_plane(edges, 0.0, float(height), 1.0, 0.0, 0.0, width, height, mode)
        assert found is not None
        seam = found[0][height - 1000, 60]
        if abs(seam - 1.0) > 1e-9:
            print(f"  a tall path's seam, {mode=}: {seam:.4f}, not 1.0")
            failed += 1
    # Edges whose x overflows after scaling, among 40 ordinary ones in the same rows: ten
    # rectangles drawn twice, so the rows are measured exactly and their edges sorted.
    ordinary = [rectangle(2.0 + 3 * k, 1.0, 3.5 + 3 * k, 40.0) for k in range(10)] * 2
    wild = [[(1e308, 5.0), (1.5e308, 30.0), (-1e308, 20.0)]]
    edges = loop_edges(ordinary + wild)
    want = reference_area(edges, 0.0, 41.0, 4.0, 0.0, 0.0, 160, 164)
    for mode in (1, 2):
        first = area_coverage_plane(edges, 0.0, 41.0, 4.0, 0.0, 0.0, 160, 164, mode)
        again = area_coverage_plane(edges, 0.0, 41.0, 4.0, 0.0, 0.0, 160, 164, mode)
        if (
            first is None
            or again is None
            or want is None
            or not numpy.array_equal(first[0], again[0])
            or not numpy.array_equal(first[0], want[0])
        ):
            print(f"  edges that overflow to infinity, {mode=}: not the reference's coverage")
            failed += 1
    # A path far past the sweep's budget: 240 staggered bars of alternating orientation, each
    # overlapping the next, whose unchecked rows are supersampled, by winding, 4x4.
    loops = [
        rectangle(1.0 + 1.0 * k, 1.0 + 0.2 * k, 2.6 + 1.0 * k, 150.0 + 0.2 * k, reverse=k % 2 == 1)
        for k in range(240)
    ]
    failed += check_sampled("staggered bars past the budget", loops, 245, 200, 2)
    # Rows with windings meeting and more than 256 active edges, which are supersampled
    # rather than measured exactly: 140 bars crossed by 20 near-horizontal hatches.
    loops = [rectangle(1.0 + 1.5 * k, 5.0, 2.0 + 1.5 * k, 195.0) for k in range(140)] + [
        [(0.5, 10.0 + 9 * j), (215.0, 16.0 + 9 * j), (215.0, 17.0 + 9 * j), (0.5, 11.0 + 9 * j)]
        for j in range(20)
    ]
    failed += check_sampled("hatched bars", loops, 216, 200, 3)
    # An outline with an edge whose x is finite but whose slope is not, taken as SimpleEdges:
    # the dense path must drop it as the engine's table does.
    arrays = outline_arrays(
        (
            ((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)),
            ((-1e308, 40.0), (1e308, 40.5), (20.0, 60.0)),
        )
    )
    assert arrays is not None
    placed = []
    for simple in (True, False):
        found_edges = glyph_outline_edges(
            arrays.xs, arrays.ys, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, arrays.spans, simple
        )[2]
        assert found_edges is not None
        target = numpy.full((70, 70, 4), 200, numpy.uint8)
        fill_glyph_coverage_at(
            found_edges, 0.0, 70.0, 1.0, 0, 0, 70, 70, (20, 40, 60, 200), target, None, None, 1.0
        )
        placed.append(target)
    if not numpy.array_equal(placed[0], placed[1]):
        print("  an edge with an infinite slope: SimpleEdges and plain rows differ")
        failed += 1
    # Simple outlines filled from the device edges (SimpleEdges) and from the edge table.
    rng = numpy.random.default_rng(20261121)
    for case in range(200):
        count = int(rng.integers(5, 30))
        # Angles spread around the centre, so the hole, scaled about it, stays inside.
        angles = (numpy.arange(count) + rng.uniform(0, 0.8, count)) * (2 * math.pi / count)
        radius = float(rng.uniform(2, 30))
        cx, cy = rng.uniform(radius, 64 - radius, 2)
        outer = [(cx + radius * math.cos(t), cy + radius * math.sin(t)) for t in angles]
        inner = [(cx + radius * 0.4 * math.cos(t), cy + radius * 0.4 * math.sin(t)) for t in angles]
        arrays = outline_arrays((tuple(outer), tuple(inner[::-1])))
        assert arrays is not None
        shift = rng.normal(0, 1, 2)
        placed = []
        for simple in (True, False):
            edges = glyph_outline_edges(
                arrays.xs, arrays.ys, 1.0, 0.0, 0.0, 1.0, *shift, arrays.spans, simple
            )[2]
            assert edges is not None
            target = numpy.full((70, 70, 4), 200, numpy.uint8)
            fill_glyph_coverage_at(
                edges, 0.0, 70.0, 1.0, 0, 0, 70, 70, (20, 40, 60, 200), target, None, None, 1.0
            )
            placed.append(target)
        if not arrays.simple or not numpy.array_equal(placed[0], placed[1]):
            print(f"  simple outline {case}: the two paths differ (simple={arrays.simple})")
            failed += 1
    print(f"6 engine cases and 200 outlines on both paths: {failed} failures")
    return failed


def memory_page(content: bytes) -> bytes:
    """A US Letter page of content."""
    return document(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
            stream(b"", content),
        ]
    )


def diagonals(count: int, paint: bytes) -> bytes:
    """count slivers 0.6 wide that cross the page diagonally, each its own fill."""
    out = []
    for k in range(count):
        x = -300 + k * 1212 / count
        corners = numbers((x, 0, x + 0.6, 0, x + 300.6, 792, x + 300, 792)).split()
        out.append(b"%s %s m %s %s l %s %s l %s %s l h %s" % (*corners, paint))
    return b"\n".join(out)


def page_fill(paint: bytes) -> bytes:
    """A nonagon that covers most of the page."""
    corners = [
        (
            306 + 330 * math.cos(0.3 + 2 * math.pi * k / 9),
            396 + 420 * math.sin(0.3 + 2 * math.pi * k / 9),
        )
        for k in range(9)
    ]
    path = b"%s m " % numbers(corners[0]) + b" ".join(b"%s l" % numbers(c) for c in corners[1:])
    return b"0.2 0.4 0.8 rg %s h %s" % (path, paint)


def traced_peak(pdf: bytes, scale: float) -> int:
    """The most memory Python's allocators held at once while rasterizing pdf at scale."""
    with PdfDocument(pdf) as opened:
        rendered = opened.pages[0].render()
        tracemalloc.start()
        try:
            rendered.rasterize(scale=scale)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()


def check_memory() -> int:
    """A fill's memory must not follow its pixel box: the peak while rasterizing a page of
    large fills may exceed an empty page's by at most MEMORY_ALLOWANCE."""
    failed = 0
    for name, content, scale in (
        ("a fill over the page at scale 8", page_fill(b"f"), 8.0),
        ("an even-odd fill over the page at scale 8", page_fill(b"f*"), 8.0),
        ("200 diagonal slivers at scale 4", diagonals(200, b"f"), 4.0),
        ("200 even-odd diagonal slivers at scale 4", diagonals(200, b"f*"), 4.0),
    ):
        extra = traced_peak(memory_page(content), scale) - traced_peak(memory_page(b""), scale)
        if extra > MEMORY_ALLOWANCE:
            print(
                f"  {name}: {extra / 2**20:.1f} MB beyond an empty page's peak, more than "
                f"{MEMORY_ALLOWANCE / 2**20:.0f} MB"
            )
            failed += 1
    print(f"4 pages of large fills: {failed} over the memory allowance")
    return failed


def main(argv: Sequence[str]) -> int:
    pdfs = Path(argv[argv.index("--pdfs") + 1]) if "--pdfs" in argv else None
    failures = (
        check_kernel()
        + check_area_kernel()
        + check_glyph_outlines()
        + check_engine_edges()
        + check_memory()
    )
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

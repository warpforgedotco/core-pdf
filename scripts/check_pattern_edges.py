# SPDX-License-Identifier: AGPL-3.0-only

"""Check that a path filled with a pattern has the antialiased edges of a plain fill.

A fill whose colour is a pattern paints the pattern through the path (ISO 32000-2 8.7.3.1,
8.7.4.1), with the same coverage a fill in a plain colour has. This writes pages that fill a
skewed pentagon, and a pentagon with a triangular hole under the even-odd rule, with a
shading pattern and with a coloured tiling pattern, each of a single colour, and checks:

- every pixel's alpha against the exact area of the pixel the path covers, worked out by
  clipping the path's polygon to the pixel in Python: within COVERAGE_TOLERANCE of it, and
  at least MIN_EDGE_PIXELS pixels must be partly covered;
- every pixel of the page against the same page with the pattern replaced by its colour,
  which core paints as a plain fill: within MATCH_TOLERANCE in each channel. This also runs
  under a hard clip from `W n` of a rotated triangle, at fill opacity 0.5, in the Multiply
  blend mode over a backdrop, and in a knockout group, where the pixels of the path's box
  the path does not cover must keep what an earlier member painted (11.4.6).

uv run --all-packages python scripts/check_pattern_edges.py
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pdf import PdfDocument  # noqa: E402
from scripts.check_patterns import document, numbers, stream  # noqa: E402

type Point = tuple[float, float]

SIZE = 200
# The pattern's single colour.
COLOR = (0.2, 0.4, 0.8)
# A pixel's alpha may differ this much from 255 times the area the path covers: core
# samples coverage on a grid, so its edges are within a sixteenth or so of the exact area.
COVERAGE_TOLERANCE = 24
# A pixel may differ this much in a channel from the same page painted in a plain colour.
MATCH_TOLERANCE = 2
# Each fill must have at least this many pixels its path covers in part.
MIN_EDGE_PIXELS = 100

# LARGE's pixel box is over 10000 pixels, where core once covered a pixel by its centre.
PENTAGON: list[Point] = [(68.1, 59.3), (131.7, 68.5), (136.8, 104.1), (100.0, 132.7), (62.3, 110.1)]
HOLE: list[Point] = [(86.2, 77.1), (117.5, 84.1), (97.5, 113.2)]
LARGE: list[Point] = [(30.3, 20.7), (171.6, 41.2), (182.9, 120.4), (101.15, 183.8), (17.45, 133.6)]
RECTANGLE: list[Point] = [(10.3, 12.6), (185.7, 12.6), (185.7, 190.2), (10.3, 190.2)]
CLIP: list[Point] = [(40.2, 20.5), (180.4, 60.3), (70.8, 120.9)]


def path_operators(polygons: Sequence[Sequence[Point]]) -> bytes:
    out = b""
    for polygon in polygons:
        (x, y), *rest = polygon
        out += b"%s m " % numbers((x, y))
        for point in rest:
            out += b"%s l " % numbers(point)
        out += b"h "
    return out


def shading_pattern() -> bytes:
    function = b"<< /FunctionType 2 /Domain [0 1] /C0 [%s] /C1 [%s] /N 1 >>" % (
        numbers(COLOR),
        numbers(COLOR),
    )
    return (
        b"<< /PatternType 2 /Shading << /ShadingType 2 /ColorSpace /DeviceRGB "
        b"/Coords [0 0 200 0] /Extend [true true] /Function %s >> >>" % function
    )


def tiling_pattern() -> bytes:
    return stream(
        b"/PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 10 10] /XStep 10 /YStep 10 "
        b"/Resources << >>",
        b"%s rg 0 0 10 10 re f" % numbers(COLOR),
    )


class Case:
    """A page that fills polygons (with the even-odd rule when evenodd) with a pattern, or
    with its colour when plain, inside `before` and `after` operators."""

    def __init__(
        self,
        name: str,
        polygons: list[list[Point]],
        *,
        tiling: bool = False,
        evenodd: bool = False,
        before: bytes = b"",
        after: bytes = b"",
        group: bytes | None = None,
        analytic: bool = True,
    ) -> None:
        self.name = name
        self.polygons = polygons
        self.tiling = tiling
        self.evenodd = evenodd
        self.before = before
        self.after = after
        self.group = group
        self.analytic = analytic

    def pdf(self, *, plain: bool) -> bytes:
        colour = b"%s rg" % numbers(COLOR) if plain else b"/Pattern cs /P scn"
        paint = b"f*" if self.evenodd else b"f"
        body = b"%s %s %s%s %s" % (
            self.before,
            colour,
            path_operators(self.polygons),
            paint,
            self.after,
        )
        pattern = tiling_pattern() if self.tiling else shading_pattern()
        resources = (
            b"<< /Pattern << /P 6 0 R >> /ExtGState << /G0 7 0 R >> /XObject << /F 8 0 R >> >>"
        )
        if self.group is None:
            content = body
            form = stream(b"/Type /XObject /Subtype /Form /BBox [0 0 1 1]", b"")
        else:
            content = b"1 1 0 rg 0 0 100 200 re f /F Do"
            form = stream(
                b"/Type /XObject /Subtype /Form /BBox [0 0 %d %d] /Resources %s /Group %s"
                % (SIZE, SIZE, resources, self.group),
                body,
            )
        return document(
            [
                b"<< /Type /Catalog /Pages 2 0 R >>",
                b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Resources %s "
                b"/Contents 4 0 R >>" % (SIZE, SIZE, resources),
                stream(b"", content),
                b"<< >>",
                pattern,
                b"<< /Type /ExtGState /ca 0.5 /BM /Multiply >>",
                form,
            ]
        )


def render(pdf: bytes) -> numpy.ndarray:
    with PdfDocument(pdf) as document_:
        image = document_.pages[0].render().rasterize()
    return numpy.frombuffer(bytes(image.pixels), numpy.uint8).reshape(image.height, image.width, 4)


def clipped_area(polygon: Sequence[Point], x0: float, y0: float) -> float:
    """The area of polygon (in device space) inside the unit pixel at (x0, y0)."""
    points = list(polygon)
    for axis, bound, keep_above in (
        (0, x0, True),
        (0, x0 + 1, False),
        (1, y0, True),
        (1, y0 + 1, False),
    ):
        if not points:
            return 0.0
        clipped: list[Point] = []
        for index, current in enumerate(points):
            previous = points[index - 1]
            current_in = current[axis] >= bound if keep_above else current[axis] <= bound
            previous_in = previous[axis] >= bound if keep_above else previous[axis] <= bound
            if current_in != previous_in:
                t = (bound - previous[axis]) / (current[axis] - previous[axis])
                clipped.append(
                    (
                        previous[0] + t * (current[0] - previous[0]),
                        previous[1] + t * (current[1] - previous[1]),
                    )
                )
            if current_in:
                clipped.append(current)
        points = clipped
    area = 0.0
    for index, (x, y) in enumerate(points):
        px, py = points[index - 1]
        area += px * y - x * py
    return abs(area) / 2


def coverage(polygons: list[list[Point]]) -> numpy.ndarray:
    """The exact coverage of each pixel by the outer polygon minus the holes inside it."""
    plane = numpy.zeros((SIZE, SIZE))
    for index, polygon in enumerate(polygons):
        device = [(x, SIZE - y) for x, y in polygon]
        xs = [x for x, _ in device]
        ys = [y for _, y in device]
        sign = 1.0 if index == 0 else -1.0
        for row in range(int(min(ys)), int(max(ys)) + 1):
            for column in range(int(min(xs)), int(max(xs)) + 1):
                plane[row, column] += sign * clipped_area(device, column, row)
    return plane


def cases() -> list[Case]:
    clip = path_operators([CLIP]) + b"W n"
    found: list[Case] = []
    for tiling in (False, True):
        kind = "tiling" if tiling else "shading"
        found += [
            Case(f"{kind}, pentagon", [PENTAGON], tiling=tiling),
            Case(f"{kind}, even-odd ring", [PENTAGON, HOLE], tiling=tiling, evenodd=True),
            Case(f"{kind}, rectangle", [RECTANGLE], tiling=tiling),
            Case(f"{kind}, large pentagon", [LARGE], tiling=tiling),
            Case(
                f"{kind}, clipped by W n",
                [PENTAGON],
                tiling=tiling,
                before=b"q " + clip,
                after=b"Q",
                analytic=False,
            ),
            Case(
                f"{kind}, large, clipped by W n",
                [LARGE],
                tiling=tiling,
                before=b"q " + clip,
                after=b"Q",
                analytic=False,
            ),
            Case(
                f"{kind}, opacity 0.5 and Multiply",
                [PENTAGON],
                tiling=tiling,
                before=b"1 1 0 rg 0 0 100 200 re f /G0 gs",
                analytic=False,
            ),
            Case(
                f"{kind}, knockout group",
                [PENTAGON, HOLE],
                tiling=tiling,
                evenodd=True,
                before=b"1 0 0 rg 0 0 200 90 re f",
                group=b"<< /S /Transparency /K true >>",
                analytic=False,
            ),
        ]
    return found


def check(case: Case) -> int:
    pattern = render(case.pdf(plain=False)).astype(numpy.int32)
    plain = render(case.pdf(plain=True)).astype(numpy.int32)
    failed = 0
    difference = numpy.abs(pattern - plain).max(axis=2)
    mismatched = int((difference > MATCH_TOLERANCE).sum())
    if mismatched:
        row, column = numpy.argwhere(difference > MATCH_TOLERANCE)[0]
        print(
            f"  {case.name}: {mismatched} pixels differ from the plain fill by more than "
            f"{MATCH_TOLERANCE}, the first at {column},{row}: "
            f"{tuple(pattern[row, column])} != {tuple(plain[row, column])}"
        )
        failed += mismatched
    if case.analytic:
        expected = coverage(case.polygons) * 255
        partial = (expected > 0.5) & (expected < 254.5)
        edges = int(partial.sum())
        if edges < MIN_EDGE_PIXELS:
            print(f"  {case.name}: only {edges} edge pixels")
            failed += 1
        for name, pixels in (("pattern", pattern), ("plain", plain)):
            error = numpy.abs(pixels[..., 3] - expected)
            bad = int((error > COVERAGE_TOLERANCE).sum())
            if bad:
                row, column = numpy.argwhere(error > COVERAGE_TOLERANCE)[0]
                print(
                    f"  {case.name}: {bad} {name} pixels' alpha is off the exact coverage by more "
                    f"than {COVERAGE_TOLERANCE}, the first at {column},{row}: "
                    f"{pixels[row, column, 3]} != {expected[row, column]:.1f}"
                )
                failed += bad
    return failed


def main() -> int:
    failures = 0
    selected = cases()
    for case in selected:
        failures += check(case)
    print(f"{len(selected)} pattern fill pages: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

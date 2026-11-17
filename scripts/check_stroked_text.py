# SPDX-License-Identifier: AGPL-3.0-only

"""Check that stroked text is stroked as a path is, with the graphics state's parameters.

Stroking a glyph has the same effect as stroking a path (ISO 32000-2 9.3.6): its outline is
stroked with the line width, line cap, line join, miter limit and dash pattern of the
graphics state, and those parameters are interpreted in user space, not text space. The pen
is a circle in user space: a CTM that skews or scales unevenly makes it an ellipse on the
page (8.4.3.2), while the text matrix and horizontal scaling, which shape the glyphs, leave
it a circle.

This writes PDFs that show Helvetica-Bold in render modes 1, 2, 5 and 6 with every line
join, miter limits above and below the glyphs' corners, dashes with each cap, line widths
from 0 to thick, at a constant alpha and in Multiply, under CTMs that stretch, skew and
rotate, and under text matrices and horizontal scaling that stretch and skew the glyphs.
Each page is rendered with core, and so is a twin page that draws each glyph's outline, as
core placed it on the page, as a path under the same CTM and graphics state, and paints it
with S, or B for the modes that fill too. Every pixel of the two must match within a byte.

uv run --all-packages python scripts/check_stroked_text.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pdf import PdfDocument  # noqa: E402
from core_pdf.impl.render.model import PathPaintItem  # noqa: E402
from scripts.check_strokes import SIZE, Affine, document, numbers, rotation  # noqa: E402

type Pixels = numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]]
type Point = tuple[float, float]

FONT = b"/Font << /F << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> >>"
IDENTITY: Affine = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
MITER, ROUND, BEVEL = 0, 1, 2
BUTT, ROUND_CAP, SQUARE = 0, 1, 2
# Glyphs with acute, right and obtuse corners and curves; the letter spacing keeps their
# strokes apart.
TEXT = b"(VAMWS) Tj"


class Case:
    """A page showing TEXT in render mode `mode`, at `size`, from `origin` in text space,
    under `ctm`, the text matrix `tm` and horizontal scaling `scaling`, stroked with
    `width`, `join`, `cap`, `limit` and `dash`, at the constant alpha `alpha` and in the
    blend mode `blend`."""

    def __init__(
        self,
        name: str,
        *,
        mode: int = 1,
        width: float = 4.0,
        join: int = MITER,
        cap: int = BUTT,
        limit: float = 10.0,
        dash: tuple[list[float], float] | None = None,
        ctm: Affine = IDENTITY,
        tm: Affine = IDENTITY,
        scaling: float = 100.0,
        size: float = 64.0,
        origin: Point = (14.137, 120.291),
        alpha: float = 1.0,
        blend: str | None = None,
    ) -> None:
        self.name = name
        self.mode = mode
        self.width = width
        self.join = join
        self.cap = cap
        self.limit = limit
        self.dash = dash
        self.ctm = ctm
        self.tm = tm
        self.scaling = scaling
        self.size = size
        self.origin = origin
        self.alpha = alpha
        self.blend = blend

    def state(self) -> bytes:
        dash = b"[] 0 d"
        if self.dash is not None:
            dash = b"[%s] %s d" % (numbers(self.dash[0]), numbers([self.dash[1]]))
        paint = []
        if self.alpha != 1.0:
            paint.append(b"/CA %s /ca %s" % (numbers([self.alpha]), numbers([self.alpha])))
        if self.blend is not None:
            paint.append(b"/BM /%s" % self.blend.encode())
        out = b"0.1 0.3 0.7 RG 0.9 0.6 0.2 rg %s w %d J %d j %s M %s " % (
            numbers([self.width]),
            self.cap,
            self.join,
            numbers([self.limit]),
            dash,
        )
        return out + (b"/G gs " if paint else b"")

    def resources(self) -> bytes:
        paint = []
        if self.alpha != 1.0:
            paint.append(b"/CA %s /ca %s" % (numbers([self.alpha]), numbers([self.alpha])))
        if self.blend is not None:
            paint.append(b"/BM /%s" % self.blend.encode())
        return b"<< %s /ExtGState << /G << %s >> >> >>" % (FONT, b" ".join(paint))

    def pdf(self) -> bytes:
        body = b"q %s cm %sBT /F %s Tf 14 Tc %s Tz %s Tm %s Td %d Tr %s ET Q" % (
            numbers(self.ctm),
            self.state(),
            numbers([self.size]),
            numbers([self.scaling]),
            numbers(self.tm),
            numbers(self.origin),
            self.mode,
            TEXT,
        )
        return document(body, self.resources())

    def twin(self, outlines: list[list[tuple[list[Point], bool]]]) -> bytes:
        """The page drawing each glyph's outline, given on the page, as a path in user
        space under the same CTM and graphics state."""
        a, b, c, d, e, f = self.ctm
        determinant = a * d - b * c
        paint = b"B" if self.mode in {2, 6} else b"S"
        body = b"q %s cm %s" % (numbers(self.ctm), self.state())
        for glyph in outlines:
            for points, closed in glyph:
                user = [
                    (
                        (d * (x - e) - c * (y - f)) / determinant,
                        (a * (y - f) - b * (x - e)) / determinant,
                    )
                    for x, y in points
                ]
                body += b"%s m " % exact(user[0])
                body += b"".join(b"%s l " % exact(point) for point in user[1:])
                if closed:
                    body += b"h "
            body += paint + b" "
        return document(body + b"Q", self.resources())


def exact(point: Point) -> bytes:
    """A point as the shortest decimals that read back as the same floats."""
    return b"%s %s" % (repr(point[0]).encode(), repr(point[1]).encode())


def render(data: bytes) -> tuple[Pixels, list[list[tuple[list[Point], bool]]]]:
    """The page's pixels, and the outline of each glyph it strokes, on the page."""
    with PdfDocument(data) as pdf:
        page = pdf.pages[0].render()
        outlines = [
            [
                ([(float(x), float(y)) for x, y in subpath.points], bool(subpath.closed))
                for subpath in item.path.subpaths
            ]
            for item in page.display_list.items
            if type(item) is PathPaintItem
        ]
        image = page.rasterize()
    pixels = numpy.frombuffer(bytes(image.pixels), numpy.uint8).reshape(
        image.height, image.width, 4
    )
    return pixels, outlines


# Away from whole and quarter pixels, where a sample on a stroke's edge could fall either side
# of it with the last bit of the twin's arithmetic.
STRETCH: Affine = (1.6, 0.0, 0.0, 0.55, 0.13, 60.37)
SKEW: Affine = (1.0, 0.0, 0.5, 1.0, -40.21, 20.43)


def stretched_rotation(degrees: float, x_scale: float, y_scale: float) -> Affine:
    a, b, c, d, _, _ = rotation(degrees)
    return (a * x_scale, b * x_scale, c * y_scale, d * y_scale, 40.29, 10.11)


def cases() -> list[Case]:
    found = [
        Case(f"join {join}, miter limit {limit}", join=join, limit=limit)
        for join in (MITER, ROUND, BEVEL)
        for limit in (10.0, 1.2)
    ]
    found += [
        Case("miter limit 2.2", limit=2.2, width=6.0),
        Case("dashes, butt caps", dash=([9.0, 5.0], 3.0)),
        Case("dashes, round caps", dash=([7.0, 6.0], 0.0), cap=ROUND_CAP, join=ROUND),
        Case("dashes, square caps", dash=([6.0, 7.0], 1.0), cap=SQUARE, join=BEVEL),
        Case("zero-length dashes, round caps", dash=([0.0, 7.0], 0.0), cap=ROUND_CAP),
        Case("thick", width=9.0, limit=3.0),
        Case("width 0.12", width=0.12),
        Case("width 0.4", width=0.4),
        Case("width 0", width=0.0),
        Case("alpha 0.5", alpha=0.5, width=6.0, join=BEVEL),
        Case("Multiply", blend="Multiply", width=6.0, join=ROUND),
        Case("fill, then stroke", mode=2, width=5.0, limit=1.5),
        Case("stroke and clip", mode=5, width=5.0, join=BEVEL),
        Case("fill, stroke and clip", mode=6, width=5.0, join=ROUND),
    ]
    # The pen is in user space: an ellipse on the page under a CTM that scales unevenly
    # or skews, with the joins, caps, miter limit and dashes worked out in user space.
    found += [
        Case(f"{name}, join {join}", ctm=ctm, join=join, limit=2.0, width=5.0)
        for name, ctm in (
            ("stretched", STRETCH),
            ("skewed", SKEW),
            ("stretched and rotated", stretched_rotation(20.0, 1.4, 0.7)),
        )
        for join in (MITER, ROUND, BEVEL)
    ]
    found += [
        Case("stretched, dashes", ctm=STRETCH, dash=([8.0, 5.0], 2.0), cap=SQUARE),
        Case("skewed, alpha 0.5", ctm=SKEW, alpha=0.5, width=6.0),
        Case("stretched, width 0.12", ctm=STRETCH, width=0.12),
        Case("skewed, fill, then stroke", ctm=SKEW, mode=2, width=4.0),
    ]
    # The text matrix and horizontal scaling shape the glyphs, not the pen.
    found += [
        Case("horizontal scaling 50", scaling=50.0, width=5.0, limit=2.0),
        Case(
            "horizontal scaling 160", scaling=160.0, size=40.0, width=5.0, origin=(4.137, 120.291)
        ),
        Case("skewed text matrix", tm=(1.0, 0.0, 0.45, 1.0, 0.0, 0.0), width=5.0, limit=1.5),
        Case("stretched text matrix", tm=(1.0, 0.0, 0.0, 1.7, 0.0, 0.0), size=48.0, width=5.0),
        Case(
            "rotated text matrix, horizontal scaling 70, stretched CTM",
            tm=(*rotation(-15.0)[:4], 0.0, 0.0),
            scaling=70.0,
            ctm=STRETCH,
            width=5.0,
            dash=([10.0, 4.0], 0.0),
        ),
    ]
    return found


def check(case: Case) -> int:
    pixels, outlines = render(case.pdf())
    if not outlines:
        print(f"  {case.name}: no glyph outlines")
        return 1
    twin, _ = render(case.twin(outlines))
    difference = numpy.abs(pixels.astype(int) - twin.astype(int)).max(axis=2)
    wrong = numpy.argwhere(difference > 1)
    for row, column in wrong[:3]:
        print(
            f"  {case.name}: pixel {column},{row}: {tuple(int(v) for v in pixels[row, column])},"
            f" not {tuple(int(v) for v in twin[row, column])} as the path strokes it"
        )
    return len(wrong)


def main() -> int:
    selected = cases()
    failed_pages = failures = 0
    for case in selected:
        failed = check(case)
        failures += failed
        if failed:
            failed_pages += 1
            print(f"{case.name}: {failed} of {SIZE * SIZE} pixels differ")
    print(f"{len(selected)} stroked text pages: {failures} failures on {failed_pages} pages")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

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
with S, or B for the modes that fill too. Every pixel of the two must be the same, or,
where the text is filled too, within a byte: the text object's glyphs are composited in a
group, and the path's fill and stroke straight onto the page, which can round a pixel that
both partly cover, where a dash leaves the fill's edge bare, a byte apart. The twin maps the
outlines back to user space with the CTM as the page writes it, rounded to six decimals:
inverting the unrounded CTM would move each point by up to 1e-4 of a pixel, enough to move a
sample across a stroke's edge. The rest of the twin's arithmetic moves a point by an ulp or
so, which must not move a sample: a seeded sweep of rotating, unevenly scaling CTMs, with
each line width, join, cap, miter limit and dashes, checks that it does not. The sweep's
pages are allowed a byte, as the filled pages are: a sample of an opaque stroke is 16 levels,
so the byte hides no sample, while it spares the sweep's hundred-odd random pages any last-
bit difference between one platform's maths library and another's that rounds a blended
pixel the other way. The fixed pages, stroked only, must match exactly.
A thin stroke is floored at one device pixel across the pen's narrower axis, as a path's
is (8.4.3.2, 10.7.4): text stroked at a line width of 0 must also match, pixel for pixel,
the same text stroked at a tiny width, upright and under a rotating, stretching CTM.

uv run --all-packages python scripts/check_stroked_text.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pdf import PdfDocument  # noqa: E402
from core_pdf.impl.render.model import PathPaintItem  # noqa: E402
from scripts.check_strokes import (  # noqa: E402
    SIZE,
    Affine,
    apply,
    document,
    numbers,
    rotation,
    stretched_rotation,
)

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
        exact: bool = True,
    ) -> None:
        self.name = name
        self.mode = mode
        self.width = width
        self.join = join
        self.cap = cap
        self.limit = limit
        self.dash = dash
        # The CTM as the page writes it, so that the twin maps its outlines back to user
        # space with the matrix the page is drawn under.
        self.ctm = written(ctm)
        self.tm = tm
        self.scaling = scaling
        self.size = size
        self.origin = origin
        self.alpha = alpha
        self.blend = blend
        # Whether the twin must match byte for byte where the text is only stroked.
        self.exact = exact

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
            matrix(self.ctm),
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
        body = b"q %s cm %s" % (matrix(self.ctm), self.state())
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


def written(ctm: Affine) -> Affine:
    """ctm with each entry rounded to the decimals matrix writes."""
    a, b, c, d, e, f = (float(f"{value:.6f}") for value in ctm)
    return a, b, c, d, e, f


def matrix(ctm: Affine) -> bytes:
    """ctm's entries with six decimals, never an exponent, which a PDF number cannot have."""
    return " ".join(f"{value:.6f}" for value in ctm).encode()


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


def turned(degrees: float, x_scale: float, y_scale: float) -> Affine:
    """stretched_rotation, moved away from whole and quarter pixels."""
    return stretched_rotation(degrees, x_scale, y_scale, 40.29, 10.11)


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
        Case("width 0.05", width=0.05),
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
            ("stretched and rotated", turned(20.0, 1.4, 0.7)),
            ("stretched and rotated about the centre", stretched_rotation(20.0, 1.4, 0.7)),
        )
        for join in (MITER, ROUND, BEVEL)
    ]
    found += [
        Case("stretched, dashes", ctm=STRETCH, dash=([8.0, 5.0], 2.0), cap=SQUARE),
        Case("skewed, alpha 0.5", ctm=SKEW, alpha=0.5, width=6.0),
        Case("stretched, width 0.12", ctm=STRETCH, width=0.12),
        Case("stretched, width 0", ctm=STRETCH, width=0.0),
        Case("stretched and rotated, width 0.05", ctm=turned(47.0, 1.2, 0.75), width=0.05),
        Case("skewed, fill, then stroke", ctm=SKEW, mode=2, width=4.0),
        # A dash running on through a corner, where the position along the segment before
        # it, plus the rest of that segment, rounds an ulp short of its length.
        Case(
            "rotated, dashes through a corner, miter joins",
            ctm=(-0.693409, -0.720766, 0.683607, -0.65766, 117.245828, 307.255709),
            dash=([7.07256, 4.83665], 0.331093),
            cap=ROUND_CAP,
            width=5.0,
        ),
        # Dashes ending in segments along the pen's axes shorter than half its width.
        Case(
            "stretched and rotated, short dashes, square caps",
            ctm=(0.466183, 0.228833, -0.573215, 1.167767, 189.253987, -33.174055),
            dash=([1.19024, 6.10734], 3.53978),
            cap=SQUARE,
            join=BEVEL,
            limit=1.2,
            width=5.0,
        ),
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
    return found + sweep()


# Seeded, so that a failure is found again.
SWEEP_SEED = 457
SWEEP_PAGES = 120


def sweep() -> list[Case]:
    """Pages under CTMs that rotate by any angle and scale each axis by its own factor, with
    the text centred on the page, a quarter of them dashed and a quarter filled too."""
    rng = random.Random(SWEEP_SEED)
    found = []
    for _ in range(SWEEP_PAGES):
        degrees = rng.uniform(0.0, 360.0)
        x_scale, y_scale = rng.uniform(0.5, 1.6), rng.uniform(0.5, 1.6)
        x, y = apply(stretched_rotation(degrees, x_scale, y_scale, 0.0, 0.0), (90.0, 140.0))
        ctm = stretched_rotation(
            degrees, x_scale, y_scale, SIZE / 2.0 - x + rng.random(), SIZE / 2.0 - y + rng.random()
        )
        width = rng.choice((0.0, 0.05, 0.3, 1.0, 2.5, 5.0, 8.0))
        join = rng.choice((MITER, ROUND, BEVEL))
        cap = rng.choice((BUTT, ROUND_CAP, SQUARE))
        limit = rng.choice((1.2, 2.0, 10.0))
        dash = ([rng.uniform(0.0, 9.0), rng.uniform(2.0, 8.0)], rng.uniform(0.0, 5.0))
        dashed = rng.random() < 0.25
        mode = 2 if rng.random() < 0.25 else 1
        found.append(
            Case(
                f"sweep: rotated {degrees:.2f}, scaled {x_scale:.3f} by {y_scale:.3f},"
                f" width {width}, join {join}, cap {cap}, miter limit {limit}"
                + (", dashed" if dashed else "")
                + (", filled" if mode == 2 else ""),
                mode=mode,
                width=width,
                join=join,
                cap=cap,
                limit=limit,
                dash=dash if dashed else None,
                ctm=ctm,
                exact=False,
            )
        )
    return found


def check(case: Case) -> int:
    pixels, outlines = render(case.pdf())
    if not outlines:
        print(f"  {case.name}: no glyph outlines")
        return 1
    twin, _ = render(case.twin(outlines))
    difference = numpy.abs(pixels.astype(int) - twin.astype(int)).max(axis=2)
    wrong = numpy.argwhere(difference > (0 if case.exact and case.mode in {1, 5} else 1))
    for row, column in wrong[:3]:
        print(
            f"  {case.name}: pixel {column},{row}: {tuple(int(v) for v in pixels[row, column])},"
            f" not {tuple(int(v) for v in twin[row, column])} as the path strokes it"
        )
    return len(wrong)


def check_zero_width() -> int:
    """Text stroked at a line width of 0 paints the pixels it does at a tiny width: both are
    floored at one device pixel. Returns the pages that differ."""
    failures = 0
    for name, ctm in (("upright", IDENTITY), ("stretched and rotated", turned(20.0, 1.4, 0.7))):
        zero, _ = render(Case(name, width=0.0, ctm=ctm).pdf())
        tiny, _ = render(Case(name, width=1e-4, ctm=ctm).pdf())
        if not numpy.array_equal(zero, tiny):
            failures += 1
            print(f"  {name}: text stroked at width 0 and width 0.0001 differs")
    return failures


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
    zero_failures = check_zero_width()
    print(f"width 0 against a tiny width: {zero_failures} failures")
    return 1 if failures or zero_failures else 0


if __name__ == "__main__":
    sys.exit(main())

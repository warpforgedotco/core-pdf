# SPDX-License-Identifier: AGPL-3.0-only

"""Check text painted with tiling and shading patterns against a reference of pattern space.

Text whose fill or stroke colour is a pattern shows the pattern through its glyphs, in the
pattern's own space: its Matrix on the default space of the content stream that names it
(ISO 32000-2 8.7.2, 8.7.3.1, 9.3.6). This writes PDFs that show Helvetica-Bold and a Type 3
font in render modes 0, 1, 2 and 4 to 6, with coloured and uncoloured tiling patterns and an
axial shading pattern, under text matrices and CTMs that scale, rotate and skew, in forms
with a Matrix, and inside a tiling pattern's cell. Each page is also rendered with the
pattern replaced by opaque black, which gives the glyphs' coverage: a pixel whose
neighbourhood that rendering covers whole is inside the glyphs, one whose neighbourhood it
leaves empty is outside, and the rest are edges and skipped. Inside, the pixel must show the
pattern's colour at its centre, worked out back through the transforms in Python (tiling
colours exactly, more than a pixel and a half from a cell's edge; shading colours within 2
of 255); outside, nothing. In modes 4 to 6 a green fill of the right half of the page after
ET must show only inside the glyphs' outlines. A Type 3 glyph that sets its own colour after
d0 keeps it (9.6.4), one after d1 takes the text's pattern. Text is a knockout group
(9.3.8), so a stroked glyph, whose pattern is painted over its padded box, must leave the
neighbours that box overlaps as they were.

uv run --all-packages python scripts/check_pattern_text.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pdf import PdfDocument  # noqa: E402
from scripts.check_patterns import (  # noqa: E402
    AXIS,
    C0,
    C1,
    IDENTITY,
    RED,
    SIZE,
    Affine,
    Tiling,
    apply,
    document,
    invert,
    multiply,
    numbers,
    rotation,
    scale_of,
    shading_color,
    stream,
)

type Color = tuple[float, ...] | None

GREEN = (0.0, 255.0, 0.0)
# Pixels this many device pixels from a glyph's edge, or from where the clip fill starts,
# are skipped: antialiasing blends colours there.
EDGE = 2
# Each page must check at least this many pixels of the text's paint.
MIN_PAINTED = 300
# The x at which the fill after ET starts, in modes 4 to 6.
SPLIT = 150.0
# The axial shading the shading patterns paint.
SHADING = b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [%s] /Extend [true true] " % numbers(
    AXIS
) + b"/Function << /FunctionType 2 /Domain [0 1] /C0 [%s] /C1 [%s] /N 1 >> >>" % (
    numbers(C0),
    numbers(C1),
)
# A Type 3 font of three glyphs, 1000 units wide in 1000-unit glyph space: a after d1 fills
# a triangle and tries to set red, which d1 ignores; b after d0 fills a ring in red; c after
# d0 fills a square without setting a colour.
TYPE3_GLYPHS = {
    b"a": b"1000 0 0 0 900 900 d1 1 0 0 rg 50 50 m 850 50 l 450 850 l f",
    b"b": b"1000 0 d0 1 0 0 rg 50 50 800 800 re 250 250 400 400 re f*",
    b"c": b"1000 0 d0 50 50 800 800 re f",
}


class Pattern:
    """A pattern resource, and the colour it paints at a point of its own space with the
    distance of that point, in pattern units, to where the colour changes."""

    def __init__(
        self,
        body: bytes,
        color: Callable[[float, float], tuple[Color, float]],
        *,
        uncoloured: tuple[int, int, int] | None = None,
        tolerance: int = 0,
        inner: bytes = b"<< >>",
        matrix: Affine = IDENTITY,
    ) -> None:
        self.body = body
        self.color = color
        self.uncoloured = uncoloured
        self.tolerance = tolerance
        self.inner = inner
        self.matrix = matrix

    def select(self, *, stroke: bool) -> bytes:
        space, colour = (b"CS", b"SCN") if stroke else (b"cs", b"scn")
        if self.uncoloured is None:
            return b"/Pattern %s /P %s" % (space, colour)
        components = numbers([value / 255 for value in self.uncoloured])
        return b"/PCS %s %s /P %s" % (space, components, colour)


def tiling_pattern(tiling: Tiling) -> Pattern:
    def color(x: float, y: float) -> tuple[Color, float]:
        found = tiling.color(x, y)
        return (None if found is None else tuple(map(float, found))), tiling.distance_to_edge(x, y)

    return Pattern(tiling.stream(), color, uncoloured=tiling.uncoloured, matrix=tiling.matrix)


def shading_pattern(matrix: Affine = IDENTITY) -> Pattern:
    def color(x: float, y: float) -> tuple[Color, float]:
        return shading_color(x, y), float("inf")

    body = b"<< /Type /Pattern /PatternType 2 /Matrix [%s] /Shading 8 0 R >>" % numbers(matrix)
    return Pattern(body, color, tolerance=2, matrix=matrix)


class Case:
    """A page whose content places a form with `placement`; the form, with Matrix `form`,
    shows text with `text`, painted with `pattern` in render mode `mode`. reference_space
    maps a page point to the pattern's space; own_color, when given, is the colour a Type 3
    glyph paints there itself, or None where the text's paint shows."""

    def __init__(
        self,
        name: str,
        pattern: Pattern,
        mode: int,
        text: bytes,
        *,
        placement: Affine = IDENTITY,
        form: Affine = IDENTITY,
        type3: bool = False,
        own_color: Callable[[float, float], Color] | None = None,
    ) -> None:
        self.name = name
        self.pattern = pattern
        self.mode = mode
        self.text = text
        self.placement = placement
        self.form = form
        self.type3 = type3
        self.own_color = own_color

    def paint(self, *, black: bool) -> bytes:
        fills = self.mode in {0, 2, 4, 6}
        strokes = self.mode in {1, 2, 5, 6}
        if black:
            return b"%s %s" % (b"0 g" if fills else b"", b"0 G" if strokes else b"")
        return b" ".join(
            [
                *([self.pattern.select(stroke=False)] if fills else []),
                *([self.pattern.select(stroke=True)] if strokes else []),
            ]
        )

    def content(self, mode: int, *, black: bool, clip_fill: bool) -> bytes:
        body = b"%s 10 w %d Tr %s ET" % (self.paint(black=black), mode, self.text)
        body = b"BT /F 1 Tf " + body
        if clip_fill:
            # The clip cases place the form with the identity: the form's space is the page's.
            body += b" 0 1 0 rg %s 0 %d %d re f" % (numbers([SPLIT]), SIZE, SIZE)
        return body

    def pdf(self, mode: int, *, black: bool, clip_fill: bool) -> bytes:
        font = (
            b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] "
            b"/FontMatrix [0.001 0 0 0.001 0 0] /CharProcs << /a 9 0 R /b 10 0 R /c 11 0 R >> "
            b"/Encoding << /Differences [97 /a /b /c] >> /FirstChar 97 /LastChar 99 "
            b"/Widths [1000 1000 1000] /Resources << >> >>"
            if self.type3
            else b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"
        )
        resources = (
            b"<< /Pattern << /P 6 0 R >> /Font << /F 7 0 R >> "
            b"/ColorSpace << /PCS [/Pattern /DeviceRGB] >> /XObject << /Fm 5 0 R >> >>"
        )
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Contents 4 0 R /Resources "
            % (SIZE, SIZE)
            + resources
            + b" >>",
            stream(b"", b"q %s cm /Fm Do Q" % numbers(self.placement)),
            stream(
                b"/Type /XObject /Subtype /Form /BBox [-5000 -5000 5000 5000] /Matrix [%s] "
                % numbers(self.form)
                + b"/Resources "
                + resources,
                self.content(mode, black=black, clip_fill=clip_fill),
            ),
            self.pattern.body,
            font,
            SHADING,
            *(stream(b"", TYPE3_GLYPHS[name]) for name in (b"a", b"b", b"c")),
            self.pattern.inner,
        ]
        return document(objects)

    def reference(self, x: float, y: float) -> tuple[Color, float]:
        """The colour the pattern paints at page point (x, y), and the distance in device
        pixels to where it changes."""
        space = multiply(self.pattern.matrix, multiply(self.form, self.placement))
        px, py = apply(invert(space), x, y)
        color, distance = self.pattern.color(px, py)
        return color, distance * scale_of(space)


def render(pdf: bytes) -> numpy.ndarray:
    with PdfDocument(pdf) as document_:
        image = document_.pages[0].render().rasterize()
    return (
        numpy.frombuffer(bytes(image.pixels), numpy.uint8)
        .reshape(image.height, image.width, 4)
        .astype(numpy.int32)
    )


def neighbourhood(alpha: numpy.ndarray, *, full: bool) -> numpy.ndarray:
    """Where every pixel within EDGE has alpha 255 (full) or 0."""
    hit = alpha == 255 if full else alpha == 0
    padded = numpy.pad(hit, EDGE, constant_values=not full)
    result = numpy.ones_like(hit)
    height, width = hit.shape
    for dy in range(2 * EDGE + 1):
        for dx in range(2 * EDGE + 1):
            result &= padded[dy : dy + height, dx : dx + width]
    return result


def check(case: Case) -> tuple[int, int, int]:
    """The pixels checked, those that differ, and those checked that show the text's paint."""
    clips = case.mode >= 4
    paint_mask = render(case.pdf(case.mode, black=True, clip_fill=False))[..., 3]
    inside = neighbourhood(paint_mask, full=True)
    outside = neighbourhood(paint_mask, full=False)
    if clips:
        # The clip is the glyphs' outlines, whatever the mode paints.
        clip_mask = render(case.pdf(4, black=True, clip_fill=False))[..., 3]
        inside_clip = neighbourhood(clip_mask, full=True)
        outside_clip = neighbourhood(clip_mask, full=False)
    pixels = render(case.pdf(case.mode, black=False, clip_fill=clips))
    tolerance = case.pattern.tolerance
    checked = failed = painted = 0
    for row in range(pixels.shape[0]):
        y = SIZE - (row + 0.5)
        for column in range(pixels.shape[1]):
            x = column + 0.5
            expected: Color
            if clips and abs(x - SPLIT) <= EDGE:
                continue
            if clips and x > SPLIT and not outside_clip[row, column]:
                if not inside_clip[row, column]:
                    continue
                expected = GREEN
            elif inside[row, column]:
                own = case.own_color(x, y) if case.own_color is not None else None
                if own is not None:
                    expected = own
                else:
                    expected, distance = case.reference(x, y)
                    if distance < 1.5:
                        continue
                    painted += expected is not None
            elif outside[row, column]:
                expected = None
            else:
                continue
            checked += 1
            pixel = pixels[row, column]
            if expected is None:
                ok = pixel[3] == 0
            else:
                ok = pixel[3] == 255 and all(
                    abs(int(pixel[index]) - round(expected[index])) <= tolerance
                    for index in range(3)
                )
            if not ok:
                if failed < 3:
                    print(f"  {case.name}: pixel {column},{row}: {tuple(pixel)} != {expected}")
                failed += 1
    return checked, failed, painted


def text_case(
    name: str,
    pattern: Pattern,
    mode: int,
    *,
    matrix: Affine = (110, 0, 0, 110, 12, 100),
    placement: Affine = IDENTITY,
    form: Affine = IDENTITY,
) -> Case:
    """Show HO, 110 units high, with the text matrix `matrix`."""
    return Case(
        name, pattern, mode, b"%s Tm (HO) Tj" % numbers(matrix), placement=placement, form=form
    )


def type3_case(name: str, pattern: Pattern, *, placement: Affine = IDENTITY) -> Case:
    """Show the Type 3 glyphs abc, 90 units wide, from (15, 100) of the form: b keeps its red."""
    size = 90.0
    origin = (15.0, 100.0)
    to_form = invert(placement)

    def own_color(x: float, y: float) -> Color:
        fx, fy = apply(to_form, x, y)
        # Glyph b fills from origin + size * (1.05, 0.05) to origin + size * (1.85, 0.85).
        gx = (fx - origin[0]) / size - 1.0
        gy = (fy - origin[1]) / size
        if 0.0 <= gx <= 1.0 and 0.0 <= gy <= 1.0:
            return tuple(map(float, RED))
        return None

    matrix = (size, 0.0, 0.0, size, *origin)
    return Case(
        name,
        pattern,
        0,
        b"%s Tm (abc) Tj" % numbers(matrix),
        placement=placement,
        type3=True,
        own_color=own_color,
    )


def cell_case(name: str, *, placement: Affine = IDENTITY) -> Case:
    """Fill the page with a tiling pattern whose 100-wide cell shows H, 80 high, painted with
    a shading pattern: each cell shows the shading in the cell's space."""
    shading_matrix = (100.0 / AXIS[2], 0.0, 0.0, 100.0 / AXIS[2], 0.0, 0.0)
    cell = stream(
        b"/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 100 100] "
        b"/XStep 110 /YStep 110 /Resources << /Pattern << /S 12 0 R >> "
        b"/Font << /G << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> >> >>",
        b"BT /G 80 Tf /Pattern cs /S scn 10 15 Td (H) Tj ET",
    )
    inner = shading_pattern(shading_matrix)
    to_form = invert(placement)
    scale = scale_of(placement)

    def color(x: float, y: float) -> tuple[Color, float]:
        u = x - (x // 110.0) * 110.0
        v = y - (y // 110.0) * 110.0
        if u >= 100.0 or v >= 100.0:
            return None, 0.0
        return shading_color(u * AXIS[2] / 100.0, v), min(u, v, 100 - u, 100 - v)

    pattern = Pattern(cell, color, tolerance=3, inner=inner.body)

    class CellCase(Case):
        def content(self, mode: int, *, black: bool, clip_fill: bool) -> bytes:  # noqa: ARG002
            return b"/Pattern cs /P scn 0 0 %d %d re f" % (SIZE, SIZE)

        def pdf(self, mode: int, *, black: bool, clip_fill: bool) -> bytes:
            pdf = super().pdf(mode, black=black, clip_fill=clip_fill)
            return pdf.replace(b"/Pattern cs /S scn", b"0 g") if black else pdf

        def reference(self, x: float, y: float) -> tuple[Color, float]:
            px, py = apply(to_form, x, y)
            found, distance = color(px, py)
            return found, distance * scale

    return CellCase(name, pattern, 0, b"", placement=placement)


def cases() -> list[Case]:
    shading = shading_pattern()
    shifted = shading_pattern((1.0, 0.0, 0.0, 1.0, 40.0, 0.0))
    tiling = tiling_pattern(Tiling())
    skewed = tiling_pattern(Tiling((1.5, 0.3, 0.2, 1.2, 5, 7)))
    uncoloured = tiling_pattern(Tiling(uncoloured=(204, 25, 127)))
    rotated = rotation(20, 1.0, 60, 10)
    return [
        text_case("shading fill", shading, 0),
        text_case("tiling fill", tiling, 0),
        text_case("uncoloured tiling fill", uncoloured, 0),
        text_case(
            "skewed tiling fill, rotated text", skewed, 0, matrix=rotation(-12, 100, 30, 120)
        ),
        text_case("shading stroke", shading, 1),
        text_case("tiling stroke", tiling, 1),
        text_case("shading fill and stroke", shifted, 2),
        text_case("tiling fill and stroke", tiling, 2),
        text_case("shading fill and clip", shading, 4),
        text_case("tiling stroke and clip", tiling, 5),
        text_case("uncoloured fill, stroke and clip", uncoloured, 6),
        text_case("shading fill, scaled CTM", shading, 0, placement=(1.2, 0, 0, 0.8, -5, 20)),
        text_case("tiling fill, rotated form", tiling, 0, placement=rotated),
        text_case(
            "tiling fill, form Matrix",
            skewed,
            0,
            placement=(1.1, 0, 0, 1.1, 10, 0),
            form=(1.0, 0.0, 0.2, 1.0, -20.0, 0.0),
        ),
        text_case("shading stroke, rotated form", shifted, 1, placement=rotated),
        type3_case("Type 3, shading", shading),
        type3_case("Type 3, tiling, scaled", tiling, placement=(1.1, 0, 0, 1.1, 0, -20)),
        cell_case("text in a tiling cell"),
        cell_case("text in a tiling cell, scaled", placement=(1.2, 0, 0, 1.2, 5, 5)),
    ]


def main() -> int:
    total = failures = 0
    selected = cases()
    for case in selected:
        checked, failed, painted = check(case)
        total += checked
        if painted < MIN_PAINTED:
            print(f"{case.name}: only {painted} pixels inside the glyphs show the pattern")
            failed += 1
        failures += failed
        if failed:
            print(f"{case.name}: {failed} of {checked} pixels differ")
    print(f"{len(selected)} pattern text pages, {total} pixels: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

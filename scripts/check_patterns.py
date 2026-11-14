# SPDX-License-Identifier: AGPL-3.0-only

"""Check rendered tiling and shading patterns against a Python reference of pattern space.

A pattern's space is its Matrix on the default space of the content stream that names it:
the page, a form XObject as the Do places it, or the cell of an outer pattern (ISO 32000-2
8.7.2). Tiling cells sit at multiples of XStep and YStep there, clipped by BBox there
(8.7.3.1), and a stroke painted with a pattern shows the pattern through the stroke
(8.7.1). This writes PDFs that fill and stroke with tiling and shading patterns on the page,
in forms that scale, rotate and skew, inside other patterns' cells and with uncoloured
cells, renders each with core, and works out in Python what each pixel centre shows: back
through the transforms to pattern space, into the cell that holds it, and the colour the
cell paints there. Pixels within a device pixel and a half of an edge, where coverage blends
colours, are skipped; every other pixel must match, tiling colours exactly and shading
colours within 2 of 255 (3 for a shading inside a cell).

uv run --all-packages python scripts/check_patterns.py
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence

import numpy

from core_pdf import PdfDocument

type Affine = tuple[float, float, float, float, float, float]
type Color = tuple[int, int, int] | None

IDENTITY: Affine = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
SIZE = 300
RED = (255, 0, 0)
BLUE = (0, 0, 255)
# Half the side of the square around a pixel centre whose corners must agree.
MARGIN = 1.5
# The axial shading the shading patterns use: C0 to C1 along Coords in shading space.
AXIS = (0.0, 0.0, 240.0, 0.0)
C0 = (1.0, 0.8, 0.0)
C1 = (0.0, 0.2, 1.0)


def multiply(left: Sequence[float], right: Sequence[float]) -> Affine:
    """left then right, as PDF concatenates matrices."""
    a, b, c, d, e, f = left
    g, h, i, j, k, m = right
    return (
        a * g + b * i,
        a * h + b * j,
        c * g + d * i,
        c * h + d * j,
        e * g + f * i + k,
        e * h + f * j + m,
    )


def invert(matrix: Sequence[float]) -> Affine:
    a, b, c, d, e, f = matrix
    det = a * d - b * c
    return (d / det, -b / det, -c / det, a / det, (c * f - d * e) / det, (b * e - a * f) / det)


def apply(matrix: Sequence[float], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def scale_of(matrix: Sequence[float]) -> float:
    """The smallest factor the matrix scales a length by."""
    a, b, c, d, _, _ = matrix
    return float(numpy.linalg.svd(numpy.array(((a, b), (c, d))), compute_uv=False)[-1])


def numbers(values: Sequence[float]) -> bytes:
    return " ".join(f"{value:g}" for value in values).encode()


class Tiling:
    """A tiling pattern whose cell paints a red square at (0, 0) and a blue one at (8, 8),
    each 8 wide, in a 16-wide BBox; an uncoloured one paints both in its scn colour."""

    def __init__(
        self,
        matrix: Affine = IDENTITY,
        steps: tuple[float, float] = (20.0, 20.0),
        *,
        uncoloured: Color = None,
        tiling_type: int = 1,
    ) -> None:
        self.matrix = matrix
        self.steps = steps
        self.uncoloured = uncoloured
        self.tiling_type = tiling_type

    def stream(self) -> bytes:
        if self.uncoloured is not None:
            body = b"0 0 8 8 re f 8 8 8 8 re f"
        else:
            body = b"1 0 0 rg 0 0 8 8 re f 0 0 1 rg 8 8 8 8 re f"
        x_step, y_step = self.steps
        head = b"/Type /Pattern /PatternType 1 /PaintType %d /TilingType %d /BBox [0 0 16 16] " % (
            2 if self.uncoloured is not None else 1,
            self.tiling_type,
        ) + b"/XStep %s /YStep %s /Matrix [%s] /Resources << >>" % (
            numbers([x_step]),
            numbers([y_step]),
            numbers(self.matrix),
        )
        return stream(head, body)

    def color(self, x: float, y: float) -> Color:
        """The colour at (x, y) in pattern space, or None where no cell paints."""
        x_step, y_step = (abs(step) for step in self.steps)
        u = x - math.floor(x / x_step) * x_step
        v = y - math.floor(y / y_step) * y_step
        if u < 8.0 and v < 8.0:
            return self.uncoloured or RED
        if 8.0 <= u < 16.0 and 8.0 <= v < 16.0:
            return self.uncoloured or BLUE
        return None

    def distance_to_edge(self, x: float, y: float) -> float:
        x_step, y_step = (abs(step) for step in self.steps)
        u = x - math.floor(x / x_step) * x_step
        v = y - math.floor(y / y_step) * y_step
        return min(
            min(abs(u - edge) for edge in (0.0, 8.0, 16.0, x_step)),
            min(abs(v - edge) for edge in (0.0, 8.0, 16.0, y_step)),
        )


def shading_color(x: float, y: float) -> tuple[float, ...]:
    """The axial shading at (x, y) in shading space, extended at both ends."""
    x0, y0, x1, y1 = AXIS
    dx, dy = x1 - x0, y1 - y0
    t = ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)
    t = min(1.0, max(0.0, t))
    return tuple(255.0 * (a + t * (b - a)) for a, b in zip(C0, C1, strict=True))


def stream(dictionary: bytes, data: bytes) -> bytes:
    return b"<< " + dictionary + b" /Length %d >>\nstream\n" % len(data) + data + b"\nendstream"


def document(objects: list[bytes]) -> bytes:
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
    """A page that places a form with `placement`; the form, with Matrix `form`, runs
    `form_body`, which paints with pattern P (object 6; `extra` objects follow it). reference
    gives a point of the page the colour it shows, or None, and its distance to an edge."""

    def __init__(
        self,
        name: str,
        reference: Callable[[float, float], tuple[Color | tuple[float, ...], float]],
        pattern: bytes,
        form_body: bytes,
        *,
        placement: Affine = IDENTITY,
        form: Affine = IDENTITY,
        extra: tuple[bytes, ...] = (),
        tolerance: int = 0,
    ) -> None:
        self.name = name
        self.reference = reference
        self.pattern = pattern
        self.form_body = form_body
        self.placement = placement
        self.form = form
        self.extra = extra
        self.tolerance = tolerance

    def pdf(self) -> bytes:
        resources = (
            b"<< /Pattern << /P 6 0 R >> /ColorSpace << /PCS [/Pattern /DeviceRGB] >> "
            b"/XObject << /Fm 5 0 R >> >>"
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
                self.form_body,
            ),
            self.pattern,
            *self.extra,
        ]
        return document(objects)


def fill_case(
    name: str,
    tiling: Tiling,
    *,
    placement: Affine = IDENTITY,
    form: Affine = IDENTITY,
    side: float = 200.0,
) -> Case:
    """Fill a square of the form with the tiling pattern."""
    space = multiply(form, placement)
    to_form = invert(space)
    # Pattern space maps to the form's default space, which the Do placed on the page.
    pattern_space = multiply(tiling.matrix, space)
    to_pattern = invert(pattern_space)
    form_scale = scale_of(space)
    pattern_scale = scale_of(pattern_space)
    paint = b"/Pattern cs /P scn"
    if tiling.uncoloured is not None:
        paint = b"/PCS cs %s /P scn" % numbers([value / 255 for value in tiling.uncoloured])

    def reference(x: float, y: float) -> tuple[Color, float]:
        fx, fy = apply(to_form, x, y)
        inside = 0.0 <= fx <= side and 0.0 <= fy <= side
        edge = min(abs(fx), abs(fy), abs(fx - side), abs(fy - side)) * form_scale
        if not inside:
            return None, edge
        px, py = apply(to_pattern, x, y)
        return tiling.color(px, py), min(edge, tiling.distance_to_edge(px, py) * pattern_scale)

    body = paint + b" 0 0 %s %s re f" % (numbers([side]), numbers([side]))
    return Case(name, reference, tiling.stream(), body, placement=placement, form=form)


def shading_stroke_case(
    name: str, *, placement: Affine = IDENTITY, form: Affine = IDENTITY, matrix: Affine = IDENTITY
) -> Case:
    """Stroke a polyline of the form, 16 wide, with an axial shading pattern."""
    points = ((20.0, 20.0), (200.0, 110.0), (20.0, 200.0))
    width = 16.0
    space = multiply(form, placement)
    scale = scale_of(space)
    to_form = invert(space)
    to_shading = invert(multiply(matrix, space))

    def reference(x: float, y: float) -> tuple[tuple[float, ...] | None, float]:
        fx, fy = apply(to_form, x, y)
        distance = min(segment_distance(fx, fy, a, b) for a, b in zip(points, points[1:]))
        corner = min(math.hypot(fx - px, fy - py) for px, py in points)
        # Joins and caps are another check's business: only the straight runs count.
        edge = min(abs(distance - width / 2), max(0.0, corner - width)) * scale
        if distance > width / 2:
            return None, edge
        return shading_color(*apply(to_shading, x, y)), edge

    path = b"%s m %s l %s l" % tuple(numbers(point) for point in points)
    body = b"/Pattern CS /P SCN %s w " % numbers([width]) + path + b" S"
    pattern = b"<< /Type /Pattern /PatternType 2 /Matrix [%s] /Shading 7 0 R >>" % numbers(matrix)
    shading = (
        b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [%s] /Extend [true true] "
        % numbers(AXIS)
        + b"/Function << /FunctionType 2 /Domain [0 1] /C0 [%s] /C1 [%s] /N 1 >> >>"
        % (numbers(C0), numbers(C1))
    )
    return Case(
        name,
        reference,
        pattern,
        body,
        placement=placement,
        form=form,
        extra=(shading,),
        tolerance=2,
    )


def tiling_stroke_case(name: str, tiling: Tiling, *, placement: Affine = IDENTITY) -> Case:
    """Stroke a polyline, 24 wide, with the tiling pattern."""
    points = ((30.0, 30.0), (250.0, 140.0), (30.0, 250.0))
    width = 24.0
    scale = scale_of(placement)
    to_form = invert(placement)
    to_pattern = invert(multiply(tiling.matrix, placement))

    def reference(x: float, y: float) -> tuple[Color, float]:
        fx, fy = apply(to_form, x, y)
        distance = min(segment_distance(fx, fy, a, b) for a, b in zip(points, points[1:]))
        corner = min(math.hypot(fx - px, fy - py) for px, py in points)
        edge = min(abs(distance - width / 2), max(0.0, corner - width)) * scale
        if distance > width / 2:
            return None, edge
        px, py = apply(to_pattern, x, y)
        return tiling.color(px, py), min(edge, tiling.distance_to_edge(px, py) * scale)

    path = b"%s m %s l %s l" % tuple(numbers(point) for point in points)
    body = b"/Pattern CS /P SCN %s w " % numbers([width]) + path + b" S"
    return Case(name, reference, tiling.stream(), body, placement=placement)


def nested_shading_case(name: str, *, placement: Affine = IDENTITY) -> Case:
    """Fill with a tiling pattern whose 20-wide cell fills (0, 0)-(16, 16) with a shading
    pattern whose axis runs 16 along the cell: each cell shows the whole gradient."""
    cell = stream(
        b"/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 16 16] "
        b"/XStep 20 /YStep 20 /Resources << /Pattern << /S 7 0 R >> >>",
        b"/Pattern cs /S scn 0 0 16 16 re f",
    )
    shading_pattern = b"<< /Type /Pattern /PatternType 2 /Matrix [%s] /Shading 8 0 R >>" % numbers(
        (16.0 / AXIS[2], 0.0, 0.0, 16.0 / AXIS[2], 0.0, 0.0)
    )
    shading = (
        b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [%s] /Extend [true true] "
        % numbers(AXIS)
        + b"/Function << /FunctionType 2 /Domain [0 1] /C0 [%s] /C1 [%s] /N 1 >> >>"
        % (numbers(C0), numbers(C1))
    )
    side = 200.0
    to_form = invert(placement)
    scale = scale_of(placement)

    def reference(x: float, y: float) -> tuple[tuple[float, ...] | None, float]:
        fx, fy = apply(to_form, x, y)
        edge = min(abs(fx), abs(fy), abs(fx - side), abs(fy - side)) * scale
        if not (0.0 <= fx <= side and 0.0 <= fy <= side):
            return None, edge
        u = fx - math.floor(fx / 20.0) * 20.0
        v = fy - math.floor(fy / 20.0) * 20.0
        edge = min(edge, min(abs(u), abs(u - 16), abs(u - 20), abs(v), abs(v - 16)) * scale)
        if u >= 16.0 or v >= 16.0:
            return None, edge
        return shading_color(u * AXIS[2] / 16.0, 0.0), edge

    body = b"/Pattern cs /P scn 0 0 %s %s re f" % (numbers([side]), numbers([side]))
    return Case(
        name,
        reference,
        cell,
        body,
        placement=placement,
        extra=(shading_pattern, shading),
        tolerance=3,
    )


def segment_distance(x: float, y: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(x - ax - t * dx, y - ay - t * dy)


def rotation(degrees: float, scale: float = 1.0, e: float = 0.0, f: float = 0.0) -> Affine:
    radians = math.radians(degrees)
    c, s = math.cos(radians) * scale, math.sin(radians) * scale
    return (c, s, -s, c, e, f)


def cases() -> list[Case]:
    return [
        fill_case("page, identity", Tiling()),
        fill_case("page, translated Matrix", Tiling((1, 0, 0, 1, 30, 40))),
        fill_case("page, scaled and skewed Matrix", Tiling((1.5, 0.3, 0.2, 1.2, 5, 7))),
        fill_case("page, negative steps", Tiling(steps=(-22.0, -18.0))),
        fill_case("page, gapped steps", Tiling(steps=(30.0, 24.0), tiling_type=2)),
        fill_case("scaled form", Tiling(), placement=(1.3, 0, 0, 0.7, 20, 30)),
        fill_case("rotated form", Tiling(), placement=rotation(30, 1.2, 150, 10)),
        fill_case(
            "rotated form with a form Matrix and a pattern Matrix",
            Tiling((1, 0.25, 0, 1, 3, 0), tiling_type=3),
            placement=(1.1, 0, 0, 1.1, 60, 20),
            form=rotation(-20, 1.0, 10, 60),
        ),
        fill_case(
            "uncoloured, rotated form",
            Tiling(uncoloured=(204, 25, 127)),
            placement=rotation(25, 1.1, 140, 0),
        ),
        shading_stroke_case("shading stroke, page"),
        shading_stroke_case(
            "shading stroke, rotated form",
            placement=rotation(15, 1.1, 60, 0),
            matrix=(1, 0, 0, 1, 20, 0),
        ),
        tiling_stroke_case("tiling stroke, page", Tiling()),
        tiling_stroke_case("tiling stroke, scaled", Tiling(), placement=(0.9, 0, 0, 0.9, 10, 10)),
        nested_shading_case("shading pattern in a tiling cell"),
        nested_shading_case(
            "shading pattern in a tiling cell, scaled", placement=(1.2, 0, 0, 1.2, 10, 5)
        ),
    ]


def check(case: Case) -> tuple[int, int]:
    with PdfDocument(case.pdf()) as pdf:
        image = pdf.pages[0].render().rasterize()
    pixels = numpy.frombuffer(bytes(image.pixels), numpy.uint8).reshape(
        image.height, image.width, 4
    )
    checked = failed = 0
    for row in range(image.height):
        for column in range(image.width):
            x = column + 0.5
            y = SIZE - (row + 0.5)
            expected, edge = case.reference(x, y)
            if edge < MARGIN:
                continue
            checked += 1
            pixel = pixels[row, column]
            if expected is None:
                ok = pixel[3] == 0
            else:
                ok = pixel[3] == 255 and all(
                    abs(int(pixel[index]) - round(expected[index])) <= case.tolerance
                    for index in range(3)
                )
            if not ok:
                if failed < 3:
                    print(f"  {case.name}: pixel {column},{row}: {tuple(pixel)} != {expected}")
                failed += 1
    return checked, failed


def main() -> int:
    total = failures = 0
    selected = cases()
    for case in selected:
        checked, failed = check(case)
        total += checked
        failures += failed
        if failed:
            print(f"{case.name}: {failed} of {checked} pixels differ")
    print(f"{len(selected)} pattern pages, {total} pixels: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

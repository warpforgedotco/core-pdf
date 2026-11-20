# SPDX-License-Identifier: AGPL-3.0-only

"""Check that a rectangle or a disc renders the same whichever route paints it.

A filled path is antialiased: a pixel's alpha is the share of it the path covers, and under
a blend mode or a clip path that share is counted on a 4 x 4 grid of samples, a sample on
an edge inside on the left and bottom (core's fill_path). Core also paints rectangles and
discs on their own, for the stroke of an axis-aligned segment, a square or round dot or
cap, and the cells of a glyph bitmap; each must cover what the same shape filled as one
path covers, whatever its blend mode and clip, so that pieces meeting inside a pixel cover
it once and leave no seam.

This renders, at scales 1 and 2.0833, pages that stroke horizontal and vertical segments
with butt caps at fractional coordinates, in Multiply over a backdrop, at a constant alpha
of 0.5 under a triangular clip, and opaque under it, and the same rectangles filled as
paths: each pair must be byte-identical. It strokes segments with projecting square caps,
opaque inside a transparency group under a triangular clip, which core paints piece by
piece, against the union of the pieces filled as one path; and, under the clip, strokes
whose edges fall on the samples at scale 1: segments 1.25 wide, dashes at alpha 0.5 and a
segment with square caps at alpha 0.5, each against its shape filled as one path. Then it
paints, on a raster target directly, square and round dots (fill_terminal), glyph bitmaps
(draw_glyph_bitmap) whose cells have fractional device edges, each glyph against its cells
filled as one path, and rectangles in a blend mode named Normal, against a rectangle
filled in none: in Multiply, in Multiply at alpha 0.5, at alpha 0.5, and at alpha 0.5,
opaque and in Screen under a triangular clip. The stroke kernel's round caps under the
clip, sampled discs, must be within three samples of the discs filled as polygons.

Rectangles fill_rect already painted as fill_path does, opaque in the Normal mode under no
clip or a rectangular one, and in any mode with every device edge on a pixel edge, must
paint the same pixels as before: their pages' digests are recorded.

uv run --all-packages python scripts/check_rect_routes.py
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pdf import PdfDocument  # noqa: E402
from core_pdf.impl.capture.records import CapturedPath, CapturedSubpath  # noqa: E402
from core_pdf.impl.render.clipping import ClipState, DeviceGrid  # noqa: E402
from core_pdf.impl.render.paths import circle_path, stroke_half_width  # noqa: E402
from core_pdf.impl.render.target import RasterTarget  # noqa: E402
from scripts.check_patterns import document, stream  # noqa: E402

SIZE = 200
SCALES = (1.0, 2.0833)
BACKDROP = b"1 0.9 0.2 rg 0 0 200 200 re f "
TRIANGLE_CLIP = b"10.2 15.7 m 190.4 30.3 l 60.9 185.1 l h W n "
# Horizontal and vertical segments, with the rectangles their strokes cover, line width
# 1.3: device edges inside pixels at both scales.
SEGMENTS = (
    (b"20.3 50.37 m 180.6 50.37 l", b"20.3 49.72 160.3 1.3 re"),
    (b"170.41 70.3 m 170.41 160.6 l", b"169.76 70.3 1.3 90.3 re"),
    (b"30.3 120.37 m 150.6 120.37 l", b"30.3 119.72 120.3 1.3 re"),
)
# Segments stroked with projecting caps 3.1 wide, opaque inside a transparency group, which
# core paints piece by piece: the rectangle around each segment, then the square around
# each end; and the union of those, one path.
CAPPED = (
    (
        b"50.3 60.6 m 90.7 60.6 l",
        b"50.3 59.05 40.4 3.1 re 48.75 59.05 3.1 3.1 re 89.15 59.05 3.1 3.1 re",
    ),
    (
        b"120.41 50.3 m 120.41 100.7 l",
        b"118.86 50.3 3.1 50.4 re 118.86 48.75 3.1 3.1 re 118.86 99.15 3.1 3.1 re",
    ),
)
# Strokes whose edges fall on the 4 x 4 samples at scale 1, each against its shape filled
# as one path: a sample on an edge is inside on the left and bottom, as a filled path's is.
DASHES = (
    b" ".join(b"%.6g 79.65 3 1.3 re" % (20.125 + 5 * index) for index in range(26))
    + b" "
    + b" ".join(b"59.65 %.6g 1.3 3 re" % (40.125 + 5 * index) for index in range(20))
)
ON_SAMPLES = (
    (
        "width 1.25",
        b"q " + TRIANGLE_CLIP,
        b"1.25 w 30 50 m 170 50 l S 100 40 m 100 150 l S",
        b"30 49.375 140 1.25 re f 99.375 40 1.25 110 re f",
    ),
    (
        "dashes at alpha 0.5",
        b"q " + TRIANGLE_CLIP + b"/A gs ",
        b"1.3 w [3 2] 0 d 20.125 80.3 m 150.125 80.3 l 60.3 40.125 m 60.3 140.125 l S",
        DASHES + b" f",
    ),
    (
        "square caps at alpha 0.5",
        b"q " + TRIANGLE_CLIP + b"/A gs ",
        b"2 J 1.25 w 40.5 140.25 m 100.25 140.25 l S",
        b"40.5 139.625 59.75 1.25 re 39.875 139.625 1.25 1.25 re 99.625 139.625 1.25 1.25 re f",
    ),
)

# The digests of pages fill_rect painted as fill_path before; see unchanged_pages.
UNCHANGED = {
    "opaque-normal-strokes@1.0": "75624eb6ef845073",
    "opaque-normal-strokes@2.0833": "38070778bcec1315",
    "opaque-normal-fills@1.0": "75624eb6ef845073",
    "opaque-normal-fills@2.0833": "38070778bcec1315",
    "aligned-multiply-strokes@1.0": "6bcd5e8a4133d905",
    "aligned-multiply-fills@1.0": "6bcd5e8a4133d905",
    "aligned-triangle-alpha-strokes@1.0": "f7b9500984d397e5",
}


def page(content: bytes) -> bytes:
    return document(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Resources << /ExtGState "
            b"<< /M 5 0 R /A 6 0 R >> /XObject << /F 7 0 R >> >> /Contents 4 0 R >>" % (SIZE, SIZE),
            stream(b"", content),
            b"<< /Type /ExtGState /BM /Multiply >>",
            b"<< /Type /ExtGState /ca 0.5 /CA 0.5 >>",
            stream(
                b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] "
                b"/Group << /S /Transparency >> /Resources << >>",
                b"",
            ),
        ]
    )


def group_page(content: bytes) -> bytes:
    """A page that paints content in a non-isolated transparency group over the backdrop."""
    return document(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Resources << /XObject "
            b"<< /F 5 0 R >> >> /Contents 4 0 R >>" % (SIZE, SIZE),
            stream(b"", BACKDROP + b"/F Do"),
            stream(
                b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] "
                b"/Group << /S /Transparency >> /Resources << >>",
                content,
            ),
        ]
    )


def render(pdf: bytes, scale: float) -> numpy.ndarray:
    with PdfDocument(pdf) as document_:
        image = document_.pages[0].render().rasterize(scale=scale)
    return numpy.frombuffer(bytes(image.pixels), numpy.uint8).reshape(image.height, image.width, 4)


def strokes(before: bytes, width: bytes, cap: int, pairs: tuple[tuple[bytes, bytes], ...]) -> bytes:
    body = b" S ".join(stroke for stroke, _ in pairs)
    return before + b"0.1 0.3 0.8 RG %s w %d J %s S" % (width, cap, body)


def fills(before: bytes, pairs: tuple[tuple[bytes, bytes], ...]) -> bytes:
    """Each pair's shape filled, one path each."""
    return before + b"0.1 0.3 0.8 rg " + b" f ".join(fill for _, fill in pairs) + b" f"


def compare(name: str, got: numpy.ndarray, want: numpy.ndarray) -> int:
    if got.shape == want.shape and numpy.array_equal(got, want):
        return 0
    if got.shape != want.shape:
        print(f"{name}: {got.shape} pixels, {want.shape} wanted")
        return 1
    differ = numpy.argwhere((got != want).any(axis=2))
    y, x = differ[0]
    print(
        f"{name}: {len(differ)} pixels differ from the shape filled as a path, first "
        f"({x}, {y}): {tuple(got[y, x])}, {tuple(want[y, x])} wanted"
    )
    return 1


def pdf_pairs() -> int:
    """Strokes against the same shapes filled as paths."""
    failed = 0
    states = {
        "multiply": BACKDROP + b"/M gs ",
        "triangle-alpha": BACKDROP + b"q " + TRIANGLE_CLIP + b"/A gs ",
        "triangle-opaque": BACKDROP + b"q " + TRIANGLE_CLIP,
    }
    for scale in SCALES:
        for state, before in states.items():
            stroked = render(page(strokes(before, b"1.3", 0, SEGMENTS)), scale)
            filled = render(page(fills(before, SEGMENTS)), scale)
            failed += compare(f"{state} segments @{scale}", stroked, filled)
        clip = b"q " + TRIANGLE_CLIP
        stroked = render(group_page(strokes(clip, b"3.1", 2, CAPPED)), scale)
        filled = render(group_page(fills(clip, CAPPED)), scale)
        failed += compare(f"grouped triangle square caps @{scale}", stroked, filled)
        for name, before, stroke, fill in ON_SAMPLES:
            stroked = render(page(before + b"0.1 0.3 0.8 RG " + stroke), scale)
            filled = render(page(before + b"0.1 0.3 0.8 rg " + fill), scale)
            failed += compare(f"triangle {name} @{scale}", stroked, filled)
    return failed


def target(scale: float) -> RasterTarget:
    width = round(SIZE * scale)
    pixels = bytearray(b"\xff\xe6\x33\xff") * (width * width)
    view = numpy.frombuffer(pixels, numpy.uint8).reshape(width, width, 4)
    clip = ClipState(DeviceGrid(0.0, 0.0, float(SIZE), scale, width, width))
    return RasterTarget(pixels, None, clip=clip, page_view=view)


def triangle() -> CapturedPath:
    return CapturedPath(
        [CapturedSubpath([(10.2, 15.7), (190.4, 30.3), (60.9, 185.1)], closed=True)]
    )


def rectangle(x0: float, y0: float, x1: float, y1: float) -> CapturedPath:
    return CapturedPath([CapturedSubpath([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True)])


# A glyph bitmap: rows top first, bit i of a row the cell i from the left.
BITMAP = (0b0111110, 0b1100011, 0b1011101, 0b1111111, 0b1000001, 0b0110110, 0b0011100)
BITMAP_SIZE = 7
GLYPH_BOXES = ((40.3, 50.45, 61.27, 73.9), (100.13, 90.71, 131.9, 112.36))


def glyph_path(box: tuple[float, float, float, float]) -> CapturedPath:
    """BITMAP in box as one path: a subpath for each set cell."""
    return CapturedPath(
        [
            CapturedSubpath([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True)
            for x0, y0, x1, y1 in glyph_cells(box)
        ]
    )


def glyph_cells(box: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    """The set cells of BITMAP in box, as draw_glyph_bitmap places them."""
    x0, y0, x1, y1 = box
    cell_w = (x1 - x0) / BITMAP_SIZE
    cell_h = (y1 - y0) / BITMAP_SIZE
    cells = []
    for index, row in enumerate(BITMAP):
        top = y1 - index * cell_h
        bottom = y1 - (index + 1) * cell_h
        column = 0
        while column < BITMAP_SIZE:
            if not row & (1 << column):
                column += 1
                continue
            end = column + 1
            cells.append((x0 + column * cell_w, bottom, x0 + end * cell_w, top))
            column = end
    return cells


Paint = Callable[[RasterTarget, tuple[int, int, int, int], str | None], None]


def direct_pairs() -> int:
    """Square dots and glyph bitmaps painted on a raster target, against the same squares
    and cells filled as paths."""
    centres = ((50.3, 60.6), (91.17, 80.43), (60.55, 110.29), (120.06, 70.71))
    width = 3.1

    def dots(paint: RasterTarget, rgba: tuple[int, int, int, int], mode: str | None) -> None:
        for x, y in centres:
            paint.fill_terminal(x, y, width, rgba, False, mode)

    def dot_fills(paint: RasterTarget, rgba: tuple[int, int, int, int], mode: str | None) -> None:
        for x, y in centres:
            half = width / 2
            paint.fill_path(rectangle(x - half, y - half, x + half, y + half), rgba, mode)

    def glyphs(paint: RasterTarget, rgba: tuple[int, int, int, int], mode: str | None) -> None:
        for box in GLYPH_BOXES:
            paint.draw_glyph_bitmap(box, list(BITMAP), rgba, mode, BITMAP_SIZE, BITMAP_SIZE)

    def glyph_fills(paint: RasterTarget, rgba: tuple[int, int, int, int], mode: str | None) -> None:
        for box in GLYPH_BOXES:
            paint.fill_path(glyph_path(box), rgba, mode)

    def round_dots(paint: RasterTarget, rgba: tuple[int, int, int, int], mode: str | None) -> None:
        for x, y in centres:
            paint.fill_terminal(x, y, width, rgba, True, mode)

    def round_fills(paint: RasterTarget, rgba: tuple[int, int, int, int], mode: str | None) -> None:
        for x, y in centres:
            paint.fill_path(circle_path(x, y, stroke_half_width(paint.scale, width)), rgba, mode)

    def named_normal(
        paint: RasterTarget, rgba: tuple[int, int, int, int], mode: str | None
    ) -> None:
        # A blend mode named Normal is no blend mode.
        for x, y in centres:
            paint.fill_rect((x - 2.3, y - 1.7, x + 2.3, y + 1.7), rgba, mode or "Normal")

    def normal_fills(
        paint: RasterTarget, rgba: tuple[int, int, int, int], mode: str | None
    ) -> None:
        for x, y in centres:
            paint.fill_path(rectangle(x - 2.3, y - 1.7, x + 2.3, y + 1.7), rgba, mode)

    pairs: list[tuple[str, Paint, Paint]] = [
        ("square dots", dots, dot_fills),
        ("glyph bitmaps", glyphs, glyph_fills),
        ("round dots", round_dots, round_fills),
        ("rectangles named Normal", named_normal, normal_fills),
    ]
    states: list[tuple[str, tuple[int, int, int, int], str | None, bool]] = [
        ("multiply", (26, 77, 204, 255), "Multiply", False),
        ("multiply alpha", (26, 77, 204, 128), "Multiply", False),
        ("triangle alpha", (26, 77, 204, 128), None, True),
        ("triangle opaque", (26, 77, 204, 255), None, True),
        ("triangle screen", (26, 77, 204, 255), "Screen", True),
        ("alpha", (26, 77, 204, 128), None, False),
    ]
    failed = 0
    for scale in SCALES:
        for shape, paint, reference in pairs:
            for state, rgba, mode, clipped in states:
                images = []
                for draw in (paint, reference):
                    raster = target(scale)
                    if clipped:
                        raster.clip.push(triangle(), "nonzero")
                    draw(raster, rgba, mode)
                    images.append(raster.pixel_array.copy())
                failed += compare(f"{state} {shape} @{scale}", images[0], images[1])
    return failed + kernel_round_caps()


def kernel_round_caps() -> int:
    """Opaque round-capped segments under a triangular clip, which the stroke kernel paints
    piece by piece, a disc for each cap, against the rectangles and discs filled as one
    path: the kernel samples a disc itself rather than the polygon fill_path fills, so a
    pixel may differ by up to three samples along the arc, and never by more."""
    width = 3.1
    segments = (((50.3, 100.6), (90.7, 100.6)), ((140.41, 60.3), (140.41, 100.7)))
    failed = 0
    for scale in SCALES:
        half = stroke_half_width(scale, width)
        images = []
        for stroked in (True, False):
            raster = target(scale)
            raster.clip.push(triangle(), "nonzero")
            rgba = (26, 77, 204, 255)
            if stroked:
                for start, end in segments:
                    path = CapturedPath([CapturedSubpath([start, end])])
                    raster.stroke_path(path, width, rgba, None, None, 1, 0, 10.0)
            else:
                for (x0, y0), (x1, y1) in segments:
                    # The segments run along the axes, from low to high.
                    across = half if x0 == x1 else 0.0
                    along = half if y0 == y1 else 0.0
                    subpaths = rectangle(x0 - across, y0 - along, x1 + across, y1 + along).subpaths
                    subpaths += circle_path(x0, y0, half).subpaths
                    subpaths += circle_path(x1, y1, half).subpaths
                    raster.fill_path(CapturedPath(subpaths), rgba, None)
            images.append(raster.pixel_array.astype(numpy.int32))
        difference = numpy.abs(images[0] - images[1]).max(axis=2)
        # Three samples of sixteen, rounded.
        worst = int(difference.max())
        if worst > 3 * 255 // 16 + 1:
            y, x = numpy.unravel_index(int(difference.argmax()), difference.shape)
            print(f"triangle round caps @{scale}: ({x}, {y}) differs by {worst}")
            failed += 1
    return failed


def unchanged_pages() -> dict[str, numpy.ndarray]:
    """Pages of rectangles fill_rect painted as fill_path before."""
    aligned = ((b"20 50 m 180 50 l", b""), (b"170 70 m 170 160 l", b""))
    aligned_fills = ((b"", b"20 49 160 2 re"), (b"", b"169 70 2 90 re"))
    return {
        "opaque-normal-strokes@1.0": render(page(strokes(BACKDROP, b"1.3", 0, SEGMENTS)), 1.0),
        "opaque-normal-strokes@2.0833": render(
            page(strokes(BACKDROP, b"1.3", 0, SEGMENTS)), 2.0833
        ),
        "opaque-normal-fills@1.0": render(page(fills(BACKDROP, SEGMENTS)), 1.0),
        "opaque-normal-fills@2.0833": render(page(fills(BACKDROP, SEGMENTS)), 2.0833),
        "aligned-multiply-strokes@1.0": render(
            page(strokes(BACKDROP + b"/M gs ", b"2", 0, aligned)), 1.0
        ),
        "aligned-multiply-fills@1.0": render(page(fills(BACKDROP + b"/M gs ", aligned_fills)), 1.0),
        "aligned-triangle-alpha-strokes@1.0": render(
            page(strokes(BACKDROP + b"q " + TRIANGLE_CLIP + b"/A gs ", b"2", 0, aligned)), 1.0
        ),
    }


def digest(image: numpy.ndarray) -> str:
    return hashlib.sha256(f"{image.shape}".encode() + image.tobytes()).hexdigest()[:16]


def main() -> int:
    failures = pdf_pairs() + direct_pairs()
    pages = unchanged_pages()
    if "--record" in sys.argv:
        for name, image in pages.items():
            print(f'    "{name}": "{digest(image)}",')
        return 0
    for name, image in pages.items():
        if digest(image) != UNCHANGED[name]:
            print(f"{name}: pixels differ from those recorded, {digest(image)}")
            failures += 1
    print(f"rectangle routes: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

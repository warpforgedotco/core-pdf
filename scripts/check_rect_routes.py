# SPDX-License-Identifier: AGPL-3.0-only

"""Check that a rectangle renders the same whichever route paints it.

A filled path is antialiased: a pixel's alpha is the share of it the path covers (core's
fill_path). Core also paints rectangles on their own, for the stroke of an axis-aligned
segment, a square dot and a cell of a glyph bitmap; each such rectangle must be the
rectangle filled as a path, whatever its blend mode and clip, even where fill_rect cannot
give its partly covered pixels their coverage (a blend mode other than Normal, or a clip
other than rectangles) and its device edges fall inside pixels.

This renders, at scales 1 and 2.0833, pages that stroke horizontal and vertical segments
with butt caps at fractional coordinates, in Multiply over a backdrop, at a constant alpha
of 0.5 under a triangular clip, and opaque under it, and the same rectangles filled as
paths: each pair must be byte-identical. It strokes segments with projecting square caps,
opaque inside a transparency group under a triangular clip, which core paints piece by
piece, against the same rectangles and end squares filled one by one. Then it paints, on a
raster target directly, square dots (fill_terminal) and glyph bitmaps (draw_glyph_bitmap)
whose cells have fractional device edges, in Multiply, in Multiply at alpha 0.5, and at
alpha 0.5, opaque and in Screen under a triangular clip, against the same squares and cells
filled as paths one by one.

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
# each end; and the same rectangles and squares.
CAPPED = (
    (
        b"50.3 60.6 m 90.7 60.6 l",
        b"50.3 59.05 40.4 3.1 re f 48.75 59.05 3.1 3.1 re f 89.15 59.05 3.1 3.1 re",
    ),
    (
        b"120.41 50.3 m 120.41 100.7 l",
        b"118.86 50.3 3.1 50.4 re f 118.86 48.75 3.1 3.1 re f 118.86 99.15 3.1 3.1 re",
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


def glyph_cells(
    box: tuple[float, float, float, float], runs: bool
) -> list[tuple[float, float, float, float]]:
    """The rectangles draw_glyph_bitmap paints for BITMAP in box: each set cell, or each run
    of set cells in a row when runs."""
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
            if runs:
                while end < BITMAP_SIZE and row & (1 << end):
                    end += 1
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
        runs = rgba[3] == 255 and mode is None
        for box in GLYPH_BOXES:
            for cell in glyph_cells(box, runs):
                paint.fill_path(rectangle(*cell), rgba, mode)

    pairs: list[tuple[str, Paint, Paint]] = [
        ("square dots", dots, dot_fills),
        ("glyph bitmaps", glyphs, glyph_fills),
    ]
    states: list[tuple[str, tuple[int, int, int, int], str | None, bool]] = [
        ("multiply", (26, 77, 204, 255), "Multiply", False),
        ("multiply alpha", (26, 77, 204, 128), "Multiply", False),
        ("triangle alpha", (26, 77, 204, 128), None, True),
        ("triangle opaque", (26, 77, 204, 255), None, True),
        ("triangle screen", (26, 77, 204, 255), "Screen", True),
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

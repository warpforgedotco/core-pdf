# SPDX-License-Identifier: AGPL-3.0-only

"""Check that an uncoloured tiling pattern's cell is a stencil the scn colour pours through.

An uncoloured tiling pattern (PaintType 2) has no colour of its own: each time it is used,
its whole cell is painted in the colour scn gave with it (ISO 32000-2 8.7.3.3). Its content
stream, and every stream it invokes, may not set colour: the colour operators, ri and sh are
ignored there, and of the images only image masks are painted (8.6.8, Table 75), in the
pattern's colour like the cell's fills, strokes and text.

Each page here fills itself with one uncoloured pattern in the colour (0.2, 0.6, 0.9), whose
40 by 40 cell paints a 16-unit square at (2, 2) -- by an image mask, an inline image mask, a
fill after colour operators, a form XObject that sets colours and paints an image mask, or
a Type 3 glyph -- and tries to paint the square at (22, 22) with a shading, an image or
inline image that is not a mask, which must leave it empty. Every pixel at least a device
pixel and a half from a square's edge is checked: inside the first square it must be the
pattern's colour, opaque, within 1 of 255, inside the second transparent, and elsewhere in
the cell transparent too, except where the case paints more. Text in the pattern's colour is
also checked: every pixel it marks must be the pattern's colour.

uv run --all-packages python scripts/check_uncoloured_patterns.py
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy

from core_pdf import PdfDocument

SIZE = 200
CELL = 40.0
COLOR = (51, 153, 230)
TOLERANCE = 1
MARGIN = 1.5
PAINTED = (2.0, 18.0)
IGNORED = (22.0, 38.0)

PLACE_PAINTED = b"q 16 0 0 16 2 2 cm "
PLACE_IGNORED = b"q 16 0 0 16 22 22 cm "
MASK = b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /ImageMask true"
IMAGE = (
    b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 /ColorSpace /DeviceRGB"
)
INLINE_MASK = b"BI /W 1 /H 1 /IM true /F /AHx ID 00> EI"
INLINE_IMAGE = b"BI /W 1 /H 1 /CS /RGB /BPC 8 /F /AHx ID E61A1A> EI"
SHADING = (
    b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 40 0] /Extend [true true] "
    b"/Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 1 0] /N 1 >> >>"
)
TYPE3_FONT = (
    b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] "
    b"/FontMatrix [0.001 0 0 0.001 0 0] /CharProcs << /square 10 0 R >> "
    b"/Encoding << /Type /Encoding /Differences [65 /square] >> "
    b"/FirstChar 65 /LastChar 65 /Widths [1000] >>"
)


def stream(dictionary: bytes, data: bytes) -> bytes:
    return b"<< " + dictionary + b" /Length %d >>\nstream\n" % len(data) + data + b"\nendstream"


def document(cell: bytes, extra: Sequence[bytes]) -> bytes:
    """A page filled with an uncoloured tiling pattern (object 5) whose cell runs cell, with
    resources: /M an image mask (6), /I an image (7), /Fm extra[0] (8) and /T3 extra[1] (9),
    /Sh a shading and /F Helvetica."""
    resources = (
        b"<< /XObject << /M 6 0 R /I 7 0 R /Fm 8 0 R >> /Shading << /Sh %s >> " % SHADING
        + b"/Font << /F << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> /T3 9 0 R >> "
        + b"/ExtGState << /BP << /UseBlackPtComp /ON >> >> >>"
    )
    pattern = stream(
        b"/Type /Pattern /PatternType 1 /PaintType 2 /TilingType 1 /BBox [0 0 40 40] "
        b"/XStep 40 /YStep 40 /Resources " + resources,
        cell,
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Contents 4 0 R " % (SIZE, SIZE)
        + b"/Resources << /Pattern << /P 5 0 R >> "
        + b"/ColorSpace << /PCS [/Pattern /DeviceRGB] >> >> >>",
        stream(b"", b"/PCS cs 0.2 0.6 0.9 /P scn 0 0 %d %d re f" % (SIZE, SIZE)),
        pattern,
        stream(MASK, b"\x00"),
        stream(IMAGE, b"\xe6\x1a\x1a"),
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


def form(body: bytes) -> bytes:
    return stream(
        b"/Type /XObject /Subtype /Form /BBox [0 0 40 40] /Resources << /XObject << /M 6 0 R >> >>",
        body,
    )


class Case(NamedTuple):
    name: str
    cell: bytes
    # Where in the cell the pattern's colour must show besides the painted square.
    painted: tuple[tuple[float, float], ...] = (PAINTED,)
    extra: tuple[bytes, ...] = (form(b""), TYPE3_FONT)
    # Whether every marked pixel must be the pattern's colour, wherever it is.
    text: bool = False

    def pdf(self) -> bytes:
        return document(self.cell, self.extra)


def cases() -> list[Case]:
    ignored = (
        b"q 22 22 16 16 re W n /Sh sh Q "
        + PLACE_IGNORED
        + b"/I Do Q "
        + PLACE_IGNORED
        + INLINE_IMAGE
        + b" Q"
    )
    glyph = stream(b"", b"1000 0 d0 1 0 0 rg 0 0 1000 1000 re f")
    return [
        Case("an image mask", PLACE_PAINTED + b"/M Do Q " + ignored),
        Case("an inline image mask", PLACE_PAINTED + INLINE_MASK + b" Q " + ignored),
        Case(
            "a fill after colour operators",
            b"1 0 0 rg 0 1 0 RG 0 0 1 0 k /DeviceRGB cs 1 0 0 sc 0.5 g /Perceptual ri /BP gs "
            b"2 2 16 16 re f " + ignored,
        ),
        Case(
            "a form that sets colours and paints an image mask",
            b"/Fm Do " + ignored,
            painted=(PAINTED, IGNORED),
            extra=(
                form(
                    b"1 0 0 rg 2 2 16 16 re f 0 1 0 rg q 16 0 0 16 22 22 cm /M Do Q "
                    b"q 22 22 16 16 re W n /Sh sh Q"
                ),
                TYPE3_FONT,
            ),
        ),
        Case(
            "a Type 3 glyph that begins with d0 and sets a colour",
            b"BT /T3 16 Tf 2 2 Td (A) Tj ET " + ignored,
            extra=(form(b""), TYPE3_FONT, glyph),
        ),
        Case(
            "text after a colour operator",
            b"BT /F 30 Tf 1 0 0 rg 0 1 0 RG 2 Tr 1 w 3 6 Td (IH) Tj ET " + ignored,
            painted=(),
            text=True,
        ),
    ]


def square(u: float, v: float, span: tuple[float, float]) -> tuple[bool, float]:
    """Whether (u, v) is inside the square span by span, and its distance to the edge."""
    low, high = span
    if low <= u < high and low <= v < high:
        return True, min(u - low, high - u, v - low, high - v)
    return False, math.hypot(max(low - u, 0.0, u - high), max(low - v, 0.0, v - high))


def expected(case: Case, x: float, y: float) -> bool | None:
    """Whether the cell paints the pattern's colour at (x, y), or None near an edge."""
    u, v = x % CELL, y % CELL
    painted = False
    for span in {*case.painted, IGNORED}:
        inside, edge = square(u, v, span)
        if edge < MARGIN:
            return None
        painted = painted or (inside and span in case.painted)
    return painted


def check(case: Case) -> tuple[int, int]:
    with PdfDocument(case.pdf()) as pdf:
        image = pdf.pages[0].render().rasterize()
    pixels = numpy.frombuffer(bytes(image.pixels), numpy.uint8).reshape(
        image.height, image.width, 4
    )
    checked = failed = 0
    for row in range(image.height):
        for column in range(image.width):
            pixel = pixels[row, column]
            colored = all(abs(int(pixel[i]) - COLOR[i]) <= TOLERANCE for i in range(3))
            if case.text:
                ok = pixel[3] == 0 or colored
            else:
                want = expected(case, column + 0.5, SIZE - (row + 0.5))
                if want is None:
                    continue
                ok = (pixel[3] == 255 and colored) if want else pixel[3] == 0
            checked += 1
            if not ok:
                if failed < 3:
                    print(f"  {case.name}: pixel {column},{row}: {tuple(int(v) for v in pixel)}")
                failed += 1
    return checked, failed


def main() -> int:
    total = failures = 0
    selected: list[Case] = cases()
    check_case: Callable[[Case], tuple[int, int]] = check
    for case in selected:
        checked, failed = check_case(case)
        total += checked
        failures += failed
        if failed:
            print(f"{case.name}: {failed} of {checked} pixels differ")
    print(f"{len(selected)} uncoloured pattern pages, {total} pixels: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

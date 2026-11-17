# SPDX-License-Identifier: AGPL-3.0-only

"""Check images drawn with Interpolate against bilinear interpolation worked out in Python.

An image whose Interpolate is true asks to be smoothed when drawn at a resolution lower than
the device's (ISO 32000-2 8.9.5.3); an image mask's interpolation smooths the mask's edges,
not the painted colour (8.9.6.2). Core interpolates bilinearly between the four sample
centres nearest a pixel's centre, clamped at the image's edges, as MuPDF does, and only when
the image is drawn larger than its samples along one of its axes. This writes pages that draw
small images upscaled, under axis-aligned and rotated placements: grey and RGB images, an
image mask, an inline image with /I true, an image at fill opacity 0.5, an image with an
SMask, and one with a colour key Mask, whose colours are interpolated weighted by alpha, so
that a masked-out sample lends its neighbours nothing. An SMask is an image of its own, its
Interpolate its own (11.6.5.2, Table 143): an image whose SMask alone interpolates, one
interpolated over an SMask that is not, and an SMask of another size than its image, each
plane sampled by its own flag and at its own resolution. Each pixel whose centre is more than
EDGE device pixels inside the image must be within TOLERANCE of the value interpolated at
its centre, worked out back through the placement in Python. With Interpolate false or
absent each must equal the nearest sample, and an image drawn smaller than its samples must
render as it does without Interpolate.

uv run --all-packages python scripts/check_image_interpolation.py
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pdf import PdfDocument  # noqa: E402
from scripts.check_patterns import Affine, apply, document, invert, numbers, stream  # noqa: E402

SIZE = 200
# A pixel may differ this much in a channel from the value worked out in Python.
TOLERANCE = 2
# Pixels whose centre is within this many device pixels of the image's edge are skipped.
EDGE = 1.5
# Each page must check at least this many pixels.
MIN_CHECKED = 400

GREY = [[0, 255, 40, 200], [255, 0, 220, 10], [90, 160, 0, 255], [30, 250, 120, 60]]
RGB = [[(255, 0, 0), (0, 255, 0), (0, 0, 255)], [(255, 255, 0), (20, 40, 60), (250, 128, 4)]]
# The bits of a 4 by 4 image mask, 1 where it paints with the default Decode [0 1] off.
MASK_BITS = [[1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
SOFT_MASK = [[255, 0, 255, 0], [0, 128, 64, 255], [255, 32, 0, 200], [10, 255, 90, 0]]
SMALL_MASK = [[0, 255, 60], [200, 20, 255]]
RED = (255, 0, 0)
# A 4 by 4 RGB image whose black samples a colour key Mask [0 0 0 0 0 0] masks out.
KEYED = [
    [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 0)],
    [(0, 255, 0), (0, 0, 0), (250, 200, 10), (40, 40, 40)],
    [(0, 0, 0), (0, 0, 0), (200, 0, 200), (0, 128, 255)],
    [(255, 255, 0), (90, 30, 60), (0, 0, 0), (255, 255, 255)],
]


type Grid = Sequence[Sequence[float]]


def bilinear(grid: Grid, u: float, v: float) -> float:
    """grid (rows from the top) at image-space u, v, between the nearest sample centres."""
    height = len(grid)
    width = len(grid[0])
    x = u * width - 0.5
    y = (1.0 - v) * height - 0.5
    column = math.floor(x)
    row = math.floor(y)
    fx = x - column
    fy = y - row

    def at(r: int, c: int) -> float:
        return grid[min(max(r, 0), height - 1)][min(max(c, 0), width - 1)]

    top = at(row, column) * (1 - fx) + at(row, column + 1) * fx
    bottom = at(row + 1, column) * (1 - fx) + at(row + 1, column + 1) * fx
    return top * (1 - fy) + bottom * fy


def nearest(grid: Grid, u: float, v: float) -> float:
    height = len(grid)
    width = len(grid[0])
    return grid[min(int((1.0 - v) * height), height - 1)][min(int(u * width), width - 1)]


def rotation(degrees: float, sx: float, sy: float, e: float, f: float) -> Affine:
    c = math.cos(math.radians(degrees))
    s = math.sin(math.radians(degrees))
    return (sx * c, sx * s, -sy * s, sy * c, e, f)


class Case:
    """A page that draws image object 5 (a stream, or inline when `inline`) placed by
    placement after `before`; expected gives the RGB a pixel shows at image-space u, v."""

    def __init__(
        self,
        name: str,
        image: bytes,
        placement: Affine,
        expected: Callable[[float, float], Sequence[float]] | None,
        *,
        before: bytes = b"",
        inline: bytes | None = None,
        extra: tuple[bytes, ...] = (),
        same_as: Case | None = None,
    ) -> None:
        self.name = name
        self.image = image
        self.placement = placement
        self.expected = expected
        self.before = before
        self.inline = inline
        self.extra = extra
        self.same_as = same_as

    def pdf(self) -> bytes:
        draw = self.inline if self.inline is not None else b"/Im Do"
        content = b"q %s %s cm %s Q" % (self.before, numbers(self.placement), draw)
        return document(
            [
                b"<< /Type /Catalog /Pages 2 0 R >>",
                b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Resources << /XObject "
                b"<< /Im 5 0 R >> /ExtGState << /G0 << /ca 0.5 >> >> >> /Contents 4 0 R >>"
                % (SIZE, SIZE),
                stream(b"", content),
                self.image,
                *self.extra,
            ]
        )


def render(pdf: bytes) -> numpy.ndarray:
    with PdfDocument(pdf) as document_:
        image = document_.pages[0].render().rasterize(background=(255, 255, 255, 255))
    return numpy.frombuffer(bytes(image.pixels), numpy.uint8).reshape(image.height, image.width, 4)


def grey_image(*, interpolate: bool | None, width: int = 4, rows: Grid | None = None) -> bytes:
    rows = GREY if rows is None else rows
    data = bytes(int(value) for row in rows for value in row)
    return stream(
        b"/Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray "
        b"/BitsPerComponent 8%s" % (width, len(rows), flag(interpolate)),
        data,
    )


def grey(
    rule: Callable[[Grid, float, float], float], grid: Grid = GREY
) -> Callable[[float, float], Sequence[float]]:
    def expected(u: float, v: float) -> Sequence[float]:
        value = rule(grid, u, v)
        return (value, value, value)

    return expected


def rgb_expected(u: float, v: float) -> Sequence[float]:
    return tuple(bilinear([[pixel[k] for pixel in row] for row in RGB], u, v) for k in range(3))


def painted_by(alpha: Grid) -> Callable[[float, float], Sequence[float]]:
    """Red painted over white through alpha (0 to 255) interpolated at u, v."""

    def expected(u: float, v: float) -> Sequence[float]:
        a = bilinear(alpha, u, v) / 255
        return tuple(255 * (1 - a) + channel * a for channel in RED)

    return expected


type Rule = Callable[[Grid, float, float], float]


def masked(
    colors: Sequence[Sequence[tuple[int, int, int]]],
    color_rule: Rule,
    alpha: Grid,
    alpha_rule: Rule,
) -> Callable[[float, float], Sequence[float]]:
    """colors, sampled at u, v by color_rule, painted over white through alpha (0 to 255)
    sampled by alpha_rule: each plane by its own rule, on its own grid."""

    def expected(u: float, v: float) -> Sequence[float]:
        a = alpha_rule(alpha, u, v) / 255
        return tuple(
            255 * (1 - a) + color_rule([[pixel[k] for pixel in row] for row in colors], u, v) * a
            for k in range(3)
        )

    return expected


def flag(interpolate: bool | None) -> bytes:
    if interpolate is None:
        return b""
    return b" /Interpolate %s" % (b"true" if interpolate else b"false")


def masked_image(
    colors: Sequence[Sequence[tuple[int, int, int]]], interpolate: bool | None
) -> bytes:
    return stream(
        b"/Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8%s /SMask 6 0 R" % (len(colors[0]), len(colors), flag(interpolate)),
        bytes(channel for row in colors for pixel in row for channel in pixel),
    )


def soft_mask_image(alpha: Grid, interpolate: bool | None) -> bytes:
    return stream(
        b"/Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray "
        b"/BitsPerComponent 8%s" % (len(alpha[0]), len(alpha), flag(interpolate)),
        bytes(int(value) for row in alpha for value in row),
    )


def keyed_expected(u: float, v: float) -> Sequence[float]:
    """KEYED over white, its colours interpolated weighted by alpha, as MuPDF does."""
    alpha = [[0.0 if pixel == (0, 0, 0) else 1.0 for pixel in row] for row in KEYED]
    coverage = bilinear(alpha, u, v)
    return tuple(
        255 * (1 - coverage)
        + bilinear(
            [
                [a * pixel[k] for a, pixel in zip(alphas, row, strict=True)]
                for alphas, row in zip(alpha, KEYED, strict=True)
            ],
            u,
            v,
        )
        for k in range(3)
    )


def mask_bytes(bits: list[list[int]]) -> bytes:
    # 1 bit per sample, rows padded to a byte; sample 0 paints under Decode [0 1].
    return bytes(sum((1 - bit) << (7 - index) for index, bit in enumerate(row)) for row in bits)


def cases() -> list[Case]:
    axis: Affine = (48.0, 0.0, 0.0, 36.0, 20.0, 30.0)
    turned = rotation(30, 64.0, 48.0, 110.0, 20.0)
    mask = stream(
        b"/Type /XObject /Subtype /Image /Width 4 /Height 4 /ImageMask true "
        b"/BitsPerComponent 1 /Interpolate true",
        mask_bytes(MASK_BITS),
    )
    mask_alpha = [[255.0 * bit for bit in row] for row in MASK_BITS]
    rgb = stream(
        b"/Type /XObject /Subtype /Image /Width 3 /Height 2 /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8 /Interpolate true",
        bytes(channel for row in RGB for pixel in row for channel in pixel),
    )
    soft = stream(
        b"/Type /XObject /Subtype /Image /Width 4 /Height 4 /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8 /Interpolate true /SMask 6 0 R",
        bytes(RED) * 16,
    )
    soft_mask = stream(
        b"/Type /XObject /Subtype /Image /Width 4 /Height 4 /ColorSpace /DeviceGray "
        b"/BitsPerComponent 8 /Interpolate true",
        bytes(value for row in SOFT_MASK for value in row),
    )
    keyed = stream(
        b"/Type /XObject /Subtype /Image /Width 4 /Height 4 /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8 /Interpolate true /Mask [0 0 0 0 0 0]",
        bytes(channel for row in KEYED for pixel in row for channel in pixel),
    )
    inline = (
        b"BI /W 4 /H 4 /CS /G /BPC 8 /I true ID "
        + bytes(value for row in GREY for value in row)
        + b" EI"
    )
    large = [[(17 * r + 29 * c) % 256 for c in range(64)] for r in range(64)]
    downscaled = Case(
        "downscaled, Interpolate true",
        grey_image(interpolate=True, width=64, rows=large),
        (40.0, 0.0, 0.0, 30.0, 20.0, 30.0),
        None,
    )
    downscaled.same_as = Case(
        "downscaled, no Interpolate",
        grey_image(interpolate=None, width=64, rows=large),
        downscaled.placement,
        None,
    )

    def faded(u: float, v: float) -> Sequence[float]:
        value = bilinear(GREY, u, v)
        return (127.5 + value / 2,) * 3

    return [
        Case("grey, axis-aligned", grey_image(interpolate=True), axis, grey(bilinear)),
        Case("grey, rotated", grey_image(interpolate=True), turned, grey(bilinear)),
        Case(
            "grey, flipped",
            grey_image(interpolate=True),
            (-48, 0, 0, -36, 160, 180),
            grey(bilinear),
        ),
        Case("grey, Interpolate absent", grey_image(interpolate=None), axis, grey(nearest)),
        Case("grey, Interpolate false", grey_image(interpolate=False), turned, grey(nearest)),
        Case("RGB, rotated", rgb, rotation(-20, 90.0, 60.0, 40.0, 100.0), rgb_expected),
        Case("image mask, axis-aligned", mask, axis, painted_by(mask_alpha), before=b"1 0 0 rg"),
        Case("image mask, rotated", mask, turned, painted_by(mask_alpha), before=b"1 0 0 rg"),
        Case("inline image, /I true", b"<< >>", axis, grey(bilinear), inline=inline),
        Case(
            "grey at fill opacity 0.5",
            grey_image(interpolate=True),
            turned,
            faded,
            before=b"/G0 gs",
        ),
        Case("RGB with an SMask", soft, axis, painted_by(SOFT_MASK), extra=(soft_mask,)),
        Case("RGB with a colour key Mask", keyed, turned, keyed_expected),
        Case(
            "RGB not interpolated, its SMask interpolated",
            masked_image(KEYED, False),
            axis,
            masked(KEYED, nearest, SOFT_MASK, bilinear),
            extra=(soft_mask_image(SOFT_MASK, True),),
        ),
        Case(
            "RGB interpolated, its SMask's Interpolate absent",
            masked_image(KEYED, True),
            turned,
            masked(KEYED, bilinear, SOFT_MASK, nearest),
            extra=(soft_mask_image(SOFT_MASK, None),),
        ),
        Case(
            "RGB interpolated, its SMask's Interpolate false",
            masked_image(KEYED, True),
            axis,
            masked(KEYED, bilinear, SOFT_MASK, nearest),
            extra=(soft_mask_image(SOFT_MASK, False),),
        ),
        Case(
            "a 3 by 2 SMask interpolated over a 4 by 4 image that is not",
            masked_image(KEYED, None),
            turned,
            masked(KEYED, nearest, SMALL_MASK, bilinear),
            extra=(soft_mask_image(SMALL_MASK, True),),
        ),
        downscaled,
    ]


def check(case: Case) -> int:
    pixels = render(case.pdf()).astype(numpy.int32)
    if case.same_as is not None:
        other = render(case.same_as.pdf()).astype(numpy.int32)
        differing = int((numpy.abs(pixels - other).max(axis=2) > 0).sum())
        if differing:
            print(f"  {case.name}: {differing} pixels differ from {case.same_as.name}")
        return differing
    assert case.expected is not None
    inverse = invert(case.placement)
    a, b, c, d, _, _ = case.placement
    # Device pixels per unit of u and of v: a pixel EDGE inside the image in device space.
    margin_u = EDGE / math.hypot(a, b)
    margin_v = EDGE / math.hypot(c, d)
    checked = failed = 0
    for row in range(SIZE):
        for column in range(SIZE):
            u, v = apply(inverse, column + 0.5, SIZE - (row + 0.5))
            if not (margin_u < u < 1 - margin_u and margin_v < v < 1 - margin_v):
                continue
            checked += 1
            expected = case.expected(u, v)
            pixel = pixels[row, column]
            if any(abs(int(pixel[k]) - round(expected[k])) > TOLERANCE for k in range(3)):
                if failed < 3:
                    shown = tuple(round(value) for value in expected)
                    print(f"  {case.name}: pixel {column},{row}: {tuple(pixel[:3])} != {shown}")
                failed += 1
    if checked < MIN_CHECKED:
        print(f"  {case.name}: only {checked} pixels checked")
        failed += 1
    return failed


def main() -> int:
    failures = 0
    selected = cases()
    for case in selected:
        failures += check(case)
    print(f"{len(selected)} image interpolation pages: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: AGPL-3.0-only

"""Check core_pdf_cythonized's spot plate kernels against their Python reference.

Rendering keeps a spot colorant's plate in a CMYK group (core_pdf.impl.render.raster_spots)
and composites it with composite_spot_plate, one plate over one box, and prints it over the
process inks with apply_spot_plate. reference_composite and reference_apply below are the
same arithmetic in Python, pixel by pixel in doubles: the specification the kernels
implement. This compares them bit for bit on random plates, alphas, shapes and layers, and
on edge cases: alpha 0 and 1 everywhere, empty boxes, widths of 1 and odd widths, plates of
one and three components, every input that may be missing missing, NaN and infinite
floats, views that are not contiguous, and a window composited in bands of
raster_spots.BAND rows, as composite_plates does, against the whole.

uv run --all-packages python scripts/check_spot_plates.py [random cases]
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Callable
from typing import Any

import numpy

from core_pdf.impl.render.raster_spots import BAND, bands
from core_pdf_cythonized._spot_plates import (
    ERASED,
    GROUP,
    KEPT,
    NAMED,
    apply_spot_plate,
    composite_spot_plate,
)

MODES = (NAMED, ERASED, KEPT, GROUP)
SPECIAL_FLOATS = (math.nan, math.inf, -math.inf, -0.0, 0.0, 1.0, 1.5, -0.5)

type Bytes = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]
type Floats = numpy.ndarray[Any, numpy.dtype[numpy.float32]]


def unit(byte: int) -> float:
    return byte / 255.0


def clamp_unit(value: float) -> float:
    if not value >= 0.0:
        return 0.0
    return min(value, 1.0)


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return clamp_unit(numerator / denominator)


def reference_composite(
    values: Bytes,
    below: Bytes | None,
    inner: Bytes | None,
    previous: Bytes | None,
    layer: Bytes | None,
    marked: int,
    source: Bytes,
    result: Bytes,
    current: Bytes,
    backdrop_alpha: Bytes,
    element_alpha: Bytes,
    group_alpha: Floats | None,
    cover: Floats | None,
    mode: int,
) -> None:
    """composite_spot_plate in Python: see its docstring."""
    rows, columns, channels = values.shape
    for i in range(rows):
        for j in range(columns):
            s = unit(int(source[i, j]))
            ea = unit(int(element_alpha[i, j]))
            ba = unit(int(backdrop_alpha[i, j]))
            if mode == GROUP and group_alpha is not None:
                share = ratio(s, float(group_alpha[i, j]))
            else:
                share = ratio(s, ea)
            kept = ratio(ba, ea) if ea > 0.0 else 1.0
            covered = 0.0
            if cover is not None:
                covered = clamp_unit(float(cover[i, j]))
                covered = max(covered, s)
            ar = unit(int(result[i, j]))
            ca = unit(int(current[i, j]))
            for k in range(channels):
                b = unit(int(below[i, j, k])) if below is not None else 0.0
                if mode == NAMED:
                    assert layer is not None
                    t = 0.0 if marked & (1 << k) else 1.0 - unit(int(layer[i, j, k]))
                    e = b + share * (t - b)
                elif mode == ERASED:
                    e = b * (1.0 - share)
                elif mode == KEPT:
                    e = b * kept
                else:
                    own = unit(int(inner[i, j, k])) if inner is not None else 0.0
                    e = b + share * (own - b)
                if cover is not None:
                    assert previous is not None
                    p = unit(int(previous[i, j, k]))
                    if covered > 0.0:
                        e = (e * ea + (1.0 - covered) * (p * ca - b * ba)) / ar if ar > 0.0 else 0.0
                    else:
                        e = p
                values[i, j, k] = round(clamp_unit(e) * 255.0)


def reference_apply(pixels: Bytes, ink: Bytes) -> None:
    """apply_spot_plate in Python: each of the first three components less the ink, at
    least 0, a plate of one component standing for all three."""
    rows, columns, channels = ink.shape
    for i in range(rows):
        for j in range(columns):
            for k in range(3):
                amount = int(ink[i, j, k if channels == 3 else 0])
                pixels[i, j, k] = max(int(pixels[i, j, k]) - amount, 0)


def random_bytes(generator: numpy.random.Generator, shape: tuple[int, ...], style: str) -> Bytes:
    """Bytes of shape: random, all 0, all 255, or only 0 and 255."""
    if style == "zero":
        return numpy.zeros(shape, numpy.uint8)
    if style == "full":
        return numpy.full(shape, 255, numpy.uint8)
    if style == "binary":
        return (generator.integers(0, 2, shape) * 255).astype(numpy.uint8)
    return generator.integers(0, 256, shape, dtype=numpy.uint8)


def random_floats(generator: numpy.random.Generator, shape: tuple[int, int]) -> Floats:
    """Floats of shape, mostly 0 to 1, with special values among them."""
    values = generator.random(shape).astype(numpy.float32)
    if values.size:
        picks = generator.random(shape) < 0.1
        specials = generator.choice(numpy.array(SPECIAL_FLOATS, numpy.float32), shape)
        values = numpy.where(picks, specials, values).astype(numpy.float32)
    return values


def strided(array: numpy.ndarray[Any, Any], generator: numpy.random.Generator) -> Any:
    """array, or the same values as a view that is not contiguous."""
    if generator.random() < 0.5 or array.ndim == 0:
        return array
    padded = numpy.zeros((array.shape[0], array.shape[1] * 2, *array.shape[2:]), array.dtype)
    padded[:, ::2] = array
    return padded[:, ::2]


def composite_case(
    generator: numpy.random.Generator, rows: int, columns: int
) -> tuple[list[Any], int]:
    """Random arguments of composite_spot_plate, values and all, and the mode."""
    channels = int(generator.choice((1, 3)))
    alpha_style = str(generator.choice(("random", "zero", "full", "binary")))
    mode = int(generator.choice(MODES))
    knockout = bool(generator.random() < 0.5)

    def plate() -> Bytes | None:
        if generator.random() < 0.25:
            return None
        return strided(random_bytes(generator, (rows, columns, channels), "random"), generator)

    def alpha() -> Bytes:
        style = alpha_style if generator.random() < 0.7 else "random"
        return strided(random_bytes(generator, (rows, columns), style), generator)

    layer = strided(random_bytes(generator, (rows, columns, 4), "random"), generator)
    previous = plate() if knockout else None
    if knockout and previous is None:
        previous = numpy.zeros((rows, columns, channels), numpy.uint8)
    arguments = [
        random_bytes(generator, (rows, columns, channels), "random"),
        plate(),
        plate() if mode == GROUP else None,
        previous,
        layer if mode == NAMED else (layer if generator.random() < 0.3 else None),
        int(generator.integers(0, 8)),
        alpha(),
        alpha(),
        alpha(),
        alpha(),
        alpha(),
        random_floats(generator, (rows, columns)) if generator.random() < 0.5 else None,
        random_floats(generator, (rows, columns)) if knockout else None,
        mode,
    ]
    return arguments, mode


def same_composite(arguments: list[Any]) -> str | None:
    """Run the kernel and the reference on copies of values; None when they agree."""
    ours = arguments[0].copy()
    theirs = arguments[0].copy()
    composite_spot_plate(ours, *arguments[1:])
    reference_composite(theirs, *arguments[1:])
    if numpy.array_equal(ours, theirs):
        return None
    where = tuple(int(v) for v in numpy.argwhere(ours != theirs)[0])
    return f"at {where} kernel {ours[where]} reference {theirs[where]}"


def slice_rows(argument: Any, band: slice) -> Any:
    return argument[band] if isinstance(argument, numpy.ndarray) else argument


def banded_composite(arguments: list[Any]) -> str | None:
    """Composite a window in bands of BAND rows, as composite_plates does, and compare with
    the reference over the whole window."""
    rows = arguments[0].shape[0]
    ours = arguments[0].copy()
    for band in bands(slice(0, rows)):
        composite_spot_plate(
            ours[band], *(slice_rows(argument, band) for argument in arguments[1:])
        )
    theirs = arguments[0].copy()
    reference_composite(theirs, *arguments[1:])
    if numpy.array_equal(ours, theirs):
        return None
    where = numpy.argwhere(ours != theirs)[0]
    return f"banded, at {tuple(int(v) for v in where)}"


def same_apply(pixels: Bytes, ink: Bytes) -> str | None:
    ours = pixels.copy()
    theirs = pixels.copy()
    apply_spot_plate(ours, ink)
    reference_apply(theirs, ink)
    return None if numpy.array_equal(ours, theirs) else "apply_spot_plate differs"


def edge_shapes() -> list[tuple[int, int]]:
    """Boxes of no rows or columns, one pixel wide or high, odd widths, and around a band."""
    return [
        (0, 0),
        (0, 5),
        (5, 0),
        (1, 1),
        (1, 7),
        (7, 1),
        (3, 127),
        (BAND - 1, 3),
        (BAND, 1),
        (BAND + 1, 2),
        (2 * BAND + 3, 1),
    ]


def run(check: Callable[[], str | None], name: str, failures: list[str]) -> None:
    message = check()
    if message is not None:
        failures.append(f"{name}: {message}")


def main(argv: list[str]) -> int:
    cases = int(argv[0]) if argv else 300
    generator = numpy.random.default_rng(468)
    random.seed(468)
    failures: list[str] = []
    checked = 0
    shapes = [*edge_shapes()]
    shapes += [
        (int(generator.integers(1, 24)), int(generator.integers(1, 24))) for _ in range(cases)
    ]
    for rows, columns in shapes:
        arguments, mode = composite_case(generator, rows, columns)
        run(
            lambda arguments=arguments: same_composite(arguments),
            f"{rows}x{columns} mode {mode}",
            failures,
        )
        checked += 1
    # Each mode, with and without a knockout shape, at alpha 0 and 1 everywhere.
    for mode in MODES:
        for style in ("zero", "full"):
            for knockout in (False, True):
                rows, columns = 4, 5
                plate = random_bytes(generator, (rows, columns, 3), "random")
                alpha = random_bytes(generator, (rows, columns), style)
                arguments = [
                    plate.copy(),
                    plate,
                    plate,
                    plate if knockout else None,
                    random_bytes(generator, (rows, columns, 4), "random"),
                    2,
                    alpha,
                    alpha,
                    alpha,
                    alpha,
                    alpha,
                    None,
                    random_floats(generator, (rows, columns)) if knockout else None,
                    mode,
                ]
                run(
                    lambda arguments=arguments: same_composite(arguments),
                    f"mode {mode} {style}",
                    failures,
                )
                checked += 1
    # Windows taller than a band, composited band by band.
    for rows in (BAND, BAND + 1, 2 * BAND + 5):
        for _ in range(3):
            arguments, mode = composite_case(generator, rows, int(generator.integers(1, 6)))
            run(
                lambda arguments=arguments: banded_composite(arguments),
                f"{rows} rows banded",
                failures,
            )
            checked += 1
    for rows, columns in shapes[:60]:
        for channels in (1, 3):
            pixels = strided(random_bytes(generator, (rows, columns, 4), "random"), generator)
            ink = random_bytes(
                generator,
                (rows, columns, channels),
                str(generator.choice(("random", "zero", "full"))),
            )
            run(
                lambda pixels=pixels, ink=ink: same_apply(pixels, ink),
                f"apply {rows}x{columns}x{channels}",
                failures,
            )
            checked += 1
    for failure in failures[:20]:
        print(failure)
    print(f"{checked} spot plate kernel cases: {len(failures)} differ from the reference")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

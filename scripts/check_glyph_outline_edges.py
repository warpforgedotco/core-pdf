# SPDX-License-Identifier: AGPL-3.0-only

"""Check core_pdf_cythonized.glyph_outline_edges against the numpy and Python it replaced.

Rendering places each glyph's outline under the glyph's matrix with glyph_outline_edges:
the x and y columns, each point x * a + y * c + e and x * b + y * d + f; the edges of the
contours it keeps, the spans it kept and whether it dropped one; and the outline's bounds.
Before it, numpy computed xs * a + ys * c and xs * b + ys * d (GlyphOutlineArrays's
linear_columns, deleted), and the translation and the edges were a second step. This
compares the kernel with that reference, bit for bit, on every outline the first pages of
the upstream test PDFs draw, under the matrix each was drawn with and under random
matrices (integers, huge and tiny scales, signed zeros, infinities and NaN among them),
and on edge-case outlines: closed and open contours, contours of one or two points, and
points that are NaN, infinite or signed zeros.

uv run --all-packages python scripts/check_glyph_outline_edges.py [random matrices per outline]
"""

from __future__ import annotations

import glob
import logging
import math
import random
import struct
import sys
import warnings
from array import array
from typing import Any

import numpy

from core_pdf import PdfDocument
from core_pdf.impl.render import commands
from core_pdf_cythonized import glyph_outline_edges, outline_coordinate_arrays

CORPUS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/**/*.pdf",
    "tests/upstream/unstructured/example-docs/pdf/*.pdf",
)
PAGES = 2
MAX_BYTES = 8_000_000
# Distinct outlines kept from the corpus, and draws of them, at most.
MAX_OUTLINES = 4000
MAX_DRAWS = 40000
SPECIAL = (math.nan, math.inf, -math.inf, 0.0, -0.0)

Matrix = tuple[float, float, float, float, float, float]


def reference(
    xs: numpy.ndarray[Any, Any], ys: numpy.ndarray[Any, Any], matrix: Matrix, spans: Any
) -> tuple[Any, ...]:
    """What rendering computed before glyph_outline_edges."""
    a, b, c, d, e, f = matrix
    xs = numpy.asarray(xs)
    ys = numpy.asarray(ys)
    column_x = (xs * a + ys * c) + e
    column_y = (xs * b + ys * d) + f
    x = column_x.tolist()
    y = column_y.tolist()
    kept: list[tuple[int, int, bool]] = []
    dropped = False
    rows: list[tuple[float, float, float, float]] = []
    for start, stop in spans:
        end = stop
        if end > start and x[start] == x[end - 1] and y[start] == y[end - 1]:
            end -= 1
        if end - start >= 2:
            closes = x[start] != x[end - 1] or y[start] != y[end - 1]
            kept.append((start, end, closes))
            rows.extend((x[i], y[i], x[i + 1], y[i + 1]) for i in range(start, end - 1))
            if closes:
                rows.append((x[end - 1], y[end - 1], x[start], y[start]))
        else:
            dropped = True
    if not kept:
        return column_x, column_y, None, kept, dropped, None
    edges = numpy.array(rows, dtype=numpy.float64).reshape(-1, 4)
    bounds = (
        float(column_x.min()),
        float(column_y.min()),
        float(column_x.max()),
        float(column_y.max()),
    )
    return column_x, column_y, edges, kept, dropped, bounds


def same_float(left: float, right: float) -> bool:
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return struct.pack("<d", left) == struct.pack("<d", right)


def same_array(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    # Bit for bit, except that any NaN matches any NaN.
    lnan = numpy.isnan(left)
    if not numpy.array_equal(lnan, numpy.isnan(right)):
        return False
    return left[~lnan].tobytes() == right[~lnan].tobytes()


def difference(xs: Any, ys: Any, matrix: Matrix, spans: Any) -> str | None:
    expected = reference(xs, ys, matrix, spans)
    actual: tuple[Any, ...] = glyph_outline_edges(xs, ys, *matrix, spans)
    names = ("column_x", "column_y", "edges", "kept", "dropped", "bounds")
    for name, want, got in zip(names, expected, actual, strict=True):
        if name in {"column_x", "column_y", "edges"}:
            same = same_array(want, got)
        elif name == "bounds":
            same = (want is None and got is None) or (
                want is not None
                and got is not None
                and all(type(value) is float for value in got)
                and all(same_float(w, g) for w, g in zip(want, got, strict=True))
            )
        else:
            same = want == got
        if not same:
            return f"{name}: expected {want!r}, got {got!r}"
    return None


def corpus_draws() -> tuple[list[tuple[Any, Any, Any]], list[tuple[Any, Any, Matrix, Any]]]:
    """The outlines the corpus pages draw, and each draw's arguments."""
    outlines: dict[int, tuple[Any, Any, Any]] = {}
    draws: list[tuple[Any, Any, Matrix, Any]] = []
    kernel = commands.glyph_outline_edges

    def recording(xs: Any, ys: Any, *rest: Any, **options: Any) -> Any:
        # Rendering passes the edges' subtype after the spans.
        *matrix, spans = rest[:7]
        if len(draws) < MAX_DRAWS:
            draws.append((xs, ys, tuple(matrix), spans))
        if len(outlines) < MAX_OUTLINES:
            outlines.setdefault(id(xs), (xs, ys, spans))
        return kernel(xs, ys, *rest, **options)

    commands.glyph_outline_edges = recording  # ty: ignore[invalid-assignment]
    try:
        paths = sorted({path for pattern in CORPUS for path in glob.glob(pattern, recursive=True)})
        for path in paths:
            with open(path, "rb") as handle:
                data = handle.read()
            if len(data) > MAX_BYTES:
                continue
            try:
                with PdfDocument.open(data) as document:
                    for page in document.pages[:PAGES]:
                        page.render()
            except Exception:  # noqa: BLE001, S112
                continue
    finally:
        commands.glyph_outline_edges = kernel
    return list(outlines.values()), draws


def random_number(rng: random.Random) -> float:
    roll = rng.random()
    if roll < 0.05:
        return rng.choice(SPECIAL)
    if roll < 0.2:
        return float(rng.randint(-3, 3))
    if roll < 0.3:
        return rng.choice((1e-300, 1e300, -1e-300, 2.0**-1074, 1e16 + 1.0))
    return rng.uniform(-1000.0, 1000.0) * 10.0 ** rng.randint(-4, 1)


def random_matrix(rng: random.Random) -> Matrix:
    numbers: list[Any] = [random_number(rng) for _ in range(6)]
    if rng.random() < 0.1:
        # Python ints, which both convert to the nearest double.
        numbers = [rng.randint(-(2**60), 2**60) if rng.random() < 0.5 else v for v in numbers]
    a, b, c, d, e, f = numbers
    return a, b, c, d, e, f


def edge_outlines() -> list[tuple[Any, Any, Any]]:
    contours: list[tuple[tuple[tuple[float, float], ...], ...]] = [
        (((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)),),  # closed
        (((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),),  # open
        (((0.0, 0.0), (0.0, 0.0)),),  # two equal points: dropped
        (((0.0, 0.0), (1.0, 1.0)), ((2.0, 2.0), (3.0, 3.0), (2.0, 2.0))),
        (((math.nan, 0.0), (1.0, 0.0), (1.0, 1.0)),),
        (((0.0, math.inf), (1.0, -math.inf), (1.0, 1.0)),),
        (((0.0, -0.0), (-0.0, 0.0), (0.0, 1.0)),),
        (((-0.0, -0.0), (-0.0, 1.0), (-0.0, 2.0)),),
        (((1e308, 1e308), (-1e308, 1e308), (0.0, -1e308)),),
    ]
    outlines = []
    for contour in contours:
        arrays = outline_coordinate_arrays(contour)
        if arrays is not None:
            outlines.append(arrays)
    # Columns that are not numpy arrays take the memoryview route.
    arrays = outline_coordinate_arrays(contours[1])
    assert arrays is not None
    xs, ys, spans = arrays
    outlines.append((array("d", xs.tolist()), array("d", ys.tolist()), spans))
    return outlines


def main(argv: list[str]) -> int:
    per_outline = int(argv[0]) if argv else 4
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    rng = random.Random(20261114)
    outlines, draws = corpus_draws()
    if not draws:
        print("no glyph outlines drawn; are the submodules checked out?")
        return 1
    cases: list[tuple[Any, Any, Matrix, Any]] = list(draws)
    edges = edge_outlines()
    for xs, ys, spans in [*outlines, *edges]:
        cases.extend((xs, ys, random_matrix(rng), spans) for _ in range(per_outline))
        cases.append((xs, ys, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), spans))
    for xs, ys, matrix, spans in cases:
        problem = difference(xs, ys, matrix, spans)
        if problem is not None:
            print(f"glyph_outline_edges differs under {matrix!r}, spans {spans!r}: {problem}")
            return 1
    print(
        f"{len(cases)} placements of {len(outlines)} corpus and {len(edges)} edge-case "
        f"outlines ({len(draws)} as drawn) match"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

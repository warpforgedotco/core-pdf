# SPDX-License-Identifier: AGPL-3.0-only

"""Check that PDF functions' batch forms give what evaluating one input at a time gives.

An exponential (type 2) or stitching (type 3) function, an array of them, and a Range around
either have a batch form (core_pdf.impl.spec.s_08_graphics.pdf_function.batch_evaluator)
that finds the outputs of many inputs at once, with the same float operations in the same
order. Shadings and Separation tints use it to find their colours. This checks it in two ways.

- Functions: thousands of random functions, compiled both as the specification module compiles
  them and as graphics.functions does, with inputs inside and outside each Domain, on the
  stitching Bounds, at -0.0, and where math.pow fails. Every output must equal the one-at-a-
  time output. Where one input fails one at a time, the batch must fail too.
- Pages: axial and radial shadings in DeviceGray, DeviceRGB, DeviceCMYK, a Separation, Lab and
  an ICCBased space, under transforms, are rendered with the batch forms and without them.
  Every pixel must be the same.

uv run --all-packages python scripts/check_pdf_functions.py [random functions]
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import numpy

from core_pdf import PdfDocument
from core_pdf.impl.graphics import functions as graphics_functions
from core_pdf.impl.graphics import image_samples, shading
from core_pdf.impl.render import raster_shading
from core_pdf.impl.spec.s_08_graphics import pdf_function
from core_pdf.impl.spec.s_08_graphics.pdf_function import batch_evaluator


def number(random: numpy.random.Generator, low: float = -1.0, high: float = 2.0) -> float:
    if random.random() < 0.15:
        return float(random.choice([0.0, 1.0, 0.5, -0.0]))
    return float(random.uniform(low, high))


def exponential(random: numpy.random.Generator, count: int) -> dict[str, Any]:
    low = float(random.choice([0.0, 0.0, -1.0, 0.25]))
    high = low + float(random.choice([1.0, 1.0, 2.0, 0.5]))
    exponent = float(random.choice([1.0, 1.0, 1.0, 2.0, 0.5, 3.7, 0.0, -1.0]))
    function: dict[str, Any] = {
        "FunctionType": 2,
        "Domain": [low, high],
        "N": exponent,
        "C0": [number(random) for _ in range(count)],
        "C1": [number(random) for _ in range(count)],
    }
    if random.random() < 0.3:
        function["Range"] = [value for _ in range(count) for value in (0.0, 1.0)]
    return function


def stitching(random: numpy.random.Generator, count: int, depth: int = 0) -> dict[str, Any]:
    parts = int(random.integers(1, 5))
    low = float(random.choice([0.0, 0.0, -1.0]))
    high = low + float(random.choice([1.0, 2.0]))
    bounds = sorted(float(value) for value in random.uniform(low, high, parts - 1))
    if len(set(bounds)) != len(bounds) or (bounds and bounds[0] <= low):
        bounds = [low + (high - low) * (index + 1) / parts for index in range(parts - 1)]
    children = [
        stitching(random, count, depth + 1)
        if depth < 1 and random.random() < 0.2
        else exponential(random, count)
        for _ in range(parts)
    ]
    encode = [float(random.choice([0.0, 1.0, 0.25])) for _ in range(2 * parts)]
    function: dict[str, Any] = {
        "FunctionType": 3,
        "Domain": [low, high],
        "Functions": children,
        "Bounds": bounds,
        "Encode": encode,
    }
    if random.random() < 0.3:
        function["Range"] = [value for _ in range(count) for value in (0.0, 1.0)]
    return function


def random_function(random: numpy.random.Generator) -> object:
    count = int(random.choice([1, 1, 3, 4]))
    if random.random() < 0.15:
        return [exponential(random, 1) for _ in range(count)]
    return stitching(random, count) if random.random() < 0.6 else exponential(random, count)


def inputs_for(random: numpy.random.Generator, function: object) -> list[float]:
    values = [float(value) for value in random.uniform(-1.5, 2.5, 40)]
    values += [0.0, -0.0, 1.0, 0.5, -1.0, 2.0]
    if isinstance(function, dict):
        values += [float(value) for value in function.get("Domain", ())]
        values += [float(value) for value in function.get("Bounds", ())]
    return values


def one_at_a_time(evaluate: Callable[..., tuple[float, ...]], values: list[float]) -> Any:
    try:
        return numpy.asarray([evaluate(value) for value in values], dtype=numpy.float64)
    except TypeError, ValueError, ArithmeticError:
        return None


def check_function(function: object, values: list[float]) -> tuple[int, list[str]]:
    """The number of batches compared and any differences, for both compilers."""
    compared = 0
    problems: list[str] = []
    compilers: list[tuple[str, Callable[[object], Any]]] = [
        ("spec", pdf_function.compile_pdf_function),
        ("graphics", graphics_functions.compile_pdf_function),
    ]
    for label, compile_function in compilers:
        try:
            evaluate = compile_function(function)
        except ValueError:
            continue
        batch = batch_evaluator(evaluate)
        if batch is None:
            problems.append(f"{label}: no batch form for {function!r}")
            continue
        expected = one_at_a_time(evaluate, values)
        try:
            found = batch(numpy.asarray(values, dtype=numpy.float64))
        except TypeError, ValueError, ArithmeticError:
            found = None
        compared += 1
        if expected is None:
            if found is not None:
                problems.append(f"{label}: the batch succeeds where one input fails: {function!r}")
            continue
        if found is None:
            if numpy.isnan(expected).any():
                continue  # a NaN is left to one at a time, which then finds it
            problems.append(f"{label}: the batch fails where every input succeeds: {function!r}")
            continue
        if found.shape != expected.shape or not numpy.array_equal(found, expected):
            problems.append(f"{label}: the batch differs for {function!r}")
    return compared, problems


def stream(dictionary: bytes, data: bytes) -> bytes:
    return b"<< " + dictionary + b" /Length %d >>\nstream\n" % len(data) + data + b"\nendstream"


def document(objects: list[bytes]) -> bytes:
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for index, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % index + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


STITCHED = (
    b"<< /FunctionType 3 /Domain [0 1] /Bounds [0.3 0.7] /Encode [0 1 1 0 0 1] /Functions ["
    b"<< /FunctionType 2 /Domain [0 1] /C0 %(c0)s /C1 %(c1)s /N 1 >> "
    b"<< /FunctionType 2 /Domain [0 1] /C0 %(c0)s /C1 %(c1)s /N 2.2 >> "
    b"<< /FunctionType 2 /Domain [0 1] /C0 %(c1)s /C1 %(c0)s /N 0.5 >>] >>"
)
TINT = b"<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0.1 0.9 0.4 0.05] /N 1 >>"
SPACES: list[tuple[bytes, bytes, bytes]] = [
    (b"/DeviceGray", b"[0.1]", b"[0.9]"),
    (b"/DeviceRGB", b"[0.9 0.1 0.2]", b"[0.1 0.4 1]"),
    (b"/DeviceCMYK", b"[0.9 0.1 0 0.2]", b"[0 0.5 1 0]"),
    (b"[/Separation /Spot /DeviceCMYK " + TINT + b"]", b"[0]", b"[1]"),
    (
        b"[/Lab << /WhitePoint [0.9505 1 1.089] /Range [-100 100 -100 100] >>]",
        b"[20 -60 40]",
        b"[90 70 -50]",
    ),
    (
        b"[/CalRGB << /WhitePoint [0.9505 1 1.089] /Gamma [2.2 2.2 2.2] >>]",
        b"[0.9 0.1 0.2]",
        b"[0.1 0.4 1]",
    ),
]
PLACEMENTS = [b"1 0 0 1 0 0", b"0 120 120 0 20 10", b"80 60 -60 80 100 20"]


def page(space: bytes, c0: bytes, c1: bytes, shading_type: int, placement: bytes) -> bytes:
    coords = b"[0 0 1 1]" if shading_type == 2 else b"[0.4 0.4 0.05 0.6 0.6 0.9]"
    function = STITCHED % {b"c0": c0, b"c1": c1}
    if placement == PLACEMENTS[0]:
        coords = b"[10 10 190 170]" if shading_type == 2 else b"[90 90 5 110 110 120]"
    shading_object = (
        b"<< /ShadingType %d /ColorSpace %s /Coords %s /Extend [true true] /Function %s >>"
        % (shading_type, space, coords, function)
    )
    return document(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R "
            b"/Resources << /Shading << /S 5 0 R >> >> >>",
            stream(b"", b"q 0 0 200 200 re W n %s cm /S sh Q" % placement),
            shading_object,
        ]
    )


def rendered(source: bytes) -> bytes:
    with PdfDocument(source) as pdf:
        return bytes(pdf.pages[0].render().rasterize().pixels)


@contextmanager
def without_batches() -> Generator[None]:
    """Find every colour one at a time, as before the batch forms."""
    modules = (raster_shading, shading, image_samples)
    saved = [module.batch_evaluator for module in modules]
    for module in modules:
        setattr(module, "batch_evaluator", lambda _evaluate: None)  # noqa: B010
    try:
        yield
    finally:
        for module, original in zip(modules, saved, strict=True):
            setattr(module, "batch_evaluator", original)  # noqa: B010


def main(argv: list[str]) -> int:
    count = int(argv[0]) if argv else 3000
    random = numpy.random.default_rng(20261114)
    compared = 0
    problems: list[str] = []
    for _ in range(count):
        function = random_function(random)
        found, issues = check_function(function, inputs_for(random, function))
        compared += found
        problems += issues
    pages = 0
    for space, c0, c1 in SPACES:
        for shading_type in (2, 3):
            for placement in PLACEMENTS:
                source = page(space, c0, c1, shading_type, placement)
                with_batch = rendered(source)
                with without_batches():
                    one_by_one = rendered(source)
                pages += 1
                if with_batch != one_by_one:
                    problems.append(
                        f"page: {space[:24]!r} type {shading_type} under {placement!r} differs"
                    )
    for problem in problems[:20]:
        print(problem)
    print(
        f"{count} random functions ({compared} batches) and {pages} shading pages: "
        f"{len(problems)} failures"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

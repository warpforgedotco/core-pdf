# SPDX-License-Identifier: AGPL-3.0-only

"""Check that PDF functions' batch forms give what evaluating one input at a time gives.

Every kind of function has a batch form (core_pdf.impl.spec.s_08_graphics.pdf_function.
batch_evaluator) that finds the outputs of many inputs at once, bit for bit as one at a time
finds them: exponential (type 2) and stitching (type 3) functions, arrays of them and a Range
around either; sampled (type 0) functions, by multilinear interpolation over every corner of
the cell; and PostScript calculator (type 4) programs, run over the whole batch with the
inputs split where their paths part. Shadings, Separation and DeviceN tints use it to find
their colours. This checks it in two ways.

- Functions: thousands of random functions, compiled both as the specification module compiles
  them and as graphics.functions does, with inputs inside and outside each Domain, on the
  stitching Bounds and a sampled function's grid points, at -0.0, NaN and infinities, and where
  an operator fails.
  Sampled functions have every BitsPerSample, one to four inputs (and one of twelve), Encode,
  Decode and Range;
  calculator programs use every operator, conditionals, copy, index and roll with counts that
  can differ between inputs, integers that overflow to reals, and operators that fail. Every
  output must have the bits of the one-at-a-time output. Where one input fails one at a
  time, the batch must fail too, and the batch of the inputs that succeed must not.
- Pages: axial and radial shadings in DeviceGray, DeviceRGB, DeviceCMYK, a Separation, Lab and
  an ICCBased space, under transforms; function-based, axial and radial shadings through
  sampled and calculator functions; Separation and DeviceN images whose tint transforms are
  sampled or calculator functions; and luminosity soft masks whose transfer functions are, are
  rendered with the batch forms and without them.
  Every pixel must be the same.

uv run --all-packages python scripts/check_pdf_functions.py [random functions]
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, cast

import numpy

from core_pdf import PdfDocument
from core_pdf.impl.graphics import functions as graphics_functions
from core_pdf.impl.graphics import image_samples, shading
from core_pdf.impl.render import raster_shading, raster_support
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics import pdf_function
from core_pdf.impl.spec.s_08_graphics.pdf_function import batch_evaluator

type Inputs = list[tuple[float, ...]]


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


def inputs_for(random: numpy.random.Generator, function: object) -> Inputs:
    values = [float(value) for value in random.uniform(-1.5, 2.5, 40)]
    values += [0.0, -0.0, 1.0, 0.5, -1.0, 2.0]
    if isinstance(function, dict):
        values += [float(value) for value in function.get("Domain", ())]
        values += [float(value) for value in function.get("Bounds", ())]
    return with_nonfinite(random, [(value,) for value in values])


SAMPLE_BITS = (1, 2, 4, 8, 12, 16, 24, 32)


def sampled(random: numpy.random.Generator) -> PdfStream:
    """A random sampled function: every bit depth, one to four inputs, Encode, Decode and
    Range, data that is short or has bytes to spare, Order 1 or 3."""
    inputs = int(random.choice([1, 1, 1, 2, 2, 3, 4]))
    sizes = [int(random.integers(1, 7 if inputs <= 2 else 4)) for _ in range(inputs)]
    outputs = int(random.choice([1, 1, 2, 3, 4]))
    bits = int(random.choice(SAMPLE_BITS))
    domain: list[float] = []
    for _ in range(inputs):
        low = float(random.choice([0.0, 0.0, -1.0, 0.25, -180.0]))
        domain += [low, low + float(random.choice([1.0, 1.0, 2.0, 0.5, 360.0]))]
    output_range: list[float] = []
    for _ in range(outputs):
        if random.random() < 0.6:
            output_range += [0.0, 1.0]
        else:
            low = number(random, -2.0, 1.0)
            output_range += [low, low + float(random.choice([0.0, 0.5, 2.0, 100.0]))]
    dictionary: dict[str, Any] = {
        "FunctionType": 0,
        "Domain": domain,
        "Range": output_range,
        "Size": sizes,
        "BitsPerSample": bits,
    }
    if random.random() < 0.15:
        dictionary["Order"] = 3
    if random.random() < 0.4:
        dictionary["Decode"] = [number(random, -1.0, 2.0) for _ in range(2 * outputs)]
    if random.random() < 0.4:
        encode: list[float] = []
        for size in sizes:
            encode += [
                float(value)
                for value in random.choice(
                    [(0, size - 1), (size - 1, 0), (-1, size + 1), (0.5, size - 1.5), (1, 1)]
                )
            ]
        dictionary["Encode"] = encode
    total = math.prod(sizes) * outputs
    length = (total * bits + 7) // 8 + int(random.choice([0, 0, 0, 0, 3, -1]))
    data = random.integers(0, 256, max(0, length), dtype=numpy.uint8)
    if random.random() < 0.3:
        data[: len(data) // 3] = 255
    return PdfStream(dictionary, bytes(data))


def sampled_inputs(random: numpy.random.Generator, function: PdfStream) -> Inputs:
    """Points inside and outside the Domain, on its bounds, on the sample grid along some
    inputs and between samples along others, and at -0.0."""
    domain = cast("list[float]", function.dictionary["Domain"])
    sizes = cast("list[int]", function.dictionary["Size"])
    pairs = list(zip(domain[::2], domain[1::2], strict=True))
    rows: Inputs = []
    for _ in range(48):
        row = []
        for (low, high), size in zip(pairs, sizes, strict=True):
            choice = random.random()
            if choice < 0.35:
                value = float(random.uniform(low - 0.3 * (high - low), high + 0.3 * (high - low)))
            elif choice < 0.7 and size > 1:
                value = low + (high - low) * int(random.integers(0, size)) / (size - 1)
            elif choice < 0.85:
                value = float(random.choice([low, high, -0.0, 0.0]))
            else:
                value = float(random.uniform(low, high))
            row.append(value)
        rows.append(tuple(row))
    return with_nonfinite(random, rows)


def with_nonfinite(random: numpy.random.Generator, rows: Inputs) -> Inputs:
    """rows, with NaN, inf or -inf for some of the inputs of a few of them."""
    changed: Inputs = []
    for row in rows:
        if random.random() < 0.1:
            row = tuple(
                float(random.choice([math.nan, math.inf, -math.inf]))
                if random.random() < 0.5
                else value
                for value in row
            )
        changed.append(row)
    return changed


INTEGERS = (0, 1, 2, 3, -1, -2, 5, 7, 10, 255, 360, 65536, 2147483647, -2147483648)
REALS = (0.0, -0.0, 0.5, 1.0, 2.5, -1.5, 0.25, 0.3, 90.0, 1e10, -1e10, 3.14159, 1e300)
UNARY = ("abs", "ceiling", "floor", "round", "truncate", "sqrt", "sin", "cos", "ln", "log")
UNARY += ("neg", "cvi", "cvr")
NUMERIC_BINARY = ("add", "sub", "mul", "div", "atan", "exp")
INTEGER_BINARY = ("idiv", "mod", "bitshift", "and", "or", "xor")
COMPARISONS = ("eq", "ne", "gt", "ge", "lt", "le")
ALL_OPERATORS = (
    UNARY
    + NUMERIC_BINARY
    + INTEGER_BINARY
    + COMPARISONS
    + ("not", "copy", "exch", "pop", "dup", "index", "roll")
)


def literal_token(value: object) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    assert isinstance(value, float)
    if value == 0.0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    text = repr(value)
    return format(value, "f") if "e" in text else text


class Program:
    """A random calculator program, written as its kinds of operands are tracked: "i", "f",
    "b", or "?" where the inputs' paths leave the kind open."""

    def __init__(self, random: numpy.random.Generator) -> None:
        self.random = random

    def choose(self, options: tuple[Any, ...] | list[Any]) -> Any:
        return options[int(self.random.integers(0, len(options)))]

    def literal(self, tokens: list[str], kinds: list[str]) -> None:
        roll = self.random.random()
        if roll < 0.45:
            value: object = self.choose(INTEGERS)
            kinds.append("i")
        elif roll < 0.9:
            value = (
                self.choose(REALS)
                if self.random.random() < 0.5
                else round(float(self.random.uniform(-3, 3)), 4)
            )
            kinds.append("f")
        else:
            value = bool(self.random.random() < 0.5)
            kinds.append("b")
        tokens.append(literal_token(value))

    def numeric(self, kinds: list[str], count: int) -> bool:
        return len(kinds) >= count and all(kind in "if?" for kind in kinds[-count:])

    def step(self, tokens: list[str], kinds: list[str], depth: int) -> None:
        random = self.random
        options = ["literal"]
        if self.numeric(kinds, 1):
            options += ["unary", "unary", "compare_literal"]
        if self.numeric(kinds, 2):
            options += ["binary", "binary", "compare"]
        if len(kinds) >= 2 and kinds[-1] == kinds[-2] == "i":
            options += ["integer"]
        if kinds and kinds[-1] == "i":
            options += ["integer_literal", "integer_literal"]
        if kinds and kinds[-1] == "f":
            options += ["to_integer"]
        if kinds and kinds[-1] in "ib":
            options += ["not"]
        if len(kinds) >= 2 and kinds[-1] == kinds[-2] == "b":
            options += ["logical"]
        if kinds and kinds[-1] == "b" and depth < 3:
            options += ["conditional", "conditional"]
        if kinds:
            options += ["pop", "dup", "copy", "index", "roll", "roll"]
        if any(kind == "f" for kind in kinds) and depth < 3:
            options += ["conditional_from", "varying_roll", "varying_index"]
        if len(kinds) >= 2:
            options += ["exch"]
        if random.random() < 0.03:
            options = ["anything"]
        action = self.choose(options)
        if action == "literal":
            self.literal(tokens, kinds)
        elif action == "unary":
            name = self.choose(UNARY)
            tokens.append(name)
            kind = kinds.pop()
            if name == "cvi":
                kinds.append("i")
            elif name in {"abs", "neg", "ceiling", "floor", "round", "truncate"}:
                kinds.append(kind)
            else:
                kinds.append("f")
        elif action == "binary":
            name = self.choose(NUMERIC_BINARY)
            tokens.append(name)
            right, left = kinds.pop(), kinds.pop()
            both = left == right == "i"
            kinds.append("i" if both and name in {"add", "sub", "mul"} else "f")
        elif action == "integer_literal":
            tokens += [
                str(self.choose((1, 2, 3, -3, 7, 31, 32, -5, 255))),
                self.choose(INTEGER_BINARY),
            ]
        elif action == "to_integer":
            tokens += [self.choose(("10", "1000", "3")), "mul", "cvi"]
            kinds[-1] = "i"
        elif action == "integer":
            tokens.append(self.choose(INTEGER_BINARY))
            kinds.pop()
        elif action == "not":
            tokens.append("not")
        elif action == "logical":
            tokens.append(self.choose(("and", "or", "xor")))
            kinds.pop()
        elif action == "compare":
            tokens.append(self.choose(COMPARISONS))
            kinds.pop()
            kinds[-1] = "b"
        elif action == "compare_literal":
            tokens += [
                literal_token(round(float(random.uniform(-1, 2)), 2)),
                self.choose(COMPARISONS),
            ]
            kinds[-1] = "b"
        elif action == "conditional":
            kinds.pop()
            self.conditional(tokens, kinds, depth)
        elif action == "conditional_from":
            position = self.choose([index for index, kind in enumerate(kinds) if kind == "f"])
            threshold = literal_token(round(float(random.uniform(-0.5, 1.5)), 2))
            tokens += [str(len(kinds) - 1 - position), "index", threshold, self.choose(COMPARISONS)]
            self.conditional(tokens, kinds, depth)
        elif action in {"pop", "dup"}:
            tokens.append(action)
            if action == "pop":
                kinds.pop()
            else:
                kinds.append(kinds[-1])
        elif action == "exch":
            tokens.append("exch")
            kinds[-1], kinds[-2] = kinds[-2], kinds[-1]
        elif action == "copy":
            count = int(random.integers(0, len(kinds) + (2 if random.random() < 0.1 else 1)))
            tokens += [str(count), "copy"]
            if count <= len(kinds):
                kinds.extend(kinds[len(kinds) - count :])
        elif action == "index":
            index = int(random.integers(-1 if random.random() < 0.05 else 0, len(kinds)))
            tokens += [str(index), "index"]
            kinds.append(kinds[-index - 1] if index >= 0 else "?")
        elif action == "roll":
            count = int(random.integers(0, len(kinds) + 1))
            shift = int(random.integers(-4, 5))
            tokens += [str(count), str(shift), "roll"]
            if count and shift % count:
                moved = kinds[-count:]
                kinds[-count:] = moved[-(shift % count) :] + moved[: -(shift % count)]
        elif action == "varying_roll":
            # A shift found from an input: each input rolls its own way.
            count = int(random.integers(1, len(kinds) + 1))
            position = self.choose([index for index, kind in enumerate(kinds) if kind == "f"])
            tokens += [str(count), str(len(kinds) - position), "index", "7", "mul", "cvi"]
            tokens.append("roll")
            if len(set(kinds[-count:])) > 1:
                kinds[-count:] = ["?"] * count
        elif action == "varying_index":
            position = self.choose([index for index, kind in enumerate(kinds) if kind == "f"])
            tokens += [str(len(kinds) - 1 - position), "index", "0.5", "gt"]
            first, second = (int(value) for value in random.integers(0, len(kinds), 2))
            tokens += ["{", str(first), "}", "{", str(second), "}", "ifelse", "index"]
            kinds.append(kinds[-first - 1] if kinds[-first - 1] == kinds[-second - 1] else "?")
        else:
            tokens.append(self.choose(ALL_OPERATORS))
            kinds[:] = ["?"] * max(0, len(kinds) - 1)

    def conditional(self, tokens: list[str], kinds: list[str], depth: int) -> None:
        """A boolean has been pushed: an if or ifelse whose branches leave as many operands."""
        branches = [self.branch(kinds, depth)]
        if self.random.random() < 0.6:
            branches.append(self.branch(kinds, depth))
        for branch, _ in branches:
            tokens += ["{", *branch, "}"]
        tokens.append("ifelse" if len(branches) == 2 else "if")
        ends = [found for _, found in branches] + ([] if len(branches) == 2 else [list(kinds)])
        kinds[:] = [column[0] if len(set(column)) == 1 else "?" for column in zip(*ends)]

    def branch(self, kinds: list[str], depth: int) -> tuple[list[str], list[str]]:
        tokens: list[str] = []
        found = list(kinds)
        for _ in range(int(self.random.integers(1, 5))):
            self.step(tokens, found, depth + 1)
        self.settle(tokens, found, len(kinds))
        return tokens, found

    def settle(self, tokens: list[str], kinds: list[str], height: int) -> None:
        while len(kinds) > height:
            tokens.append("pop")
            kinds.pop()
        while len(kinds) < height:
            self.literal(tokens, kinds)

    def write(self, inputs: int, outputs: int) -> bytes:
        tokens: list[str] = []
        kinds = ["f"] * inputs
        for _ in range(int(self.random.integers(1, 24))):
            self.step(tokens, kinds, 0)
        if self.random.random() < 0.01:
            tokens += ["1"] * 101
            kinds += ["i"] * 101
        self.settle(tokens, kinds, outputs)
        # An output must be a number: each boolean becomes 1.0 or 0.0, turning the stack round.
        if "b" in kinds:
            for _ in range(outputs):
                if kinds[-1] == "b":
                    tokens += ["{", "1.0", "}", "{", "0.0", "}", "ifelse"]
                    kinds[-1] = "f"
                tokens += [str(outputs), "1", "roll"]
                kinds.insert(0, kinds.pop())
        return ("{ " + " ".join(tokens) + " }").encode()


def calculator(random: numpy.random.Generator) -> PdfStream:
    inputs = int(random.choice([1, 1, 1, 2, 3]))
    outputs = int(random.choice([1, 1, 3, 4]))
    domain: list[float] = []
    for _ in range(inputs):
        domain += [
            float(value)
            for value in random.choice(
                [(0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 360.0), (-1e6, 1e6)]
            )
        ]
    output_range: list[float] = []
    for _ in range(outputs):
        output_range += [
            float(value)
            for value in random.choice([(0.0, 1.0), (0.0, 1.0), (-1e12, 1e12), (-0.5, 0.25)])
        ]
    dictionary = {"FunctionType": 4, "Domain": domain, "Range": output_range}
    return PdfStream(dictionary, Program(random).write(inputs, outputs))


def calculator_inputs(random: numpy.random.Generator, function: PdfStream) -> Inputs:
    domain = cast("list[float]", function.dictionary["Domain"])
    pairs = list(zip(domain[::2], domain[1::2], strict=True))
    special = [0.0, -0.0, 0.5, 0.3, 1.0, -1.0, 0.25, 90.0, 3e9, -3e9]
    rows: Inputs = []
    for _ in range(48):
        row = []
        for low, high in pairs:
            choice = random.random()
            if choice < 0.5:
                span = high - low
                value = float(random.uniform(low - 0.3 * span, high + 0.3 * span))
            elif choice < 0.8:
                value = float(random.choice(special))
            else:
                value = float(random.choice([low, high]))
            row.append(value)
        rows.append(tuple(row))
    return with_nonfinite(random, rows)


WIDE_INPUTS = 12


def wide_sampled_inputs(random: numpy.random.Generator) -> Inputs:
    """2000 points of a 12-input function: on the sample grid along most inputs and between
    samples along a few, as an image's colorants mostly are, and between samples along all."""
    on_grid = random.integers(0, 2, (2000, WIDE_INPUTS)).astype(numpy.float64)
    between = random.random((2000, WIDE_INPUTS))
    rows = numpy.where(random.random((2000, WIDE_INPUTS)) < 0.85, on_grid, between)
    rows[:100] = between[:100]
    rows[100:200] = on_grid[100:200]
    return [tuple(row) for row in rows.tolist()]


def wide_image_samples() -> bytes:
    """A 64 by 64 image of 12 colorants, most of them 0 or 255 at each pixel."""
    random = numpy.random.default_rng(12)
    levels = numpy.where(
        random.random((64 * 64, WIDE_INPUTS)) < 0.9,
        random.integers(0, 2, (64 * 64, WIDE_INPUTS)) * 255,
        random.integers(0, 256, (64 * 64, WIDE_INPUTS)),
    )
    return levels.astype(numpy.uint8).tobytes()


def check_wide_sampled(random: numpy.random.Generator) -> tuple[int, list[str]]:
    """A sampled function of 12 inputs (Size [2] * 12, 4096 cells' corners), as a 12-colorant
    DeviceN tint transform has: the batch interpolates along the inputs each row moves on."""
    total = 2**WIDE_INPUTS * 3
    function = PdfStream(
        {
            "FunctionType": 0,
            "Domain": [0.0, 1.0] * WIDE_INPUTS,
            "Range": [0.0, 1.0] * 3,
            "Size": [2] * WIDE_INPUTS,
            "BitsPerSample": 8,
        },
        random.integers(0, 256, total, dtype=numpy.uint8).tobytes(),
    )
    return check_function(function, wide_sampled_inputs(random))


def check_bit_depths() -> list[str]:
    """A sampled function of each bit depth reads its packed samples and maps the largest to
    the top of Decode (7.10.2); before, every depth but 8 was refused."""
    problems: list[str] = []
    for bits in SAMPLE_BITS:
        top = (1 << bits) - 1
        samples = [top // 3, top, 0]
        bit_string = "".join(format(value, f"0{bits}b") for value in samples)
        bit_string += "0" * (-len(bit_string) % 8)
        data = int(bit_string, 2).to_bytes(len(bit_string) // 8, "big")
        function = PdfStream(
            {
                "FunctionType": 0,
                "Domain": [0.0, 1.0],
                "Range": [0.0, 1.0],
                "Size": [3],
                "BitsPerSample": bits,
            },
            data,
        )
        try:
            evaluate = pdf_function.compile_pdf_function(function)
            found = [evaluate(value)[0] for value in (0.0, 0.5, 1.0, 0.25)]
        except ValueError as error:
            problems.append(f"{bits} bits per sample: {error}")
            continue
        wanted = [top // 3 / top, 1.0, 0.0, (top // 3 * 0.5 + top * 0.5) / top]
        if found != wanted:
            problems.append(f"{bits} bits per sample: {found} rather than {wanted}")
    return problems


def one_at_a_time(evaluate: Callable[..., tuple[float, ...]], rows: Inputs) -> list[Any]:
    """Each input's outputs, or None where it fails."""
    found: list[Any] = []
    for row in rows:
        try:
            found.append(evaluate(*row))
        except TypeError, ValueError, ArithmeticError:
            found.append(None)
    return found


def batch_of(batch: Callable[[Any], Any], rows: Inputs) -> Any:
    try:
        return batch(numpy.asarray(rows, dtype=numpy.float64))
    except TypeError, ValueError, ArithmeticError:
        return None


def same_bits(found: Any, expected: Any) -> bool:
    """Equal bit for bit: -0.0 is not 0.0, and a NaN only matches the same NaN."""
    return found.shape == expected.shape and numpy.array_equal(
        found.view(numpy.uint64), expected.view(numpy.uint64)
    )


def check_function(function: object, rows: Inputs) -> tuple[int, list[str]]:
    """The number of batches compared and any differences, for both compilers."""
    compared = 0
    problems: list[str] = []
    # A sampled function's batch must find NaN and infinite inputs too.
    exact = isinstance(function, PdfStream) and function.dictionary.get("FunctionType") == 0
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
        outputs = one_at_a_time(evaluate, rows)
        succeeded = [row for row, output in zip(rows, outputs, strict=True) if output is not None]
        expected = [output for output in outputs if output is not None]
        if len(succeeded) < len(rows):
            compared += 1
            if batch_of(batch, rows) is not None:
                problems.append(f"{label}: the batch succeeds where one input fails: {function!r}")
        if not succeeded:
            continue
        compared += 1
        wanted = numpy.asarray(expected, dtype=numpy.float64)
        found = batch_of(batch, succeeded)
        if found is None:
            if numpy.isnan(wanted).any():
                continue  # a NaN is left to one at a time, which then finds it
            if not exact and not numpy.isfinite(numpy.asarray(succeeded)).all():
                continue  # so is an input that is not finite, but a sampled function's batch
            problems.append(f"{label}: the batch fails where every input succeeds: {function!r}")
        elif not same_bits(found, wanted):
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
# A type 1 shading's Matrix places it on the page already: these turn it about there.
TYPE1_PLACEMENTS = [b"1 0 0 1 0 0", b"0 1 1 0 0 0", b"0.7 0.4 -0.4 0.7 80 -10"]


def page_with(resources: bytes, content: bytes, extra: list[bytes]) -> bytes:
    """A 200 by 200 page with these resources and content, and objects 5 on."""
    return document(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R "
            b"/Resources << " + resources + b" >> >>",
            stream(b"", content),
            *extra,
        ]
    )


def page(space: bytes, c0: bytes, c1: bytes, shading_type: int, placement: bytes) -> bytes:
    coords = b"[0 0 1 1]" if shading_type == 2 else b"[0.4 0.4 0.05 0.6 0.6 0.9]"
    function = STITCHED % {b"c0": c0, b"c1": c1}
    if placement == PLACEMENTS[0]:
        coords = b"[10 10 190 170]" if shading_type == 2 else b"[90 90 5 110 110 120]"
    shading_object = (
        b"<< /ShadingType %d /ColorSpace %s /Coords %s /Extend [true true] /Function %s >>"
        % (shading_type, space, coords, function)
    )
    return page_with(
        b"/Shading << /S 5 0 R >>",
        b"q 0 0 200 200 re W n %s cm /S sh Q" % placement,
        [shading_object],
    )


def sampled_stream(domain: bytes, sizes: list[int], outputs: int, bits: int, extra: bytes) -> bytes:
    """A sampled function with a smooth table: each output a different blend of the inputs."""
    grid = numpy.stack(
        numpy.meshgrid(*[numpy.linspace(0, 1, size) for size in sizes], indexing="ij"), axis=-1
    )
    # The first input varies fastest in a sampled function's table.
    grid = grid.transpose(*reversed(range(len(sizes))), len(sizes)).reshape(-1, len(sizes))
    columns = []
    for output in range(outputs):
        weights = numpy.cos(numpy.arange(len(sizes)) * (output + 1.3)) * 0.5 + 0.5
        columns.append(numpy.clip((grid * weights).sum(axis=1) / weights.sum(), 0, 1))
    samples = (numpy.stack(columns, axis=1).reshape(-1) * ((1 << bits) - 1)).round().astype(int)
    bit_string = "".join(format(value, f"0{bits}b") for value in samples.tolist())
    bit_string += "0" * (-len(bit_string) % 8)
    data = int(bit_string, 2).to_bytes(len(bit_string) // 8, "big") if bit_string else b""
    size_text = b" ".join(b"%d" % size for size in sizes)
    range_text = b" ".join(b"0 1" for _ in range(outputs))
    return stream(
        b"/FunctionType 0 /Domain %s /Range [%s] /Size [%s] /BitsPerSample %d %s"
        % (domain, range_text, size_text, bits, extra),
        data,
    )


def calculator_stream(domain: bytes, outputs: int, program: bytes) -> bytes:
    range_text = b" ".join(b"0 1" for _ in range(outputs))
    return stream(b"/FunctionType 4 /Domain %s /Range [%s]" % (domain, range_text), program)


# A type 1 shading's function of (x, y): rings with a branch, integer rounding and a roll.
RINGS = (
    b"{ 2 copy dup mul exch dup mul add sqrt 12 mul dup floor sub "
    b"dup 0.5 gt { 1 exch sub } if 3 1 roll 2 copy gt { exch } if "
    b"0.5 mul 3 -1 roll 0.25 add 1 index 4 mul cvi 2 mod 1 eq { neg 1 add } if }"
)
# An axial shading's function of t: sine bands, then three components and a fourth from cvi.
BANDS = b"{ dup 720 mul sin 0.5 mul 0.5 add exch dup 3 mul cvi 2 mod 0.4 mul exch dup mul 0 }"
# A Separation tint transform to DeviceCMYK, with a branch on the tint.
SPOT = b"{ dup 0.6 gt { dup 0.6 sub 2.5 mul } { 0 } ifelse exch dup 0.8 mul exch 0.3 mul 0 }"


def function_pages() -> list[tuple[str, bytes]]:
    """Shadings and images whose colours go through sampled and calculator functions."""
    pages: list[tuple[str, bytes]] = []
    two_in = b"[-1 1 -1 1]"
    type1_functions = [
        ("calculator", calculator_stream(two_in, 3, RINGS), b"/DeviceRGB"),
        ("sampled 8", sampled_stream(two_in, [5, 4], 3, 8, b""), b"/DeviceRGB"),
        (
            "sampled 16",
            sampled_stream(two_in, [9, 3], 4, 16, b"/Decode [0 1 1 0 0 1 0.2 0.8]"),
            b"/DeviceCMYK",
        ),
        ("sampled 4", sampled_stream(two_in, [3, 3], 1, 4, b"/Encode [0 2 2 0]"), b"/DeviceGray"),
    ]
    for label, function, space in type1_functions:
        for placement in TYPE1_PLACEMENTS:
            shading_object = (
                b"<< /ShadingType 1 /ColorSpace %s /Domain [-1 1 -1 1] "
                b"/Matrix [60 0 0 60 100 100] /Function 6 0 R >>" % space
            )
            source = page_with(
                b"/Shading << /S 5 0 R >>",
                b"q 0 0 200 200 re W n %s cm /S sh Q" % placement,
                [shading_object, function],
            )
            pages.append((f"type 1 {label} under {placement!r}", source))
    one_in = b"[0 1]"
    separation = b"[/Separation /Spot /DeviceCMYK 7 0 R]"
    axial_functions = [
        ("calculator", calculator_stream(one_in, 4, BANDS), b"/DeviceCMYK", b"<< >>"),
        ("sampled 12", sampled_stream(one_in, [6], 3, 12, b""), b"/DeviceRGB", b"<< >>"),
        ("sampled 1", sampled_stream(one_in, [7], 1, 1, b""), b"/DeviceGray", b"<< >>"),
        (
            "separation",
            sampled_stream(one_in, [33], 1, 8, b""),
            separation,
            calculator_stream(one_in, 4, SPOT),
        ),
    ]
    for label, function, space, tint in axial_functions:
        for shading_type in (2, 3):
            coords = b"[10 10 190 170]" if shading_type == 2 else b"[90 90 5 110 110 120]"
            shading_object = (
                b"<< /ShadingType %d /ColorSpace %s /Coords %s /Extend [true true] "
                b"/Function 6 0 R >>" % (shading_type, space, coords)
            )
            source = page_with(
                b"/Shading << /S 5 0 R >>",
                b"q 0 0 200 200 re W n 0.9 0.2 -0.2 0.9 12 8 cm /S sh Q",
                [shading_object, function, tint],
            )
            pages.append((f"type {shading_type} {label}", source))
    # A luminosity soft mask of 40 grays, through a transfer function.
    grays = b" ".join(b"%.3f g %d 0 5 200 re f" % (index / 39, index * 5) for index in range(40))
    group = stream(
        b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] "
        b"/Group << /S /Transparency /CS /DeviceGray >>",
        grays,
    )
    transfers = [
        (
            "calculator",
            calculator_stream(one_in, 1, b"{ dup mul 1 exch sub dup 0.5 lt { 2 div } if }"),
        ),
        ("sampled 16", sampled_stream(one_in, [11], 1, 16, b"/Decode [1 0]")),
    ]
    for label, transfer in transfers:
        source = page_with(
            b"/ExtGState << /GS << /SMask << /S /Luminosity /G 5 0 R /TR 6 0 R >> >> >>",
            b"/GS gs 0.9 0.2 0.1 rg 0 0 200 200 re f",
            [group, transfer],
        )
        pages.append((f"soft mask transfer {label}", source))
    ramp = bytes((x * 4 + y) % 256 for y in range(64) for x in range(64))
    images = [
        (
            "separation calculator",
            b"[/Separation /Spot /DeviceCMYK 6 0 R]",
            calculator_stream(one_in, 4, SPOT),
            ramp,
        ),
        (
            "devicen sampled",
            b"[/DeviceN [/A /B] /DeviceRGB 6 0 R]",
            sampled_stream(b"[0 1 0 1]", [4, 5], 3, 8, b""),
            bytes(value for pair in zip(ramp, ramp[::-1], strict=True) for value in pair),
        ),
        (
            "devicen calculator",
            b"[/DeviceN [/A /B /C] /DeviceCMYK 6 0 R]",
            calculator_stream(
                b"[0 1 0 1 0 1]", 4, b"{ 3 copy add add 3 div 4 1 roll 0.5 gt { exch } if 0.1 }"
            ),
            bytes(
                value
                for triple in zip(ramp, ramp[::-1], ramp[::3] * 3, strict=False)
                for value in triple
            ),
        ),
        (
            "devicen 12 colorants sampled",
            b"[/DeviceN [%s] /DeviceRGB 6 0 R]"
            % b" ".join(b"/C%d" % index for index in range(WIDE_INPUTS)),
            sampled_stream(
                b"[%s]" % b" ".join([b"0 1"] * WIDE_INPUTS), [2] * WIDE_INPUTS, 3, 8, b""
            ),
            wide_image_samples(),
        ),
    ]
    for label, space, function, data in images:
        components = len(data) // (64 * 64)
        image = stream(
            b"/Type /XObject /Subtype /Image /Width 64 /Height 64 /ColorSpace %s "
            b"/BitsPerComponent 8" % space,
            data[: 64 * 64 * components],
        )
        source = page_with(
            b"/XObject << /Im 5 0 R >>", b"q 180 0 0 180 10 10 cm /Im Do Q", [image, function]
        )
        pages.append((f"image {label}", source))
    return pages


def rendered(source: bytes) -> bytes:
    """The page's pixels, or the error rendering it raised."""
    with PdfDocument(source) as pdf:
        try:
            return bytes(pdf.pages[0].render().rasterize().pixels)
        except Exception as error:  # noqa: BLE001 - the error must be the same both ways
            return f"{type(error).__name__}: {error}".encode()


@contextmanager
def without_batches() -> Generator[None]:
    """Find every colour one at a time, as before the batch forms."""
    modules = (raster_shading, raster_support, shading, image_samples)
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
    generators: list[tuple[Callable[[numpy.random.Generator], Any], Callable[..., Inputs]]] = [
        (random_function, inputs_for),
        (sampled, sampled_inputs),
        (calculator, calculator_inputs),
    ]
    for make, inputs in generators:
        for _ in range(count):
            function = make(random)
            found, issues = check_function(function, inputs(random, function))
            compared += found
            problems += issues
    problems += check_bit_depths()
    found, issues = check_wide_sampled(random)
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
    for label, source in function_pages():
        with_batch = rendered(source)
        with without_batches():
            one_by_one = rendered(source)
        pages += 1
        if with_batch != one_by_one:
            problems.append(f"page: {label} differs")
    for problem in problems[:20]:
        print(problem)
    print(
        f"{3 * count} random functions ({compared} batches) and {pages} pages: "
        f"{len(problems)} failures"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

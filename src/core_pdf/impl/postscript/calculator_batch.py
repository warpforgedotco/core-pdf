# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 7.10.5: PostScript calculator (Type 4) functions run over many inputs at once.

A calculator program is run once over a whole batch of inputs instead of once for each: every
operand stack entry holds one numpy array, a value for each input, all of one kind (a
boolean, an integer or a real), as the one-at-a-time interpreter of calculator.py keeps a
Python bool, int or float. Where the inputs part ways (a condition true for some and false
for others, an integer sum that leaves the integer range for some, a copy, index or roll
count that differs) the batch splits into groups that go on separately, so each input meets
the operators and operand kinds that it meets alone. Each operator gives what calculator.py
gives bit for bit: the arithmetic numpy does in IEEE double precision as Python does, and
the operators whose results numpy may round differently (sin, cos, atan, exp, ln and log)
call the scalar operator on each value. The batch raises ValueError where one of the inputs
would raise alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

lazy import numpy

from core_pdf.impl.postscript.calculator import (
    INT_MASK,
    INT_MAX,
    INT_MIN,
    STACK_LIMIT,
    Conditional,
    Instruction,
    Operand,
    binary,
    unary,
)

# An operand stack entry: its kind ("b" for booleans, "i" for integers, "f" for reals) and
# its values, one for each input of the group.
type Entry = tuple[str, Any]
# A group of the batch: the positions of its inputs in the batch, and its operand stack.
type Group = tuple[Any, list[Entry]]
# An operator's result for a group: parts that each cover some of its inputs (None for all
# of them), with that part's kind and values (given for every input of the group).
type Parts = list[tuple[Any, str, Any]]

KIND_OF: dict[type, str] = {bool: "b", int: "i", float: "f"}
DTYPE_OF = {"b": "bool", "i": "int64", "f": "float64"}
# The scalar operators whose results numpy's own functions may round otherwise.
SCALAR_UNARY = frozenset(("sin", "cos", "ln", "log"))
SCALAR_BINARY = frozenset(("atan", "exp"))


def failure(message: str = "calculator batch input fails one at a time") -> ValueError:
    return ValueError(message)


def pop(stack: list[Entry]) -> Entry:
    if not stack:
        raise failure("calculator operand stack underflow")
    return stack.pop()


def numeric(entry: Entry) -> Any:
    kind, values = entry
    if kind == "b":
        raise failure("calculator operator requires numeric operands")
    return values


def integers(entry: Entry) -> Any:
    kind, values = entry
    if kind != "i":
        raise failure("calculator operator requires integer operands")
    return values


def integer_parts(values: Any) -> Parts:
    """What promote gives an integer result: an integer, or a real outside the 32-bit range."""
    inside = (values >= INT_MIN) & (values <= INT_MAX)
    if inside.all():
        return [(None, "i", values)]
    reals = values.astype(numpy.float64)
    if not inside.any():
        return [(None, "f", reals)]
    return [(inside, "i", values), (~inside, "f", reals)]


def real_parts(values: Any) -> Parts:
    """What promote gives a real result, which it requires to be finite."""
    if not numpy.isfinite(values).all():
        raise failure("calculator arithmetic exceeds real representation")
    return [(None, "f", values)]


def scalar_parts(function: Callable[..., Any], *entries: Entry) -> Parts:
    """function on each input's Python operand values, as the one-at-a-time interpreter
    calls it, with its results grouped by kind."""
    columns = [values.tolist() for _, values in entries]
    try:
        results = [function(*operands) for operands in zip(*columns, strict=True)]
    except (ArithmeticError, TypeError, ValueError) as error:
        raise failure("undefined calculator arithmetic result") from error
    kinds = [KIND_OF[type(result)] for result in results]
    present = sorted(set(kinds))
    if len(present) == 1:
        kind = present[0]
        return [(None, kind, numpy.asarray(results, dtype=DTYPE_OF[kind]))]
    labels = numpy.asarray(kinds)
    parts: Parts = []
    for kind in present:
        mask = labels == kind
        values = numpy.zeros(len(results), dtype=DTYPE_OF[kind])
        values[mask] = [result for result, chosen in zip(results, mask.tolist()) if chosen]
        parts.append((mask, kind, values))
    return parts


def subgroup(rows: Any, stack: list[Entry], mask: Any) -> Group:
    return rows[mask], [(kind, values[mask]) for kind, values in stack]


def push_parts(rows: Any, stack: list[Entry], parts: Parts) -> list[Group]:
    if len(parts) == 1:
        _, kind, values = parts[0]
        stack.append((kind, values))
        return [(rows, stack)]
    groups: list[Group] = []
    for mask, kind, values in parts:
        if mask.any():
            part_rows, part_stack = subgroup(rows, stack, mask)
            part_stack.append((kind, values[mask]))
            groups.append((part_rows, part_stack))
    return groups


def arithmetic(name: str, left: Entry, right: Entry) -> Parts:
    first, second = numeric(left), numeric(right)
    if name in SCALAR_BINARY:
        return scalar_parts(lambda a, b: binary(name, a, b), left, right)
    if name in {"idiv", "mod"}:
        numerator, denominator = integers(left), integers(right)
        if not denominator.all():
            raise failure("undefined calculator arithmetic result")
        quotient = numpy.abs(numerator) // numpy.abs(denominator)
        quotient = numpy.where((numerator < 0) != (denominator < 0), -quotient, quotient)
        if name == "mod":
            return [(None, "i", numerator - quotient * denominator)]
        if ((quotient < INT_MIN) | (quotient > INT_MAX)).any():
            raise failure("calculator idiv exceeds integer representation")
        return [(None, "i", quotient)]
    with numpy.errstate(all="ignore"):
        if name == "div":
            if not second.all():
                raise failure("undefined calculator arithmetic result")
            return real_parts(numpy.true_divide(first, second, dtype=numpy.float64))
        both_integers = left[0] == "i" and right[0] == "i"
        if name == "add":
            values = first + second
        elif name == "sub":
            values = first - second
        else:
            values = first * second
        return integer_parts(values) if both_integers else real_parts(values)


def unary_parts(name: str, entry: Entry) -> Parts:
    kind, values = entry
    numeric(entry)
    if name in SCALAR_UNARY:
        return scalar_parts(lambda value: unary(name, value), entry)
    if name in {"abs", "neg"}:
        found = numpy.abs(values) if name == "abs" else numpy.negative(values)
        return integer_parts(found) if kind == "i" else real_parts(found)
    if name == "cvr":
        return [(None, "f", values.astype(numpy.float64))]
    if kind == "i" and name in {"cvi", "ceiling", "floor", "round", "truncate"}:
        return [(None, "i", values)]
    if name == "cvi":
        truncated = numpy.trunc(values)
        if ((truncated < INT_MIN) | (truncated > INT_MAX)).any():
            raise failure("calculator cvi exceeds integer representation")
        return [(None, "i", truncated.astype(numpy.int64))]
    if name == "sqrt":
        if (values < 0).any():
            raise failure("undefined calculator arithmetic result")
        return [(None, "f", numpy.sqrt(values.astype(numpy.float64)))]
    if name == "ceiling":
        rounded = numpy.ceil(values)
    elif name == "floor":
        rounded = numpy.floor(values)
    elif name == "truncate":
        rounded = numpy.trunc(values)
    else:
        lower = numpy.floor(values)
        rounded = lower + (values - lower >= 0.5)
    # float(math.floor(-0.5) + 1) is 0.0, where numpy's rounding keeps the sign of -0.0.
    return [(None, "f", rounded + 0.0)]


def stack_operator(name: str, rows: Any, stack: list[Entry]) -> list[Group]:
    """pop, dup, exch, copy, index and roll; those that take a count from the stack split
    the group by its value, as each input moves its own stack."""
    if name == "pop":
        pop(stack)
        return [(rows, stack)]
    if name == "dup":
        if not stack:
            raise failure("calculator operand stack underflow")
        stack.append(stack[-1])
        return [(rows, stack)]
    if name == "exch":
        if len(stack) < 2:
            raise failure("calculator operand stack underflow")
        stack[-2], stack[-1] = stack[-1], stack[-2]
        return [(rows, stack)]
    if name == "roll":
        shift_values = integers(pop(stack))
        count_values = integers(pop(stack))
        keys = numpy.stack((count_values, shift_values), axis=1)
    else:
        count_values = integers(pop(stack))
        keys = count_values[:, None]
    groups: list[Group] = []
    if len(rows) and (keys == keys[0]).all():
        splits = [(None, keys[0].tolist())]
    else:
        distinct, inverse = numpy.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        splits = [(inverse == index, key) for index, key in enumerate(distinct.tolist())]
    for mask, key in splits:
        part_rows, part_stack = (rows, stack) if mask is None else subgroup(rows, stack, mask)
        part_stack = list(part_stack)
        depth = len(part_stack)
        if name == "copy":
            count = key[0]
            if not 0 <= count <= depth:
                raise failure("invalid calculator copy count")
            if depth + count > STACK_LIMIT:
                raise failure("calculator operand stack limit exceeded")
            if count:
                part_stack.extend(part_stack[-count:])
        elif name == "index":
            index = key[0]
            if not 0 <= index < depth:
                raise failure("invalid calculator index")
            part_stack.append(part_stack[-index - 1])
        else:
            count, shift = key
            if not 0 <= count <= depth:
                raise failure("invalid calculator roll count")
            if count and (shift := shift % count):
                moved = part_stack[-count:]
                part_stack[-count:] = moved[-shift:] + moved[:-shift]
        groups.append((part_rows, part_stack))
    return groups


def logical(name: str, rows: Any, stack: list[Entry]) -> list[Group]:
    if name == "not":
        kind, values = pop(stack)
        if kind == "b":
            stack.append(("b", ~values))
        else:
            stack.append(("i", ~integers((kind, values))))
        return [(rows, stack)]
    right, left = pop(stack), pop(stack)
    if name in {"and", "or", "xor"}:
        if left[0] == "b" and right[0] == "b":
            kind = "b"
        else:
            kind = "i"
            integers(left)
            integers(right)
        if name == "and":
            values = left[1] & right[1]
        elif name == "or":
            values = left[1] | right[1]
        else:
            values = left[1] ^ right[1]
        stack.append((kind, values))
        return [(rows, stack)]
    if name == "bitshift":
        shift, value = integers(right), integers(left)
        bits = value & INT_MASK
        up = (bits << numpy.clip(shift, 0, 31)) & INT_MASK
        down = bits >> numpy.clip(-shift, 0, 31)
        bits = numpy.where(numpy.abs(shift) >= 32, 0, numpy.where(shift >= 0, up, down))
        stack.append(("i", numpy.where(bits > INT_MAX, bits - (1 << 32), bits)))
        return [(rows, stack)]
    if name in {"eq", "ne"}:
        if (left[0] == "b") != (right[0] == "b"):
            equal = numpy.zeros(len(rows), dtype=numpy.bool_)
        else:
            equal = left[1] == right[1]
        stack.append(("b", equal if name == "eq" else ~equal))
        return [(rows, stack)]
    first, second = numeric(left), numeric(right)
    if name == "gt":
        found = first > second
    elif name == "ge":
        found = first >= second
    elif name == "lt":
        found = first < second
    else:
        found = first <= second
    stack.append(("b", found))
    return [(rows, stack)]


STACK_OPERATORS = frozenset(("pop", "dup", "exch", "copy", "index", "roll"))
LOGICAL_OPERATORS = frozenset(
    ("not", "and", "or", "xor", "bitshift", "eq", "ne", "gt", "ge", "lt", "le")
)
BINARY_OPERATORS = frozenset(("add", "sub", "mul", "div", "idiv", "mod", "atan", "exp"))


def apply(instruction: str, rows: Any, stack: list[Entry]) -> list[Group]:
    if instruction in STACK_OPERATORS:
        return stack_operator(instruction, rows, stack)
    if instruction in LOGICAL_OPERATORS:
        return logical(instruction, rows, stack)
    if instruction in BINARY_OPERATORS:
        right, left = pop(stack), pop(stack)
        return push_parts(rows, stack, arithmetic(instruction, left, right))
    return push_parts(rows, stack, unary_parts(instruction, pop(stack)))


def literal(value: Operand, count: int) -> Entry:
    kind = KIND_OF[type(value)]
    return kind, numpy.full(count, value, dtype=DTYPE_OF[kind])


def run(block: Sequence[Instruction], groups: list[Group]) -> list[Group]:
    for instruction in block:
        following: list[Group] = []
        for rows, stack in groups:
            if isinstance(instruction, Conditional):
                kind, condition = pop(stack)
                if kind != "b":
                    raise failure("calculator condition requires a boolean")
                if condition.all():
                    found = run(instruction.when_true, [(rows, stack)])
                elif not condition.any():
                    found = run(instruction.when_false, [(rows, stack)])
                else:
                    found = run(instruction.when_true, [subgroup(rows, stack, condition)]) + run(
                        instruction.when_false, [subgroup(rows, stack, ~condition)]
                    )
            elif isinstance(instruction, str):
                found = apply(instruction, rows, stack)
            else:
                stack.append(literal(instruction, len(rows)))
                found = [(rows, stack)]
            for _, found_stack in found:
                if len(found_stack) > STACK_LIMIT:
                    raise failure("calculator operand stack limit exceeded")
            following.extend(found)
        groups = following
    return groups


def clipped(values: Any, lower: float, upper: float) -> Any:
    """max(lower, min(upper, value)) of each value, as Python finds it, -0.0 and NaN included:
    min keeps its first argument unless the second is less, and max unless it is greater."""
    below = numpy.where(values < upper, values, upper)
    return numpy.where(below > lower, below, lower)


def batch_calculator(
    program: tuple[Instruction, ...],
    domains: Sequence[tuple[float, float]],
    ranges: Sequence[tuple[float, float]],
) -> Callable[[Any], Any] | None:
    """The batch form of a compiled calculator program: from a float64 array of inputs, (n,)
    for one input or (n, m) for m, to a float64 array of outputs, (n, len(ranges)), equal to
    calling it on each input. None when a bound is not a float, which can make an input an
    integer where it is clipped."""
    bounds = [*domains, *ranges]
    if any(type(lower) is not float or type(upper) is not float for lower, upper in bounds):
        return None
    input_count = len(domains)

    def batch(inputs: Any) -> Any:
        values = numpy.asarray(inputs, dtype=numpy.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != input_count:
            raise failure("invalid calculator input count")
        if not numpy.isfinite(values).all():
            raise failure("invalid calculator input")
        count = len(values)
        outputs = numpy.empty((count, len(ranges)), dtype=numpy.float64)
        if not count:
            return outputs
        stack: list[Entry] = [
            ("f", clipped(values[:, index], lower, upper))
            for index, (lower, upper) in enumerate(domains)
        ]
        for rows, found in run(program, [(numpy.arange(count), stack)]):
            if len(found) != len(ranges):
                raise failure("invalid calculator output count")
            for index, (entry, (lower, upper)) in enumerate(zip(found, ranges, strict=True)):
                outputs[rows, index] = clipped(numeric(entry).astype(numpy.float64), lower, upper)
        return outputs

    return batch


__all__ = ("batch_calculator",)

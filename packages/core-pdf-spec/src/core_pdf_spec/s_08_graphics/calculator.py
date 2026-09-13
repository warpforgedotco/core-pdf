# SPDX-License-Identifier: AGPL-3.0-only
"""Restricted Type 4 expressions, ISO 32000-1/2 7.10.5 and PLRM 3rd ed. 8.2."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    require_pdf_number,
    require_pdf_number_array,
)

internal_STACK_LIMIT = 100
internal_NESTING_LIMIT = 255
internal_INT_MIN = -(1 << 31)
internal_INT_MAX = (1 << 31) - 1
internal_INT_MASK = (1 << 32) - 1
internal_NUMBER = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\Z")
internal_WHITESPACE = b"\x00\t\n\f\r "
internal_UNARY = frozenset(
    (
        "abs",
        "ceiling",
        "floor",
        "round",
        "truncate",
        "sqrt",
        "sin",
        "cos",
        "ln",
        "log",
        "neg",
        "cvi",
        "cvr",
    )
)
internal_BINARY = frozenset(("add", "sub", "mul", "div", "idiv", "mod", "atan", "exp"))
internal_OPERATORS = (
    internal_UNARY
    | internal_BINARY
    | frozenset(
        (
            "eq",
            "ne",
            "gt",
            "ge",
            "lt",
            "le",
            "and",
            "or",
            "xor",
            "not",
            "bitshift",
            "copy",
            "exch",
            "pop",
            "dup",
            "index",
            "roll",
        )
    )
)

type internal_Operand = int | float | bool
type internal_Instruction = internal_Operand | str | internal_Conditional


@dataclass(frozen=True, slots=True)
class internal_Conditional:
    when_true: tuple[internal_Instruction, ...]
    when_false: tuple[internal_Instruction, ...] = ()


def internal_tokens(source: bytes) -> Iterator[bytes]:
    position = 0
    while position < len(source):
        byte = source[position]
        if byte in internal_WHITESPACE:
            position += 1
        elif byte == 37:  # PDF comments end at either line-ending character.
            position += 1
            while position < len(source) and source[position] not in b"\r\n":
                position += 1
        elif byte in b"{}":
            yield source[position : position + 1]
            position += 1
        else:
            start = position
            while position < len(source) and source[position] not in internal_WHITESPACE + b"{}%":
                position += 1
            yield source[start:position]


def internal_parse(source: bytes) -> tuple[internal_Instruction, ...]:
    tokens = iter(internal_tokens(source))

    def block(depth: int) -> tuple[internal_Instruction, ...]:
        if depth > internal_NESTING_LIMIT:
            raise ValueError("calculator brace nesting limit exceeded")
        instructions: list[internal_Instruction] = []
        for token in tokens:
            if token == b"}":
                return tuple(instructions)
            if token == b"{":
                when_true = block(depth + 1)
                operator = next(tokens, None)
                when_false: tuple[internal_Instruction, ...] = ()
                if operator == b"{":
                    when_false = block(depth + 1)
                    if next(tokens, None) != b"ifelse":
                        raise ValueError("calculator blocks require ifelse")
                elif operator != b"if":
                    raise ValueError("calculator block requires if")
                instructions.append(internal_Conditional(when_true, when_false))
            elif token in {b"true", b"false"}:
                instructions.append(token == b"true")
            elif internal_NUMBER.fullmatch(token):
                number = float(token)
                if not math.isfinite(number):
                    raise ValueError("calculator number exceeds real representation")
                # Preserve integer operands within this implementation's
                # signed 32-bit representation; larger literals use reals.
                instructions.append(
                    int(number)
                    if b"." not in token and internal_INT_MIN <= number <= internal_INT_MAX
                    else number
                )
            else:
                try:
                    name = token.decode("ascii")
                except UnicodeDecodeError as error:
                    raise ValueError("invalid calculator token") from error
                if name not in internal_OPERATORS:
                    raise ValueError(f"unsupported calculator token: {name!r}")
                instructions.append(name)
        raise ValueError("unterminated calculator block")

    if next(tokens, None) != b"{":
        raise ValueError("calculator function requires outer braces")
    program = block(1)
    if next(tokens, None) is not None:
        raise ValueError("trailing calculator function content")
    return program


def internal_integer(value: internal_Operand) -> int:
    if type(value) is not int:
        raise ValueError("calculator operator requires integer operands")
    return value


def internal_number(value: internal_Operand) -> int | float:
    if isinstance(value, bool):
        raise ValueError("calculator operator requires numeric operands")
    return value


def internal_promote(value: int | float) -> int | float:
    if not math.isfinite(value):
        raise ValueError("calculator arithmetic exceeds real representation")
    if isinstance(value, int) and not internal_INT_MIN <= value <= internal_INT_MAX:
        return float(value)
    return value


def internal_unary(operator: str, value: internal_Operand) -> int | float:
    number = internal_number(value)
    if operator == "abs":
        return internal_promote(abs(number))
    if operator == "neg":
        return internal_promote(-number)
    if operator == "cvr":
        return float(number)
    if operator == "cvi":
        integer = math.trunc(number)
        if not internal_INT_MIN <= integer <= internal_INT_MAX:
            raise ValueError("calculator cvi exceeds integer representation")
        return integer
    if operator in {"ceiling", "floor", "round", "truncate"}:
        if isinstance(number, int):
            return number
        if operator == "ceiling":
            rounded = math.ceil(number)
        elif operator == "floor":
            rounded = math.floor(number)
        elif operator == "truncate":
            rounded = math.trunc(number)
        else:
            # PLRM round chooses the greater integer at a half, including
            # negative halves, and retains the original numeric type.
            lower = math.floor(number)
            rounded = lower + (number - lower >= 0.5)
        return float(rounded)
    if operator == "sqrt":
        return math.sqrt(number)
    if operator == "sin":
        return math.sin(math.radians(number))
    if operator == "cos":
        return math.cos(math.radians(number))
    if operator == "ln":
        return math.log(number)
    return math.log10(number)


def internal_binary(operator: str, left: internal_Operand, right: internal_Operand) -> int | float:
    first, second = internal_number(left), internal_number(right)
    if operator == "add":
        return internal_promote(first + second)
    if operator == "sub":
        return internal_promote(first - second)
    if operator == "mul":
        return internal_promote(first * second)
    if operator == "div":
        return internal_promote(first / second)
    if operator in {"idiv", "mod"}:
        numerator, denominator = internal_integer(left), internal_integer(right)
        quotient = abs(numerator) // abs(denominator)
        if (numerator < 0) != (denominator < 0):
            quotient = -quotient
        if operator == "mod":
            return numerator - quotient * denominator
        if not internal_INT_MIN <= quotient <= internal_INT_MAX:
            raise ValueError("calculator idiv exceeds integer representation")
        return quotient
    if operator == "atan":
        if first == second == 0:
            raise ValueError("calculator atan is undefined at zero")
        return math.degrees(math.atan2(first, second)) % 360.0
    return internal_promote(math.pow(first, second))


def internal_operator(operator: str, stack: list[internal_Operand]) -> None:
    if operator == "pop":
        stack.pop()
    elif operator == "dup":
        stack.append(stack[-1])
    elif operator == "exch":
        stack[-2], stack[-1] = stack[-1], stack[-2]
    elif operator == "copy":
        count = internal_integer(stack.pop())
        if not 0 <= count <= len(stack):
            raise ValueError("invalid calculator copy count")
        if len(stack) + count > internal_STACK_LIMIT:
            raise ValueError("calculator operand stack limit exceeded")
        if count:
            stack.extend(stack[-count:])
    elif operator == "index":
        index = internal_integer(stack.pop())
        if not 0 <= index < len(stack):
            raise ValueError("invalid calculator index")
        stack.append(stack[-index - 1])
    elif operator == "roll":
        shift = internal_integer(stack.pop())
        count = internal_integer(stack.pop())
        if not 0 <= count <= len(stack):
            raise ValueError("invalid calculator roll count")
        if count and (shift := shift % count):
            values = stack[-count:]
            stack[-count:] = values[-shift:] + values[:-shift]
    elif operator == "not":
        value = stack.pop()
        stack.append(not value if isinstance(value, bool) else ~internal_integer(value))
    elif operator in {"and", "or", "xor"}:
        right, left = stack.pop(), stack.pop()
        if not (isinstance(left, bool) and isinstance(right, bool)):
            left, right = internal_integer(left), internal_integer(right)
        stack.append(
            left & right
            if operator == "and"
            else left | right
            if operator == "or"
            else left ^ right
        )
    elif operator == "bitshift":
        shift, value = internal_integer(stack.pop()), internal_integer(stack.pop())
        bits = value & internal_INT_MASK
        if abs(shift) >= 32:
            bits = 0
        else:
            bits = (bits << shift) & internal_INT_MASK if shift >= 0 else bits >> -shift
        stack.append(bits if bits <= internal_INT_MAX else bits - (1 << 32))
    elif operator in {"eq", "ne"}:
        right, left = stack.pop(), stack.pop()
        equal = isinstance(left, bool) == isinstance(right, bool) and left == right
        stack.append(equal if operator == "eq" else not equal)
    elif operator in {"gt", "ge", "lt", "le"}:
        right, left = internal_number(stack.pop()), internal_number(stack.pop())
        stack.append(
            left > right
            if operator == "gt"
            else left >= right
            if operator == "ge"
            else left < right
            if operator == "lt"
            else left <= right
        )
    elif operator in internal_UNARY:
        stack.append(internal_unary(operator, stack.pop()))
    else:
        right, left = stack.pop(), stack.pop()
        stack.append(internal_binary(operator, left, right))


def internal_bounds(value: object, kind: str) -> tuple[tuple[float, float], ...]:
    values = require_pdf_number_array(value, f"invalid calculator {kind}")
    if not values or len(values) % 2:
        raise ValueError(f"invalid calculator {kind}")
    bounds = tuple(zip(values[::2], values[1::2], strict=True))
    if any(lower > upper for lower, upper in bounds):
        raise ValueError(f"invalid calculator {kind}")
    return bounds


def internal_compile_calculator_function(function: PdfStream) -> Callable[..., tuple[float, ...]]:
    """Compile once, using 32-bit integers, finite reals and a 100-entry stack.

    Braces form conditional syntax only: blocks cannot be put on the operand
    stack or called indirectly. Execution therefore visits each instruction at
    most once, and the 255-level syntax limit also bounds execution nesting.
    """
    domains = internal_bounds(function.dictionary.get("Domain"), "domain")
    ranges = internal_bounds(function.dictionary.get("Range"), "range")
    if max(len(domains), len(ranges)) > internal_STACK_LIMIT:
        raise ValueError("calculator operand stack limit exceeded")
    program = internal_parse(function.data)

    def evaluate(*inputs: float) -> tuple[float, ...]:
        if len(inputs) != len(domains):
            raise ValueError("invalid calculator input count")
        stack: list[internal_Operand] = [
            max(lower, min(upper, require_pdf_number(value, "invalid calculator input")))
            for value, (lower, upper) in zip(inputs, domains, strict=True)
        ]
        frames = [iter(program)]
        try:
            while frames:
                instruction = next(frames[-1], None)
                if instruction is None:
                    frames.pop()
                elif isinstance(instruction, internal_Conditional):
                    condition = stack.pop()
                    if not isinstance(condition, bool):
                        raise ValueError("calculator condition requires a boolean")
                    frames.append(
                        iter(instruction.when_true if condition else instruction.when_false)
                    )
                elif isinstance(instruction, str):
                    internal_operator(instruction, stack)
                else:
                    stack.append(instruction)
                if len(stack) > internal_STACK_LIMIT:
                    raise ValueError("calculator operand stack limit exceeded")
        except IndexError as error:
            raise ValueError("calculator operand stack underflow") from error
        except ArithmeticError as error:
            raise ValueError("undefined calculator arithmetic result") from error
        if len(stack) != len(ranges):
            raise ValueError("invalid calculator output count")
        return tuple(
            max(lower, min(upper, float(internal_number(value))))
            for value, (lower, upper) in zip(stack, ranges, strict=True)
        )

    return evaluate


__all__ = ()

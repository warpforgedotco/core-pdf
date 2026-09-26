# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Sequence
from typing import ClassVar

from core_records import Record, frozen_setattr

STACK_LIMIT = 100
NESTING_LIMIT = 255
INT_MIN = -(1 << 31)
INT_MAX = (1 << 31) - 1
INT_MASK = (1 << 32) - 1
NUMBER = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\Z")
WHITESPACE = b"\x00\t\n\f\r "
UNARY = frozenset(
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
BINARY = frozenset(("add", "sub", "mul", "div", "idiv", "mod", "atan", "exp"))
OPERATORS = (
    UNARY
    | BINARY
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

type Operand = int | float | bool
type Instruction = Operand | str | Conditional


class Conditional(Record):
    __slots__ = ("when_true", "when_false")

    when_true: tuple[Instruction, ...]
    when_false: tuple[Instruction, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("when_true", "when_false")
    __match_args__ = ("when_true", "when_false")

    def __init__(
        self,
        when_true: tuple[Instruction, ...],
        when_false: tuple[Instruction, ...] = (),
    ) -> None:
        frozen_setattr(self, "when_true", when_true)
        frozen_setattr(self, "when_false", when_false)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.when_true == other.when_true and self.when_false == other.when_false

    def __hash__(self) -> int:
        return hash((self.when_true, self.when_false))


def iter_tokens(source: bytes) -> Iterator[bytes]:
    position = 0
    while position < len(source):
        byte = source[position]
        if byte in WHITESPACE:
            position += 1
        elif byte == 37:
            position += 1
            while position < len(source) and source[position] not in b"\r\n":
                position += 1
        elif byte in b"{}":
            yield source[position : position + 1]
            position += 1
        else:
            start = position
            while position < len(source) and source[position] not in WHITESPACE + b"{}%":
                position += 1
            yield source[start:position]


def parse(source: bytes) -> tuple[Instruction, ...]:
    tokens = iter(iter_tokens(source))

    def block(depth: int) -> tuple[Instruction, ...]:
        if depth > NESTING_LIMIT:
            raise ValueError("calculator brace nesting limit exceeded")
        instructions: list[Instruction] = []
        for token in tokens:
            if token == b"}":
                return tuple(instructions)
            if token == b"{":
                when_true = block(depth + 1)
                operator = next(tokens, None)
                when_false: tuple[Instruction, ...] = ()
                if operator == b"{":
                    when_false = block(depth + 1)
                    if next(tokens, None) != b"ifelse":
                        raise ValueError("calculator blocks require ifelse")
                elif operator != b"if":
                    raise ValueError("calculator block requires if")
                instructions.append(Conditional(when_true, when_false))
            elif token in {b"true", b"false"}:
                instructions.append(token == b"true")
            elif NUMBER.fullmatch(token):
                number = float(token)
                if not math.isfinite(number):
                    raise ValueError("calculator number exceeds real representation")
                instructions.append(
                    int(number) if b"." not in token and INT_MIN <= number <= INT_MAX else number
                )
            else:
                try:
                    name = token.decode("ascii")
                except UnicodeDecodeError as error:
                    raise ValueError("invalid calculator token") from error
                if name not in OPERATORS:
                    raise ValueError(f"unsupported calculator token: {name!r}")
                instructions.append(name)
        raise ValueError("unterminated calculator block")

    if next(tokens, None) != b"{":
        raise ValueError("calculator function requires outer braces")
    program = block(1)
    if next(tokens, None) is not None:
        raise ValueError("trailing calculator function content")
    return program


def integer(value: Operand) -> int:
    if type(value) is not int:
        raise ValueError("calculator operator requires integer operands")
    return value


def require_number(value: Operand) -> int | float:
    if isinstance(value, bool):
        raise ValueError("calculator operator requires numeric operands")
    return value


def promote(value: int | float) -> int | float:
    if not math.isfinite(value):
        raise ValueError("calculator arithmetic exceeds real representation")
    if isinstance(value, int) and not INT_MIN <= value <= INT_MAX:
        return float(value)
    return value


def unary(operator: str, value: Operand) -> int | float:
    number = require_number(value)
    if operator == "abs":
        return promote(abs(number))
    if operator == "neg":
        return promote(-number)
    if operator == "cvr":
        return float(number)
    if operator == "cvi":
        integer = math.trunc(number)
        if not INT_MIN <= integer <= INT_MAX:
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


def binary(operator: str, left: Operand, right: Operand) -> int | float:
    first, second = require_number(left), require_number(right)
    if operator == "add":
        return promote(first + second)
    if operator == "sub":
        return promote(first - second)
    if operator == "mul":
        return promote(first * second)
    if operator == "div":
        return promote(first / second)
    if operator in {"idiv", "mod"}:
        numerator, denominator = integer(left), integer(right)
        quotient = abs(numerator) // abs(denominator)
        if (numerator < 0) != (denominator < 0):
            quotient = -quotient
        if operator == "mod":
            return numerator - quotient * denominator
        if not INT_MIN <= quotient <= INT_MAX:
            raise ValueError("calculator idiv exceeds integer representation")
        return quotient
    if operator == "atan":
        if first == second == 0:
            raise ValueError("calculator atan is undefined at zero")
        return math.degrees(math.atan2(first, second)) % 360.0
    return promote(math.pow(first, second))


def operator(operator: str, stack: list[Operand]) -> None:
    match operator:
        case "pop":
            stack.pop()
        case "dup":
            stack.append(stack[-1])
        case "exch":
            stack[-2], stack[-1] = stack[-1], stack[-2]
        case "copy":
            count = integer(stack.pop())
            if not 0 <= count <= len(stack):
                raise ValueError("invalid calculator copy count")
            if len(stack) + count > STACK_LIMIT:
                raise ValueError("calculator operand stack limit exceeded")
            if count:
                stack.extend(stack[-count:])
        case "index":
            index = integer(stack.pop())
            if not 0 <= index < len(stack):
                raise ValueError("invalid calculator index")
            stack.append(stack[-index - 1])
        case "roll":
            shift = integer(stack.pop())
            count = integer(stack.pop())
            if not 0 <= count <= len(stack):
                raise ValueError("invalid calculator roll count")
            if count and (shift := shift % count):
                values = stack[-count:]
                stack[-count:] = values[-shift:] + values[:-shift]
        case "not":
            value = stack.pop()
            stack.append(not value if isinstance(value, bool) else ~integer(value))
        case "and" | "or" | "xor":
            right, left = stack.pop(), stack.pop()
            if not (isinstance(left, bool) and isinstance(right, bool)):
                left, right = integer(left), integer(right)
            stack.append(
                left & right
                if operator == "and"
                else left | right
                if operator == "or"
                else left ^ right
            )
        case "bitshift":
            shift, value = integer(stack.pop()), integer(stack.pop())
            bits = value & INT_MASK
            if abs(shift) >= 32:
                bits = 0
            else:
                bits = (bits << shift) & INT_MASK if shift >= 0 else bits >> -shift
            stack.append(bits if bits <= INT_MAX else bits - (1 << 32))
        case "eq" | "ne":
            right, left = stack.pop(), stack.pop()
            equal = isinstance(left, bool) == isinstance(right, bool) and left == right
            stack.append(equal if operator == "eq" else not equal)
        case "gt" | "ge" | "lt" | "le":
            right, left = require_number(stack.pop()), require_number(stack.pop())
            stack.append(
                left > right
                if operator == "gt"
                else left >= right
                if operator == "ge"
                else left < right
                if operator == "lt"
                else left <= right
            )
        case _ if operator in UNARY:
            stack.append(unary(operator, stack.pop()))
        case _:
            right, left = stack.pop(), stack.pop()
            stack.append(binary(operator, left, right))


def require_real(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("invalid calculator input")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("invalid calculator input") from error
    if not math.isfinite(result):
        raise ValueError("invalid calculator input")
    return result


def bounds(value: Sequence[tuple[float, float]], kind: str) -> tuple[tuple[float, float], ...]:
    bounds = tuple((lower, upper) for lower, upper in value)
    if not bounds:
        raise ValueError(f"invalid calculator {kind}")
    for lower, upper in bounds:
        try:
            ordered = require_real(lower) <= require_real(upper)
        except ValueError:
            raise ValueError(f"invalid calculator {kind}") from None
        if not ordered:
            raise ValueError(f"invalid calculator {kind}")
    return bounds


def compile_calculator(
    source: bytes,
    domains: Sequence[tuple[float, float]],
    ranges: Sequence[tuple[float, float]],
) -> Callable[..., tuple[float, ...]]:
    domains = bounds(domains, "domain")
    ranges = bounds(ranges, "range")
    if max(len(domains), len(ranges)) > STACK_LIMIT:
        raise ValueError("calculator operand stack limit exceeded")
    program = parse(source)

    def evaluate(*inputs: float) -> tuple[float, ...]:
        if len(inputs) != len(domains):
            raise ValueError("invalid calculator input count")
        stack: list[Operand] = [
            max(lower, min(upper, require_real(value)))
            for value, (lower, upper) in zip(inputs, domains, strict=True)
        ]
        frames = [iter(program)]
        try:
            while frames:
                instruction = next(frames[-1], None)
                if instruction is None:
                    frames.pop()
                elif isinstance(instruction, Conditional):
                    condition = stack.pop()
                    if not isinstance(condition, bool):
                        raise ValueError("calculator condition requires a boolean")
                    frames.append(
                        iter(instruction.when_true if condition else instruction.when_false)
                    )
                elif isinstance(instruction, str):
                    operator(instruction, stack)
                else:
                    stack.append(instruction)
                if len(stack) > STACK_LIMIT:
                    raise ValueError("calculator operand stack limit exceeded")
        except IndexError as error:
            raise ValueError("calculator operand stack underflow") from error
        except ArithmeticError as error:
            raise ValueError("undefined calculator arithmetic result") from error
        if len(stack) != len(ranges):
            raise ValueError("invalid calculator output count")
        return tuple(
            max(lower, min(upper, float(require_number(value))))
            for value, (lower, upper) in zip(stack, ranges, strict=True)
        )

    return evaluate


__all__ = ("STACK_LIMIT", "NESTING_LIMIT", "OPERATORS", "compile_calculator")

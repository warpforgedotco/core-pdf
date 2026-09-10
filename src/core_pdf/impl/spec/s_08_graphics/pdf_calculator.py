# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded numeric stack machine for PDF Type 4 calculator functions."""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from core_pdf.impl.spec.s_07_syntax_primitives.scanning import (
    is_integer_word,
    is_number_word_bytes,
)
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import WHITESPACE

Number: TypeAlias = int | float
UnaryOperator: TypeAlias = Callable[[Number], Number]


def internal_round(value: Number) -> int:
    """PostScript round chooses the greater integer at an exact half tie."""
    lower = math.floor(value)
    return lower + (value - lower >= 0.5)


def internal_atan(numerator: Number, denominator: Number) -> float:
    if numerator == denominator == 0:
        raise ValueError("undefined calculator angle")
    return math.degrees(math.atan2(numerator, denominator)) % 360


def internal_idiv(dividend: Number, divisor: Number) -> int:
    quotient = math.trunc(dividend / divisor)
    if not -(1 << 31) <= quotient < 1 << 31:
        raise ValueError("calculator integer division out of range")
    return quotient


@dataclass(frozen=True, slots=True)
class internal_Procedure:
    tokens: tuple[internal_Token, ...]


internal_Token: TypeAlias = Number | bool | str | internal_Procedure
internal_TOKEN = re.compile(
    rb"[" + re.escape(WHITESPACE) + rb"]+|%[^\r\n]*|[{}]|[^" + re.escape(WHITESPACE) + rb"{}%]+"
)
internal_UNARY: dict[str, UnaryOperator] = {
    "abs": abs,
    "ceiling": math.ceil,
    "cos": lambda x: math.cos(math.radians(x)),
    "cvi": int,
    "cvr": float,
    "floor": math.floor,
    "ln": math.log,
    "log": math.log10,
    "neg": operator.neg,
    "round": internal_round,
    "sin": lambda x: math.sin(math.radians(x)),
    "sqrt": math.sqrt,
    "truncate": math.trunc,
}
internal_BINARY: dict[str, Callable[[Number, Number], Number]] = {
    "add": operator.add,
    "atan": internal_atan,
    "div": operator.truediv,
    "exp": math.pow,
    "idiv": internal_idiv,
    "mod": lambda x, y: x - math.trunc(x / y) * y,
    "mul": operator.mul,
    "sub": operator.sub,
}
internal_COMPARE: dict[str, Callable[[Number, Number], bool]] = {
    "eq": operator.eq,
    "ge": operator.ge,
    "gt": operator.gt,
    "le": operator.le,
    "lt": operator.lt,
    "ne": operator.ne,
}
internal_OPERATOR_NAMES = (
    frozenset(internal_UNARY)
    | frozenset(internal_BINARY)
    | frozenset(internal_COMPARE)
    | {
        "and",
        "bitshift",
        "false",
        "not",
        "or",
        "true",
        "xor",
        "if",
        "ifelse",
        "copy",
        "dup",
        "exch",
        "index",
        "pop",
        "roll",
    }
)


def internal_validate_calculator(procedure: internal_Procedure) -> None:
    """Braces are conditional syntax, never manipulable procedure operands."""
    pending = [procedure]
    while pending:
        tokens = pending.pop().tokens
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if isinstance(token, internal_Procedure):
                following = tokens[index + 1 : index + 3]
                if following and following[0] == "if":
                    pending.append(token)
                    index += 2
                    continue
                if (
                    len(following) == 2
                    and isinstance(following[0], internal_Procedure)
                    and following[1] == "ifelse"
                ):
                    pending.extend((token, following[0]))
                    index += 3
                    continue
                raise ValueError("calculator procedure requires an adjacent conditional")
            if isinstance(token, str):
                if token not in internal_OPERATOR_NAMES:
                    raise ValueError(f"unsupported calculator operator: {token}")
                if token in {"if", "ifelse"}:
                    raise ValueError("calculator conditional requires braced expressions")
            index += 1


def internal_parse_calculator(data: bytes) -> internal_Procedure:
    if len(data) > 1_048_576:
        raise ValueError("calculator function exceeds program limit")
    frames: list[list[internal_Token]] = [[]]
    for match in internal_TOKEN.finditer(data):
        token = match.group()
        if token[0] in WHITESPACE or token.startswith(b"%"):
            continue
        if token == b"{":
            if len(frames) >= 64:
                raise ValueError("calculator procedure nesting limit exceeded")
            frames.append([])
        elif token == b"}":
            if len(frames) == 1:
                raise ValueError("unbalanced calculator procedure")
            frames[-2].append(internal_Procedure(tuple(frames.pop())))
        else:
            value: internal_Token
            if is_number_word_bytes(token):
                value = int(token) if is_integer_word(token) else float(token)
            else:
                if token[0] in b"+-.0123456789":
                    raise ValueError("invalid calculator numeric literal")
                value = token.decode("ascii")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("nonfinite calculator literal")
            frames[-1].append(value)
    if len(frames) != 1 or len(frames[0]) != 1 or not isinstance(frames[0][0], internal_Procedure):
        raise ValueError("calculator function must contain one procedure")
    internal_validate_calculator(frames[0][0])
    return frames[0][0]


def internal_signed_word(value: int) -> int:
    return ((value + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


class internal_Calculator:
    def __init__(
        self,
        inputs: tuple[float, ...],
        unary_operators: Mapping[str, UnaryOperator] = internal_UNARY,
    ) -> None:
        if len(inputs) > 100:
            raise ValueError("calculator stack limit exceeded")
        self.stack: list[internal_Token] = list(inputs)
        self.fuel = 100_000
        self.unary_operators = unary_operators

    def pop(self) -> internal_Token:
        if not self.stack:
            raise ValueError("calculator stack underflow")
        return self.stack.pop()

    def number(self) -> Number:
        value = self.pop()
        if type(value) not in (int, float):
            raise ValueError("calculator requires a number")
        assert isinstance(value, (int, float))
        return value

    def integer(self) -> int:
        value = self.pop()
        if type(value) is not int:
            raise ValueError("calculator requires an integer")
        assert isinstance(value, int)
        return value

    def push(self, value: internal_Token) -> None:
        if len(self.stack) >= 100:
            raise ValueError("calculator stack limit exceeded")
        if type(value) in (int, float):
            assert isinstance(value, (int, float))
            if not math.isfinite(value):
                raise ValueError("nonfinite calculator result")
            if type(value) is int and not -(1 << 31) <= value < 1 << 31:
                value = float(value)
        self.stack.append(value)

    def execute(self, procedure: internal_Procedure) -> None:
        pending = [iter(procedure.tokens)]
        while pending:
            token = next(pending[-1], None)
            if token is None:
                pending.pop()
                continue
            self.fuel -= 1
            if self.fuel < 0:
                raise ValueError("calculator operation limit exceeded")
            if not isinstance(token, str):
                self.push(token)
            elif token in {"if", "ifelse"}:
                alternative = self.pop() if token == "ifelse" else internal_Procedure(())
                consequent, condition = self.pop(), self.pop()
                if (
                    type(condition) is not bool
                    or not isinstance(consequent, internal_Procedure)
                    or not isinstance(alternative, internal_Procedure)
                ):
                    raise ValueError("invalid calculator conditional")
                pending.append(iter((consequent if condition else alternative).tokens))
            else:
                self.operator(token)

    def operator(self, name: str) -> None:  # noqa: C901 - calculator operator dispatch
        if name in self.unary_operators:
            number_value = self.number()
            result = self.unary_operators[name](number_value)
            if name in {"ceiling", "floor", "round", "truncate"} and type(number_value) is float:
                result = float(result)
            if name == "cvi" and not -(1 << 31) <= result < 1 << 31:
                raise ValueError("calculator integer conversion out of range")
            self.push(result)
        elif name in internal_BINARY:
            right_number, left_number = self.number(), self.number()
            if name in {"idiv", "mod"} and (
                type(left_number) is not int or type(right_number) is not int
            ):
                raise ValueError("calculator integer arithmetic requires integers")
            self.push(internal_BINARY[name](left_number, right_number))
        elif name in internal_COMPARE:
            right, left = self.pop(), self.pop()
            if type(left) not in (int, float, bool) or type(right) not in (int, float, bool):
                raise ValueError("invalid calculator comparison")
            assert isinstance(left, (int, float))
            assert isinstance(right, (int, float))
            if type(left) is bool or type(right) is bool:
                if name not in {"eq", "ne"}:
                    raise ValueError("calculator ordering requires numbers")
                equal = type(left) is type(right) and left == right
                self.push(equal if name == "eq" else not equal)
                return
            self.push(internal_COMPARE[name](left, right))
        elif name in {"true", "false"}:
            self.push(name == "true")
        elif name == "not":
            value = self.pop()
            if type(value) is bool:
                self.push(not value)
            elif type(value) is int:
                self.push(internal_signed_word(~value))
            else:
                raise ValueError("invalid calculator logical operand")
        elif name in {"and", "or", "xor"}:
            right, left = self.pop(), self.pop()
            if type(left) is not type(right) or type(left) not in (int, bool):
                raise ValueError("invalid calculator logical operands")
            assert isinstance(left, int)
            assert isinstance(right, int)
            result = {"and": operator.and_, "or": operator.or_, "xor": operator.xor}[name](
                left, right
            )
            self.push(result if type(left) is bool else internal_signed_word(result))
        elif name == "bitshift":
            shift, value = self.integer(), self.integer()
            shifted = (
                0
                if abs(shift) >= 32
                else (
                    (value & 0xFFFFFFFF) << shift if shift >= 0 else (value & 0xFFFFFFFF) >> -shift
                )
            )
            self.push(internal_signed_word(shifted))
        elif name == "pop":
            self.pop()
        elif name == "dup":
            value = self.pop()
            self.push(value)
            self.push(value)
        elif name == "exch":
            first, second = self.pop(), self.pop()
            self.push(first)
            self.push(second)
        elif name in {"copy", "index", "roll"}:
            shift = self.integer() if name == "roll" else 0
            count = self.integer()
            if (
                count < 0
                or count > len(self.stack)
                or (name == "index" and count == len(self.stack))
            ):
                raise ValueError("invalid calculator stack index")
            if name == "copy" and count:
                for value in self.stack[-count:]:
                    self.push(value)
            elif name == "index":
                self.push(self.stack[-count - 1])
            elif name == "roll" and count:
                shift %= count
                if shift:
                    self.stack[-count:] = self.stack[-shift:] + self.stack[-count:-shift]
        else:
            raise ValueError(f"unsupported calculator operator: {name}")


def compile_calculator_function(
    data: bytes,
    domains: tuple[float, ...],
    ranges: tuple[float, ...],
    *,
    unary_operators: Mapping[str, UnaryOperator] | None = None,
) -> Callable[..., tuple[float, ...]]:
    if not domains or len(domains) % 2 or not ranges or len(ranges) % 2:
        raise ValueError("invalid calculator domain or range")
    if any(
        values[index] > values[index + 1]
        for values in (domains, ranges)
        for index in range(0, len(values), 2)
    ):
        raise ValueError("reversed calculator domain or range")
    procedure = internal_parse_calculator(data)
    selected_unary = internal_UNARY | dict(unary_operators or {})

    def evaluate(*inputs: float) -> tuple[float, ...]:
        if len(inputs) * 2 != len(domains):
            raise ValueError("invalid calculator input count")
        machine = internal_Calculator(
            tuple(
                max(domains[2 * i], min(domains[2 * i + 1], value))
                for i, value in enumerate(inputs)
            ),
            selected_unary,
        )
        try:
            machine.execute(procedure)
        except (ArithmeticError, TypeError) as exc:
            raise ValueError("invalid calculator arithmetic") from exc
        count = len(ranges) // 2
        if len(machine.stack) < count:
            raise ValueError("missing calculator outputs")
        output = [machine.number() for _ in range(count)][::-1]
        return tuple(
            max(ranges[2 * i], min(ranges[2 * i + 1], value)) for i, value in enumerate(output)
        )

    return evaluate

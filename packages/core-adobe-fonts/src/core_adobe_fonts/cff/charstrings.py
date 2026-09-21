# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from math import isfinite, sqrt

from core_adobe_fonts.cff.font import CFFFont

TYPE2_MAX_SUBR_DEPTH = 10
TYPE2_MAX_STACK = 48
TYPE2_TRANSIENT_SIZE = 32


def cubic_extrema_times(p0: float, p1: float, p2: float, p3: float) -> tuple[float, ...]:
    if p0 <= p1 <= p2 <= p3 or p0 >= p1 >= p2 >= p3:
        return ()
    a = -p0 + 3.0 * p1 - 3.0 * p2 + p3
    b = 2.0 * (p0 - 2.0 * p1 + p2)
    c = p1 - p0
    epsilon = 1e-12
    if abs(a) <= epsilon:
        if abs(b) <= epsilon:
            return ()
        root = -c / b
        return (root,) if 0.0 < root < 1.0 else ()
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return ()
    root_delta = sqrt(discriminant)
    roots = ((-b - root_delta) / (2.0 * a), (-b + root_delta) / (2.0 * a))
    return tuple(dict.fromkeys(root for root in roots if 0.0 < root < 1.0))


def cubic_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    mt = 1.0 - t
    mt3 = mt**3
    t3 = t**3
    mt2t = 3.0 * mt * mt * t
    mtt2 = 3.0 * mt * t * t
    return (
        mt3 * p0[0] + mt2t * p1[0] + mtt2 * p2[0] + t3 * p3[0],
        mt3 * p0[1] + mt2t * p1[1] + mtt2 * p2[1] + t3 * p3[1],
    )


def internal_execute_type2_flex(
    operator: int,
    operands: list[float],
    curve: Callable[[float, float, float, float, float, float], None],
) -> None:
    match operator:
        case 34:
            dx1, dx2, dy2, dx3, dx4, dx5, dx6 = operands
            curve(dx1, 0.0, dx2, dy2, dx3, 0.0)
            curve(dx4, 0.0, dx5, -dy2, dx6, 0.0)
        case 35:
            (
                dx1,
                dy1,
                dx2,
                dy2,
                dx3,
                dy3,
                dx4,
                dy4,
                dx5,
                dy5,
                dx6,
                dy6,
                ignored_flex_depth,
            ) = operands
            curve(dx1, dy1, dx2, dy2, dx3, dy3)
            curve(dx4, dy4, dx5, dy5, dx6, dy6)
        case 36:
            dx1, dy1, dx2, dy2, dx3, dx4, dx5, dy5, dx6 = operands
            dy6 = -(dy1 + dy2 + dy5)
            curve(dx1, dy1, dx2, dy2, dx3, 0.0)
            curve(dx4, 0.0, dx5, dy5, dx6, dy6)
        case 37:
            dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5, d6 = operands
            dx = dx1 + dx2 + dx3 + dx4 + dx5
            dy = dy1 + dy2 + dy3 + dy4 + dy5
            if abs(dx) > abs(dy):
                dx6, dy6 = d6, -dy
            else:
                dx6, dy6 = -dx, d6
            curve(dx1, dy1, dx2, dy2, dx3, dy3)
            curve(dx4, dy4, dx5, dy5, dx6, dy6)
        case _:
            raise ValueError("invalid Type 2 flex operator")


def internal_type2_subr_bias(count: int) -> int:
    if count < 1240:
        return 107
    if count < 33900:
        return 1131
    return 32768


def execute_type2_charstring(  # noqa: C901
    charstring: bytes,
    *,
    local_subrs: tuple[bytes, ...],
    global_subrs: tuple[bytes, ...],
    move: Callable[[float, float], None],
    line: Callable[[float, float], None],
    curve: Callable[[float, float, float, float, float, float], None],
    flush_contour: Callable[[], None],
    has_current_point: Callable[[], bool],
    seac: Callable[[int, int, float, float], None],
    random_value: Callable[[], float],
) -> bool:
    stack: list[float] = []
    transient = [0.0] * TYPE2_TRANSIENT_SIZE
    stem_count = 0
    width_resolved = False
    subr_bias = internal_type2_subr_bias(len(local_subrs))
    gsubr_bias = internal_type2_subr_bias(len(global_subrs))

    def push(value: float) -> None:
        if len(stack) >= TYPE2_MAX_STACK or not isfinite(value):
            raise ValueError("invalid Type 2 operand stack")
        stack.append(float(value))

    def require_integer(value: float) -> int:
        integer = int(value)
        if value != integer:
            raise ValueError("Type 2 operator requires an integer")
        return integer

    def pop_integer() -> int:
        return require_integer(stack.pop())

    def execute_escaped_operator(operator: int) -> None:
        match operator:
            case 0:
                stack.clear()
            case 3:
                second = stack.pop()
                first = stack.pop()
                push(float(first != 0.0 and second != 0.0))
            case 4:
                second = stack.pop()
                first = stack.pop()
                push(float(first != 0.0 or second != 0.0))
            case 5:
                push(float(stack.pop() == 0.0))
            case 9:
                push(abs(stack.pop()))
            case 10:
                second = stack.pop()
                push(stack.pop() + second)
            case 11:
                second = stack.pop()
                push(stack.pop() - second)
            case 12:
                second = stack.pop()
                push(stack.pop() / second)
            case 14:
                push(-stack.pop())
            case 15:
                second = stack.pop()
                push(float(stack.pop() == second))
            case 18:
                stack.pop()
            case 20:
                index = pop_integer()
                value = stack.pop()
                if not 0 <= index < len(transient):
                    raise ValueError("invalid Type 2 transient-array index")
                transient[index] = value
            case 21:
                index = pop_integer()
                if not 0 <= index < len(transient):
                    raise ValueError("invalid Type 2 transient-array index")
                push(transient[index])
            case 22:
                value2 = stack.pop()
                value1 = stack.pop()
                choice2 = stack.pop()
                choice1 = stack.pop()
                push(choice1 if value1 <= value2 else choice2)
            case 23:
                push(random_value())
            case 24:
                second = stack.pop()
                push(stack.pop() * second)
            case 26:
                push(sqrt(stack.pop()))
            case 27:
                push(stack[-1])
            case 28:
                stack[-1], stack[-2] = stack[-2], stack[-1]
            case 29:
                index = max(pop_integer(), 0)
                if index >= len(stack):
                    raise ValueError("invalid Type 2 stack index")
                push(stack[-index - 1])
            case 30:
                shift = pop_integer()
                count = pop_integer()
                if count < 0 or count > len(stack):
                    raise ValueError("invalid Type 2 roll count")
                if count:
                    shift %= count
                    if shift:
                        values = stack[-count:]
                        stack[-count:] = values[-shift:] + values[:-shift]
            case 34 | 35 | 36 | 37:
                if not has_current_point():
                    raise ValueError("Type 2 flex operator has no current point")
                internal_execute_type2_flex(operator, stack, curve)
                stack.clear()
            case _:
                raise ValueError("unsupported Type 2 escaped operator")

    def execute(  # noqa: C901
        program: bytes, depth: int = 0
    ) -> bool:
        nonlocal stem_count, width_resolved
        if depth > TYPE2_MAX_SUBR_DEPTH:
            raise ValueError("invalid Type 2 charstring")
        pos = 0
        try:
            while pos < len(program):
                byte = program[pos]
                if byte > 31 or byte in {28, 255}:
                    value, pos = CFFFont.parse_number(program, pos)
                    push(value)
                    continue
                pos += 1
                match byte:
                    case 1 | 3 | 18 | 23:
                        operand_count = len(stack)
                        if not width_resolved and operand_count % 2:
                            operand_count -= 1
                        if operand_count < 2 or operand_count % 2:
                            raise ValueError("invalid Type 2 charstring")
                        stem_count += operand_count // 2
                        if stem_count > 96:
                            raise ValueError("invalid Type 2 charstring")
                        width_resolved = True
                        stack.clear()
                    case 4 | 22:
                        if len(stack) == 1:
                            displacement = stack[0]
                        elif not width_resolved and len(stack) == 2:
                            displacement = stack[1]
                        else:
                            raise ValueError("invalid Type 2 charstring")
                        width_resolved = True
                        if byte == 4:
                            move(0.0, displacement)
                        else:
                            move(displacement, 0.0)
                        stack.clear()
                    case 5:
                        if not has_current_point() or len(stack) < 2 or len(stack) % 2:
                            raise ValueError("invalid Type 2 charstring")
                        for i in range(0, len(stack) - 1, 2):
                            line(stack[i], stack[i + 1])
                        stack.clear()
                    case 6 | 7:
                        if not has_current_point() or not stack:
                            raise ValueError("invalid Type 2 charstring")
                        horizontal = byte == 6
                        for value in stack:
                            line(value, 0.0) if horizontal else line(0.0, value)
                            horizontal = not horizontal
                        stack.clear()
                    case 8:
                        if not has_current_point() or len(stack) < 6 or len(stack) % 6:
                            raise ValueError("invalid Type 2 charstring")
                        for i in range(0, len(stack) - 5, 6):
                            curve(*stack[i : i + 6])
                        stack.clear()
                    case 10 | 29:
                        if not stack:
                            raise ValueError("invalid Type 2 charstring")
                        subrs, bias = (
                            (local_subrs, subr_bias) if byte == 10 else (global_subrs, gsubr_bias)
                        )
                        subr_index = pop_integer() + bias
                        if not 0 <= subr_index < len(subrs):
                            raise ValueError("invalid Type 2 charstring")
                        if not execute(subrs[subr_index], depth + 1):
                            return False
                    case 11:
                        return True
                    case 12:
                        if pos >= len(program):
                            raise ValueError("invalid Type 2 charstring")
                        escaped_operator = program[pos]
                        pos += 1
                        execute_escaped_operator(escaped_operator)
                    case 14:
                        arguments = list(stack)
                        if not width_resolved:
                            if len(arguments) in {1, 5}:
                                arguments = arguments[1:]
                            elif len(arguments) not in {0, 4}:
                                raise ValueError("invalid Type 2 charstring")
                            width_resolved = True
                        elif len(arguments) not in {0, 4}:
                            raise ValueError("invalid Type 2 charstring")
                        stack.clear()
                        flush_contour()
                        if arguments:
                            seac(
                                require_integer(arguments[2]),
                                require_integer(arguments[3]),
                                arguments[0],
                                arguments[1],
                            )
                        return False
                    case 19 | 20:
                        operand_count = len(stack)
                        if not width_resolved and operand_count % 2:
                            operand_count -= 1
                        if operand_count % 2:
                            raise ValueError("invalid Type 2 charstring")
                        stem_count += operand_count // 2
                        if stem_count <= 0 or stem_count > 96:
                            raise ValueError("invalid Type 2 charstring")
                        mask_bytes = (stem_count + 7) // 8
                        if pos + mask_bytes > len(program):
                            raise ValueError("invalid Type 2 charstring")
                        width_resolved = True
                        stack.clear()
                        pos += mask_bytes
                    case 21:
                        if len(stack) == 2:
                            dx, dy = stack
                        elif not width_resolved and len(stack) == 3:
                            dx, dy = stack[1:]
                        else:
                            raise ValueError("invalid Type 2 charstring")
                        width_resolved = True
                        move(dx, dy)
                        stack.clear()
                    case 24:
                        if not has_current_point() or len(stack) < 8 or (len(stack) - 2) % 6:
                            raise ValueError("invalid Type 2 charstring")
                        curve_args = stack[:-2]
                        for i in range(0, len(curve_args) - 5, 6):
                            curve(*curve_args[i : i + 6])
                        line(stack[-2], stack[-1])
                        stack.clear()
                    case 25:
                        if not has_current_point() or len(stack) < 8 or (len(stack) - 6) % 2:
                            raise ValueError("invalid Type 2 charstring")
                        line_args = stack[:-6]
                        for i in range(0, len(line_args) - 1, 2):
                            line(line_args[i], line_args[i + 1])
                        curve(*stack[-6:])
                        stack.clear()
                    case 26 | 27:
                        if (
                            not has_current_point()
                            or len(stack) < 4
                            or len(stack) % 4 not in {0, 1}
                        ):
                            raise ValueError("invalid Type 2 charstring")
                        first_offset = stack.pop(0) if len(stack) % 2 else 0.0
                        for i in range(0, len(stack) - 3, 4):
                            first, dx2, dy2, last = stack[i : i + 4]
                            if byte == 26:
                                curve(first_offset, first, dx2, dy2, 0.0, last)
                            else:
                                curve(first, first_offset, dx2, dy2, last, 0.0)
                            first_offset = 0.0
                        stack.clear()
                    case 30 | 31:
                        if (
                            not has_current_point()
                            or len(stack) < 4
                            or len(stack) % 4 not in {0, 1}
                        ):
                            raise ValueError("invalid Type 2 charstring")
                        horizontal = byte == 31
                        args = list(stack)
                        stack.clear()
                        while len(args) >= 4:
                            if horizontal:
                                dx1 = args.pop(0)
                                dy1 = 0.0
                                dx2 = args.pop(0)
                                dy2 = args.pop(0)
                                dy3 = args.pop(0)
                                dx3 = args.pop(0) if len(args) == 1 else 0.0
                            else:
                                dx1 = 0.0
                                dy1 = args.pop(0)
                                dx2 = args.pop(0)
                                dy2 = args.pop(0)
                                dx3 = args.pop(0)
                                dy3 = args.pop(0) if len(args) == 1 else 0.0
                            curve(dx1, dy1, dx2, dy2, dx3, dy3)
                            horizontal = not horizontal
                    case _:
                        raise ValueError("invalid Type 2 charstring")
            return True
        except (ArithmeticError, IndexError, ValueError) as exc:
            raise ValueError("invalid Type 2 charstring") from exc

    return execute(charstring)


__all__ = (
    "TYPE2_MAX_SUBR_DEPTH",
    "TYPE2_MAX_STACK",
    "TYPE2_TRANSIENT_SIZE",
    "cubic_extrema_times",
    "cubic_point",
    "execute_type2_charstring",
)

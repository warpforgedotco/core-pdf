# SPDX-License-Identifier: AGPL-3.0-only

import math
from collections.abc import Callable, Sequence

import pytest

from core_postscript.calculator import compile_calculator

UNBOUNDED = ((-1e100, 1e100),)


def build_calculator(
    program: bytes,
    domains: Sequence[tuple[float, float]] = UNBOUNDED,
    ranges: Sequence[tuple[float, float]] = UNBOUNDED,
) -> Callable[..., tuple[float, ...]]:
    return compile_calculator(program, domains, ranges)


def constant(expression: str, outputs: int = 1) -> Callable[..., tuple[float, ...]]:
    return build_calculator(
        ("{ pop " + expression + " }").encode("ascii"), ranges=UNBOUNDED * outputs
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("-7 abs", (7,)),
        ("8 5 add", (13,)),
        ("1 -1 atan", (135,)),
        ("-7.8 ceiling", (-7,)),
        ("60 cos", (0.5,)),
        ("-7.8 cvi", (-7,)),
        ("7 cvr", (7,)),
        ("9 4 div", (2.25,)),
        ("4 .5 exp", (2,)),
        ("-7.2 floor", (-8,)),
        ("-9 4 idiv", (-2,)),
        ("1 ln", (0,)),
        ("1000 log", (3,)),
        ("-9 4 mod", (-1,)),
        ("-7 3 mul", (-21,)),
        ("-7 neg", (7,)),
        ("-5.5 round", (-5,)),
        ("30 sin", (0.5,)),
        ("81 sqrt", (9,)),
        ("8 13 sub", (-5,)),
        ("-7.8 truncate", (-7,)),
        ("12 10 and", (8,)),
        ("-1 -1 bitshift", (2147483647,)),
        ("7 7.0 eq { 1 } { 0 } ifelse", (1,)),
        ("7 7.0 ge { 1 } { 0 } ifelse", (1,)),
        ("8 7 gt { 1 } { 0 } ifelse", (1,)),
        ("7 7.0 le { 1 } { 0 } ifelse", (1,)),
        ("7 8 lt { 1 } { 0 } ifelse", (1,)),
        ("7 8 ne { 1 } { 0 } ifelse", (1,)),
        ("0 not", (-1,)),
        ("12 10 or", (14,)),
        ("12 10 xor", (6,)),
        ("true { 17 } if", (17,)),
        ("false { 13 } { 17 } ifelse", (17,)),
        ("2 5 2 copy", (2, 5, 2, 5)),
        ("7 dup", (7, 7)),
        ("2 5 exch", (5, 2)),
        ("2 5 1 index", (2, 5, 2)),
        ("2 5 pop", (2,)),
        ("10 20 30 3 1 roll", (30, 10, 20)),
    ],
)
def test_calculator_implements_every_table_42_operator(
    expression: str, expected: tuple[float, ...]
) -> None:
    result = constant(expression, len(expected))(0)
    assert result == pytest.approx(expected)
    assert all(type(value) is float for value in result)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("-9 -4 idiv", 2),
        ("9 -4 idiv", -2),
        ("-9 -4 mod", -1),
        ("9 -4 mod", 1),
        ("5.5 round", 6),
        ("-5.5 round", -5),
        ("-.5 round", 0),
        (".5 round", 1),
        ("-5.6 round", -6),
        ("0 1 atan", 0),
        ("1 0 atan", 90),
        ("0 -1 atan", 180),
        ("-1 0 atan", 270),
        ("-1 1 atan", 315),
        ("-1 -1 atan", 225),
        ("-30 sin", -0.5),
        ("420 cos", 0.5),
        ("-3 3 exp", -27),
        ("-3 2.0 exp", 9),
        ("0 0 exp", 1),
        ("-2 -3 exp", -0.125),
        ("4 -1.5 exp", 0.125),
        ("2.718281828459045 ln", 1),
        (".01 log", -2),
        ("-7.0 ceiling", -7),
        ("7.8 floor", 7),
        ("7.8 truncate", 7),
    ],
)
def test_calculator_arithmetic_uses_postscript_signs_angles_and_rounding(
    expression: str, expected: float
) -> None:
    assert constant(expression)(0) == pytest.approx((expected,))


@pytest.mark.parametrize(
    "expression",
    [
        "7 abs",
        "7 2 add",
        "7 ceiling",
        "7 floor",
        "7 2 idiv",
        "7 2 mod",
        "7 2 mul",
        "7 neg",
        "7 round",
        "7 2 sub",
        "7 truncate",
        "7.8 cvi",
        "-2147483648",
        "2147483647",
    ],
)
def test_calculator_integer_operations_preserve_integer_type(expression: str) -> None:
    assert len(constant(expression + " 1 idiv")(0)) == 1


@pytest.mark.parametrize(
    "expression",
    [
        "7.0 abs",
        "7 2.0 add",
        "7.0 ceiling",
        "7.0 floor",
        "7 1 div",
        "7 2.0 mul",
        "7.0 neg",
        "7.0 round",
        "7 2.0 sub",
        "7.0 truncate",
        "7 cvr",
        "49 sqrt",
        "0 sin",
        "0 cos",
        "0 1 atan",
        "7 1 exp",
        "0 0 exp",
        "1 ln",
        "1 log",
    ],
)
def test_calculator_real_results_are_not_implicitly_integer_operands(expression: str) -> None:
    with pytest.raises(ValueError):
        constant(expression + " 1 idiv")(0)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2147483647 1 add", 2147483648),
        ("-2147483648 1 sub", -2147483649),
        ("1073741824 2 mul", 2147483648),
        ("-2147483648 neg", 2147483648),
        ("-2147483648 abs", 2147483648),
        ("2147483648", 2147483648),
        ("-2147483649", -2147483649),
    ],
)
def test_calculator_signed_32_bit_integer_overflow_promotes_to_real(
    expression: str, expected: int
) -> None:
    assert constant(expression)(0) == (expected,)
    with pytest.raises(ValueError):
        constant(expression + " 1 idiv")(0)


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("2147483647.9", 2147483647),
        ("-2147483648.9", -2147483648),
        ("-0.9", 0),
        ("0.9", 0),
    ],
)
def test_calculator_cvi_checks_range_after_truncation(literal: str, expected: int) -> None:
    assert constant(literal + " cvi 1 idiv")(0) == (expected,)


@pytest.mark.parametrize("literal", ["2147483648.0", "-2147483649.0", "99999999999999999999"])
def test_calculator_cvi_rejects_out_of_range_integer_results(literal: str) -> None:
    with pytest.raises(ValueError):
        constant(literal + " cvi")(0)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("-1 -1 bitshift", 2147483647),
        ("-2147483648 -31 bitshift", 1),
        ("1073741824 1 bitshift", -2147483648),
        ("-1 1 bitshift", -2),
        ("-2147483648 0 bitshift", -2147483648),
        ("-1 -32 bitshift", 0),
        ("1 32 bitshift", 0),
        ("1 2147483647 bitshift", 0),
        ("-1 -2147483648 bitshift", 0),
        ("-2147483648 not", 2147483647),
        ("-1 2147483647 xor", -2147483648),
        ("-1 2147483647 and", 2147483647),
        ("-2147483648 1 or", -2147483647),
    ],
)
def test_calculator_bitwise_operations_use_32_bits_and_zero_fill(
    expression: str, expected: int
) -> None:
    assert constant(expression)(0) == (expected,)


@pytest.mark.parametrize("left", [False, True])
@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("operator", ["and", "or", "xor", "eq", "ne"])
def test_calculator_boolean_truth_tables(left: bool, right: bool, operator: str) -> None:
    expected = {
        "and": left and right,
        "or": left or right,
        "xor": left != right,
        "eq": left == right,
        "ne": left != right,
    }[operator]
    expression = f"{str(left).lower()} {str(right).lower()} {operator} {{ 1 }} {{ 0 }} ifelse"
    assert constant(expression)(0) == (int(expected),)


@pytest.mark.parametrize(
    ("comparison", "expected"),
    [
        ("true 1 eq", 0),
        ("false 0 eq", 0),
        ("1 true eq", 0),
        ("true 1 ne", 1),
        ("1 1.0 eq", 1),
        ("1 1.0 ne", 0),
        ("2 1 ge", 1),
        ("1 2 ge", 0),
        ("2 2 gt", 0),
        ("2 1 le", 0),
        ("1 2 le", 1),
        ("2 2 lt", 0),
        ("true not", 0),
        ("false not", 1),
    ],
)
def test_calculator_comparison_and_boolean_operands_are_not_python_numbers(
    comparison: str, expected: int
) -> None:
    assert constant(comparison + " { 1 } { 0 } ifelse")(0) == (expected,)


@pytest.mark.parametrize(
    "expression",
    [
        "true abs",
        "true 1 add",
        "1 false sub",
        "false neg",
        "true 2 mul",
        "true 1 div",
        "true cvi",
        "true cvr",
        "true 1 eq 1 add",
        "true 1 and",
        "1 true or",
        "false 1 xor",
        "1.0 1 and",
        "1 1.0 or",
        "1.0 not",
        "true 1 gt",
        "false true ge",
        "1 true le",
        "false 1 lt",
        "1.0 1 bitshift",
        "1 1.0 bitshift",
        "true 1 bitshift",
        "3.0 2 idiv",
        "3 2.0 mod",
        "1 { 7 } if",
        "0 { 7 } { 8 } ifelse",
    ],
)
def test_calculator_rejects_wrong_operand_types(expression: str) -> None:
    with pytest.raises(ValueError):
        constant(expression)(0)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("10 20 30 3 -1 roll", (20, 30, 10)),
        ("10 20 30 3 4 roll", (30, 10, 20)),
        ("10 20 30 3 -4 roll", (20, 30, 10)),
        ("10 20 30 3 0 roll", (10, 20, 30)),
        ("0 2147483647 roll 9", (9,)),
        ("0 copy 9", (9,)),
        ("9 1 -2147483648 roll", (9,)),
        ("9 0 index", (9, 9)),
        ("2 5 7 1 copy", (2, 5, 7, 7)),
    ],
)
def test_calculator_stack_order_and_zero_count_controls(
    expression: str, expected: tuple[float, ...]
) -> None:
    assert constant(expression, len(expected))(0) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "pop",
        "dup",
        "exch",
        "1 add",
        "1 2 3 roll",
        "1 copy",
        "0 index",
        "9 -1 copy",
        "9 2 copy",
        "9 1.0 copy",
        "9 true copy",
        "9 -1 index",
        "9 1 index",
        "9 .0 index",
        "9 false index",
        "9 -1 1 roll",
        "9 2 1 roll",
        "9 1.0 1 roll",
        "9 1 1.0 roll",
        "9 true 1 roll",
        "9 1 false roll",
    ],
)
def test_calculator_stack_underflow_and_invalid_stack_arguments(expression: str) -> None:
    with pytest.raises(ValueError):
        constant(expression)(0)


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        (b"{ 4 lt { 7 } { 9 } ifelse }", 7),
        (b"{ 4 gt { 7 } { 9 } ifelse }", 9),
        (b"{ pop 7 false { pop 1 0 div } if }", 7),
        (b"{ pop true { 7 } { 1 0 div } ifelse }", 7),
        (b"{ pop false { -1 sqrt } { 7 } ifelse }", 7),
        (b"{ pop true { false { 2 } { true { 7 } if } ifelse } if }", 7),
        (b"{ pop true { } if 7 }", 7),
    ],
)
def test_calculator_conditionals_execute_only_selected_branch(
    program: bytes, expected: int
) -> None:
    assert build_calculator(program)(3) == (expected,)


@pytest.mark.parametrize(
    "program",
    [
        b"1",
        b"",
        b"{} {}",
        b"{} 1",
        b"1 {}",
        b"{ pop 1",
        b"{ pop 1 }}",
        b"{ pop { 1 } }",
        b"{ pop { 1 } dup }",
        b"{ pop true { 1 } dup if }",
        b"{ pop true { 1 } pop 2 }",
        b"{ pop true { 1 } { 2 } if }",
        b"{ pop true { 1 } ifelse }",
        b"{ pop true if }",
        b"{ pop true false ifelse }",
        b"{ pop true { 1 } { 2 } { 3 } ifelse }",
        b"{ pop false { forbidden } { 1 } ifelse }",
        b"{ pop false { [1] } { 1 } ifelse }",
        b"{ pop true { 1 } If }",
    ],
)
def test_calculator_rejects_malformed_program_and_first_class_procedures(program: bytes) -> None:
    with pytest.raises(ValueError):
        build_calculator(program)


@pytest.mark.parametrize(
    "operand",
    [
        b"1e2",
        b"1E2",
        b"16#ff",
        b"0x10",
        b".",
        b"+",
        b"--1",
        b"1.2.3",
        b"[1]",
        b"(1)",
        b"<31>",
        b"<< /A 1 >>",
        b"/name",
        b"name",
        b"null",
        b"NaN",
        b"inf",
        b"1\x0b2",
        b"\xff",
        b"9" * 400,
    ],
)
def test_calculator_uses_pdf_number_syntax_and_forbids_other_object_types(operand: bytes) -> None:
    with pytest.raises(ValueError):
        build_calculator(b"{ pop " + operand + b" }")


@pytest.mark.parametrize(
    ("literal", "expected"),
    [(b"+.5", 0.5), (b"-.5", -0.5), (b"5.", 5), (b"+005", 5), (b"-0", 0)],
)
def test_calculator_accepts_pdf_signed_and_decimal_number_forms(
    literal: bytes, expected: float
) -> None:
    assert build_calculator(b"{ pop " + literal + b" }")(0) == (expected,)


@pytest.mark.parametrize("whitespace", [b"\x00", b"\t", b"\n", b"\f", b"\r", b" "])
def test_calculator_accepts_each_pdf_whitespace_byte(whitespace: bytes) -> None:
    program = whitespace.join([b"{", b"pop", b"2", b"3", b"add", b"}"])
    assert build_calculator(program)(0) == (5,)


@pytest.mark.parametrize("line_end", [b"\r", b"\n", b"\r\n"])
def test_calculator_comments_ignore_braces_and_unsupported_tokens(line_end: bytes) -> None:
    program = (
        b"% leading comment"
        + line_end
        + b"{pop% } /name 1e2 [7]"
        + line_end
        + b"true{3}% separator"
        + line_end
        + b"{9}ifelse}% trailing comment"
    )
    assert build_calculator(program)(0) == (3,)


def test_calculator_unterminated_comment_does_not_supply_a_closing_brace() -> None:
    with pytest.raises(ValueError):
        build_calculator(b"{ pop 7 % }")


@pytest.mark.parametrize(
    "expression",
    [
        "1 0 div",
        "0 0 div",
        "1 0 idiv",
        "1 0 mod",
        "-2147483648 -1 idiv",
        "-1 sqrt",
        "0 ln",
        "-1 ln",
        "0 log",
        "-1 log",
        "0 0 atan",
        "-4 .5 exp",
        "0 -1 exp",
        "10 1000 exp",
        "1" + "0" * 200 + " dup mul",
        "1 ." + "0" * 308 + "1 div",
    ],
)
def test_calculator_rejects_undefined_or_nonfinite_arithmetic(expression: str) -> None:
    with pytest.raises(ValueError):
        constant(expression)(0)


def test_calculator_clips_inputs_before_execution_and_outputs_after_execution() -> None:
    function = build_calculator(b"{ sqrt }", domains=((0, 9),), ranges=((0, 2),))
    assert function(-4) == (0,)
    assert function(1) == (1,)
    assert function(4) == (2,)
    assert function(16) == (2,)


def test_calculator_multiple_inputs_and_outputs_follow_stack_order() -> None:
    function = build_calculator(
        b"{ exch }",
        domains=(
            (0, 2),
            (3, 5),
        ),
        ranges=(
            (0, 4),
            (1, 2),
        ),
    )
    assert function(-10, 10) == (4, 1)
    assert function(1.5, 3.5) == (3.5, 1.5)


@pytest.mark.parametrize("value", [5, 5.0])
def test_calculator_inputs_are_real_even_when_python_caller_passes_an_integer(value: float) -> None:
    with pytest.raises(ValueError):
        build_calculator(b"{ 2 idiv }")(value)
    assert build_calculator(b"{ cvi 2 idiv }")(value) == (2,)


@pytest.mark.parametrize(
    "value", [True, False, "1", None, [], math.nan, math.inf, -math.inf, 10**400]
)
def test_calculator_rejects_nonfinite_or_nonnumeric_call_inputs(value: object) -> None:
    function = build_calculator(b"{}")
    with pytest.raises(ValueError):
        function(value)


@pytest.mark.parametrize("inputs", [(), (1, 2)])
def test_calculator_rejects_input_arity_mismatch(inputs: tuple[float, ...]) -> None:
    function = build_calculator(b"{}")
    with pytest.raises(ValueError):
        function(*inputs)


@pytest.mark.parametrize("program", [b"{ pop }", b"{ dup }", b"{ pop true }", b"{ pop false }"])
def test_calculator_requires_exact_numeric_output_stack(program: bytes) -> None:
    with pytest.raises(ValueError):
        build_calculator(program)(0)


def test_calculator_reuses_compiled_program_without_leaking_operand_stack() -> None:
    function = build_calculator(b"{ dup 0 eq { pop -1 sqrt } if 2 mul }")
    assert function(3) == (6,)
    with pytest.raises(ValueError):
        function(0)
    assert function(4) == (8,)
    assert function(3) == (6,)


def test_calculator_supports_the_required_100_operand_stack_entries() -> None:
    program = b"{ " + b"0 " * 99 + b"pop " * 99 + b"}"
    assert build_calculator(program)(7) == (7,)
    identity = build_calculator(b"{}", domains=((0, 1),) * 100, ranges=((0, 1),) * 100)
    assert identity(*([0.5] * 100)) == (0.5,) * 100


@pytest.mark.parametrize(
    "program",
    [b"{ " + b"0 " * 100 + b"pop " * 100 + b"}", b"{ pop " + b"0 " * 51 + b"51 copy }"],
)
def test_calculator_rejects_operand_stack_growth_over_implementation_limit(program: bytes) -> None:
    with pytest.raises(ValueError):
        build_calculator(program)(7)


def test_calculator_rejects_initial_inputs_over_operand_stack_limit() -> None:
    with pytest.raises(ValueError):
        function = build_calculator(b"{}", domains=((0, 1),) * 101, ranges=((0, 1),) * 101)
        function(*([0.5] * 101))


def test_calculator_allows_255_total_brace_levels() -> None:
    expression = "true { " * 254 + "7" + " } if" * 254
    assert constant(expression)(0) == (7,)


def test_calculator_rejects_256_total_brace_levels() -> None:
    expression = "true { " * 255 + "7" + " } if" * 255
    with pytest.raises(ValueError):
        constant(expression)

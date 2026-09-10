"""Affine composition shares arithmetic across PDF and CFF consumers."""

import math
from typing import Any, cast

import pytest

from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix, multiply_affine
from core_pdf_spec.s_09_fonts.font_program import DEFAULT_CFF_FONT_MATRIX, CFFFont, cff_font_matrix


def test_unconditional_affine_product_preserves_operand_order() -> None:
    first = Matrix(2, 1, 3, 4, 5, 6)
    second = Matrix(7, 8, 9, 10, 11, 12)
    expected = Matrix(23, 26, 57, 64, 100, 112)
    assert multiply_affine(first, second) == expected
    assert first.multiply(second) == expected
    assert second.multiply(first) != expected


def test_identity_shortcut_is_only_used_by_matrix_method() -> None:
    matrix = Matrix(2, 0, 0, 3, -0.0, -0.0)
    assert matrix.multiply(IDENTITY_MATRIX) is matrix
    assert IDENTITY_MATRIX.multiply(matrix) is matrix
    result = multiply_affine(matrix, IDENTITY_MATRIX)
    assert result == matrix
    assert result is not matrix
    assert all(type(value) is float for value in result)
    assert math.copysign(1, result.e) == 1
    assert math.copysign(1, matrix.e) == -1
    integer_result = multiply_affine((2, 0, 0, 3, 4, 5), (1, 0, 0, 1, 0, 0))
    assert all(type(value) is int for value in integer_result)


@pytest.mark.parametrize("value", [True, "1", b"1", float("inf"), float("nan"), 10**400])
def test_pdf_and_cff_matrix_numbers_are_strict(value: object) -> None:
    values = [value, 0, 0, 1, 0, 0]
    with pytest.raises(ValueError, match="matrix operand"):
        Matrix.from_operand(values)
    with pytest.raises(ValueError, match="CFF FontMatrix"):
        cff_font_matrix(cast(Any, {(12, 7): values}))


@pytest.mark.parametrize(
    ("top", "child", "expected"),
    [
        (None, None, DEFAULT_CFF_FONT_MATRIX),
        ([2, 1, 3, 4, 5, 6], None, Matrix(2, 1, 3, 4, 5, 6)),
        (None, [7, 8, 9, 10, 11, 12], Matrix(7, 8, 9, 10, 11, 12)),
        ([7, 8, 9, 10, 11, 12], [2, 1, 3, 4, 5, 6], Matrix(23, 26, 57, 64, 100, 112)),
    ],
)
def test_cff_top_and_font_dictionary_matrix_defaults_and_composition(
    top: list[float] | None, child: list[float] | None, expected: Matrix
) -> None:
    font = CFFFont.__new__(CFFFont)
    font.charstrings = [b"\x0e"]
    font.top_dict = {} if top is None else {(12, 7): top}
    font.fd_select = (0,)
    font.font_dicts = ({},) if child is None else ({(12, 7): child},)
    result = font.font_matrix(0)
    assert isinstance(result, Matrix)
    assert result == expected

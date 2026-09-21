"""Affine composition shares arithmetic across PDF and CFF consumers."""

import math

import pytest

from core_adobe_fonts.cff.font import DEFAULT_CFF_FONT_MATRIX
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix, multiply_affine


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
def test_pdf_matrix_numbers_are_strict(value: object) -> None:
    with pytest.raises(ValueError, match="matrix operand"):
        Matrix.from_operand([value, 0, 0, 1, 0, 0])


def test_cff_font_matrix_converts_to_the_pdf_matrix_type() -> None:
    # A CFF FontMatrix composes with PDF matrices once core lifts it into Matrix.
    assert Matrix(*DEFAULT_CFF_FONT_MATRIX) == Matrix(0.001, 0.0, 0.0, 0.001, 0.0, 0.0)
    assert Matrix(*DEFAULT_CFF_FONT_MATRIX).multiply(IDENTITY_MATRIX) == Matrix(
        *DEFAULT_CFF_FONT_MATRIX
    )

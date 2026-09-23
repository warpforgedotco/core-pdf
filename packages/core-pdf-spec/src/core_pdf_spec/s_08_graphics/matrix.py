# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, NamedTuple

from core_pdf_spec.s_07_syntax_primitives.coercion import require_pdf_number_array


class Matrix(NamedTuple):
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    @classmethod
    def from_operand(cls, operands: object) -> Matrix:
        values = require_pdf_number_array(operands, "invalid matrix operand")
        if len(values) != 6:
            raise ValueError("invalid matrix operand")
        return cls(*values)

    def multiply(self, right: Matrix) -> Matrix:
        if right == IDENTITY_MATRIX:
            return self
        if self == IDENTITY_MATRIX:
            return right
        return multiply_affine(self, right)


IDENTITY_MATRIX: Final = Matrix(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


# int | float rather than float: PDF distinguishes integers from reals, and
# these operands keep whichever they arrived as. Writing float alone invites
# a compiler to read the numeric tower literally and coerce.
def multiply_affine(left: Sequence[int | float], right: Sequence[int | float]) -> Matrix:
    return Matrix(
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
        left[4] * right[0] + left[5] * right[2] + right[4],
        left[4] * right[1] + left[5] * right[3] + right[5],
    )


__all__ = (
    "Matrix",
    "IDENTITY_MATRIX",
    "multiply_affine",
)

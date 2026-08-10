# SPDX-License-Identifier: AGPL-3.0-only
"""Native affine matrix value type."""

from __future__ import annotations

from typing import Any, Final, NamedTuple


class Matrix(NamedTuple):
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    @classmethod
    def from_operand(cls, operands: Any) -> Matrix:
        if not isinstance(operands, (list, tuple)) or len(operands) != 6:
            raise ValueError("invalid matrix operand")
        a0, a1, a2, a3, a4, a5 = (
            operands[0],
            operands[1],
            operands[2],
            operands[3],
            operands[4],
            operands[5],
        )
        if not all(type(v) in (int, float) for v in (a0, a1, a2, a3, a4, a5)):
            raise ValueError("invalid matrix operand")
        return cls(float(a0), float(a1), float(a2), float(a3), float(a4), float(a5))

    def multiply(self, right: Matrix) -> Matrix:
        if right == IDENTITY_MATRIX:
            return self
        if self == IDENTITY_MATRIX:
            return right
        a2, b2, c2, d2, e2, f2 = right
        return Matrix(
            self.a * a2 + self.b * c2,
            self.a * b2 + self.b * d2,
            self.c * a2 + self.d * c2,
            self.c * b2 + self.d * d2,
            self.e * a2 + self.f * c2 + e2,
            self.e * b2 + self.f * d2 + f2,
        )


IDENTITY_MATRIX: Final = Matrix(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

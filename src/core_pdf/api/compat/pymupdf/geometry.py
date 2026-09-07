"""Native geometry for the PyMuPDF compatibility facade."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, cast, overload

from core_pdf.api.compat._shared import float32


class Matrix:
    """Six-coefficient affine transform using PyMuPDF's row-vector convention."""

    def __init__(self, *args: Any) -> None:
        self.a = self.b = self.c = self.d = self.e = self.f = 0.0
        if not args:
            return
        if len(args) == 1:
            value = args[0]
            if isinstance(value, (int, float)):
                angle = math.radians(value % 360)
                self.a = self.d = round(math.cos(angle), 8)
                self.b = round(math.sin(angle), 8)
                self.c = -self.b
                return
            args = tuple(value)
        if len(args) == 6:
            self.a, self.b, self.c, self.d, self.e, self.f = map(float, args)
        elif len(args) == 2 or (len(args) == 3 and args[2] == 0):
            self.a, self.d = float(args[0]), float(args[1])
        elif len(args) == 3 and args[2] == 1:
            self.a = self.d = 1.0
            self.c, self.b = float(args[0]), float(args[1])
        else:
            raise ValueError("Matrix: bad args")

    def __iter__(self) -> Iterator[float]:
        return iter((self.a, self.b, self.c, self.d, self.e, self.f))

    def __len__(self) -> int:
        return 6

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[float, ...]: ...

    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        return tuple(self)[index]

    def __setitem__(self, index: int, value: float) -> None:
        if not 0 <= index < 6:
            raise IndexError("index out of range")
        setattr(self, ("a", "b", "c", "d", "e", "f")[index], float(value))

    def __repr__(self) -> str:
        return f"Matrix({', '.join(map(str, self))})"

    def __bool__(self) -> bool:
        return any(self)

    def __eq__(self, other: object) -> bool:
        try:
            return tuple(self) == tuple(cast(Any, other))
        except TypeError:
            return False

    def norm(self) -> float:
        return math.sqrt(sum(value * value for value in self))

    def __abs__(self) -> float:
        return self.norm()

    def __pos__(self) -> Matrix:
        return Matrix(self)

    def __neg__(self) -> Matrix:
        return Matrix(*(-value for value in self))

    def __add__(self, other: Any) -> Matrix:
        values = (other,) * 6 if isinstance(other, (int, float)) else Matrix(other)
        return Matrix(*(a + b for a, b in zip(self, values, strict=True)))

    def __sub__(self, other: Any) -> Matrix:
        values = (other,) * 6 if isinstance(other, (int, float)) else Matrix(other)
        return Matrix(*(a - b for a, b in zip(self, values, strict=True)))

    def __mul__(self, other: Any) -> Matrix:
        if isinstance(other, (int, float)):
            return Matrix(*(value * other for value in self))
        return Matrix().concat(self, other)

    def __truediv__(self, other: Any) -> Matrix:
        if isinstance(other, (int, float)):
            return Matrix(*(value / other for value in self))
        inverse = Matrix()
        if inverse.invert(other):
            raise ZeroDivisionError("matrix not invertible")
        return self * inverse

    def __invert__(self) -> Matrix:
        result = Matrix()
        result.invert(self)
        return result

    @property
    def is_rectilinear(self) -> bool:
        return (abs(self.b) < 1e-8 and abs(self.c) < 1e-8) or (
            abs(self.a) < 1e-8 and abs(self.d) < 1e-8
        )

    def concat(self, one: Any, two: Any) -> Matrix:
        a, b, c, d, e, f = map(float32, Matrix(one))
        g, h, i, j, tx, ty = map(float32, Matrix(two))

        def product_sum(x: float, y: float, z: float, w: float) -> float:
            return float32(float32(x * y) + float32(z * w))

        self.a = product_sum(a, g, b, i)
        self.b = product_sum(a, h, b, j)
        self.c = product_sum(c, g, d, i)
        self.d = product_sum(c, h, d, j)
        self.e = float32(product_sum(e, g, f, i) + tx)
        self.f = float32(product_sum(e, h, f, j) + ty)
        return self

    def prescale(self, sx: float, sy: float) -> Matrix:
        self.a *= sx
        self.b *= sx
        self.c *= sy
        self.d *= sy
        return self

    def preshear(self, h: float, v: float) -> Matrix:
        a, b, c, d = self.a, self.b, self.c, self.d
        self.a, self.b = a + v * c, b + v * d
        self.c, self.d = c + h * a, d + h * b
        return self

    def pretranslate(self, tx: float, ty: float) -> Matrix:
        self.e += tx * self.a + ty * self.c
        self.f += tx * self.b + ty * self.d
        return self

    def prerotate(self, theta: float) -> Matrix:
        angle = math.radians(theta % 360)
        cosine, sine = math.cos(angle), math.sin(angle)
        if theta % 90 == 0:
            cosine, sine = round(cosine), round(sine)
        a, b, c, d = self.a, self.b, self.c, self.d
        self.a, self.b = cosine * a + sine * c, cosine * b + sine * d
        self.c, self.d = cosine * c - sine * a, cosine * d - sine * b
        return self

    def invert(self, src: Any = None) -> int:
        a, b, c, d, e, f = map(float32, self if src is None else Matrix(src))
        determinant = a * d - b * c
        if abs(determinant) <= 1e-30:
            return 1
        inverse = 1.0 / determinant
        self.a, self.b = float32(d * inverse), float32(-b * inverse)
        self.c, self.d = float32(-c * inverse), float32(a * inverse)
        self.e = float32(-e * self.a - f * self.c)
        self.f = float32(-e * self.b - f * self.d)
        return 0


__all__ = ("Matrix",)

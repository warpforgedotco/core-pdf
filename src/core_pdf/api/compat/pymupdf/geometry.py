"""Native geometry for the PyMuPDF compatibility facade."""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Self, cast, overload

from core_pdf.api.compat._shared import float32

#: PyMuPDF's geometry tolerance, used for the same comparisons it guards there.
EPSILON = 1e-5


class Matrix:
    """Six-coefficient affine transform using PyMuPDF's row-vector convention."""

    def __init__(self, *args: Any) -> None:
        self.a = self.b = self.c = self.d = self.e = self.f = 0.0
        if not args:
            return
        if len(args) == 1:
            value = args[0]
            if isinstance(value, (int, float)):
                angle = math.radians(value)
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
        return (abs(self.b) < EPSILON and abs(self.c) < EPSILON) or (
            abs(self.a) < EPSILON and abs(self.d) < EPSILON
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
        if abs(determinant) <= sys.float_info.epsilon:
            return 1
        inverse = 1.0 / determinant
        self.a, self.b = float32(d * inverse), float32(-b * inverse)
        self.c, self.d = float32(-c * inverse), float32(a * inverse)
        self.e = float32(-e * self.a - f * self.c)
        self.f = float32(-e * self.b - f * self.d)
        return 0


class internal_Coordinates(ABC):
    internal_fields: tuple[str, ...] = ()

    @abstractmethod
    def __init__(self, *args: Any) -> None: ...

    @abstractmethod
    def transform(self, matrix: Any) -> internal_Coordinates: ...

    def __iter__(self) -> Iterator[float]:
        return iter(getattr(self, key) for key in self.internal_fields)

    def __len__(self) -> int:
        return len(self.internal_fields)

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[float, ...]: ...

    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        return tuple(self)[index]

    def __setitem__(self, index: int, value: float) -> None:
        if not 0 <= index < len(self):
            raise IndexError("index out of range")
        setattr(self, self.internal_fields[index], float(value))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(str, self))})"

    def __bool__(self) -> bool:
        return any(self)

    def __eq__(self, other: object) -> bool:
        try:
            return tuple(self) == tuple(cast(Any, other))
        except TypeError:
            return False

    def norm(self) -> float:
        return math.sqrt(sum(value * value for value in self))

    def __pos__(self) -> Any:
        return type(self)(self)

    def __neg__(self) -> Any:
        return type(self)(*(-value for value in self))

    def __add__(self, other: Any) -> Any:
        values = (other,) * len(self) if isinstance(other, (int, float)) else other
        return type(self)(*(a + b for a, b in zip(self, values, strict=True)))

    def __sub__(self, other: Any) -> Any:
        values = (other,) * len(self) if isinstance(other, (int, float)) else other
        return type(self)(*(a - b for a, b in zip(self, values, strict=True)))

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, (int, float)):
            return type(self)(*(value * other for value in self))
        return type(self)(self).transform(other)

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, (int, float)):
            return type(self)(*(value / other for value in self))
        inverse = Matrix()
        if inverse.invert(other):
            raise ZeroDivisionError("matrix not invertible")
        return self * inverse


internal_UNIT_SCALE = {"px": 1.0, "in": 1 / 72, "cm": 2.54 / 72, "mm": 25.4 / 72}


class Point(internal_Coordinates):
    internal_fields = ("x", "y")

    def __init__(self, *args: Any) -> None:
        if not args:
            args = (0.0, 0.0)
        elif len(args) == 1:
            try:
                args = tuple(args[0])
            except TypeError:
                raise ValueError("Point: bad args") from None
        if len(args) != 2:
            raise ValueError("Point: bad seq len")
        self.x, self.y = map(float, args)

    def __abs__(self) -> float:
        return self.norm()

    @property
    def unit(self) -> Point:
        length = self.norm()
        return Point(self.x / length, self.y / length) if length else Point()

    @property
    def abs_unit(self) -> Point:
        unit = self.unit
        return Point(abs(unit.x), abs(unit.y))

    def transform(self, matrix: Any) -> Point:
        a, b, c, d, e, f = map(float32, Matrix(matrix))
        x, y = float32(self.x), float32(self.y)
        self.x = float32(float32(float32(x * a) + float32(y * c)) + e)
        self.y = float32(float32(float32(x * b) + float32(y * d)) + f)
        return self

    def distance_to(self, other: Any, unit: str = "px") -> float:
        if len(other) == 2:
            point = Point(other)
            distance = math.hypot(self.x - point.x, self.y - point.y)
        else:
            rect = Rect(other)
            dx = max(rect.x0 - self.x, 0, self.x - rect.x1)
            dy = max(rect.y0 - self.y, 0, self.y - rect.y1)
            distance = math.hypot(dx, dy)
        return distance * internal_UNIT_SCALE[unit]


class internal_RectCoordinates(internal_Coordinates):
    internal_fields = ("x0", "y0", "x1", "y1")

    def __init__(
        self,
        *args: Any,
        x0: float | None = None,
        y0: float | None = None,
        x1: float | None = None,
        y1: float | None = None,
    ) -> None:
        values: tuple[float, ...]
        if not args:
            values = (0.0, 0.0, 0.0, 0.0)
        else:
            flattened: list[float] = []
            for value in args:
                if isinstance(value, (int, float)):
                    flattened.append(float(value))
                else:
                    flattened.extend(value)
            if len(flattened) != 4:
                raise ValueError("Rect: bad args")
            values = tuple(map(float, flattened))
        self.x0, self.y0, self.x1, self.y1 = values
        for key, value in zip(self.internal_fields, (x0, y0, x1, y1), strict=True):
            if value is not None:
                setattr(self, key, float(value))

    @property
    def width(self) -> float:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0, self.y1 - self.y0)

    @property
    def is_valid(self) -> bool:
        return self.x0 <= self.x1 and self.y0 <= self.y1

    @property
    def is_empty(self) -> bool:
        return self.x0 >= self.x1 or self.y0 >= self.y1

    @property
    def is_infinite(self) -> bool:
        return tuple(self) == (-2147483648, -2147483648, 2147483520, 2147483520)

    @property
    def tl(self) -> Point:
        return Point(self.x0, self.y0)

    @property
    def tr(self) -> Point:
        return Point(self.x1, self.y0)

    @property
    def bl(self) -> Point:
        return Point(self.x0, self.y1)

    @property
    def br(self) -> Point:
        return Point(self.x1, self.y1)

    top_left = tl
    top_right = tr
    bottom_left = bl
    bottom_right = br

    def get_area(self, unit: str = "px") -> float:
        return self.width * self.height * internal_UNIT_SCALE[unit] ** 2

    def contains(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return other in tuple(self)
        if len(other) == 2:
            x, y = other
            return self.x0 <= x < self.x1 and self.y0 <= y < self.y1
        rect = Rect(other)
        return self.x0 <= rect.x0 <= rect.x1 <= self.x1 and self.y0 <= rect.y0 <= rect.y1 <= self.y1

    def __contains__(self, other: Any) -> bool:
        return self.contains(other)

    def intersects(self, other: Any) -> bool:
        rect = Rect(other)
        return not self.is_empty and not rect.is_empty and not (self & rect).is_empty

    def intersect(self, other: Any) -> Self:
        rect = Rect(other)
        if rect.is_empty or self.is_infinite:
            self.x0, self.y0, self.x1, self.y1 = rect
            return self
        if self.is_empty or rect.is_infinite:
            return self
        self.x0 = float32(max(float32(self.x0), float32(rect.x0)))
        self.y0 = float32(max(float32(self.y0), float32(rect.y0)))
        self.x1 = float32(min(float32(self.x1), float32(rect.x1)))
        self.y1 = float32(min(float32(self.y1), float32(rect.y1)))
        return self

    def include_rect(self, other: Any) -> Self:
        rect = Rect(other)
        if rect.is_empty:
            return self
        if self.is_empty:
            self.x0, self.y0, self.x1, self.y1 = map(float32, rect)
            return self
        self.x0 = float32(min(self.x0, rect.x0))
        self.y0 = float32(min(self.y0, rect.y0))
        self.x1 = float32(max(self.x1, rect.x1))
        self.y1 = float32(max(self.y1, rect.y1))
        return self

    def include_point(self, other: Any) -> Self:
        point = Point(other)
        self.x0 = float32(min(self.x0, point.x))
        self.y0 = float32(min(self.y0, point.y))
        self.x1 = float32(max(self.x1, point.x))
        self.y1 = float32(max(self.y1, point.y))
        return self

    def __and__(self, other: Any) -> Self:
        return type(self)(self).intersect(other)

    def __or__(self, other: Any) -> Self:
        return type(self)(self).include_rect(other)

    def normalize(self) -> Self:
        self.x0, self.x1 = sorted((self.x0, self.x1))
        self.y0, self.y1 = sorted((self.y0, self.y1))
        return self

    def transform(self, matrix: Any) -> Self:
        if self.is_infinite:
            return self
        a, b, c, d, e, f = map(float32, Matrix(matrix))
        x0, y0, x1, y1 = map(float32, self)
        # Select contributions by coefficient sign, preserving invalid input ordering.
        xa, xb = (x0, x1) if a >= 0 else (x1, x0)
        yc, yd = (y0, y1) if c >= 0 else (y1, y0)
        xe, xf = (x0, x1) if b >= 0 else (x1, x0)
        yg, yh = (y0, y1) if d >= 0 else (y1, y0)
        self.x0 = float32(float32(float32(xa * a) + float32(yc * c)) + e)
        self.x1 = float32(float32(float32(xb * a) + float32(yd * c)) + e)
        self.y0 = float32(float32(float32(xe * b) + float32(yg * d)) + f)
        self.y1 = float32(float32(float32(xf * b) + float32(yh * d)) + f)
        return self

    def torect(self, other: Any) -> Matrix:
        rect = Rect(other)
        if self.is_empty or self.is_infinite or rect.is_empty or rect.is_infinite:
            raise ValueError("rectangles must be finite and not empty")
        sx, sy = rect.width / self.width, rect.height / self.height
        return (
            Matrix(1, 0, 0, 1, -self.x0, -self.y0)
            * Matrix(sx, sy)
            * Matrix(1, 0, 0, 1, rect.x0, rect.y0)
        )

    @property
    def quad(self) -> Quad:
        return Quad(self.tl, self.tr, self.bl, self.br)

    def morph(self, fixpoint: Point, matrix: Any) -> Quad:
        return self.quad.morph(fixpoint, matrix)


class Rect(internal_RectCoordinates):
    def __abs__(self) -> float:
        return self.get_area()

    def round(self) -> IRect:
        values = [max(-16777216, min(16777216, float32(value))) for value in self]
        return IRect(
            math.floor(values[0] + 0.001),
            math.floor(values[1] + 0.001),
            math.ceil(values[2] - 0.001),
            math.ceil(values[3] - 0.001),
        )

    @property
    def irect(self) -> IRect:
        return self.round()


class IRect(internal_RectCoordinates):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.x0, self.y0 = math.floor(self.x0), math.floor(self.y0)
        self.x1, self.y1 = math.ceil(self.x1), math.ceil(self.y1)

    @property
    def rect(self) -> Rect:
        return Rect(self)

    def __mul__(self, other: Any) -> IRect:
        rect = self.rect * other
        return IRect(rect) if isinstance(other, (int, float)) else rect.round()

    def __truediv__(self, other: Any) -> IRect:
        rect = self.rect / other
        return IRect(rect) if isinstance(other, (int, float)) else rect.round()

    def __setitem__(self, index: int, value: float) -> None:
        if not 0 <= index < 4:
            raise IndexError("index out of range")
        setattr(self, self.internal_fields[index], int(value))

    def include_rect(self, other: Any) -> IRect:
        return self.rect.include_rect(other).round()

    def include_point(self, other: Any) -> IRect:
        return self.rect.include_point(other).round()

    def __and__(self, other: Any) -> IRect:
        return self.rect.intersect(other).round()

    def __or__(self, other: Any) -> IRect:
        return self.include_rect(other)

    def intersect(self, other: Any) -> IRect:
        super().intersect(other)
        # PyMuPDF 1.28.2 mutates first, then fails because IRect has no round method.
        return getattr(self, "round")()

    def transform(self, matrix: Any) -> IRect:
        super().transform(matrix)
        return getattr(self, "round")()


class Quad:
    def __init__(self, *args: Any) -> None:
        if not args:
            args = ((0, 0),) * 4
        elif len(args) == 1:
            args = tuple(args[0])
        if len(args) != 4:
            raise ValueError("Quad: bad seq len")
        self.ul, self.ur, self.ll, self.lr = (Point(value) for value in args)

    def __iter__(self) -> Iterator[Point]:
        return iter((self.ul, self.ur, self.ll, self.lr))

    def __len__(self) -> int:
        return 4

    @overload
    def __getitem__(self, index: int) -> Point: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Point, ...]: ...

    def __getitem__(self, index: int | slice) -> Point | tuple[Point, ...]:
        return tuple(self)[index]

    def __setitem__(self, index: int, value: Any) -> None:
        if not 0 <= index < 4:
            raise IndexError("index out of range")
        setattr(self, ("ul", "ur", "ll", "lr")[index], Point(value))

    def __repr__(self) -> str:
        return f"Quad({', '.join(map(repr, self))})"

    def __eq__(self, other: object) -> bool:
        try:
            return tuple(self) == tuple(cast(Any, other))
        except TypeError:
            return False

    def __bool__(self) -> bool:
        return any(self)

    def __abs__(self) -> float:
        return self.width * self.height

    def __pos__(self) -> Quad:
        return Quad(self)

    def __neg__(self) -> Quad:
        return Quad(*(-point for point in self))

    def __add__(self, other: Any) -> Quad:
        if isinstance(other, (int, float)):
            return Quad(*(point + other for point in self))
        return Quad(*(a + b for a, b in zip(self, Quad(other), strict=True)))

    def __sub__(self, other: Any) -> Quad:
        if isinstance(other, (int, float)):
            return Quad(*(point - other for point in self))
        return Quad(*(a - b for a, b in zip(self, Quad(other), strict=True)))

    def __mul__(self, other: Any) -> Quad:
        return Quad(*(point * other for point in self))

    def __truediv__(self, other: Any) -> Quad:
        return Quad(*(point / other for point in self))

    @property
    def rect(self) -> Rect:
        return Rect(
            min(point.x for point in self),
            min(point.y for point in self),
            max(point.x for point in self),
            max(point.y for point in self),
        )

    @property
    def width(self) -> float:
        return max(self.ul.distance_to(self.ur), self.ll.distance_to(self.lr))

    @property
    def height(self) -> float:
        return max(self.ul.distance_to(self.ll), self.ur.distance_to(self.lr))

    @property
    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    @property
    def is_infinite(self) -> bool:
        return self.rect.is_infinite

    def _edges(self) -> tuple[Point, ...]:
        return (self.ur - self.ul, self.lr - self.ur, self.ll - self.lr, self.ul - self.ll)

    @property
    def is_convex(self) -> bool:
        edges = self._edges()
        crosses = [
            a.x * b.y - a.y * b.x for a, b in zip(edges, (*edges[1:], edges[0]), strict=True)
        ]
        return all(value >= 0 for value in crosses) or all(value <= 0 for value in crosses)

    @property
    def is_rectangular(self) -> bool:
        edges = self._edges()
        for a, b in zip(edges, (*edges[1:], edges[0]), strict=True):
            length = a.norm() * b.norm()
            if not length or abs(a.x * b.x + a.y * b.y) / length > 1e-4:
                return False
        return True

    def transform(self, matrix: Any) -> Quad:
        for point in self:
            point.transform(matrix)
        return self

    def morph(self, fixpoint: Point, matrix: Any) -> Quad:
        to_origin = Matrix(1, 0, 0, 1, -fixpoint.x, -fixpoint.y)
        from_origin = Matrix(1, 0, 0, 1, fixpoint.x, fixpoint.y)
        return self * (to_origin * matrix * from_origin)


__all__ = ("IRect", "Matrix", "Point", "Quad", "Rect")

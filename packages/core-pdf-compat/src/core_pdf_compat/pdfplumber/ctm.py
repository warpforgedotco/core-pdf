import math
from typing import NamedTuple


class CTM(NamedTuple):
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    @property
    def scale_x(self) -> float:
        return math.hypot(self.a, self.b)

    @property
    def scale_y(self) -> float:
        return math.hypot(self.c, self.d)

    @property
    def skew_x(self) -> float:
        return math.degrees(math.atan2(self.d, self.c)) - 90

    @property
    def skew_y(self) -> float:
        return math.degrees(math.atan2(self.b, self.a))

    @property
    def translation_x(self) -> float:
        return self.e

    @property
    def translation_y(self) -> float:
        return self.f


__all__ = ("CTM",)

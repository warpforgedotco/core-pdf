"""CID width-map parsing helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # numpy is imported lazily at each call site to keep module import cheap.
    import numpy


class FontWidthMap(Mapping[int, float]):
    def width_for(self, code: int, default: float) -> float:
        width = self.get(code)
        return default if width is None else width

    @property
    def explicit_count(self) -> int:
        return len(self)

    def iter_explicit_widths(self) -> Iterator[tuple[int, float]]:
        return iter(self.items())

    def fast_256(self, default_width: float) -> tuple[float, ...]:
        default_positive = default_width if default_width > 0.0 else 1000.0
        space_width = default_width if default_width > 0.0 else 250.0
        return tuple(
            self.width_for(code, space_width if code == 32 else default_positive)
            for code in range(256)
        )

    def fast_256_array(self, default_width: float) -> numpy.ndarray[Any, Any]:
        """Return a dense NumPy table for bulk width indexing."""
        import numpy

        return numpy.asarray(self.fast_256(default_width), dtype=numpy.float64)


class SparseFontWidthMap(FontWidthMap):
    __slots__ = ("widths",)

    widths: dict[int, float]

    def __init__(self, widths: dict[int, float] | None = None) -> None:
        self.widths = widths if widths is not None else {}

    def __getitem__(self, key: int) -> float:
        return self.widths[key]

    def __iter__(self) -> Iterator[int]:
        return iter(self.widths)

    def __len__(self) -> int:
        return len(self.widths)

    def get(self, key: object, default: Any = None) -> float | Any:
        if type(key) is not int:
            return default
        return self.widths.get(key, default)


class CompactCIDWidthMap(FontWidthMap):
    __slots__ = ("start", "widths")

    start: int
    widths: tuple[float, ...]

    def __init__(self, start: int, widths: tuple[int | float, ...]) -> None:
        self.start = start
        self.widths = tuple(float(width) for width in widths)

    def __getitem__(self, key: int) -> float:
        index = key - self.start
        if 0 <= index < len(self.widths):
            return self.widths[index]
        raise KeyError(key)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.start + len(self.widths)))

    def __len__(self) -> int:
        return len(self.widths)

    def get(self, key: object, default: Any = None) -> float | Any:
        if type(key) is not int:
            return default
        index = key - self.start
        if 0 <= index < len(self.widths):
            return self.widths[index]
        return default

    def width_for(self, code: int, default: float) -> float:
        index = code - self.start
        if 0 <= index < len(self.widths):
            return self.widths[index]
        return default

    def fast_256(self, default_width: float) -> tuple[float, ...]:
        default_positive = default_width if default_width > 0.0 else 1000.0
        space_width = default_width if default_width > 0.0 else 250.0
        start = self.start
        end = start + len(self.widths)
        if start >= 256 or end <= 0:
            return tuple(space_width if code == 32 else default_positive for code in range(256))
        return tuple(
            self.widths[code - start]
            if start <= code < end
            else space_width
            if code == 32
            else default_positive
            for code in range(256)
        )


def scale_font_widths(widths: Mapping[int, float], scale: float) -> FontWidthMap:
    return SparseFontWidthMap({code: width * scale for code, width in widths.items()})


def require_cid_int(value: Any, message: str) -> int:
    if type(value) is int:
        return value
    if type(value) is bool:
        raise ValueError(message)
    if type(value) is float and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    if isinstance(value, bytes):
        try:
            return int(value)
        except ValueError:
            pass
    raise ValueError(message)


def require_cid_float(value: Any, message: str) -> float:
    if type(value) is float:
        return value
    if type(value) is int:
        return float(value)
    if type(value) is bool:
        raise ValueError(message)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    if isinstance(value, bytes):
        try:
            return float(value)
        except ValueError:
            pass
    raise ValueError(message)


def parse_cid_widths(value: Any) -> FontWidthMap:
    if value is None:
        return SparseFontWidthMap()
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid CID widths array")
    if len(value) == 2 and type(value[0]) is int:
        contiguous_widths = value[1]
        if isinstance(contiguous_widths, (list, tuple)) and set(
            map(type, contiguous_widths)
        ).issubset({int, float}):
            return CompactCIDWidthMap(value[0], tuple(contiguous_widths))
    widths: dict[int, float] = {}
    index = 0
    while index < len(value):
        try:
            first = require_cid_int(value[index], "invalid CID widths array")
        except ValueError:
            index += 1
            continue
        index += 1
        if index >= len(value):
            break
        nxt = value[index]
        if isinstance(nxt, (list, tuple)):
            code = first
            for w in nxt:
                if type(w) is int:
                    widths[code] = float(w)
                elif type(w) is float:
                    widths[code] = w
                else:
                    with suppress(ValueError):
                        widths[code] = require_cid_float(w, "invalid CID widths array")
                code += 1
            index += 1
        else:
            if index + 1 >= len(value):
                break
            try:
                last = require_cid_int(nxt, "invalid CID widths array")
                width = require_cid_float(value[index + 1], "invalid CID widths array")
            except ValueError:
                index += 2
                continue
            if last < first:
                index += 2
                continue
            for i in range(first, last + 1):
                widths[i] = width
            index += 2
    return SparseFontWidthMap(widths)

"""CID width-map parsing helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_float_strict

internal_MIN_CID = 0


internal_MAX_CID = 0xFFFF


class FontWidthMap(Mapping[int, float]):
    def width_for(self, code: int, default: float) -> float:
        width = self.get(code)
        return default if width is None else width

    def iter_explicit_widths(self) -> Iterator[tuple[int, float]]:
        return iter(self.items())

    def fast_256(self, default_width: float, space_width: float) -> tuple[float, ...]:
        return tuple(
            self.width_for(code, space_width if code == 32 else default_width)
            for code in range(256)
        )


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

    def width_for(self, code: int, default: float) -> float:
        # The base implementation routes through get(), adding two frames to
        # the hottest simple-font lookup. CompactCIDWidthMap already overrides
        # this; every caller passes an int, so the guard in get() is moot here.
        return self.widths.get(code, default)


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

    def fast_256(self, default_width: float, space_width: float) -> tuple[float, ...]:
        default_positive = default_width
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


def parse_cid_widths(value: Any) -> FontWidthMap:
    if value is None:
        return SparseFontWidthMap()
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid CID widths array")
    if len(value) == 2 and type(value[0]) is int and isinstance(value[1], (list, tuple)):
        first, entries = value
        if (
            0 <= first <= 65535
            and first + len(entries) <= 65536
            and all(type(item) in {int, float} for item in entries)
        ):
            return CompactCIDWidthMap(first, tuple(entries))
    widths: dict[int, float] = {}
    index = 0
    while index < len(value):
        first = value[index]
        if type(first) is not int or not 0 <= first <= 65535 or index + 1 >= len(value):
            raise ValueError("invalid CID widths array")
        item = value[index + 1]
        if isinstance(item, (list, tuple)):
            if first + len(item) > 65536:
                raise ValueError("invalid CID width range")
            for offset, width in enumerate(item):
                widths[first + offset] = parse_float_strict(width, "invalid CID width")
            index += 2
        else:
            if type(item) is not int or not first <= item <= 65535 or index + 2 >= len(value):
                raise ValueError("invalid CID width range")
            width = parse_float_strict(value[index + 2], "invalid CID width")
            widths.update((cid, width) for cid in range(first, item + 1))
            index += 3
    return SparseFontWidthMap(widths)

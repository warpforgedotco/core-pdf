from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

from core_pdf_spec.s_07_syntax_primitives.coercion import require_pdf_integer, require_pdf_number
from core_pdf_spec.s_09_fonts.dictionaries import (
    font_descriptor,
    get_descendant,
)
from core_records import Record, frozen_setattr

MIN_CID = 0
MAX_CID = 0xFFFF


class CompactCIDWidthMap(Mapping[int, float]):
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


def parse_cid_widths(value: Any) -> Mapping[int, float]:
    if value is None:
        return {}
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid CID widths array")
    if len(value) == 2 and type(value[0]) is int and isinstance(value[1], (list, tuple)):
        first, entries = value
        if 0 <= first <= MAX_CID and first + len(entries) <= (MAX_CID + 1):
            return CompactCIDWidthMap(
                first, tuple(require_pdf_number(entry, "invalid CID width") for entry in entries)
            )
    widths: dict[int, float] = {}
    index = 0
    while index < len(value):
        first = require_pdf_integer(value[index], "invalid CID widths array")
        if not 0 <= first <= MAX_CID or index + 1 >= len(value):
            raise ValueError("invalid CID widths array")
        item = value[index + 1]
        if isinstance(item, (list, tuple)):
            if first + len(item) > (MAX_CID + 1):
                raise ValueError("invalid CID width range")
            for offset, width in enumerate(item):
                widths[first + offset] = require_pdf_number(width, "invalid CID width")
            index += 2
        else:
            last = require_pdf_integer(item, "invalid CID width range")
            if not first <= last <= MAX_CID or index + 2 >= len(value):
                raise ValueError("invalid CID width range")
            width = require_pdf_number(value[index + 2], "invalid CID width")
            widths.update((cid, width) for cid in range(first, last + 1))
            index += 3
    return widths


class FontMetrics(Record):
    __slots__ = (
        "widths",
        "default_width",
        "default_width_explicit",
        "default_vertical_displacement_y",
        "default_vertical_origin_y",
        "vertical_metrics",
    )

    widths: Mapping[int, float]
    default_width: float
    default_width_explicit: bool
    default_vertical_displacement_y: float
    default_vertical_origin_y: float
    vertical_metrics: dict[int, tuple[float, float, float]]

    __fields__: ClassVar[tuple[str, ...]] = (
        "widths",
        "default_width",
        "default_width_explicit",
        "default_vertical_displacement_y",
        "default_vertical_origin_y",
        "vertical_metrics",
    )
    __match_args__ = (
        "widths",
        "default_width",
        "default_width_explicit",
        "default_vertical_displacement_y",
        "default_vertical_origin_y",
        "vertical_metrics",
    )

    def __init__(
        self,
        widths: Mapping[int, float],
        default_width: float,
        default_width_explicit: bool,
        default_vertical_displacement_y: float,
        default_vertical_origin_y: float,
        vertical_metrics: dict[int, tuple[float, float, float]],
    ) -> None:
        frozen_setattr(self, "widths", widths)
        frozen_setattr(self, "default_width", default_width)
        frozen_setattr(self, "default_width_explicit", default_width_explicit)
        frozen_setattr(self, "default_vertical_displacement_y", default_vertical_displacement_y)
        frozen_setattr(self, "default_vertical_origin_y", default_vertical_origin_y)
        frozen_setattr(self, "vertical_metrics", vertical_metrics)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.widths == other.widths
            and self.default_width == other.default_width
            and self.default_width_explicit == other.default_width_explicit
            and self.default_vertical_displacement_y == other.default_vertical_displacement_y
            and self.default_vertical_origin_y == other.default_vertical_origin_y
            and self.vertical_metrics == other.vertical_metrics
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.widths,
                self.default_width,
                self.default_width_explicit,
                self.default_vertical_displacement_y,
                self.default_vertical_origin_y,
                self.vertical_metrics,
            )
        )


def parse_font_widths(font: dict[Any, Any], subtype: str | None) -> FontMetrics:
    vertical: dict[int, tuple[float, float, float]] = {}
    vy, dy = 880.0, -1000.0
    if subtype == "Type0":
        descendant = get_descendant(font)
        if descendant is None:
            raise ValueError("missing descendant font")
        dw = descendant.get("DW")
        default = 1000.0 if dw is None else require_pdf_number(dw, "invalid CID DW")
        dw2 = descendant.get("DW2")
        if dw2 is None:
            dw2 = [880, -1000]
        if not isinstance(dw2, (list, tuple)) or len(dw2) != 2:
            raise ValueError("invalid CID DW2")
        vy, dy = (require_pdf_number(item, "invalid CID DW2") for item in dw2)
        w2 = descendant.get("W2")
        if w2 is None:
            w2 = []
        if not isinstance(w2, (list, tuple)):
            raise ValueError("invalid CID W2")
        index = 0
        while index < len(w2):
            first = require_pdf_integer(w2[index], "invalid CID W2")
            if not 0 <= first <= 65535 or index + 1 >= len(w2):
                raise ValueError("invalid CID W2")
            item = w2[index + 1]
            if isinstance(item, (list, tuple)):
                if len(item) % 3 or first + len(item) // 3 > 65536:
                    raise ValueError("invalid CID W2 range")
                for offset in range(len(item) // 3):
                    a, b, c = item[offset * 3 : offset * 3 + 3]
                    vertical[first + offset] = (
                        require_pdf_number(a, "invalid W2"),
                        require_pdf_number(b, "invalid W2"),
                        require_pdf_number(c, "invalid W2"),
                    )
                index += 2
            else:
                last = require_pdf_integer(item, "invalid CID W2")
                if not first <= last <= 65535 or index + 4 >= len(w2):
                    raise ValueError("invalid CID W2 range")
                a, b, c = w2[index + 2 : index + 5]
                metric = (
                    require_pdf_number(a, "invalid W2"),
                    require_pdf_number(b, "invalid W2"),
                    require_pdf_number(c, "invalid W2"),
                )
                vertical.update((cid, metric) for cid in range(first, last + 1))
                index += 5
        return FontMetrics(
            parse_cid_widths(descendant.get("W")),
            default,
            dw is not None,
            dy,
            vy,
            vertical,
        )
    descriptor = font_descriptor(font.get("FontDescriptor")) or {}
    missing_width = descriptor.get("MissingWidth")
    default = (
        0.0 if missing_width is None else require_pdf_number(missing_width, "invalid MissingWidth")
    )
    values = font.get("Widths")
    if values is None:
        values = []
    if not isinstance(values, (list, tuple)):
        raise ValueError("invalid font widths array")
    if values and (font.get("FirstChar") is None or font.get("LastChar") is None):
        raise ValueError("missing font widths range")
    first_value = font.get("FirstChar")
    first = 0 if first_value is None else require_pdf_integer(first_value, "invalid FirstChar")
    last_value = font.get("LastChar")
    last = (
        first + len(values) - 1
        if last_value is None
        else require_pdf_integer(last_value, "invalid LastChar")
    )
    if values and (not 0 <= first <= last <= 255 or len(values) != last - first + 1):
        raise ValueError("invalid font widths range")
    widths = {
        first + index: require_pdf_number(value, "invalid font width")
        for index, value in enumerate(values)
    }
    return FontMetrics(widths, default, missing_width is not None, dy, vy, vertical)


__all__ = [
    "MIN_CID",
    "MAX_CID",
    "CompactCIDWidthMap",
    "FontMetrics",
    "parse_cid_widths",
    "parse_font_widths",
]

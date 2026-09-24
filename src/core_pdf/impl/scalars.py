# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import overload


def scalar_text(value: object) -> str | None:
    if type(value) is bool:
        return None
    if type(value) is memoryview:
        value = value.tobytes()
    if type(value) is bytearray:
        value = bytes(value)
    if type(value) is bytes:
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    return value if type(value) is str else None


@overload
def parse_int(value: object, default: None = None) -> int | None: ...


@overload
def parse_int(value: object, default: int) -> int: ...


def parse_int(value: object, default: int | None = None) -> int | None:
    if type(value) is int:
        return value
    text = scalar_text(value)
    if text is None:
        return default
    try:
        return int(text)
    except ValueError, OverflowError:
        return default


def parse_int_strict(value: object, message: str | None = None) -> int:
    parsed = parse_int(value)
    if parsed is None:
        raise ValueError(message or f"invalid integer {value!r}")
    return parsed


@overload
def parse_float(value: object, default: None) -> float | None: ...


@overload
def parse_float(value: object, default: float = 0.0) -> float: ...


def parse_float(value: object, default: float | None = 0.0) -> float | None:
    if type(value) is float:
        return value
    if type(value) is int:
        try:
            return float(value)
        except OverflowError:
            return default
    text = scalar_text(value)
    if text is None:
        return default
    try:
        return float(text)
    except ValueError, OverflowError:
        return default


def parse_float_strict(value: object, message: str | None = None) -> float:
    parsed = parse_float(value, default=None)
    if parsed is None:
        raise ValueError(message or f"invalid float {value!r}")
    return parsed


def parse_box(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return (
            parse_float_strict(value[0]),
            parse_float_strict(value[1]),
            parse_float_strict(value[2]),
            parse_float_strict(value[3]),
        )
    except ValueError:
        return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))

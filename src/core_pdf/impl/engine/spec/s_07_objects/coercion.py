# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import SupportsFloat, SupportsIndex, SupportsInt, TypeAlias, cast, overload

from core_pdf.impl.primitives import PdfName, PdfString

IntCoercible: TypeAlias = str | bytes | bytearray | SupportsInt | SupportsIndex
FloatCoercible: TypeAlias = str | bytes | bytearray | SupportsFloat | SupportsIndex
CoercedContainer: TypeAlias = dict[object, object] | list[object]


def is_pdf_null(value: object) -> bool:
    return value is None or type(value).__name__ == "NullObject"


@lru_cache(maxsize=4096)
def parse_name_str(value: str) -> str:
    return value


@lru_cache(maxsize=4096)
def parse_name_bytes(value: bytes) -> str:
    return value.decode("latin-1")


def parse_name(value: object, default: str | None = None) -> str | None:
    if type(value) is PdfName:
        return str(value)
    if type(value) is str:
        return parse_name_str(value)
    if type(value) is bytes:
        return parse_name_bytes(value)
    return default


def parse_int(value: object, default: int | None = None) -> int | None:
    if type(value) is int:
        return value
    if type(value) is bool:
        return default
    if type(value) is memoryview:
        value = value.tobytes()
    if type(value) is bytes:
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return default
    try:
        return int(cast(IntCoercible, value))
    except (TypeError, ValueError, OverflowError):
        return default


def parse_int_strict(value: object) -> int:
    parsed = parse_int(value)
    if parsed is None:
        raise ValueError(f"invalid integer {value!r}")
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
    if type(value) is bool:
        return default
    if type(value) is memoryview:
        value = value.tobytes()
    if type(value) is bytes:
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return default
    try:
        return float(cast(FloatCoercible, value))
    except (TypeError, ValueError, OverflowError):
        return default


def parse_float_strict(value: object) -> float:
    parsed = parse_float(value, default=None)
    if parsed is None:
        raise ValueError(f"invalid float {value!r}")
    return parsed


def normalize_pdf_name(value: object, default: str | None = None) -> str | None:
    name = parse_name(value, default)
    if name is not None and name.startswith("/"):
        return name[1:]
    return name


def coerce_to_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, PdfString):
        return value.data
    if isinstance(value, str):
        return value.encode("latin-1")
    data = getattr(value, "data", None)
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    raise TypeError(f"cannot coerce {type(value).__name__} to bytes")


def coerce_value(value: object, string_decoder: Callable[[bytes], object] | None = None) -> object:
    def scalar_or_container(item: object) -> object:
        if string_decoder is not None:
            if isinstance(item, PdfString):
                return string_decoder(item.data)
            if isinstance(item, bytes):
                return string_decoder(item)
        if isinstance(item, dict):
            return {}
        if isinstance(item, (list, tuple)):
            return []
        return item

    root = scalar_or_container(value)
    if not isinstance(value, (dict, list, tuple)):
        return root

    stack: list[tuple[CoercedContainer | None, object | None, object]] = [(None, None, value)]
    results: dict[int, object] = {id(value): root}
    processed: set[int] = set()
    while stack:
        parent, key, item = stack.pop()
        coerced = results.get(id(item))
        if coerced is None:
            coerced = scalar_or_container(item)
            if isinstance(item, (dict, list, tuple)):
                results[id(item)] = coerced
        if parent is not None:
            if isinstance(parent, dict):
                parent[key] = coerced
            else:
                parent.append(coerced)
        if isinstance(item, dict):
            marker = id(item)
            if marker in processed:
                continue
            processed.add(marker)
            target = cast(CoercedContainer, coerced)
            for child_key, child_value in reversed(tuple(item.items())):
                stack.append(
                    (
                        target,
                        normalize_pdf_name(child_key) or child_key,
                        child_value,
                    )
                )
        elif isinstance(item, (list, tuple)):
            marker = id(item)
            if marker in processed:
                continue
            processed.add(marker)
            target = cast(CoercedContainer, coerced)
            for child in reversed(item):
                stack.append((target, None, child))
    return root


__all__ = (
    "coerce_to_bytes",
    "coerce_value",
    "is_pdf_null",
    "normalize_pdf_name",
    "parse_float",
    "parse_float_strict",
    "parse_int",
    "parse_int_strict",
    "parse_name",
    "parse_name_bytes",
    "parse_name_str",
)

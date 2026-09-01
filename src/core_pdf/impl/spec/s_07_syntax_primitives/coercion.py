# SPDX-License-Identifier: AGPL-3.0-only
"""Compiled PDF scalar and container coercion kernels."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TypeAlias, overload

from core_pdf.impl.primitives import PdfName, PdfString

CoercedContainer: TypeAlias = dict[object, object] | list[object]


def is_pdf_null(value: object) -> bool:
    return value is None or type(value).__name__ == "NullObject"


@lru_cache(maxsize=4096)
def parse_name_bytes(value: bytes) -> str:
    return value.decode("latin-1")


def parse_name(value: object, default: str | None = None) -> str | None:
    if type(value) is PdfName:
        return str(value)
    if type(value) is str:
        return value
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
    if type(value) is bytearray:
        value = bytes(value)
    if type(value) is bytes:
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return default
    if type(value) is not str:
        return default
    try:
        return int(value)
    except (ValueError, OverflowError):
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
    if type(value) is bytearray:
        value = bytes(value)
    if type(value) is bytes:
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return default
    if type(value) is not str:
        return default
    try:
        return float(value)
    except (ValueError, OverflowError):
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
    raise TypeError(f"cannot coerce {type(value).__name__} to bytes")


def coerce_value(value: object, string_decoder: Callable[[bytes], object] | None = None) -> object:
    def decode_scalar(item: object) -> object:
        if string_decoder is not None:
            if isinstance(item, PdfString):
                return string_decoder(item.data)
            if isinstance(item, bytes):
                return string_decoder(item)
        return item

    def walk(item: object) -> object:
        decoded_item = decode_scalar(item)
        if decoded_item is not item:
            return decoded_item

        if isinstance(item, dict):
            changed = False
            coerced_items: list[tuple[object, object]] = []
            for child_key, child_value in item.items():
                coerced_child = walk(child_value)
                coerced_items.append((child_key, coerced_child))
                if coerced_child is not child_value:
                    changed = True
            if not changed:
                return item
            return dict(coerced_items)

        if isinstance(item, (list, tuple)):
            changed = False
            coerced_seq_items: list[object] = []
            for child_value in item:
                coerced_child = walk(child_value)
                coerced_seq_items.append(coerced_child)
                if coerced_child is not child_value:
                    changed = True
            if not changed:
                return item
            return list(coerced_seq_items)

        return decoded_item

    return walk(value)


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
)

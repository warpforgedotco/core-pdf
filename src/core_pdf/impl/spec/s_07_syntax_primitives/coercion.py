# SPDX-License-Identifier: AGPL-3.0-only
"""Compiled PDF scalar and container coercion kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeGuard, overload

from core_pdf.impl.primitives import PdfName, PdfString
from core_pdf.impl.spec.s_07_syntax_primitives.text_string import decode_pdf_text_string


def is_pdf_null(value: object) -> bool:
    return value is None or type(value).__name__ == "NullObject"


def is_pdf_number(value: object) -> TypeGuard[int | float]:
    """Whether ``value`` is a PDF numeric scalar, excluding booleans."""
    return type(value) is int or type(value) is float


def parse_name(value: object, default: str | None = None) -> str | None:
    if type(value) is PdfName:
        return str(value)
    if type(value) is str:
        return value
    if type(value) is bytes:
        return value.decode("latin-1")
    return default


def internal_scalar_text(value: object) -> str | None:
    """The text of a scalar that is not already a number, or None if it is not one.

    A number reaches this layer as bytes from the lexer, or as a memoryview or
    bytearray slice of the source buffer, so all three have to end up as str
    before int()/float() sees them. bool is rejected rather than converted:
    True would otherwise parse as 1 and silently stand in for a number.
    """
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
    text = internal_scalar_text(value)
    if text is None:
        return default
    try:
        return int(text)
    except (ValueError, OverflowError):
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
    text = internal_scalar_text(value)
    if text is None:
        return default
    try:
        return float(text)
    except (ValueError, OverflowError):
        return default


def parse_float_strict(value: object, message: str | None = None) -> float:
    parsed = parse_float(value, default=None)
    if parsed is None:
        raise ValueError(message or f"invalid float {value!r}")
    return parsed


def parse_box(value: object) -> tuple[float, float, float, float] | None:
    """Four strict numbers to a rectangle tuple, or None if ``value`` is not one."""
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


def normalize_pdf_name(value: object, default: str | None = None) -> str | None:
    # A PdfName already holds the decoded string, so read it directly rather
    # than paying for the parse_name frame and the __str__ dunder dispatch.
    # str is inlined for the same reason: these two cover nearly every caller.
    if type(value) is PdfName:
        name: str | None = value.str_value or ""
    elif type(value) is str:
        name = value
    else:
        name = parse_name(value, default)
    if name is not None and name.startswith("/"):
        return name[1:]
    return name


def parse_text_string(value: object) -> str | None:
    """A direct string object as text, or None if ``value`` is not one."""
    if isinstance(value, PdfString):
        return decode_pdf_text_string(value.data)
    if isinstance(value, bytes):
        return decode_pdf_text_string(value)
    if isinstance(value, str):
        return value
    return None


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
    "is_pdf_number",
    "is_pdf_null",
    "normalize_pdf_name",
    "parse_box",
    "parse_float",
    "parse_float_strict",
    "parse_int",
    "parse_int_strict",
    "parse_name",
    "parse_text_string",
)

# SPDX-License-Identifier: AGPL-3.0-only
"""Compiled PDF scalar and container coercion kernels."""

from __future__ import annotations

import math
from typing import TypeGuard, overload

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax_primitives.numbers import parse_integer_token, parse_real_token
from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf_spec.types import PdfName, PdfString


def is_pdf_null(value: object) -> bool:
    return value is None


def is_pdf_number(value: object) -> TypeGuard[int | float]:
    """Whether ``value`` is a PDF numeric scalar, excluding booleans."""
    return type(value) is int or type(value) is float


def decoded_name(value: object, default: str | None = None) -> str | None:
    """Read a decoded name without interpreting its characters as PDF syntax."""
    if type(value) is PdfName:
        return value.str_value
    if type(value) is str:
        return value
    if type(value) is bytes:
        return value.decode("latin-1")
    return default


def require_pdf_number(value: object, message: str = "expected PDF number") -> float:
    """Require a finite PDF numeric object, rather than a numeric token."""
    if not is_pdf_number(value):
        raise ValueError(message)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(message) from error
    if not math.isfinite(result):
        raise ValueError(message)
    return result


def require_pdf_integer(value: object, message: str = "expected PDF integer") -> int:
    """Require a PDF integer object, excluding booleans and numeric text."""
    if type(value) is not int:
        raise ValueError(message)
    return value


def require_pdf_number_array(
    value: object, message: str = "expected PDF number array"
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(message)
    return tuple(require_pdf_number(item, message) for item in value)


def internal_scalar_token(value: object) -> bytes | None:
    """Normalize textual scalar inputs once; booleans are not numeric tokens."""
    if type(value) is bytes:
        return value
    if type(value) is memoryview:
        return value.tobytes()
    if type(value) is bytearray:
        return bytes(value)
    if type(value) is str:
        try:
            return value.encode("ascii")
        except UnicodeEncodeError:
            return None
    return None


@overload
def parse_int(value: object, default: None = None) -> int | None: ...


@overload
def parse_int(value: object, default: int) -> int: ...


def parse_int(value: object, default: int | None = None) -> int | None:
    if type(value) is int:
        return value
    token = internal_scalar_token(value)
    if token is None:
        return default
    try:
        return parse_integer_token(token)
    except PdfParseError:
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
        return value if math.isfinite(value) else default
    if type(value) is int:
        try:
            return float(value)
        except OverflowError:
            return default
    token = internal_scalar_token(value)
    if token is None:
        return default
    try:
        return parse_real_token(token)
    except PdfParseError:
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
            require_pdf_number(value[0]),
            require_pdf_number(value[1]),
            require_pdf_number(value[2]),
            require_pdf_number(value[3]),
        )
    except ValueError:
        return None


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
    raise TypeError(f"cannot coerce {type(value).__name__} to bytes")


__all__ = (
    "coerce_to_bytes",
    "is_pdf_number",
    "is_pdf_null",
    "decoded_name",
    "parse_box",
    "parse_float",
    "parse_float_strict",
    "parse_int",
    "parse_int_strict",
    "require_pdf_integer",
    "require_pdf_number",
    "require_pdf_number_array",
    "parse_text_string",
)

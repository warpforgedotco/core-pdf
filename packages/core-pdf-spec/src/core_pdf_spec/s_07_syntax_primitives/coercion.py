# SPDX-License-Identifier: AGPL-3.0-only

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
    return type(value) is int or type(value) is float


def decoded_name(value: object, default: str | None = None) -> str | None:
    if type(value) is PdfName:
        return value.str_value
    if type(value) is str:
        return value
    if type(value) is bytes:
        return value.decode("latin-1")
    return default


def require_pdf_number(value: object, message: str = "expected PDF number") -> float:
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
    if type(value) is not int:
        raise ValueError(message)
    return value


def require_pdf_number_array(
    value: object, message: str = "expected PDF number array"
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(message)
    return tuple(require_pdf_number(item, message) for item in value)


def require_pdf_number_pairs(
    value: object,
    message: str,
    *,
    count: int | None = None,
    length_message: str | None = None,
) -> tuple[tuple[float, float], ...]:
    """(low, high) pairs of a number array: `count` pairs, or any nonzero number of them."""
    values = require_pdf_number_array(value, message)
    if len(values) != 2 * count if count is not None else not values or len(values) % 2:
        raise ValueError(message if length_message is None else length_message)
    return tuple(zip(values[::2], values[1::2], strict=True))


def scalar_token(value: object) -> bytes | None:
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


def scalar_text(value: object) -> str | None:
    """The text Python's int() and float() parse: str as is, bytes that are ASCII."""
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


# python_syntax=True parses text with Python's int() and float() grammar (so
# "1_000", " 7 ", "1e3" and "inf" are numbers) and keeps non-finite floats;
# core's recovery reads values that way. The default is the PDF grammar of
# 7.3.3, which admits none of those.


@overload
def parse_int(
    value: object, default: None = None, *, python_syntax: bool = False
) -> int | None: ...


@overload
def parse_int(value: object, default: int, *, python_syntax: bool = False) -> int: ...


def parse_int(
    value: object, default: int | None = None, *, python_syntax: bool = False
) -> int | None:
    if type(value) is int:
        return value
    if python_syntax:
        text = scalar_text(value)
        if text is None:
            return default
        try:
            return int(text)
        except ValueError, OverflowError:
            return default
    token = scalar_token(value)
    if token is None:
        return default
    try:
        return parse_integer_token(token)
    except PdfParseError:
        return default


def parse_int_strict(
    value: object, message: str | None = None, *, python_syntax: bool = False
) -> int:
    parsed = parse_int(value, python_syntax=python_syntax)
    if parsed is None:
        raise ValueError(message or f"invalid integer {value!r}")
    return parsed


@overload
def parse_float(value: object, default: None, *, python_syntax: bool = False) -> float | None: ...


@overload
def parse_float(value: object, default: float = 0.0, *, python_syntax: bool = False) -> float: ...


def parse_float(
    value: object, default: float | None = 0.0, *, python_syntax: bool = False
) -> float | None:
    if type(value) is float:
        return value if python_syntax or math.isfinite(value) else default
    if type(value) is int:
        try:
            return float(value)
        except OverflowError:
            return default
    if python_syntax:
        text = scalar_text(value)
        if text is None:
            return default
        try:
            return float(text)
        except ValueError, OverflowError:
            return default
    token = scalar_token(value)
    if token is None:
        return default
    try:
        return parse_real_token(token)
    except PdfParseError:
        return default


def parse_float_strict(
    value: object, message: str | None = None, *, python_syntax: bool = False
) -> float:
    parsed = parse_float(value, default=None, python_syntax=python_syntax)
    if parsed is None:
        raise ValueError(message or f"invalid float {value!r}")
    return parsed


def parse_box(value: object) -> tuple[float, float, float, float] | None:
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
    "require_pdf_number_pairs",
    "parse_text_string",
    "scalar_text",
)

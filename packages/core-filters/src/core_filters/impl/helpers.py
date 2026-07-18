# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import SupportsInt, cast


def is_pdf_null(value: object) -> bool:
    return value is None or type(value).__name__ == "NullObject"


def normalize_pdf_name(value: object, default: str | None = None) -> str | None:
    if type(value) is str:
        name = value
    elif type(value) is bytes:
        name = value.decode("latin-1")
    else:
        name = getattr(value, "value", default)
        if not isinstance(name, str):
            return default
    return name[1:] if name.startswith("/") else name


def parse_int(value: object, default: int | None = None) -> int | None:
    if type(value) is int:
        return value
    if type(value) is bool:
        return default
    if type(value) is bytes:
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return default
    try:
        return int(cast(SupportsInt, value))
    except (TypeError, ValueError, OverflowError):
        return default


def lookup_dict_key(value: object, key: str) -> object:
    if not isinstance(value, dict):
        return None
    if key in value:
        return value[key]
    normalized = key.lstrip("/")
    for candidate, item in value.items():
        candidate_name = normalize_pdf_name(candidate)
        if candidate_name == normalized:
            return item
    return None


def coerce_to_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
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

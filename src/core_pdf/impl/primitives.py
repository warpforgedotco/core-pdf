# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import threading
from functools import lru_cache
from typing import Final


class MissingObject:
    __slots__ = ()


MISSING: Final = MissingObject()
PDF_NAME_CACHE: dict[bytes, "PdfName"] = {}
internal_PDF_NAME_CACHE_LOCK = threading.RLock()


@lru_cache(maxsize=4096)
def pdf_name_from_str(value: str) -> "PdfName":
    b_value = value.encode("latin-1")
    with internal_PDF_NAME_CACHE_LOCK:
        cache = PDF_NAME_CACHE
        cached = cache.get(b_value)
        if cached is not None:
            return cached
        cached = PdfName(b_value)
        cache[b_value] = cached
        return cached


class PdfName:
    """PDF name object.

    Names are atomic identifiers in PDF syntax. This implementation stores the
    decoded Latin-1 value for fast dictionary lookup while preserving equality
    with raw bytes and strings used by recovery paths.
    """

    __slots__ = ("value_bytes", "str_value", "hash_value")

    value_bytes: bytes
    str_value: str | None
    hash_value: int | None

    def __init__(self, value_bytes: bytes) -> None:
        object.__setattr__(self, "value_bytes", value_bytes)
        object.__setattr__(self, "str_value", value_bytes.decode("latin-1"))
        object.__setattr__(self, "hash_value", hash(self.str_value))

    @property
    def value(self) -> str:
        return self.str_value or ""

    @classmethod
    def of(cls, value: str | bytes | memoryview | "PdfName") -> "PdfName":
        if type(value) is PdfName:
            return value

        if type(value) is str:
            return pdf_name_from_str(value)

        if type(value) is memoryview:
            key_bytes = bytes(value)
        elif type(value) is bytes:
            key_bytes = value
        else:
            raise TypeError("PDF names must be str, bytes, memoryview, or PdfName")
        with internal_PDF_NAME_CACHE_LOCK:
            cache = PDF_NAME_CACHE
            n = cache.get(key_bytes)
            if n is None:
                cache[key_bytes] = n = cls(key_bytes)
            return n

    def __str__(self) -> str:
        return self.str_value or ""

    def __repr__(self) -> str:
        return f"PdfName({self.value!r})"

    def __hash__(self) -> int:
        h = self.hash_value
        if h is None:
            h = hash(self.str_value)
            object.__setattr__(self, "hash_value", h)
        return h

    def __eq__(self, other: object) -> bool:
        if type(other) is PdfName:
            return self.value == other.value
        if type(other) is bytes:
            return self.value_bytes == other
        if type(other) is str:
            s = self.str_value
            if s is not None:
                return s == other
            return self.value == other.encode("latin-1")
        return False

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"cannot assign to field {name!r}")


class PdfReference:
    """Indirect object reference: object number plus generation number."""

    __slots__ = ("object_number", "generation_number")

    object_number: int
    generation_number: int

    def __init__(self, object_number: int, generation_number: int = 0) -> None:
        if object_number < 0 or generation_number < 0:
            raise ValueError("invalid PDF reference")
        self.object_number = object_number
        self.generation_number = generation_number

    def __eq__(self, other: object) -> bool:
        if type(other) is PdfReference:
            return (
                self.object_number == other.object_number
                and self.generation_number == other.generation_number
            )
        return False

    def __hash__(self) -> int:
        return hash((self.object_number, self.generation_number))


class PdfString:
    """PDF string object containing the raw byte representation."""

    __slots__ = ("data",)

    data: bytes

    def __init__(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise ValueError("invalid PDF string")
        self.data = data

    def __eq__(self, other: object) -> bool:
        if type(other) is PdfString:
            return self.data == other.data
        return False

    def __hash__(self) -> int:
        return hash(self.data)

    def __repr__(self) -> str:
        return f"PdfString({self.data!r})"


__all__ = (
    "MISSING",
    "MissingObject",
    "PdfName",
    "PdfReference",
    "PdfString",
)

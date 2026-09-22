# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import ClassVar, cast

from core_pdf.impl.records import ReplaceFields, ReprFields
from core_pdf.impl.types import PdfString


def is_pdf_null(value: object) -> bool:
    return value is None or type(value).__name__ == "NullObject"


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


class CoercionFrame(ReplaceFields, ReprFields):
    __slots__ = ("original", "entries", "values", "pending", "changed")

    original: object
    entries: Iterator[tuple[object, object]]
    values: list[tuple[object, object]]
    pending: tuple[object, object] | None
    changed: bool

    __fields__: ClassVar[tuple[str, ...]] = ("original", "entries", "values", "pending", "changed")
    __match_args__ = ("original", "entries", "values", "pending", "changed")

    def __init__(
        self,
        original: object,
        entries: Iterator[tuple[object, object]],
        values: list[tuple[object, object]] | None = None,
        pending: tuple[object, object] | None = None,
        changed: bool = False,
    ) -> None:
        self.original = original
        self.entries = entries
        self.values = [] if values is None else values
        self.pending = pending
        self.changed = changed

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.original == other.original
            and self.entries == other.entries
            and self.values == other.values
            and self.pending == other.pending
            and self.changed == other.changed
        )

    __hash__ = None  # type: ignore[assignment]

    def add(self, key: object, original: object, coerced: object) -> None:
        self.values.append((key, coerced))
        if coerced is not original:
            self.changed = True


def coerce_value(value: object, string_decoder: Callable[[bytes], object] | None = None) -> object:

    def decode_scalar(item: object) -> object:
        if string_decoder is not None:
            if isinstance(item, PdfString):
                return string_decoder(item.data)
            if isinstance(item, bytes):
                return string_decoder(item)
        return item

    active: set[int] = set()
    completed: dict[int, tuple[object, object]] = {}
    stack: list[CoercionFrame] = []

    def enter(item: object) -> tuple[bool, object]:
        decoded_item = decode_scalar(item)
        if decoded_item is not item:
            return True, decoded_item
        if not isinstance(item, (dict, list, tuple)):
            return True, item
        marker = id(item)
        if marker in active:
            return True, None
        cached = completed.get(marker)
        if cached is not None:
            return True, cached[1]
        if isinstance(item, dict):
            entries = iter(cast(dict[object, object], item).items())
        else:
            entries = iter(enumerate(cast(list[object] | tuple[object, ...], item)))
        active.add(marker)
        stack.append(CoercionFrame(item, entries))
        return False, None

    ready, result = enter(value)
    if ready:
        return result
    while stack:
        frame = stack[-1]
        try:
            key, child = next(frame.entries)
        except StopIteration:
            result = frame.original
            if frame.changed:
                result = (
                    dict(frame.values)
                    if isinstance(frame.original, dict)
                    else [item for _, item in frame.values]
                )
            marker = id(frame.original)
            completed[marker] = (frame.original, result)
            active.remove(marker)
            stack.pop()
            if not stack:
                return result
            parent = stack[-1]
            assert parent.pending is not None
            key, child = parent.pending
            parent.pending = None
            parent.add(key, child, result)
        else:
            ready, coerced = enter(child)
            if ready:
                frame.add(key, child, coerced)
            else:
                frame.pending = (key, child)
    raise AssertionError("object coercion did not produce a result")

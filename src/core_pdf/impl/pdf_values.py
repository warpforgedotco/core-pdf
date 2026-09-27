# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol

from core_pdf.impl.types import PdfReference, PdfString, RecordType, ReplaceFields, ReprFields


class ReferenceResolver(Protocol):
    def resolve(self, value: object, /) -> object: ...


class CoercionFrame(ReplaceFields, ReprFields, metaclass=RecordType, frozen=False):
    original: object
    entries: Iterator[tuple[object, object]]
    values: list[tuple[object, object]]
    pending: tuple[object, object] | None
    changed: bool

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

    __hash__ = None  # type: ignore[assignment]

    def add(self, key: object, original: object, coerced: object) -> None:
        self.values.append((key, coerced))
        if coerced is not original:
            self.changed = True


def resolve_destination_references(
    resolver: ReferenceResolver, value: object, depth: int = 0
) -> object:
    if depth > 8:
        return value
    if isinstance(value, PdfReference):
        resolved = resolver.resolve(value)
        if resolved is None or isinstance(resolved, (dict, list, tuple)):
            return value
        return resolve_destination_references(resolver, resolved, depth + 1)
    if isinstance(value, dict):
        return {
            str(key): resolve_destination_references(resolver, item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [resolve_destination_references(resolver, item, depth + 1) for item in value]
    return value


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
            entries = iter(item.items())
        else:
            entries = iter(enumerate(item))
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

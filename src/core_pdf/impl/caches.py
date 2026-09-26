# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from enum import Enum
from typing import overload


class Missing(Enum):
    MISSING = 0


MISSING = Missing.MISSING

type CacheLimit = int | Callable[[], int] | None


class BoundedDict[K, V](dict[K, V]):
    __slots__ = ("limit",)

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def put(self, key: K, value: V) -> V:
        if len(self) >= self.limit:
            self.clear()
        self[key] = value
        return value


class IdentityCache[V]:
    __slots__ = ("entries", "limit")

    def __init__(self, limit: CacheLimit = None) -> None:
        self.entries: dict[Hashable, tuple[object, tuple[object, ...], V]] = {}
        self.limit = limit

    @overload
    def get(self, obj: object, *extra: Hashable, pins: tuple[object, ...] = ()) -> V | None: ...

    @overload
    def get[D](
        self, obj: object, *extra: Hashable, pins: tuple[object, ...] = (), default: D
    ) -> V | D: ...

    def get(
        self,
        obj: object,
        *extra: Hashable,
        pins: tuple[object, ...] = (),
        default: object = None,
    ) -> object:
        key = (id(obj), *map(id, pins), *extra) if extra or pins else id(obj)
        entry = self.entries.get(key)
        if entry is None or entry[0] is not obj:
            return default
        return entry[2]

    def put(self, obj: object, value: V, *extra: Hashable, pins: tuple[object, ...] = ()) -> V:
        entries = self.entries
        limit = self.limit
        if limit is not None:
            if not isinstance(limit, int):
                limit = limit()
            if len(entries) >= limit:
                entries.clear()
        key = (id(obj), *map(id, pins), *extra) if extra or pins else id(obj)
        entries[key] = (obj, pins, value)
        return value

    def discard(self, obj: object, *extra: Hashable, pins: tuple[object, ...] = ()) -> None:
        key = (id(obj), *map(id, pins), *extra) if extra or pins else id(obj)
        entry = self.entries.get(key)
        if entry is not None and entry[0] is obj:
            del self.entries[key]

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IdentityCache):
            return NotImplemented
        return self.entries == other.entries

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.entries)


class ByteBudgetCache[K, V]:
    __slots__ = ("budget", "entries", "size")

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.entries: dict[K, tuple[V, int]] = {}
        self.size = 0

    def get(self, key: K) -> V | None:
        entry = self.entries.get(key)
        return None if entry is None else entry[0]

    def store(self, key: K, value: V, size: int) -> None:
        entries = self.entries
        previous = entries.pop(key, None)
        if previous is not None:
            self.size -= previous[1]
        if size > self.budget:
            return
        while entries and self.size + size > self.budget:
            self.size -= entries.pop(next(iter(entries)))[1]
        entries[key] = (value, size)
        self.size += size


__all__ = ("MISSING", "BoundedDict", "ByteBudgetCache", "CacheLimit", "IdentityCache", "Missing")

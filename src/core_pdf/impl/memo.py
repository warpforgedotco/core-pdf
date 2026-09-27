# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from core_pdf.impl.caches import MISSING

type MemoScope = Literal["content", "xref"]


class DocumentCaches:
    __slots__ = ("content", "xref")

    def __init__(self) -> None:
        self.content: dict[str, Any] = {}
        self.xref: dict[str, tuple[object, object]] = {}

    def get[V](self, name: str, build: Callable[[], V]) -> V:
        content = self.content
        value = content.get(name, MISSING)
        if value is MISSING:
            value = content[name] = build()
        return cast("V", value)

    def get_keyed[V](self, name: str, key: object, build: Callable[[], V]) -> V:
        entry = self.xref.get(name)
        if entry is None or entry[0] != key:
            entry = self.xref[name] = (key, build())
        return cast("V", entry[1])

    def peek(self, name: str, default: object = None) -> object:
        return self.content.get(name, default)

    def clear(self, scope: MemoScope = "content") -> None:
        if scope == "content":
            self.content.clear()
        else:
            self.xref.clear()


__all__ = ("DocumentCaches", "MemoScope")

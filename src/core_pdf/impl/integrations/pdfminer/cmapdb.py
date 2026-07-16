# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Iterator
from typing import cast


class CMap:
    def __init__(self, **attrs: object) -> None:
        self.attrs = dict(attrs)
        self.code2cid: dict[int, object] = {}

    def is_vertical(self) -> bool:
        return bool(self.attrs.get("WMode", 0))

    def decode(self, code: bytes) -> Iterator[tuple[int, int]]:
        mapping: dict[int, object] = self.code2cid
        start = 0
        for index, byte in enumerate(code):
            value = mapping.get(byte)
            if isinstance(value, dict):
                mapping = cast(dict[int, object], value)
                continue
            if isinstance(value, int):
                yield value, index + 1 - start
                mapping = self.code2cid
                start = index + 1


class CMapDB:
    class CMapNotFound(KeyError):
        pass

    @classmethod
    def get_cmap(cls, name: str) -> CMap:
        if name in {"Identity-H", "OneByteIdentityH"}:
            cmap = CMap()
            cmap.code2cid = {index: index for index in range(256)}
            return cmap
        if name in {"Identity-V", "OneByteIdentityV"}:
            cmap = CMap(WMode=1)
            cmap.code2cid = {index: index for index in range(256)}
            return cmap
        raise cls.CMapNotFound(name)


__all__ = ("CMap", "CMapDB")

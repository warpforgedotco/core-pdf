# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import struct
from collections.abc import Iterable
from typing import cast


class CMap:
    def __init__(self, **attrs: object) -> None:
        self.attrs = dict(attrs)
        self.code2cid: dict[int, object] = {}

    def is_vertical(self) -> bool:
        return bool(self.attrs.get("WMode", 0))

    def __repr__(self) -> str:
        return f"<CMap: {self.attrs.get('CMapName')}>"

    def decode(self, code: bytes) -> Iterable[int]:
        mapping: dict[int, object] = self.code2cid
        for byte in code:
            value = mapping.get(byte)
            if isinstance(value, dict):
                mapping = cast(dict[int, object], value)
                continue
            if isinstance(value, int):
                yield value
                mapping = self.code2cid
                continue
            mapping = self.code2cid


class IdentityCMap(CMap):
    def decode(self, code: bytes) -> tuple[int, ...]:
        count = len(code) // 2
        return struct.unpack(f">{count}H", code[: count * 2]) if count else ()


class IdentityCMapByte(IdentityCMap):
    def decode(self, code: bytes) -> tuple[int, ...]:
        count = len(code)
        return struct.unpack(f">{count}B", code) if count else ()


class CMapDB:
    class CMapNotFound(KeyError):
        pass

    @classmethod
    def get_cmap(cls, name: str) -> CMap:
        if name == "Identity-H":
            return IdentityCMap(WMode=0)
        if name == "Identity-V":
            return IdentityCMap(WMode=1)
        if name == "OneByteIdentityH":
            return IdentityCMapByte(WMode=0)
        if name == "OneByteIdentityV":
            return IdentityCMapByte(WMode=1)
        raise cls.CMapNotFound(name)


__all__ = ("CMap", "CMapDB", "IdentityCMap", "IdentityCMapByte")

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from functools import cache, lru_cache
from importlib import resources
from importlib.resources.abc import Traversable

from core_adobe_fonts.cmap.decoder import CMapDecoder

RESOURCE_PACKAGE = "core_adobe_fonts.cmap.data"


@lru_cache(maxsize=64)
def resolve_cmap_resource(name: str) -> bytes | None:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        return None
    deprecated: Traversable | None = None
    for directory in cmap_directories():
        child = directory.joinpath(name)
        if child.is_file():
            if "/deprecated/" not in str(child):
                return child.read_bytes()
            deprecated = child
    return deprecated.read_bytes() if deprecated is not None else None


@cache
def cmap_directories() -> tuple[Traversable, ...]:
    root = resources.files(RESOURCE_PACKAGE).joinpath("cmaps")
    if not root.is_dir():
        return ()
    found: list[Traversable] = []
    candidates = [root]
    while candidates:
        current = candidates.pop()
        if current.name == "CMap":
            found.append(current)
            continue
        candidates.extend(child for child in current.iterdir() if child.is_dir())
    return tuple(found)


def resolve_cmap_decoder(name: str) -> CMapDecoder | None:
    if name in {"Identity-H", "Identity-V"}:
        return CMapDecoder.identity(wmode=int(name.endswith("-V")))
    data = resolve_cmap_resource(name)
    return CMapDecoder(data, usecmap_resolver=resolve_cmap_resource) if data is not None else None


__all__ = [
    "RESOURCE_PACKAGE",
    "resolve_cmap_resource",
    "resolve_cmap_decoder",
]

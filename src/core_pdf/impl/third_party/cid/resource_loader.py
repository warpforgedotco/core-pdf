from __future__ import annotations

from functools import lru_cache
from importlib import resources
from importlib.resources.abc import Traversable

from core_pdf.impl.third_party.cid.cmap import CMapDecoder


RESOURCE_PACKAGE = "core_pdf.impl.third_party.cid"


@lru_cache(maxsize=1)
def cmap_resource_root() -> Traversable:
    return resources.files(RESOURCE_PACKAGE).joinpath("resources")


def normalized_cmap_name(name: str) -> str:
    return name[1:] if name.startswith("/") else name


@lru_cache(maxsize=1)
def cmap_resource_index() -> dict[str, Traversable]:
    root = cmap_resource_root()
    if not root.is_dir():
        return {}

    index: dict[str, Traversable] = {}
    candidates: list[tuple[Traversable, str | None]] = [(root, None)]
    while candidates:
        current, parent_name = candidates.pop()
        for child in current.iterdir():
            if child.is_dir():
                candidates.append((child, child.name))
                continue
            if parent_name != "CMap":
                continue
            name = child.name
            existing = index.get(name)
            if existing is None or "/deprecated/" in str(existing):
                index[name] = child
    return index


def resolve_cmap_resource(name: str) -> bytes | None:
    resource = cmap_resource_index().get(normalized_cmap_name(name))
    if resource is None:
        return None
    return resource.read_bytes()


@lru_cache(maxsize=256)
def resolve_cmap_decoder(name: str) -> CMapDecoder | None:
    normalized_name = normalized_cmap_name(name)
    if normalized_name in {"Identity-H", "Identity-V"}:
        return CMapDecoder.identity(byte_width=2)
    if normalized_name in {"OneByteIdentityH", "OneByteIdentityV"}:
        return CMapDecoder.identity(byte_width=1)
    cmap_data = resolve_cmap_resource(normalized_name)
    if cmap_data is None:
        return None
    try:
        return CMapDecoder(
            cmap_data,
            usecmap_resolver=resolve_cmap_decoder,
        )
    except ValueError:
        return None


def has_cmap_resource(name: str) -> bool:
    return normalized_cmap_name(name) in cmap_resource_index()


__all__ = (
    "cmap_resource_index",
    "cmap_resource_root",
    "has_cmap_resource",
    "normalized_cmap_name",
    "resolve_cmap_decoder",
    "resolve_cmap_resource",
)

"""Load the defined Adobe CMap resources without recovery aliases."""

from __future__ import annotations

from functools import cache
from importlib import resources
from importlib.resources.abc import Traversable

from core_pdf.impl.spec.s_09_fonts.cmap_decoder import CMapDecoder

RESOURCE_PACKAGE = "core_pdf.impl.spec.s_09_fonts.data"


def normalized_cmap_name(name: str) -> str:
    return name.removeprefix("/")


@cache
def internal_cmap_resource_index() -> dict[str, Traversable]:
    """Map each CMap resource name to the file a lookup for it resolves to.

    The bundled resource tree is fixed at install time, so it is walked once
    rather than per lookup: a miss used to stat all 207 files, and resolving the
    CMaps of one CJK page walked the tree dozens of times over.

    The walk keeps the order the per-lookup scan used, so the entry chosen for a
    name is the one that scan would have returned: the first non-deprecated
    match wins outright, and a deprecated match is only a fallback -- the last
    one seen, which is what repeated assignment to its ``deprecated`` local left
    behind.
    """
    root = resources.files(RESOURCE_PACKAGE).joinpath("cmaps")
    if not root.is_dir():
        return {}

    preferred: dict[str, Traversable] = {}
    deprecated: dict[str, Traversable] = {}
    candidates: list[tuple[Traversable, str | None]] = [(root, None)]
    while candidates:
        current, parent_name = candidates.pop()
        for child in current.iterdir():
            if child.is_dir():
                candidates.append((child, child.name))
                continue
            if parent_name != "CMap":
                continue
            if "/deprecated/" not in str(child):
                preferred.setdefault(child.name, child)
            else:
                deprecated[child.name] = child
    return deprecated | preferred


def cmap_resource_exists(name: str) -> bool:
    """Whether a CMap resource is bundled, without reading it."""
    return normalized_cmap_name(name) in internal_cmap_resource_index()


def resolve_cmap_resource(name: str) -> bytes | None:
    entry = internal_cmap_resource_index().get(normalized_cmap_name(name))
    return entry.read_bytes() if entry is not None else None


def resolve_cmap_decoder(name: str) -> CMapDecoder | None:
    normalized = normalized_cmap_name(name)
    if normalized in {"Identity-H", "Identity-V"}:
        return CMapDecoder.identity(wmode=int(normalized.endswith("-V")))
    data = resolve_cmap_resource(normalized)
    return CMapDecoder(data, usecmap_resolver=resolve_cmap_decoder) if data is not None else None

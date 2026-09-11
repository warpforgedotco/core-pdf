"""Load the defined Adobe CMap resources without recovery aliases."""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable

from core_pdf_spec.s_09_fonts.cmap_decoder import CMapDecoder

RESOURCE_PACKAGE = "core_pdf_spec.s_09_fonts.data"


def resolve_cmap_resource(name: str) -> bytes | None:
    root = resources.files(RESOURCE_PACKAGE).joinpath("cmaps")
    if not root.is_dir():
        return None

    deprecated: Traversable | None = None
    candidates: list[tuple[Traversable, str | None]] = [(root, None)]
    while candidates:
        current, parent_name = candidates.pop()
        for child in current.iterdir():
            if child.is_dir():
                candidates.append((child, child.name))
                continue
            if parent_name != "CMap" or child.name != name:
                continue
            if "/deprecated/" not in str(child):
                return child.read_bytes()
            deprecated = child
    return deprecated.read_bytes() if deprecated is not None else None


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

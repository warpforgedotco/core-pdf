"""Load the defined Adobe CMap resources without recovery aliases."""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable

from core_pdf.impl.spec.s_09_fonts.cmap_decoder import CMapDecoder

RESOURCE_PACKAGE = "core_pdf.impl.spec.s_09_fonts.data"


def normalized_cmap_name(name: str) -> str:
    return name[1:] if name.startswith("/") else name


def resolve_cmap_resource(name: str) -> bytes | None:
    root = resources.files(RESOURCE_PACKAGE).joinpath("cmaps")
    if not root.is_dir():
        return None

    target = normalized_cmap_name(name)
    deprecated: Traversable | None = None
    candidates: list[tuple[Traversable, str | None]] = [(root, None)]
    while candidates:
        current, parent_name = candidates.pop()
        for child in current.iterdir():
            if child.is_dir():
                candidates.append((child, child.name))
                continue
            if parent_name != "CMap" or child.name != target:
                continue
            if "/deprecated/" not in str(child):
                return child.read_bytes()
            deprecated = child
    return deprecated.read_bytes() if deprecated is not None else None


def resolve_cmap_decoder(name: str) -> CMapDecoder | None:
    normalized = normalized_cmap_name(name)
    if normalized in {"Identity-H", "Identity-V"}:
        return CMapDecoder.identity(wmode=int(normalized.endswith("-V")))
    data = resolve_cmap_resource(normalized)
    return CMapDecoder(data, usecmap_resolver=resolve_cmap_decoder) if data is not None else None

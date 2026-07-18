from __future__ import annotations

from core_cmap.impl.cid.cmap import CMapDecoder, ToUnicodeCMap
from core_cmap.impl.cid.resource_loader import (
    has_cmap_resource,
    resolve_cmap_resource,
)
from core_cmap.impl.cid.widths import (
    CompactCIDWidthMap,
    FontWidthMap,
    SparseFontWidthMap,
    parse_cid_widths,
    scale_font_widths,
)

__all__ = (
    "CMapDecoder",
    "CompactCIDWidthMap",
    "FontWidthMap",
    "SparseFontWidthMap",
    "ToUnicodeCMap",
    "has_cmap_resource",
    "parse_cid_widths",
    "resolve_cmap_resource",
    "scale_font_widths",
)

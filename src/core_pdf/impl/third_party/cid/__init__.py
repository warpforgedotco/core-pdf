from __future__ import annotations

from core_pdf.impl.third_party.cid.cmap import CMapDecoder, ToUnicodeCMap
from core_pdf.impl.third_party.cid.resource_loader import (
    has_cmap_resource,
    resolve_cmap_resource,
)
from core_pdf.impl.third_party.cid.widths import (
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

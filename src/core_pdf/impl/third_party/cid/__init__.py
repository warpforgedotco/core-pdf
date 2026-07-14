# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.third_party.cid.cmap import CMapDecoder
from core_pdf.impl.third_party.cid.resource_loader import (
    has_cmap_resource,
    resolve_cmap_decoder,
    resolve_cmap_resource,
)

__all__ = (
    "CMapDecoder",
    "has_cmap_resource",
    "resolve_cmap_decoder",
    "resolve_cmap_resource",
)

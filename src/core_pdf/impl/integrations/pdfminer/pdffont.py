# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_pdf.impl.integrations.pdfminer.cmapdb import CMap, CMapDB
from core_pdf.impl.integrations.pdfminer.pdftypes import PDFStream, resolve1
from core_pdf.impl.integrations.pdfminer.psparser import PSLiteral, literal_name


class PDFFontError(Exception):
    pass


class PDFCIDFont:
    def __init__(self, rsrcmgr: Any, spec: dict[str, Any], strict: bool = False) -> None:
        del rsrcmgr
        self.spec = spec
        self.cmap = self.get_cmap_from_spec(spec, strict)
        self.vertical = self.cmap.is_vertical()
        self.default_disp = (None, 880) if self.vertical else 0
        base_font = resolve1(spec.get("BaseFont", "unknown"))
        self.fontname = (
            literal_name(base_font) if isinstance(base_font, PSLiteral) else str(base_font)
        )

    def _get_cmap_name(self, spec: dict[str, Any], strict: bool) -> str:
        del strict
        encoding = resolve1(spec.get("Encoding"))
        if isinstance(encoding, PSLiteral):
            return literal_name(encoding)
        if isinstance(encoding, PDFStream):
            name = resolve1(encoding.attrs.get("CMapName"))
            if isinstance(name, PSLiteral):
                return literal_name(name)
            if isinstance(name, str):
                return name
        return "unknown"

    def get_cmap_from_spec(self, spec: dict[str, Any], strict: bool) -> CMap:
        name = self._get_cmap_name(spec, strict)
        try:
            return CMapDB.get_cmap(name)
        except CMapDB.CMapNotFound as exc:
            if strict:
                raise PDFFontError(exc) from exc
            return CMap()

    def is_vertical(self) -> bool:
        return self.vertical

    def get_descent(self) -> float:
        return 0.0


__all__ = ("PDFCIDFont", "PDFFontError")

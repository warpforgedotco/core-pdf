# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from core_pdf.impl.integrations.pdfminer.pdffont import PDFCIDFont
from core_pdf.impl.integrations.pdfminer.pdfpage import PDFPage
from core_pdf.impl.integrations.pdfminer.psparser import LIT, literal_name

LITERAL_FONT = LIT("Font")


class PDFResourceManager:
    def __init__(self, caching: bool = True) -> None:
        self.caching = caching
        self._cached_fonts: dict[int, Any] = {}

    def get_font(self, objid: int | None, spec: dict[str, Any]) -> Any:
        if objid and objid in self._cached_fonts:
            return self._cached_fonts[objid]
        subtype = literal_name(spec["Subtype"]) if "Subtype" in spec else "Type1"
        font: Any
        if subtype in {"CIDFontType0", "CIDFontType2"}:
            font = PDFCIDFont(self, spec)
        else:
            font = SimpleNamespace(
                fontname=str(spec.get("BaseFont", "unknown")),
                is_vertical=lambda: False,
                get_descent=lambda: 0.0,
            )
        if objid and self.caching:
            self._cached_fonts[objid] = font
        return font


class PDFPageInterpreter:
    def __init__(self, rsrcmgr: PDFResourceManager, device: Any) -> None:
        self.rsrcmgr = rsrcmgr
        self.device = device
        self.textstate = SimpleNamespace(render=0, font=None)
        self.graphicstate = SimpleNamespace()

    def init_resources(self, resources: dict[str, Any]) -> None:
        del resources

    def process_page(self, page: PDFPage) -> None:
        layout = page.layout
        self.device.cur_item = layout
        self.device._result = layout

    def do_TJ(self, seq: Any) -> None:
        del seq


__all__ = ("LITERAL_FONT", "PDFPageInterpreter", "PDFResourceManager")

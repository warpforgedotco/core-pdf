# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_document.page_boxes import rotate_page_rect
from core_pdf.impl.engine.extraction.tables.protocols import PageTableHost


class PageTableTextGeometryMixin:
    def display_text_span_chars(
        self: PageTableHost,
    ) -> dict[int, list[tuple[str, float, float, float, float]]]:
        rotate = self.rotation
        page_width = self.width
        page_height = self.height
        chars_by_seqno: dict[int, list[tuple[str, float, float, float, float]]] = {}

        for span in self.get_text_spans():
            seqno = span["seqno"]
            chars: list[tuple[str, float, float, float, float]] = []
            for codepoint, origin_x, origin_y, rect in span["chars"]:
                x0, y0, x1, y1 = rotate_page_rect(
                    rect.x0,
                    rect.y0,
                    rect.x1,
                    rect.y1,
                    rotate=rotate,
                    page_width=page_width,
                    page_height=page_height,
                )
                chars.append((chr(codepoint), x0, y0, x1, y1))
            chars_by_seqno[seqno] = chars

        return chars_by_seqno


__all__ = ("PageTableTextGeometryMixin",)

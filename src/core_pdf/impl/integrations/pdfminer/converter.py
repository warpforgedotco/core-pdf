# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_pdf.impl.integrations.pdfminer.layout import LAParams, LTPage


class PDFPageAggregator:
    def __init__(self, rsrcmgr: Any, pageno: int = 1, laparams: LAParams | None = None) -> None:
        del rsrcmgr, pageno
        self.laparams = laparams or LAParams()
        self.cur_item: Any = None
        self._result: LTPage | None = None

    def get_result(self) -> LTPage:
        if self._result is None:
            raise ValueError("no page has been processed")
        return self._result


__all__ = ("PDFPageAggregator",)

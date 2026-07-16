# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Container, Iterator
from contextlib import suppress
from typing import Any, BinaryIO

from core_pdf.impl.engine.extraction.document import PdfDocument
from core_pdf.impl.integrations.pdfminer.high_level import _page_layout
from core_pdf.impl.integrations.pdfminer.layout import LAParams, LTPage
from core_pdf.impl.integrations.pdfminer.psparser import LIT
from core_pdf.impl.primitives import PdfName, PdfString


def _compat_object(value: Any, page: Any) -> Any:
    resolver = page.document.resolver
    with suppress(Exception):
        value = resolver.resolve(value)
    if isinstance(value, PdfName):
        return LIT(value.value)
    if isinstance(value, PdfString):
        return value.data
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            name = key.value if isinstance(key, PdfName) else str(key)
            result[name] = _compat_object(child, page)
        return result
    if isinstance(value, (list, tuple)):
        return [_compat_object(child, page) for child in value]
    return value


class PDFPage:
    def __init__(self, page: Any) -> None:
        self.core_page = page
        self.pageid = page.page_number
        self.rotate = page.rotation
        self.mediabox = page.media_box
        self.layout: LTPage = _page_layout(page, LAParams())
        self.annots = [
            _compat_object(annotation.dict, page) for annotation in page.get_annotations()
        ]

    @classmethod
    def get_pages(
        cls,
        fp: BinaryIO,
        pagenos: Container[int] | None = None,
        maxpages: int = 0,
        password: str = "",
        caching: bool = True,
        check_extractable: bool = False,
    ) -> Iterator[PDFPage]:
        del caching, check_extractable
        with PdfDocument.open(fp, password=password) as document:
            yielded = 0
            for index, page in enumerate(document.pages):
                if pagenos is not None and index not in pagenos:
                    continue
                if maxpages and yielded >= maxpages:
                    break
                yield cls(page)
                yielded += 1


__all__ = ("PDFPage",)

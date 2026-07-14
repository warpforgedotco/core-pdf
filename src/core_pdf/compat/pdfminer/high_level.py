# SPDX-License-Identifier: AGPL-3.0-only
"""High-level pdfminer.six-compatible APIs backed by core-pdf."""

from __future__ import annotations

from collections.abc import Container
from os import PathLike, fspath
from typing import BinaryIO, TypeAlias, cast

from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument
from core_pdf.impl.engine.spec.s_07_document.page import PdfPage
from core_pdf.impl.engine.spec.s_07_syntax.primitives import PdfSource

FileOrName: TypeAlias = str | PathLike[str] | BinaryIO
PdfInput: TypeAlias = FileOrName | bytes | bytearray | memoryview


def extract_text(
    pdf_file: PdfInput,
    password: str = "",
    page_numbers: Container[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    codec: str = "utf-8",
    laparams: object | None = None,
) -> str:
    """Parse and return text contained in a PDF file.

    This mirrors ``pdfminer.high_level.extract_text`` for the high-level text
    extraction path. ``caching``, ``codec``, and ``laparams`` are accepted for
    call-site compatibility; extraction is performed by core-pdf's engine.
    """
    del caching, codec, laparams

    source = _normalize_pdf_input(pdf_file)
    document = PdfDocument.open(source, password=password)
    page_texts: list[str] = []
    selected_pages = page_numbers

    for page_index, page in enumerate(document.pages):
        if selected_pages and page_index not in selected_pages:
            continue
        page_texts.append(page.extract_text(layout=True))
        _clear_page_state(page)
        if maxpages and maxpages <= page_index + 1:
            break

    if not page_texts:
        return ""
    return "\f".join(page_texts) + "\f"


def _normalize_pdf_input(pdf_file: PdfInput) -> PdfSource:
    if isinstance(pdf_file, PathLike):
        path = fspath(pdf_file)
        if not isinstance(path, str):
            raise TypeError(f"Unsupported input type: {type(pdf_file).__name__}")
        return path
    return cast(PdfSource, pdf_file)


def _clear_page_state(page: PdfPage) -> None:
    page.state = None
    page.graphics = None
    page.grid_lines = None
    page.texttrace = None
    page.tables = {}


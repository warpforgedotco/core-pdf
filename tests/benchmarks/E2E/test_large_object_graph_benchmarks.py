# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end benchmarks for parsing PDFs with very large object graphs.

Builds a hand-authored, flat (non-nested) page tree with thousands of tiny
pages sharing one font resource. This isolates classic xref table parsing and
indirect-object resolution/caching at scale from content-stream complexity,
which is covered separately by ``test_dense_content_stream_benchmarks.py``.
"""

from __future__ import annotations

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine.writing import serialize_pdf_file
from core_pdf.impl.objects import PdfName, PdfReference, PdfStream

FONT_OBJECT_NUMBER = 3


def build_flat_page_tree_pdf(page_count: int) -> bytes:
    objects: dict[int, object] = {
        1: {PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): PdfReference(2)},
        FONT_OBJECT_NUMBER: {
            PdfName.of("Type"): PdfName.of("Font"),
            PdfName.of("Subtype"): PdfName.of("Type1"),
            PdfName.of("BaseFont"): PdfName.of("Helvetica"),
        },
    }
    kids = []
    next_object_number = FONT_OBJECT_NUMBER + 1
    for page_index in range(page_count):
        page_object_number = next_object_number
        content_object_number = next_object_number + 1
        next_object_number += 2
        kids.append(PdfReference(page_object_number))
        objects[page_object_number] = {
            PdfName.of("Type"): PdfName.of("Page"),
            PdfName.of("Parent"): PdfReference(2),
            PdfName.of("MediaBox"): [0, 0, 612, 792],
            PdfName.of("Resources"): {
                PdfName.of("Font"): {PdfName.of("F1"): PdfReference(FONT_OBJECT_NUMBER)}
            },
            PdfName.of("Contents"): PdfReference(content_object_number),
        }
        objects[content_object_number] = PdfStream(
            {}, f"BT /F1 10 Tf 36 750 Tm (Page {page_index}) Tj ET".encode()
        )
    objects[2] = {
        PdfName.of("Type"): PdfName.of("Pages"),
        PdfName.of("Kids"): kids,
        PdfName.of("Count"): page_count,
    }
    return serialize_pdf_file(objects, trailer={PdfName.of("Root"): PdfReference(1)})


OPEN_PAGE_COUNT = 8_000
EXTRACT_PAGE_COUNT = 1_000
OPEN_PDF_BYTES = build_flat_page_tree_pdf(OPEN_PAGE_COUNT)
EXTRACT_PDF_BYTES = build_flat_page_tree_pdf(EXTRACT_PAGE_COUNT)


def internal_open_and_count_pages(pdf_bytes: bytes) -> int:
    with PdfDocument.open(pdf_bytes) as document:
        return len(document.pages)


def internal_open_and_extract_all(pdf_bytes: bytes) -> int:
    with PdfDocument.open(pdf_bytes) as document:
        extracted = document.extract()
        return len(extracted.pages)


def test_large_page_tree_open_benchmark(benchmark) -> None:
    result = benchmark.pedantic(
        internal_open_and_count_pages,
        args=(OPEN_PDF_BYTES,),
        iterations=1,
        rounds=3,
    )
    assert result == OPEN_PAGE_COUNT


@pytest.mark.benchmark_high_impact
def test_large_page_tree_extract_all_benchmark(benchmark) -> None:
    result = benchmark.pedantic(
        internal_open_and_extract_all,
        args=(EXTRACT_PDF_BYTES,),
        iterations=1,
        rounds=1,
    )
    assert result == EXTRACT_PAGE_COUNT

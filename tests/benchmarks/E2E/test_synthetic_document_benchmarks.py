# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end benchmarks over synthetic, fixture-free PDFs.

Unlike the real-world fixtures in ``test_real_pdf_e2e_benchmarks.py``, these
documents are generated in-process, so they need no SCORE-Bench/LFS assets
and scale deterministically with page and content volume.
"""

from __future__ import annotations

from typing import Any

from core_pdf import PdfDocument, serialize_document_to_pdf
from core_pdf.impl.engine.rendering import RenderOptions
from core_pdf.impl.engine.structured import (
    Block,
    BlockKind,
    Document,
    Page,
    Table,
    TableCell,
    TextLine,
)

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0


def build_synthetic_document(
    *, page_count: int, paragraphs_per_page: int = 4, lines_per_paragraph: int = 6
) -> Document:
    pages = []
    for page_index in range(page_count):
        blocks = [
            Block(
                0,
                BlockKind.HEADING,
                lines=(TextLine(f"Section {page_index + 1}"),),
                level=1,
            ),
        ]
        for paragraph_index in range(paragraphs_per_page):
            lines = tuple(
                TextLine(
                    f"Paragraph {paragraph_index} line {line_index} of synthetic "
                    f"page {page_index}: the quick brown fox jumps over the lazy dog."
                )
                for line_index in range(lines_per_paragraph)
            )
            blocks.append(Block(paragraph_index + 1, BlockKind.PARAGRAPH, lines=lines))
        table = Table(
            len(blocks),
            rows=(
                (TableCell(0, 0, "Name"), TableCell(0, 1, "Value")),
                (TableCell(1, 0, "Alpha"), TableCell(1, 1, "1")),
                (TableCell(2, 0, "Beta"), TableCell(2, 1, "2")),
            ),
        )
        pages.append(
            Page(
                page_number=page_index + 1,
                width=PAGE_WIDTH,
                height=PAGE_HEIGHT,
                blocks=tuple(blocks),
                tables=(table,),
            )
        )
    return Document(pages=tuple(pages))


SMALL_DOCUMENT = build_synthetic_document(page_count=5)
LARGE_DOCUMENT = build_synthetic_document(page_count=50)
SMALL_PDF_BYTES = serialize_document_to_pdf(SMALL_DOCUMENT)
LARGE_PDF_BYTES = serialize_document_to_pdf(LARGE_DOCUMENT)


def internal_open_and_extract(pdf_bytes: bytes) -> int:
    with PdfDocument.open(pdf_bytes) as document:
        extracted = document.extract()
        return len(extracted.pages)


def internal_full_pipeline(pdf_bytes: bytes) -> dict[str, Any]:
    with PdfDocument.open(pdf_bytes) as document:
        extracted = document.extract()
        raster_pixels = 0
        for page in document.pages:
            rendered = page.render(RenderOptions(include_text=False))
            raster = rendered.rasterize(scale=1.5, cache=False)
            raster_pixels += raster.width * raster.height
        return {"pages": len(extracted.pages), "raster_pixels": raster_pixels}


def test_synthetic_pdf_serialize_small_benchmark(benchmark) -> None:
    result = benchmark(serialize_document_to_pdf, SMALL_DOCUMENT)
    assert result


def test_synthetic_pdf_serialize_large_benchmark(benchmark) -> None:
    result = benchmark(serialize_document_to_pdf, LARGE_DOCUMENT)
    assert result


def test_synthetic_pdf_open_and_extract_small_benchmark(benchmark) -> None:
    result = benchmark.pedantic(
        internal_open_and_extract,
        args=(SMALL_PDF_BYTES,),
        iterations=1,
        rounds=5,
    )
    assert result == 5


def test_synthetic_pdf_open_and_extract_large_benchmark(benchmark) -> None:
    result = benchmark.pedantic(
        internal_open_and_extract,
        args=(LARGE_PDF_BYTES,),
        iterations=1,
        rounds=3,
    )
    assert result == 50


def test_synthetic_pdf_full_pipeline_large_benchmark(benchmark) -> None:
    result = benchmark.pedantic(
        internal_full_pipeline,
        args=(LARGE_PDF_BYTES,),
        iterations=1,
        rounds=1,
    )
    assert result["pages"] == 50
    assert result["raster_pixels"] > 0

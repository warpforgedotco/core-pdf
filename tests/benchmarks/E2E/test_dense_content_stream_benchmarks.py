# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end benchmarks for operator-dense native content streams.

Hand-builds a single page whose content stream packs thousands of vector
rectangles interleaved with short text runs -- far denser than anything the
semantic ``Document``/``Block`` writer produces. This stresses the lexer and
operator dispatch loop (``operations.py``/``state.py``) plus the vector-heavy
side of native page classification, at a density that a complex diagram or a
dense ruled table would hit in the wild.

MODERATE and DENSE are kept as separate benchmarks (rather than one bigger
run) because the vector-classification cost was observed to jump sharply
somewhere between them -- tracking both sides lets CodSpeed surface a
regression in exactly where that jump moves.
"""

from __future__ import annotations

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine.render.display import RenderOptions
from core_pdf.impl.engine.writing import serialize_pdf_file
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.primitives import PdfName, PdfReference

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0
RECT_SIZE = 8.0


def build_dense_content_stream(rect_count: int, *, seed: int) -> bytes:
    rng = numpy.random.default_rng(seed)
    positions = rng.uniform(0, PAGE_HEIGHT - RECT_SIZE, size=(rect_count, 2))
    colors = rng.uniform(0, 1, size=(rect_count, 3))
    lines: list[bytes] = []
    for index in range(rect_count):
        x, y = positions[index]
        r, g, b = colors[index]
        lines.append(f"{r:.3f} {g:.3f} {b:.3f} rg".encode())
        lines.append(f"{x:.2f} {y:.2f} {RECT_SIZE:.2f} {RECT_SIZE:.2f} re".encode())
        lines.append(b"f" if index % 3 else b"S")
        if index % 25 == 0:
            lines.append(b"q")
            lines.append(f"1 0 0 1 {x:.2f} {y:.2f} cm".encode())
            lines.append(b"Q")
        if index % 10 == 0:
            lines.append(b"BT")
            lines.append(b"/F1 6 Tf")
            lines.append(f"1 0 0 1 {x:.2f} {y:.2f} Tm".encode())
            lines.append(f"(N{index}) Tj".encode())
            lines.append(b"ET")
    return b"\n".join(lines)


def build_single_page_pdf(content: bytes) -> bytes:
    return serialize_pdf_file(
        {
            1: {PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): PdfReference(2)},
            2: {
                PdfName.of("Type"): PdfName.of("Pages"),
                PdfName.of("Kids"): [PdfReference(3)],
                PdfName.of("Count"): 1,
            },
            3: {
                PdfName.of("Type"): PdfName.of("Page"),
                PdfName.of("Parent"): PdfReference(2),
                PdfName.of("MediaBox"): [0, 0, PAGE_WIDTH, PAGE_HEIGHT],
                PdfName.of("Resources"): {PdfName.of("Font"): {PdfName.of("F1"): PdfReference(5)}},
                PdfName.of("Contents"): PdfReference(4),
            },
            4: PdfStream({}, content),
            5: {
                PdfName.of("Type"): PdfName.of("Font"),
                PdfName.of("Subtype"): PdfName.of("Type1"),
                PdfName.of("BaseFont"): PdfName.of("Helvetica"),
            },
        },
        trailer={PdfName.of("Root"): PdfReference(1)},
    )


MODERATE_PDF_BYTES = build_single_page_pdf(build_dense_content_stream(400, seed=1))
DENSE_PDF_BYTES = build_single_page_pdf(build_dense_content_stream(800, seed=2))


def internal_open_extract_render(pdf_bytes: bytes) -> dict[str, int]:
    with PdfDocument.open(pdf_bytes) as document:
        extracted = document.extract()
        rendered = document.pages[0].render(RenderOptions(include_text=False))
        raster = rendered.rasterize(scale=1.0, cache=False)
        return {
            "pages": len(extracted.pages),
            "raster_pixels": raster.width * raster.height,
        }


def test_dense_content_stream_moderate_benchmark(benchmark) -> None:
    result = benchmark.pedantic(
        internal_open_extract_render,
        args=(MODERATE_PDF_BYTES,),
        iterations=1,
        rounds=3,
    )
    assert result["pages"] == 1
    assert result["raster_pixels"] > 0


@pytest.mark.benchmark_high_impact
def test_dense_content_stream_dense_benchmark(benchmark) -> None:
    result = benchmark.pedantic(
        internal_open_extract_render,
        args=(DENSE_PDF_BYTES,),
        iterations=1,
        rounds=1,
    )
    assert result["pages"] == 1
    assert result["raster_pixels"] > 0

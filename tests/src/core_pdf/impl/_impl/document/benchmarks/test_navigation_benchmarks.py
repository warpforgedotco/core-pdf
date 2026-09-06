# SPDX-License-Identifier: AGPL-3.0-only
"""Navigation and page-label operations over a complete multipage document."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from core_pdf import PdfDocument
from core_pdf.api.compat.llamaindex import load_data
from tests.helpers.pdf_bytes import assemble_pdf

PAGE_COUNT = 100
FIRST_PAGE = 6
FIRST_OUTLINE = FIRST_PAGE + PAGE_COUNT


def internal_navigation_pdf() -> bytes:
    pages = " ".join(f"{FIRST_PAGE + index} 0 R" for index in range(PAGE_COUNT))
    destinations = " ".join(
        f"(page-{index:03d}) [{FIRST_PAGE + index} 0 R /Fit]" for index in range(PAGE_COUNT)
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R /PageLabels 3 0 R "
        b"/Names << /Dests 4 0 R >> /Outlines 5 0 R >>",
        f"<< /Type /Pages /Count {PAGE_COUNT} /Kids [{pages}] /MediaBox [0 0 612 792] >>".encode(),
        b"<< /Nums [0 << /S /r >> 12 << /S /D /St 1 /P (A-) >> "
        b"75 << /S /D /St 1 /P (Appendix-) >>] >>",
        f"<< /Names [{destinations}] >>".encode(),
        f"<< /Type /Outlines /First {FIRST_OUTLINE} 0 R "
        f"/Last {FIRST_OUTLINE + PAGE_COUNT - 1} 0 R /Count {PAGE_COUNT} >>".encode(),
        *[b"<< /Type /Page /Parent 2 0 R >>" for _ in range(PAGE_COUNT)],
    ]
    for index in range(PAGE_COUNT):
        links = ""
        if index:
            links += f" /Prev {FIRST_OUTLINE + index - 1} 0 R"
        if index + 1 < PAGE_COUNT:
            links += f" /Next {FIRST_OUTLINE + index + 1} 0 R"
        destination = f"(page-{index:03d})" if index % 2 else f"[{FIRST_PAGE + index} 0 R /Fit]"
        objects.append(
            f"<< /Title (Chapter {index + 1}) /Parent 5 0 R /Dest {destination}{links} >>".encode()
        )
    return assemble_pdf(objects)


@pytest.fixture(scope="module")
def navigation_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("navigation-benchmark") / "navigation.pdf"
    path.write_bytes(internal_navigation_pdf())
    return path


@pytest.fixture(scope="module")
def navigation_document(navigation_path: Path) -> Iterator[PdfDocument]:
    with PdfDocument(navigation_path) as document:
        yield document


def test_named_destinations_benchmark(benchmark, navigation_document: PdfDocument) -> None:
    destinations = benchmark(navigation_document.named_destinations)

    assert [destinations[f"page-{index:03d}"].page_index for index in range(PAGE_COUNT)] == list(
        range(PAGE_COUNT)
    )


def test_outlines_benchmark(benchmark, navigation_document: PdfDocument) -> None:
    outlines = benchmark(navigation_document.iter_outlines)

    assert [outline.page_index for outline in outlines] == list(range(PAGE_COUNT))


def test_reader_page_labels_benchmark(benchmark, navigation_path: Path) -> None:
    # This public reader opens a fresh document and returns one item per page;
    # corpus construction and filesystem setup stay outside the measured call.
    pages = benchmark(load_data, navigation_path)

    assert len(pages) == PAGE_COUNT
    assert [pages[index].metadata["page_label"] for index in (0, 11, 12, 74, 75, 99)] == [
        "i",
        "xii",
        "A-1",
        "A-63",
        "Appendix-1",
        "Appendix-25",
    ]

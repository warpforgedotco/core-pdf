# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

from core_pdf.impl.engine.extraction.document import PdfDocument
from core_pdf.impl.types import PdfSource

TESTS_DIR = Path(__file__).parents[4]
SAMPLES_DIR = TESTS_DIR / "fixtures" / "pdfminer.six" / "samples"
ACROFORM_PDF = SAMPLES_DIR / "acroform" / "AcroForm_TEST.pdf"
PAGELABELS_PDF = SAMPLES_DIR / "contrib" / "pagelabels.pdf"
CMAP_OTHER_FONTS_PDF = SAMPLES_DIR / "contrib" / "issue-598-cmap-other-fonts.pdf"
CYCLIC_XOBJECTS_PDF = SAMPLES_DIR / "contrib" / "issue-1113-evil-xobjects.pdf"
ACROFORM_TEXT = "BUTTON\n\nCHECKBOX\n\nRADIO BUTTON\n\nDROPDOWN\n\nLIST\n\nCOMBO LIST\n\nTEXT\f"


def extract_text(
    source: PdfSource,
    *,
    page_numbers: set[int] | None = None,
    maxpages: int = 0,
) -> str:
    with PdfDocument.open(source) as document:
        doc = cast(Any, document)
        if page_numbers is not None:
            return "\f".join(doc.pages[index].extract_text() for index in page_numbers) + "\f"
        if maxpages:
            return "\f".join(page.extract_text() for page in doc.pages[:maxpages]) + "\f"
        return cast(str, doc.extract_text())


def test_extract_text_accepts_file_like_object() -> None:
    assert extract_text(io.BytesIO(ACROFORM_PDF.read_bytes())) == ACROFORM_TEXT


def test_extract_text_accepts_path_object() -> None:
    assert extract_text(ACROFORM_PDF) == ACROFORM_TEXT


def test_document_extract_text_accepts_file_like_object() -> None:
    with PdfDocument.open(io.BytesIO(ACROFORM_PDF.read_bytes())) as document:
        assert cast(Any, document).extract_text() == ACROFORM_TEXT


def test_extract_text_supports_page_numbers() -> None:
    result = extract_text(PAGELABELS_PDF, page_numbers={1})

    assert result == "2 Second section\nSome text here...\n\niv\f"


def test_extract_text_supports_maxpages() -> None:
    result = extract_text(PAGELABELS_PDF, maxpages=2)

    assert result.count("\f") == 2
    assert result.startswith("Contents\n1 First section iii")
    assert result.endswith("2 Second section\nSome text here...\n\niv\f")


def test_extract_text_decodes_cmaps_after_font_selection() -> None:
    result = extract_text(CMAP_OTHER_FONTS_PDF)

    assert "\x00" not in result
    assert "DIAGNOSTIC TOOLS AND SOFTWARE 120." in result
    assert "Clutch/Gearbox/MCR valve Fault: Signal voltage above threshold" in result


def test_distinct_cyclic_xobjects_with_identical_stream_lengths_are_both_extracted() -> None:
    assert extract_text(CYCLIC_XOBJECTS_PDF) == "Hello world\nHello world\f"

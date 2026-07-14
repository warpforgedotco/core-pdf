# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import io
from pathlib import Path

from core_pdf.compat.pdfminer import extract_text
from core_pdf.compat.pdfminer.high_level import extract_text as extract_text_high_level

TESTS_DIR = Path(__file__).parents[4]
SAMPLES_DIR = TESTS_DIR / "fixtures" / "pdfminer.six" / "samples"
ACROFORM_PDF = SAMPLES_DIR / "acroform" / "AcroForm_TEST.pdf"
PAGELABELS_PDF = SAMPLES_DIR / "contrib" / "pagelabels.pdf"
CMAP_OTHER_FONTS_PDF = SAMPLES_DIR / "contrib" / "issue-598-cmap-other-fonts.pdf"
ACROFORM_TEXT = (
    "BUTTON\n\n"
    "CHECKBOX\n\n"
    "RADIO BUTTON\n\n"
    "DROPDOWN\n\n"
    " LIST\n\n"
    "COMBO LIST\n\n"
    "TEXT\f"
)


def test_extract_text_accepts_file_like_object() -> None:
    assert extract_text(io.BytesIO(ACROFORM_PDF.read_bytes())) == ACROFORM_TEXT


def test_extract_text_accepts_path_object() -> None:
    assert extract_text(ACROFORM_PDF) == ACROFORM_TEXT


def test_high_level_module_reexports_extract_text() -> None:
    assert extract_text_high_level(io.BytesIO(ACROFORM_PDF.read_bytes())) == ACROFORM_TEXT


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

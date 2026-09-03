# SPDX-License-Identifier: AGPL-3.0-only
"""Extracted text keeps the characters the fonts actually produced."""

from __future__ import annotations

from core_pdf.impl.spec.s_07_content.text_helpers import normalize_extracted_text
from tests.helpers.pdf_bytes import first_page_runs, one_page_pdf


def page_text(data: bytes) -> str:
    return "".join(run.text for run in first_page_runs(data))


def form_feed_pdf() -> bytes:
    """WinAnsi code 014, which the font maps to a form feed and nothing else."""
    content = b"BT /F1 12 Tf 50 400 Td (A\x0cB) Tj ET\n"
    return one_page_pdf(
        content,
        font=b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    )


def test_form_feed_is_not_rewritten_to_a_ligature() -> None:
    # Code 014 used to be rewritten to U+FB01 for every font, which invented an
    # "fi" wherever a document legitimately emitted a form feed.
    text = page_text(form_feed_pdf())
    assert "ﬁ" not in text
    assert "\x0c" in text


def test_lone_surrogates_are_dropped() -> None:
    # Surrogates cannot be encoded to UTF-8, so they must not survive.
    assert normalize_extracted_text("a\ud800b") == "ab"


def test_ordinary_text_is_returned_unchanged() -> None:
    assert normalize_extracted_text("plain ascii") == "plain ascii"
    assert normalize_extracted_text("caf\u00e9 \u2014 na\u00efve") == "caf\u00e9 \u2014 na\u00efve"

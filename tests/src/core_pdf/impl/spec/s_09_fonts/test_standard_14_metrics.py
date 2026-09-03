# SPDX-License-Identifier: AGPL-3.0-only
"""The standard 14 fonts may omit /Widths, and the reader must supply metrics.

9.6.2.2 lets FirstChar, LastChar, Widths and FontDescriptor all be absent for
these fonts. Falling back to MissingWidth instead advances every glyph by a
full em, which stretched a line of Times to roughly twice its true width and
pushed the tail of each line off the page.
"""

from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder
from tests.helpers.pdf_bytes import first_page_runs, one_page_pdf

# Helvetica advance widths, in 1/1000 em, from the shipped AFM tables.
HELVETICA = {"H": 722, "e": 556, "l": 222, "o": 556, " ": 278}
TIMES_ROMAN = {"A": 722, "e": 444, "i": 278, "l": 278, ".": 250, " ": 250}


def one_line_pdf(base_font: bytes, text: bytes, size: int = 10) -> bytes:
    content = b"BT /F1 %d Tf 50 700 Td (%s) Tj ET\n" % (size, text)
    return one_page_pdf(
        content, font=b"<< /Type /Font /Subtype /Type1 /BaseFont /" + base_font + b" >>"
    )


def run_width(data: bytes) -> float:
    runs = first_page_runs(data)
    assert len(runs) == 1
    return runs[0].bbox[2] - runs[0].bbox[0]


def test_helvetica_without_widths_uses_the_built_in_metrics() -> None:
    expected = sum(HELVETICA[c] for c in "Hello") / 1000 * 10
    assert run_width(one_line_pdf(b"Helvetica", b"Hello")) == pytest.approx(expected, abs=0.05)


def test_times_roman_without_widths_uses_the_built_in_metrics() -> None:
    expected = sum(TIMES_ROMAN[c] for c in "Aeil.") / 1000 * 10
    assert run_width(one_line_pdf(b"Times-Roman", b"Aeil.")) == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize(
    ("base_font", "char", "expected"),
    [
        ("Helvetica", "H", 722.0),
        ("Times-Roman", "A", 722.0),
        ("Courier", "A", 600.0),
        ("Times-Italic", " ", 250.0),
    ],
)
def test_decoder_exposes_standard_14_widths(base_font: str, char: str, expected: float) -> None:
    decoder = FontDecoder({"Subtype": "Type1", "BaseFont": base_font})
    assert decoder.fast_widths[ord(char)] == pytest.approx(expected)


def test_explicit_widths_still_win_over_the_built_in_table() -> None:
    # A standard font may be overridden by supplying the entries explicitly.
    content = b"BT /F1 10 Tf 50 700 Td (HH) Tj ET\n"
    data = one_page_pdf(
        content,
        font=b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/FirstChar 72 /LastChar 72 /Widths [1000] >>",
    )
    assert run_width(data) == pytest.approx(20.0, abs=0.05)


def test_type1_without_encoding_falls_back_to_standard_encoding() -> None:
    """9.6.6.1: a font program's built-in encoding governs when the dictionary
    supplies none. Where it cannot be read back out of the program, Annex D's
    Latin text encoding is the stand-in -- PDFDocEncoding, which was used
    before, encodes text strings and has no business decoding glyphs.
    """
    decoder = FontDecoder({"Subtype": "Type1", "BaseFont": "Minion-Regular"})
    assert decoder.base_encoding == "StandardEncoding"
    # 0320 is emdash in StandardEncoding; PDFDocEncoding put LATIN CAPITAL
    # LETTER ETH there, which is what a real document rendered as 79 stray Ð.
    assert decoder.decode(bytes([0xD0])) == "—"
    assert decoder.decode(bytes([0xB1])) == "–"


def test_explicit_encoding_still_wins_for_type1() -> None:
    decoder = FontDecoder(
        {"Subtype": "Type1", "BaseFont": "Minion-Regular", "Encoding": "WinAnsiEncoding"}
    )
    assert decoder.base_encoding == "WinAnsiEncoding"
    assert decoder.decode(bytes([0xD0])) == "Ð"

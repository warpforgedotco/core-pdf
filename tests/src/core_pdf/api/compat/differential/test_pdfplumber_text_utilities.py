"""Utility and page text extraction must honor the same word options."""

from io import BytesIO

import pytest

from core_pdf.api.compat import pdfplumber as compat

reference = pytest.importorskip("pdfplumber")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("namespace", ["utils", "text"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"x_tolerance": 5},
        {"x_tolerance_ratio": 0.5},
        {"split_at_punctuation": True},
        {"expand_ligatures": False},
    ],
)
def test_text_utilities_honor_word_options(namespace, options):
    chars = [
        {
            "text": text,
            "x0": x,
            "x1": x + 5,
            "top": 10,
            "bottom": 20,
            "doctop": 10,
            "size": 10,
            "upright": True,
        }
        for text, x in [("A", 0), ("B", 9), (",", 14), ("ﬁ", 19)]
    ]
    utilities = [
        library.utils if namespace == "utils" else library.utils.text
        for library in (reference, compat)
    ]
    expected = utilities[0].extract_text(chars, **options)
    assert expected
    assert utilities[1].extract_text(chars, **options) == expected


def test_page_and_utility_text_agree_on_same_captured_characters(text_pdf_bytes):
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        page = pdf.pages[0]
        for options in ({}, {"x_tolerance": 0}, {"x_tolerance": 100}):
            expected = page.extract_text(**options)
            assert compat.utils.extract_text(page.chars, **options) == expected
            assert compat.utils.text.extract_text(page.chars, **options) == expected

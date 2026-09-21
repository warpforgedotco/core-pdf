import re
from io import BytesIO
from typing import Any

import pytest

from core_pdf.api.compat import pdfplumber as compat

reference = pytest.importorskip("pdfplumber")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize(
    ("pattern", "options"),
    [
        ("fi", {}),
        ("B", {}),
        ("(f)(i)", {}),
        ("(f)(i)", {"main_group": 2}),
        ("FI", {"case": False}),
        ("A.fi", {"regex": False}),
        (r"\s+", {}),
        (r"(?=B)", {}),
        ("", {}),
        (re.compile("fi"), {}),
    ],
)
@pytest.mark.parametrize("return_chars", [False, True])
@pytest.mark.parametrize("return_groups", [False, True])
def test_search_matches_reference_textmap_spans_and_options(
    text_pdf_bytes, pattern, options, return_chars, return_groups
):
    from pdfplumber.utils.text import TextMap

    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        page = pdf.pages[0]
        template = page.chars[0]
        chars: list[dict[str, Any]] = [
            {**template, "text": text, "x0": index * 10, "x1": index * 10 + 8}
            for index, text in enumerate(("A.", "fi", " ", "B"))
        ]
        page._objects = {"char": chars}
        pairs: list[tuple[str, dict[str, Any] | None]] = [
            (letter, char) for char in chars for letter in char["text"]
        ]
        expected = TextMap(pairs, "ttb", "ltr").search(
            pattern, return_chars=return_chars, return_groups=return_groups, **options
        )
        actual = page.search(
            pattern, return_chars=return_chars, return_groups=return_groups, layout=True, **options
        )
        assert [
            {key: value for key, value in match.items() if key != "doctop"} for match in actual
        ] == expected
        if return_chars:
            assert all(
                any(char is source for source in chars)
                for match in actual
                for char in match["chars"]
            )


@pytest.mark.parametrize("options", [{"regex": False}, {"case": False}])
def test_compiled_pattern_rejects_conflicting_options(text_pdf_bytes, options):
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        with pytest.raises(ValueError):
            pdf.pages[0].search(re.compile("Hello"), **options)


def test_compiled_pattern_with_no_match_returns_empty_result(text_pdf_bytes):
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        pattern: Any = re.compile("absent")
        assert pdf.pages[0].search(pattern) == []

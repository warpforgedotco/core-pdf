import pytest

from core_pdf_compat._text_state import (
    append_directional_text,
    is_neutral_character,
    number_operand,
    positioned_text,
)
from core_pdf_compat.pypdf import finds_startxref

TRAILER = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n"


@pytest.mark.parametrize(
    ("tail", "found"),
    [
        (b"startxref\n9\n%%EOF\n", True),
        (b"startxref\n9\n%%EOF", True),
        # An inline marker on the last line (py-pdf/pypdf#4127).
        (b"startxref\n9%%EOF", True),
        # A truncated marker on the last line.
        (b"startxref\n9\n%%E", True),
        # Duplicate markers below the one closing the revision (#4008).
        (b"startxref\n9\n%%EOF\n%%EOF\n", True),
        (b"startxref 9\n%%EOF\n", True),
        # A corrupt trailing pointer recovers an earlier intact one (#3238).
        (b"startxref\n9\n%%EOF\ntartxref\n12\n%%EOF\n", True),
        (b"startxref\n9\n", False),
        (b"9\n%%EOF\n", False),
        (b"startxref x\n%%EOF\n", False),
        # Only the last line may carry the marker after other bytes.
        (b"startxref\n9 %%EOF\ntrailing\n", False),
    ],
)
def test_startxref_search_follows_pypdf(tail, found):
    assert finds_startxref(TRAILER + tail) is found


def test_fragment_inside_the_line_span_continues_the_line():
    # A row label centered between the two baselines of a wrapped cell
    # (py-pdf/pypdf#4130): moving 6 units would break the line at size 5, but
    # the baseline lies inside the span already on the line.
    output: list[str] = []
    text, last, span = positioned_text(
        output,
        "cell",
        "",
        (5, 88.0, 100.0),
        previous_text_matrix=[1, 0, 0, 1, 0, 100],
        previous_current_matrix=[1, 0, 0, 1, 0, 0],
        text_matrix=[1, 0, 0, 1, 0, 94],
        current_matrix=[1, 0, 0, 1, 0, 0],
        line_height=5.0,
        font_size=5.0,
        space_width=125.0,
        string_width=0.0,
    )
    assert (output, text, span) == ([], "cell", (5, 88.0, 100.0))


def test_fragment_outside_the_line_span_breaks_the_line():
    output: list[str] = []
    text, last, span = positioned_text(
        output,
        "cell",
        "",
        (5, 99.0, 100.0),
        previous_text_matrix=[1, 0, 0, 1, 0, 100],
        previous_current_matrix=[1, 0, 0, 1, 0, 0],
        text_matrix=[1, 0, 0, 1, 0, 94],
        current_matrix=[1, 0, 0, 1, 0, 0],
        line_height=5.0,
        font_size=5.0,
        space_width=125.0,
        string_width=0.0,
    )
    assert (output, text, last, span) == (["cell\n"], "", "\n", (5, 94.0, 94.0))


@pytest.mark.parametrize(
    ("text", "rtl", "value", "expected"),
    [
        ("ab", False, "c", ("abc", False, "")),
        ("ab", False, "א", ("א", True, "ab")),
        ("א", True, "a", ("a", False, "א")),
        ("ב", True, "א", ("אב", True, "")),
        ("א", True, " ", (" א", True, "")),
    ],
)
def test_a_direction_change_completes_the_run_instead_of_dropping_it(text, rtl, value, expected):
    assert append_directional_text(text, rtl, value) == expected


@pytest.mark.parametrize("digit", ["٠", "٢", "٭", "۰", "۹"])
def test_arabic_indic_digits_are_neutral(digit):
    # py-pdf/pypdf#4077: they keep the insertion order instead of reversing.
    assert is_neutral_character(digit)


@pytest.mark.parametrize(
    ("operands", "expected"),
    [((3,), 3.0), ((2.5,), 2.5), ((), 7.0), (("x",), 7.0), ((True,), 7.0)],
)
def test_non_numeric_operands_take_the_default(operands, expected):
    assert number_operand(operands, 0, 7.0) == expected

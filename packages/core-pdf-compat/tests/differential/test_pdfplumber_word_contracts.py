from copy import deepcopy
from typing import Any

import pytest

from core_pdf_compat import pdfplumber as compat

reference = pytest.importorskip("pdfplumber")
pytestmark = pytest.mark.compat_differential


def internal_chars(upright):
    chars = []
    for index, text in enumerate(("A", "ﬁ", ",", " ", "B", "C")):
        along = index * 6
        x0, top = (along, 10) if upright else (10, along)
        chars.append(
            {
                "text": text,
                "x0": x0,
                "x1": x0 + 5,
                "top": top,
                "bottom": top + 5,
                "doctop": top + 100,
                "size": 10,
                "height": 5,
                "width": 5,
                "upright": upright,
                "fontname": "F1" if index < 5 else "F2",
            }
        )
    return chars


@pytest.mark.parametrize("upright", [False, True])
@pytest.mark.parametrize("split", [False, True, ","])
@pytest.mark.parametrize("keep_blank", [False, True])
@pytest.mark.parametrize("expand", [False, True])
def test_word_options_match_reference_and_preserve_source_identity(
    upright, split, keep_blank, expand
):
    chars = internal_chars(upright)
    before = deepcopy(chars)
    options = {
        "split_at_punctuation": split,
        "keep_blank_chars": keep_blank,
        "expand_ligatures": expand,
        "return_chars": True,
    }
    expected = reference.utils.extract_words(chars, **options)
    actual = compat.extract_words(chars, **options)
    assert actual == expected
    assert chars == before
    assert all(any(char is source for source in chars) for word in actual for char in word["chars"])


@pytest.mark.parametrize("upright", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {"extra_attrs": ["fontname"]},
        {"x_tolerance": 0, "y_tolerance": 0},
        {"x_tolerance_ratio": 0.05},
        {"x_tolerance": 8, "y_tolerance": 8},
    ],
)
def test_word_group_boundaries_match_reference(upright, options):
    chars = internal_chars(upright)
    assert compat.extract_words(chars, **options) == reference.utils.extract_words(chars, **options)


@pytest.mark.parametrize("upright", [False, True])
def test_page_word_projection_matches_utils_with_controlled_characters(text_pdf_bytes, upright):
    with compat.open(text_pdf_bytes) as pdf:
        page = pdf.pages[0]
        chars: list[dict[str, Any]] = internal_chars(upright)
        page._objects = {"char": chars}
        assert page.extract_words(return_chars=True) == reference.utils.extract_words(
            chars, return_chars=True
        )


def test_empty_word_input_is_empty_in_both_implementations():
    assert compat.extract_words([]) == reference.utils.extract_words([]) == []


@pytest.mark.parametrize("upright", [False, True])
@pytest.mark.parametrize("reverse_input", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_reverse_reading_directions_match_reference(upright, reverse_input, split):
    chars = internal_chars(upright)
    if reverse_input:
        chars.reverse()
    options = {
        "horizontal_ltr": False,
        "vertical_ttb": False,
        "split_at_punctuation": split,
        "return_chars": True,
    }
    assert compat.extract_words(chars, **options) == reference.utils.extract_words(chars, **options)

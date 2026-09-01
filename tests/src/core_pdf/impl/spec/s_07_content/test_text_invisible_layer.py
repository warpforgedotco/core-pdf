# SPDX-License-Identifier: AGPL-3.0-only
"""Whether an unpainted text layer is the page's real text is a whole-page property.

Render mode 3 and sub-0.1pt text paint nothing. A scan carrying an OCR layer is
made entirely of such text and must still extract; a normal page's hidden
watermark must not. Deciding that per show-text operator, from the runs captured
so far, made the answer depend on the order the operators appear in -- the first
text object on a page has no preceding runs to look at. These pin the decision to
the page instead.
"""

from __future__ import annotations

from tests.helpers.pdf_bytes import first_page_runs, first_page_text, one_page_pdf

HIDDEN_ALPHA = b"BT /F1 12 Tf 3 Tr 50 700 Td (HiddenAlpha) Tj ET\n"
HIDDEN_BETA = b"BT /F1 12 Tf 3 Tr 50 600 Td (HiddenBeta) Tj ET\n"
VISIBLE = b"BT /F1 12 Tf 0 Tr 50 500 Td (VisibleText) Tj ET\n"


def extracted_text(content: bytes) -> str:
    return first_page_text(one_page_pdf(content))


def run_visibility(content: bytes) -> dict[str, bool]:
    return {run.text: run.visible for run in first_page_runs(one_page_pdf(content))}


def test_hidden_text_before_visible_text_is_still_hidden() -> None:
    """The first text object on the page has no preceding runs to judge against."""
    assert extracted_text(HIDDEN_ALPHA + VISIBLE + HIDDEN_BETA) == "VisibleText"


def test_hidden_text_visibility_does_not_depend_on_operator_order() -> None:
    leading = run_visibility(HIDDEN_ALPHA + VISIBLE + HIDDEN_BETA)
    trailing = run_visibility(VISIBLE + HIDDEN_ALPHA + HIDDEN_BETA)
    assert leading == trailing
    assert leading == {"HiddenAlpha": False, "VisibleText": True, "HiddenBeta": False}


def test_an_unpainted_layer_survives_being_split_across_text_objects() -> None:
    """One BT per line is ordinary in an OCR layer; every line must still extract."""
    split = extracted_text(HIDDEN_ALPHA + HIDDEN_BETA)
    single = extracted_text(
        b"BT /F1 12 Tf 3 Tr 50 700 Td (HiddenAlpha) Tj 0 -20 Td (HiddenBeta) Tj ET\n"
    )
    assert "HiddenAlpha" in split
    assert "HiddenBeta" in split
    assert "HiddenAlpha" in single
    assert "HiddenBeta" in single


def test_painted_text_is_unaffected() -> None:
    content = VISIBLE + b"BT /F1 12 Tf 0 Tr 50 400 Td (MoreVisible) Tj ET\n"
    assert run_visibility(content) == {"VisibleText": True, "MoreVisible": True}

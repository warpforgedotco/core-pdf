"""Characterization tests pinning exact facade output shapes.

These pin the observable behavior of the compat facades ahead of internal
refactoring: any diff here is a behavior change, not a cleanup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from core_pdf.api.v0.compat.pdfminer import extract_text as extract_pdfminer_text
from core_pdf.api.v0.compat.pdfplumber import open as open_pdfplumber
from core_pdf.api.v0.compat.pymupdf import open as fitz_open
from core_pdf.api.v0.compat.xray import inspect as inspect_xray

FIXTURE = Path("vendor/pdfminer.six/samples/simple1.pdf")
XRAY_FIXTURE = Path("vendor/x-ray/tests/assets/rectangles_yes_2.pdf")

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="vendor fixtures not present")


def test_pdfplumber_char_envelope_keys_and_values_are_stable() -> None:
    with open_pdfplumber(FIXTURE) as pdf:
        char = pdf.pages[0].chars[0]

    assert sorted(char.keys()) == [
        "adv",
        "bottom",
        "doctop",
        "fontname",
        "height",
        "non_stroking_color",
        "object_type",
        "page_number",
        "seqno",
        "size",
        "stroking_color",
        "text",
        "top",
        "upright",
        "width",
        "x0",
        "x1",
        "y0",
        "y1",
    ]
    assert char["object_type"] == "char"
    assert char["page_number"] == 1
    assert char["text"] == "H"
    assert char["x0"] == pytest.approx(100.0)
    assert char["x1"] == pytest.approx(117.328)
    assert char["top"] == pytest.approx(74.768)
    assert char["bottom"] == pytest.approx(96.968)
    assert char["doctop"] == pytest.approx(74.768)
    assert char["width"] == pytest.approx(17.328)
    assert char["height"] == pytest.approx(22.2)


def test_pdfplumber_word_envelope_is_stable() -> None:
    with open_pdfplumber(FIXTURE) as pdf:
        word = pdf.pages[0].extract_words()[0]

    assert sorted(word.keys()) == [
        "bottom",
        "direction",
        "doctop",
        "height",
        "text",
        "top",
        "upright",
        "width",
        "x0",
        "x1",
    ]
    assert word["text"] == "Hello"
    assert word["x0"] == pytest.approx(100.0)
    assert word["x1"] == pytest.approx(154.672)
    assert word["top"] == pytest.approx(74.768)
    assert word["bottom"] == pytest.approx(96.968)


def test_pdfminer_extract_text_output_is_stable() -> None:
    assert extract_pdfminer_text(FIXTURE) == (
        "Hello \n\nWorld\n\nHello World\n\nHello World\n\n"
        "H\n\ne\n\nl\n\nl\n\no\n\n \n\nW\n\no\n\nr\n\nl\n\nd\n\n\x0c"
    )


def test_pymupdf_words_tuple_shape_is_stable() -> None:
    with fitz_open(FIXTURE) as document:
        words = cast("list[tuple[Any, ...]]", document.load_page(0).get_text("words"))

    first = words[0]
    assert len(first) == 8
    assert first[4] == "Hello"
    assert first[0] == pytest.approx(100.0)
    assert first[5:] == (0, 0, 0)


def test_pymupdf_dict_span_shape_is_stable() -> None:
    with fitz_open(FIXTURE) as document:
        payload = cast("dict[str, Any]", document.load_page(0).get_text("dict"))

    span = payload["blocks"][0]["lines"][0]["spans"][0]
    assert sorted(span.keys()) == ["text"]
    assert "Hello" in span["text"]


@pytest.mark.skipif(not XRAY_FIXTURE.exists(), reason="x-ray fixture not present")
def test_xray_inspect_output_is_stable() -> None:
    findings = inspect_xray(XRAY_FIXTURE)

    assert list(findings.keys()) == [1]
    (finding,) = findings[1]
    assert finding["text"] == "def"
    assert finding["bbox"] == pytest.approx((105.48, 75.0, 119.64, 87.0))


# Verified 2026-08-10 against the released x-ray package (PyMuPDF 1.24.14 pin):
# real x-ray reports NO bad redactions for these forms — the labels sit visibly
# on top of gray bands. core-pdf currently reports false positives because the
# capture/render pipeline mirrors XObject-drawn vector text vertically, so
# neither content-stream sequence nor raster inspection can see that the text
# paints above the fill. Tracked as a known engine defect; these tests flip to
# green when the mirrored-XObject capture is fixed.
@pytest.mark.parametrize(
    "fixture_name",
    [
        "rect_ordering_1.23.pdf",
        "rect_ordering_3.20.pdf",
        "rect_ordering_4.1.pdf",
        "rect_ordering_6.19.pdf",
    ],
)
@pytest.mark.xfail(strict=True, reason="mirrored XObject vector text defeats redaction ordering")
def test_xray_matches_real_xray_on_text_over_rect_forms(fixture_name: str) -> None:
    fixture = Path("vendor/x-ray/tests/assets") / fixture_name
    if not fixture.exists():
        pytest.skip()

    assert inspect_xray(fixture) == {}

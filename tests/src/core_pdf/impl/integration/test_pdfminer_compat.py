# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import pytest
from pdfminer.layout import LTChar as PdfMinerLTChar
from pdfminer.layout import LTFigure as PdfMinerLTFigure
from pdfminer.layout import LTLayoutContainer as PdfMinerLTLayoutContainer

from core_pdf.integrations.pdfminer.high_level import extract_pages, extract_text
from core_pdf.integrations.pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTContainer,
    LTFigure,
    LTImage,
    LTLayoutContainer,
    LTPage,
    LTTextBox,
    LTTextLine,
)

TESTS_DIR = Path(__file__).parents[4]
SAMPLES_DIR = TESTS_DIR / "fixtures" / "pdfminer.six" / "samples"
SIMPLE_PDF = SAMPLES_DIR / "simple1.pdf"


class FakeFont:
    fontname = "Helvetica"

    def is_vertical(self) -> bool:
        return False

    def get_descent(self) -> float:
        return -0.207


def walk(container: LTContainer[Any]) -> list[Any]:
    children: list[Any] = []
    for child in container:
        children.append(child)
        if isinstance(child, LTContainer):
            children.extend(walk(child))
    return children


def test_ltchar_constructor_matches_pdfminer_geometry_and_attributes() -> None:
    kwargs = {
        "matrix": (1, 0, 0, 1, 100, 700),
        "font": FakeFont(),
        "fontsize": 24,
        "scaling": 1,
        "rise": 0,
        "text": "H",
        "textwidth": 0.722,
        "textdisp": 0,
        "ncs": object(),
        "graphicstate": object(),
    }

    expected = PdfMinerLTChar(**cast(Any, kwargs))
    result = LTChar(**kwargs)

    assert result.bbox == expected.bbox
    assert result.width == expected.width
    assert result.height == expected.height
    assert result.matrix == expected.matrix
    assert result.fontname == expected.fontname
    assert result.adv == expected.adv
    assert result.size == expected.size
    assert result.upright == expected.upright
    assert result.get_text() == expected.get_text()


def test_container_and_figure_contracts_match_pdfminer() -> None:
    layout = LTLayoutContainer((0, 0, 100, 100))
    layout.add(LTLayoutContainer((1, 2, 3, 4)))
    assert len(layout) == 1
    assert list(layout)[0].bbox == (1, 2, 3, 4)

    matrix = (1, 0, 0, 1, 10, 20)
    expected = PdfMinerLTFigure("figure", (0, 0, 30, 40), matrix)
    result = LTFigure("figure", (0, 0, 30, 40), matrix)
    assert result.bbox == expected.bbox
    assert result.name == expected.name
    assert result.matrix == expected.matrix
    assert isinstance(result, LTLayoutContainer)
    assert issubclass(LTLayoutContainer, LTContainer)
    assert issubclass(PdfMinerLTLayoutContainer, object)


def test_laparams_validates_boxes_flow_like_pdfminer() -> None:
    assert LAParams().boxes_flow == 0.5
    assert LAParams(boxes_flow=None).boxes_flow is None
    with pytest.raises(ValueError, match=r"between -1 and \+1"):
        LAParams(boxes_flow=2)
    with pytest.raises(TypeError, match=r"between -1 and \+1"):
        LAParams(boxes_flow=cast(Any, "invalid"))


def test_extract_pages_returns_pdfminer_compatible_layout_tree() -> None:
    page = next(extract_pages(SIMPLE_PDF))

    assert isinstance(page, LTPage)
    assert isinstance(page, LTLayoutContainer)
    assert page.pageid == 1
    assert page.bbox == (0.0, 0.0, 612.0, 792.0)
    assert page.width == 612.0
    assert page.height == 792.0

    children = walk(page)
    assert all(isinstance(obj, LTTextBox) for obj in page)
    assert sum(isinstance(obj, LTTextLine) for obj in children) == 4
    assert sum(isinstance(obj, LTChar) for obj in children) == 44
    assert any(isinstance(obj, LTAnno) for obj in children)
    assert not any(isinstance(obj, LTImage) for obj in children)
    assert [cast(LTTextBox, obj).get_text() for obj in page] == [
        "Hello World\n",
        "Hello World\n",
        "Hello World\n",
        "H e l l o W o r l d\n",
    ]


def test_extract_pages_accepts_file_like_objects() -> None:
    page = next(extract_pages(io.BytesIO(SIMPLE_PDF.read_bytes())))

    assert page.pageid == 1
    assert page.width == 612.0


def test_extract_text_preserves_core_pdf_output_and_pdfminer_arguments() -> None:
    assert extract_text(SIMPLE_PDF, page_numbers={0}, maxpages=1) == (
        "Hello World\n\nHello World\n\nHello World\n\nH e l l o W o r l d\f"
    )

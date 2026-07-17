# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import pytest
from core_pdf.integrations.pdfminer.six.high_level import (  # ty: ignore[unresolved-import]
    extract_pages,
    extract_text,
)
from core_pdf.integrations.pdfminer.six.layout import (  # ty: ignore[unresolved-import]
    LAParams,
    LTChar,
    LTContainer,
    LTFigure,
    LTLayoutContainer,
    LTPage,
)
from pdfminer.high_level import extract_pages as pdfminer_extract_pages
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LTChar as PdfMinerLTChar
from pdfminer.layout import LTContainer as PdfMinerLTContainer
from pdfminer.layout import LTFigure as PdfMinerLTFigure
from pdfminer.layout import LTLayoutContainer as PdfMinerLTLayoutContainer

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


def layout_snapshot(obj: Any) -> tuple[Any, ...]:
    children = (
        tuple(layout_snapshot(child) for child in obj)
        if isinstance(obj, (LTContainer, PdfMinerLTContainer))
        else ()
    )
    get_text = getattr(obj, "get_text", None)
    return (
        type(obj).__name__,
        getattr(obj, "bbox", None),
        get_text() if get_text is not None else None,
        children,
    )


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
    expected_page = next(pdfminer_extract_pages(SIMPLE_PDF))
    page = next(extract_pages(SIMPLE_PDF))

    assert isinstance(page, LTPage)
    assert isinstance(page, LTLayoutContainer)
    assert page.pageid == expected_page.pageid
    assert page.bbox == expected_page.bbox
    assert page.width == expected_page.width
    assert page.height == expected_page.height
    assert layout_snapshot(page) == layout_snapshot(expected_page)


def test_extract_pages_accepts_file_like_objects() -> None:
    page = next(extract_pages(io.BytesIO(SIMPLE_PDF.read_bytes())))

    assert page.pageid == 1
    assert page.width == 612.0


def test_extract_text_matches_pdfminer_arguments_and_output() -> None:
    assert extract_text(SIMPLE_PDF, page_numbers={0}, maxpages=1) == pdfminer_extract_text(
        SIMPLE_PDF,
        page_numbers={0},
        maxpages=1,
    )

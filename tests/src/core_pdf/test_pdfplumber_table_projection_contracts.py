"""Structured table projections retain row order, geometry, and page ownership."""

from io import BytesIO
from types import SimpleNamespace

import pytest

from core_pdf.api.compat import pdfplumber as compat
from core_pdf.impl._impl.output.model import Table, TableCell


def table(label, box, cell_boxes):
    return Table(
        0,
        bbox=box,
        rows=tuple(
            (TableCell(index, 0, text, bbox=cell_box),)
            for index, (text, cell_box) in enumerate(zip(label, cell_boxes, strict=True))
        ),
    )


@pytest.mark.parametrize("geometry", ["positioned", "unpositioned", "partial"])
def test_matching_column_tables_merge_with_geometry_or_original_row_order(
    text_pdf_bytes, monkeypatch, geometry
):
    first_boxes = [(0, 20, 100, 30), (0, 10, 100, 20)]
    second_boxes = [(0, 80, 100, 90)]
    if geometry == "unpositioned":
        first_boxes, second_boxes = [None, None], [None]
    elif geometry == "partial":
        first_boxes[0] = None
    first = table(["low", "lowest"], (0, 10, 100, 30), first_boxes)
    second = table(["high"], (1, 80, 101, 90), second_boxes)
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        calls = []

        def extract(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(pages=(SimpleNamespace(tables=(first, second)),))

        monkeypatch.setattr(pdf._document, "extract", extract)
        page = pdf.pages[0]
        result = page.find_tables()
        assert calls == [{"pages": (1,)}]
        assert len(result) == 1
        assert result[0].page is page
        assert result[0].bbox == (0, 10, 100, 90)
        expected = (
            [["high"], ["low"], ["lowest"]]
            if geometry == "positioned"
            else [["low"], ["lowest"], ["high"]]
        )
        assert result[0].extract() == expected
        assert [row[0].text for row in first.rows] == ["low", "lowest"]
        assert first.bbox == (0, 10, 100, 30)


@pytest.mark.parametrize("box", [(2, 80, 102, 90), (0, 80, 103, 90), None])
def test_tables_with_different_or_missing_column_bounds_remain_separate(
    text_pdf_bytes, monkeypatch, box
):
    first = table(["one"], (0, 10, 100, 30), [None])
    second = table(["two"], box, [None])
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        monkeypatch.setattr(
            pdf._document,
            "extract",
            lambda **kwargs: SimpleNamespace(pages=(SimpleNamespace(tables=(first, second)),)),
        )
        result = pdf.pages[0].find_tables()
        assert [item.extract() for item in result] == [[["one"]], [["two"]]]
        assert all(item.page is pdf.pages[0] for item in result)


@pytest.mark.parametrize("body_count", [500, 501])
@pytest.mark.parametrize("footer", [False, True])
def test_thin_table_body_extension_requires_more_than_500_nonfooter_characters(
    text_pdf_bytes, monkeypatch, body_count, footer
):
    native = table(["cell"], (0, 10, 200, 15), [None])
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        monkeypatch.setattr(
            pdf._document,
            "extract",
            lambda **kwargs: SimpleNamespace(pages=(SimpleNamespace(tables=(native,)),)),
        )
        page = pdf.pages[0]
        chars = [{"bottom": 100}] * body_count + ([{"bottom": 180}] * 100 if footer else [])
        page._objects = {"char": chars}
        result = page.find_tables()
        assert len(result) == 1
        assert result[0].bbox == ((0, 0, 200, 100) if body_count > 500 else native.bbox)
        assert result[0].extract() == [["cell"]]
        assert native.bbox == (0, 10, 200, 15)


def test_ragged_table_columns_preserve_missing_cells_and_row_identity():
    native = Table(
        0,
        rows=(
            (TableCell(0, 0, "A", bbox=(0, 0, 5, 5)), TableCell(0, 1, "B", bbox=(5, 0, 10, 5))),
            (TableCell(1, 0, "C", bbox=(0, 5, 5, 10)),),
        ),
    )
    result = compat.Table(native)
    assert result.extract() == [["A", "B"], ["C", ""]]
    assert list(result.columns[1]) == [(5, 0, 10, 5), None]
    assert len(result.rows[0]) == 2
    assert result.rows[1][0] == (0, 5, 5, 10)
    assert len(native.rows[1]) == 1

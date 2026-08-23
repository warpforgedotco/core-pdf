from io import BytesIO

import pytest

from core_pdf.api import compat
from core_pdf.api.compat.pdfminer import (
    LTChar,
    LTTextBox,
    LTTextBoxHorizontal,
    LTTextLineHorizontal,
    _reading_order,
)
from core_pdf.api.compat.pdfplumber import _words
from core_pdf.api.compat.pypdf import PdfMerger, PdfReader, PdfWriter


def internal_pdf_with_page_widths(*widths: float) -> bytes:
    writer = PdfWriter()
    for width in widths:
        writer.add_blank_page(width=width, height=100)
    return writer.write(BytesIO())


def test_compat_unknown_attribute_uses_module_protocol() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(compat, "not_a_compatibility_export")


def test_pypdf_merger_inserts_at_flattened_page_position() -> None:
    merger = PdfMerger()
    merger.append(internal_pdf_with_page_widths(100, 101))
    merger.append(internal_pdf_with_page_widths(200, 201))

    merger.merge(1, internal_pdf_with_page_widths(300, 301))

    output = BytesIO()
    merger.write(output)
    with PdfReader(output.getvalue()) as reader:
        assert [float(page.mediabox.width) for page in reader.pages] == [
            100,
            300,
            301,
            101,
            200,
            201,
        ]


def test_pdfplumber_words_defaults_missing_upright_to_true() -> None:
    chars = [
        {
            "text": text,
            "x0": x0,
            "x1": x0 + 5.0,
            "top": 0.0,
            "bottom": 10.0,
            "doctop": 0.0,
        }
        for text, x0 in (("A", 0.0), ("B", 5.0))
    ]

    word = _words(chars)[0]

    assert word["text"] == "AB"
    assert word["upright"] is True
    assert word["direction"] == "ltr"


def test_pdfminer_reading_order_retains_every_merged_box() -> None:
    boxes: list[LTTextBox] = []
    for index in range(100):
        bbox = (
            float(index % 10),
            float(index // 10),
            float(index % 10 + 1),
            float(index // 10 + 1),
        )
        character = LTChar(bbox, str(index), "Test", 1.0)
        line = LTTextLineHorizontal(bbox, [character])
        boxes.append(LTTextBoxHorizontal(bbox, [line]))

    ordered = _reading_order(boxes, 0.5, (0.0, 0.0, 20.0, 20.0))

    assert sorted(box.get_text() for box in ordered) == sorted(str(index) for index in range(100))

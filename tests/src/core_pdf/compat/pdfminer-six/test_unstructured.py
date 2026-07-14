from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

from core_pdf.impl.engine.extraction.document import PdfDocument
from core_pdf.impl.engine.layout.models import LayoutLine, TextRun
from core_pdf.impl.engine.spec.s_07_objects.coercion import parse_name
from core_pdf.impl.models import FieldRecord
from core_pdf.impl.types import PdfDict, PdfObject

TESTS_DIR = Path(__file__).parents[4]
SAMPLES_DIR = TESTS_DIR / "fixtures" / "pdfminer.six" / "samples"
SIMPLE_PDF = SAMPLES_DIR / "simple1.pdf"


def test_pdfminer_integration_document_exposes_page_count() -> None:
    document = PdfDocument.open(str(SIMPLE_PDF))

    assert document.page_count() == 1


def test_iter_unstructured_region_layouts_yields_native_regions() -> None:
    with PdfDocument.open(SIMPLE_PDF) as document:
        lines = cast(Any, document).extract_lines(include_words=True)

    assert {line["page_number"] for line in lines} == {1}
    assert lines[0]["page_width"] == 612
    assert lines[0]["page_height"] == 792
    assert [line["text"] for line in lines[:3]] == [
        "Hello World",
        "Hello World",
        "Hello World",
    ]
    first_words = cast(list[dict[str, Any]], lines[0]["words"])
    assert first_words[0]["text"] == "Hello"
    assert first_words[0]["start_index"] == 0
    assert first_words[1]["text"] == "World"
    assert (
        str(lines[0]["text"])[
            int(first_words[1]["start_index"]) : int(first_words[1]["start_index"])
            + len(str(first_words[1]["text"]))
        ]
        == "World"
    )


def test_iter_unstructured_region_layouts_accepts_file_like_object() -> None:
    with PdfDocument.open(io.BytesIO(SIMPLE_PDF.read_bytes())) as document:
        lines = cast(Any, document).extract_lines()

    assert {line["page_number"] for line in lines} == {1}
    assert lines[0]["text"] == "Hello World"


def test_unstructured_line_render_inserts_missing_gap_space() -> None:
    line = LayoutLine(
        [
            TextRun("Hello", 0.0, 0.0, 24.0, 10.0, 0.0, 0.0, 10.0, 4.0, 0, 0, 0),
            TextRun("World", 34.0, 0.0, 62.0, 10.0, 34.0, 0.0, 10.0, 4.0, 1, 1, 0),
        ]
    )

    text, words = line.text_and_words()

    assert text == "Hello World"
    assert [word.text for word in words] == ["Hello", "World"]
    assert words[1].start_index == 6


def test_unstructured_line_render_does_not_duplicate_existing_spaces() -> None:
    line = LayoutLine(
        [
            TextRun("LayoutParser: ", 0.0, 0.0, 58.0, 10.0, 0.0, 0.0, 10.0, 4.0, 0, 0, 0),
            TextRun("A ", 72.0, 0.0, 80.0, 10.0, 72.0, 0.0, 10.0, 4.0, 1, 1, 0),
            TextRun("Unified", 92.0, 0.0, 124.0, 10.0, 92.0, 0.0, 10.0, 4.0, 2, 2, 0),
        ]
    )

    text, words = line.text_and_words()

    assert text == "LayoutParser: A Unified"
    assert [word.text for word in words] == ["LayoutParser:", "A", "Unified"]
    assert words[1].start_index == 14
    assert words[2].start_index == 16


def test_unstructured_field_regions_use_widget_values() -> None:
    widget = {
        "Subtype": "Widget",
        "Rect": [40, 700, 300, 720],
    }
    field = FieldRecord(
        "name",
        "Tx",
        b"Jane Doe",
        "Jane Doe",
        (40.0, 700.0, 300.0, 720.0),
        cast(PdfDict, widget),
        widget=cast(PdfDict, widget),
    )

    regions = [
        {
            "text": field.value_text,
            "bbox": field.rect,
            "words": [{"text": word, "start_index": start} for word, start in field.value_words()],
        }
    ]

    assert len(regions) == 1
    assert regions[0]["text"] == "Jane Doe"
    assert regions[0]["bbox"] == (40.0, 700.0, 300.0, 720.0)
    assert cast(list[dict[str, object]], regions[0]["words"])[0]["text"] == "Jane"


class FakeResolver:
    def resolve_box(self, value: PdfObject) -> tuple[float, float, float, float] | None:
        if not isinstance(value, list) or len(value) != 4:
            return None
        left, bottom, right, top = (float(item) for item in value)
        return left, bottom, right, top

    def resolve_str(self, value: PdfObject) -> str | None:
        if isinstance(value, bytes):
            return value.decode("latin-1")
        if isinstance(value, str):
            return value
        return None

    def resolve_name_like_value(self, value: PdfObject) -> str | None:
        return parse_name(value)

from __future__ import annotations

import io
from pathlib import Path

from core_pdf.compat.pdfminer.document import PdfDocument
from core_pdf.compat.pdfminer.unstructured import (
    _field_regions,
    _render_line_with_words,
    iter_unstructured_region_layouts,
)
from core_pdf.impl.engine.spec.s_07_content.models import LayoutLine, TextRun
from core_pdf.impl.engine.spec.s_07_document.models import FieldRecord
from core_pdf.impl.engine.spec.s_07_syntax.primitives import PdfObject, parse_name

TESTS_DIR = Path(__file__).parents[4]
SAMPLES_DIR = TESTS_DIR / "fixtures" / "pdfminer.six" / "samples"
SIMPLE_PDF = SAMPLES_DIR / "simple1.pdf"


def test_pdfminer_integration_document_exposes_page_count() -> None:
    document = PdfDocument.open(str(SIMPLE_PDF))

    assert document.page_count() == 1


def test_iter_unstructured_region_layouts_yields_native_regions() -> None:
    pages = list(iter_unstructured_region_layouts(SIMPLE_PDF))

    assert len(pages) == 1
    page = pages[0]
    assert page.width == 612
    assert page.height == 792
    assert [region.text for region in page.regions[:3]] == [
        "Hello World",
        "Hello World",
        "Hello World",
    ]
    assert page.regions[0].words[0].text == "Hello"
    assert page.regions[0].words[0].start_index == 0
    assert page.regions[0].words[1].text == "World"
    assert page.regions[0].text[
        page.regions[0].words[1].start_index : page.regions[0].words[1].start_index
        + len(page.regions[0].words[1].text)
    ] == "World"


def test_iter_unstructured_region_layouts_accepts_file_like_object() -> None:
    pages = list(iter_unstructured_region_layouts(io.BytesIO(SIMPLE_PDF.read_bytes())))

    assert len(pages) == 1
    assert pages[0].regions[0].text == "Hello World"


def test_unstructured_line_render_inserts_missing_gap_space() -> None:
    line = LayoutLine()
    line.add(TextRun("Hello", 0.0, 0.0, 24.0, 10.0, 0.0, 0.0, 10.0, 4.0, 0, 0, 0))
    line.add(TextRun("World", 34.0, 0.0, 62.0, 10.0, 34.0, 0.0, 10.0, 4.0, 1, 1, 0))

    text, words, visible = _render_line_with_words(line, 0)

    assert text == "Hello World"
    assert [word.text for word in words] == ["Hello", "World"]
    assert words[1].start_index == 6
    assert visible


def test_unstructured_line_render_does_not_duplicate_existing_spaces() -> None:
    line = LayoutLine()
    line.add(
        TextRun("LayoutParser: ", 0.0, 0.0, 58.0, 10.0, 0.0, 0.0, 10.0, 4.0, 0, 0, 0)
    )
    line.add(TextRun("A ", 72.0, 0.0, 80.0, 10.0, 72.0, 0.0, 10.0, 4.0, 1, 1, 0))
    line.add(
        TextRun("Unified", 92.0, 0.0, 124.0, 10.0, 92.0, 0.0, 10.0, 4.0, 2, 2, 0)
    )

    text, words, visible = _render_line_with_words(line, 0)

    assert text == "LayoutParser: A Unified"
    assert [word.text for word in words] == ["LayoutParser:", "A", "Unified"]
    assert words[1].start_index == 14
    assert words[2].start_index == 16
    assert visible


def test_unstructured_field_regions_use_widget_values() -> None:
    widget = {
        "Subtype": "Widget",
        "Rect": [40, 700, 300, 720],
    }
    field = FieldRecord("name", "Tx", b"Jane Doe", widget)

    regions = list(_field_regions([field], FakeResolver(), 612.0, 792.0))  # type: ignore[arg-type]

    assert len(regions) == 1
    assert regions[0].text == "Jane Doe"
    assert regions[0].bbox == (40.0, 700.0, 300.0, 720.0)
    assert regions[0].words[0].text == "Jane Doe"


class FakeResolver:
    def resolve_box(self, value: PdfObject) -> tuple[float, float, float, float] | None:
        if not isinstance(value, list) or len(value) != 4:
            return None
        return tuple(float(item) for item in value)  # type: ignore[return-value]

    def resolve_str(self, value: PdfObject) -> str | None:
        if isinstance(value, bytes):
            return value.decode("latin-1")
        if isinstance(value, str):
            return value
        return None

    def resolve_name_like_value(self, value: PdfObject) -> str | None:
        return parse_name(value)

"""Whole-document passes: what is built once, and what is freed page by page."""

import gc
import weakref
from typing import Any

import pytest

import core_pdf.impl.capture.page as capture_page_module
from core_pdf import PdfDocument
from core_pdf.impl.extract import selection
from core_pdf.impl.extract.pipeline import PageExtraction

PAGE_COUNT = 3


def multi_page_pdf(count: int = PAGE_COUNT) -> bytes:
    page_numbers = [3 + 2 * index for index in range(count)]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids ["
        + b" ".join(f"{number} 0 R".encode() for number in page_numbers)
        + f"] /Count {count} >>".encode(),
    }
    font_number = 3 + 2 * count
    objects[font_number] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    for index, number in enumerate(page_numbers):
        content = f"BT /F1 12 Tf 20 100 Td (Page {index + 1}) Tj ET".encode()
        objects[number] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
            + f"/Resources << /Font << /F1 {font_number} 0 R >> >> ".encode()
            + f"/Contents {number + 1} 0 R >>".encode()
        )
        objects[number + 1] = (
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        )
    data = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(data)
        data.extend(f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n")
    xref = len(data)
    size = max(objects) + 1
    data.extend(f"xref\n0 {size}\n0000000000 65535 f \n".encode())
    for number in range(1, size):
        data.extend(f"{offsets[number]:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(data)


def test_pages_are_built_once_per_document() -> None:
    with PdfDocument(multi_page_pdf()) as document:
        pages = document.pages
        assert len(pages) == PAGE_COUNT
        assert document.pages is pages


def test_extracting_every_page_groups_form_fields_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each page asks for its fields, and grouping them walks every page, so a
    # grouping per page made a whole-document pass quadratic in its length.
    calls: list[int] = []
    original = PdfDocument.group_fields_by_page

    def counted(self: Any, page_sequence: Any) -> Any:
        calls.append(len(page_sequence))
        return original(self, page_sequence)

    monkeypatch.setattr(PdfDocument, "group_fields_by_page", counted)
    with PdfDocument(multi_page_pdf()) as document:
        texts = [page.extract().text for page in document.pages]
    assert calls == [PAGE_COUNT]
    assert [text.strip() for text in texts] == [f"Page {n}" for n in range(1, PAGE_COUNT + 1)]


def test_public_fields_by_page_cannot_change_the_shared_grouping() -> None:
    with PdfDocument(multi_page_pdf()) as document:
        grouped = document.fields_by_page()
        grouped[0] = ["not a field"]  # ty: ignore[invalid-assignment]
        assert document.fields_by_page() == {}
        assert document.pages[0].get_fields() == []


def test_a_page_capture_state_is_freed_without_the_cycle_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states: list[weakref.ref[Any]] = []

    class TrackedTextState(capture_page_module.TextState):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            states.append(weakref.ref(self))

    monkeypatch.setattr(capture_page_module, "TextState", TrackedTextState)
    gc.disable()
    try:
        with PdfDocument(multi_page_pdf(1)) as document:
            program = document.pages[0].get_page_program()
            assert program.body.glyphs
            assert states
            assert all(state() is None for state in states)
    finally:
        gc.enable()


def test_document_extraction_assembles_each_page_before_capturing_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int]] = []

    class RecordingExtraction(PageExtraction):
        def __init__(self, page: Any, **kwargs: Any) -> None:
            events.append(("capture", page.page_number))
            super().__init__(page, **kwargs)

        def assembled_page(self, context: Any) -> Any:
            events.append(("assemble", int(self.page.page_number)))
            return super().assembled_page(context)

    monkeypatch.setattr(selection, "PageExtraction", RecordingExtraction)
    with PdfDocument(multi_page_pdf()) as document:
        result = document.extract()
    assert events == [
        (kind, number) for number in range(1, PAGE_COUNT + 1) for kind in ("capture", "assemble")
    ]
    assert [page.text.strip() for page in result.pages] == [
        f"Page {n}" for n in range(1, PAGE_COUNT + 1)
    ]

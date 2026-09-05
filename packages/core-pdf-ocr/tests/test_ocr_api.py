# SPDX-License-Identifier: AGPL-3.0-only
"""Public OCR integration and the native distribution's independent operation."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from core_pdf import PdfDocument as NativePdfDocument
from core_pdf import PdfPage as NativePdfPage
from core_pdf.impl.exceptions import PdfDocumentClosedError
from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.output.model import Document, Page
from core_pdf.impl.runtime.execution import ExtractionScope, internal_ExtractionCancelled
from core_pdf_ocr import PdfDocument, PdfPage
from core_pdf_ocr.impl.extract.contracts import ObservationSource, PageAnalysis, RecognitionResult
from core_pdf_ocr.impl.extract.ocr import pipeline as recognition_pipeline
from tests.helpers.pdf_bytes import one_page_pdf, stream_obj, text_pages_pdf


def internal_scan_pdf() -> bytes:
    return one_page_pdf(
        b"q 100 0 0 100 0 0 cm /Scan Do Q",
        media_box=(0, 0, 100, 100),
        resources=b"<< /XObject << /Scan 6 0 R >> >>",
        extra_objects=(
            stream_obj(
                b"\xff\xff\xff\xff",
                b"/Type /XObject /Subtype /Image /Width 2 /Height 2 "
                b"/ColorSpace /DeviceGray /BitsPerComponent 8",
            ),
        ),
    )


def internal_recognized_text(text: str) -> RecognitionResult:
    return RecognitionResult(
        ObservationBatch.from_columns(
            (text,),
            ((10.0, 40.0, 90.0, 50.0),),
            source=ObservationSource.OCR,
            confidence=(99.0,),
        )
    )


@pytest.mark.parametrize("view", ["page", "page-property", "document", "document-property"])
def test_companion_subclasses_and_structured_views_use_the_recognition_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    view: str,
) -> None:
    expected = "Recognized archival document content"
    captures: list[PageAnalysis] = []

    def recognize(capture: PageAnalysis, *args: object, **kwargs: object) -> RecognitionResult:
        captures.append(capture)
        return internal_recognized_text(expected)

    monkeypatch.setattr(recognition_pipeline, "recognize_page", recognize)
    data = internal_scan_pdf()
    with NativePdfDocument.open(io.BytesIO(data)) as native:
        result = native.extract()
        assert result.pages[0].text == ""
        assert result.pages[0].base_route == "native"
        assert captures == []

    with PdfDocument.open(io.BytesIO(data)) as document:
        page = document.pages[0]
        assert isinstance(document, NativePdfDocument)
        assert isinstance(document, PdfDocument)
        assert isinstance(page, NativePdfPage)
        assert isinstance(page, PdfPage)
        if view == "page":
            extracted = page.extract()
        elif view == "page-property":
            extracted = page.structured_view
        elif view == "document":
            extracted = document.extract().pages[0]
        else:
            extracted = document.structured_document.pages[0]

        assert isinstance(extracted, Page)
        assert extracted.text == expected
        assert extracted.base_route == "ocr"
        assert extracted.blocks[0].lines[0].source == "ocr"
        assert len(captures) == 1
        assert isinstance(captures[0].page, PdfPage)
        assert captures[0].page.document is document
        assert captures[0].page.page_number == page.page_number
        assert document.internal_active_operations == 0


def test_companion_selection_and_adapters_preserve_order_and_operation_lifetime() -> None:
    texts = (
        "First native page contains ordinary document prose.",
        "Second native page contains ordinary document prose.",
        "Third native page contains ordinary document prose.",
    )
    adapted: list[Document] = []
    with PdfDocument.open(io.BytesIO(text_pages_pdf(texts))) as document:

        class Adapter:
            def apply(self, result: Document) -> Document:
                assert document.internal_active_operations == 0
                adapted.append(result)
                return replace(result, metadata={**result.metadata, "adapted": True})

        result = document.extract(pages=[3, 1, 3], adapters=(Adapter(),))
        assert tuple(page.page_number for page in result.pages) == (3, 1)
        assert tuple(page.text for page in result.pages) == (texts[2], texts[0])
        assert all(page.base_route == "native" for page in result.pages)
        assert result.metadata["adapted"] is True
        assert len(adapted) == 1
        assert "adapted" not in adapted[0].metadata


@pytest.mark.parametrize("page_only", [False, True])
def test_companion_close_cancels_active_extraction_and_defers_resource_release(
    monkeypatch: pytest.MonkeyPatch,
    page_only: bool,
) -> None:
    document = PdfDocument.open(io.BytesIO(internal_scan_pdf()))
    page = document.pages[0]

    def close_during_recognition(
        capture: PageAnalysis,
        plan: Any,
        context: ExtractionScope,
        **kwargs: object,
    ) -> RecognitionResult:
        assert capture.page.document is document
        assert document.internal_active_operations == 1
        document.close()
        assert document.closed
        assert not document.internal_closed
        assert document.raw_data
        context.raise_if_cancelled()
        raise AssertionError("closed document did not cancel recognition")

    monkeypatch.setattr(recognition_pipeline, "recognize_page", close_during_recognition)
    try:
        with pytest.raises(internal_ExtractionCancelled, match="cancelled"):
            if page_only:
                page.extract()
            else:
                document.extract()
        assert document.internal_active_operations == 0
        assert document.internal_closed
        assert document.raw_data == b""
        with pytest.raises(PdfDocumentClosedError, match="closed"):
            document.extract()
        with pytest.raises(PdfDocumentClosedError, match="closed"):
            page.extract()
    finally:
        document.close()


def test_core_import_render_and_extraction_do_not_require_or_initialize_ocr(tmp_path: Path) -> None:
    pdf_path = tmp_path / "native.pdf"
    pdf_path.write_bytes(
        one_page_pdf(
            b"1 0 0 rg 0 0 100 100 re f 0 0 0 rg BT /F1 10 Tf 5 50 Td (Native content) Tj ET",
            media_box=(0, 0, 100, 100),
        )
    )
    script = textwrap.dedent(
        """
        import importlib.abc
        import os
        import signal
        import sys

        blocked = {"core_pdf_ocr", "tesserocr", "cysignals"}

        class RejectRecognitionImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in blocked:
                    raise AssertionError("core imported recognition dependency: " + fullname)
                return None

        sys.meta_path.insert(0, RejectRecognitionImports())
        signals = (signal.SIGINT, signal.SIGTERM, signal.SIGABRT, signal.SIGSEGV, signal.SIGFPE)
        handlers = {item: signal.getsignal(item) for item in signals}
        omp = {key: value for key, value in os.environ.items() if key.startswith("OMP_")}
        from core_pdf import PdfDocument

        with PdfDocument.open(sys.argv[1]) as document:
            assert document.extract().pages[0].text == "Native content"
            image = document.pages[0].render().rasterize()
            assert image.array().shape == (100, 100, 4)
            assert tuple(image.array()[0, 0]) == (255, 0, 0, 255)
        assert handlers == {item: signal.getsignal(item) for item in signals}
        assert omp == {key: value for key, value in os.environ.items() if key.startswith("OMP_")}
        assert not any(name.split(".", 1)[0] in blocked for name in sys.modules)
        """
    )
    environment = os.environ.copy()
    environment.pop("OMP_THREAD_LIMIT", None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script, str(pdf_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

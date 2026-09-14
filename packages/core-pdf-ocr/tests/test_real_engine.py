"""Opt-in integration with the pinned Tesseract engine and English model."""

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from core_pdf import PdfDocument as NativePdfDocument
from core_pdf.impl._impl.render.model import RasterImage
from core_pdf.impl._impl.runtime.execution import ExtractionScope, internal_ExtractionCancelled
from core_pdf_ocr import PdfDocument
from core_pdf_ocr.impl.extract.ocr import tesseract
from core_pdf_ocr.impl.extract.ocr.types import internal_OcrTask

pytestmark = pytest.mark.skipif(
    os.environ.get("CORE_PDF_TESSERACT_TESTS") != "1",
    reason="set CORE_PDF_TESSERACT_TESTS=1 to run pinned real-engine integration",
)
FIXTURES = Path(__file__).with_name("fixtures")
TESSERACT_VERSION = "tesseract 5.5.1"
ENG_SHA256 = "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"


@pytest.fixture(autouse=True)
def pinned_engine() -> None:
    engine = tesseract.internal_import_tesserocr()
    assert engine.tesseract_version().splitlines()[0] == TESSERACT_VERSION
    model = Path(tesseract.internal_tessdata_path()) / "eng.traineddata"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == ENG_SHA256


@pytest.fixture
def owned_engines(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Count real API ownership without replacing recognition or image handling."""
    factory = tesseract.internal_api
    engines: list[Any] = []

    class TrackedEngine:
        def __init__(self, mode: int) -> None:
            self.api = factory(mode)
            self.ends = 0
            self.recognitions = 0
            engines.append(self)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.api, name)

        def Recognize(self, **kwargs: Any) -> bool:
            self.recognitions += 1
            return bool(self.api.Recognize(**kwargs))

        def End(self) -> None:
            self.ends += 1
            self.api.End()

    monkeypatch.setattr(tesseract, "internal_api", TrackedEngine)
    return engines


@pytest.mark.parametrize("name", ["invoice.pdf", "rotated-image.pdf"])
def test_public_extraction_recognizes_image_only_pages_and_releases_engine(
    name: str,
    owned_engines: list[Any],
) -> None:
    source = (FIXTURES / name).read_bytes()
    with NativePdfDocument(source) as native:
        assert not native.extract().to_markdown().strip()
    with PdfDocument(source) as document:
        result = document.extract()
        text = result.to_markdown()
        assert "INVOICE 2048" in text
        assert "Blue widgets 12" in text
        assert "Total 360 dollars" in text
    assert document.closed
    assert owned_engines
    assert all(engine.ends == 1 and engine.recognitions >= 1 for engine in owned_engines)


@pytest.fixture
def invoice_task() -> internal_OcrTask:
    with Image.open(FIXTURES / "invoice.png") as image:
        raster = RasterImage(image.tobytes(), image.width, image.height, 1)
    # Rectangle coordinates remain relative to the original raster.
    return internal_OcrTask(6, raster, (30, 35, 650, 80), (10, 20, 490, 212), 150)


def test_real_crop_excludes_other_lines_and_maps_into_page(
    invoice_task: internal_OcrTask,
    owned_engines: list[Any],
) -> None:
    result = tesseract.internal_recognize(invoice_task)
    assert result.recognition_status == "ok"
    assert tuple(result.observations.text) == ("INVOICE 2048",)
    x0, y0, x1, y1 = result.observations.bbox[0]
    assert 10 < x0 < x1 < 490
    assert 156 < y0 < y1 < 196
    assert len(owned_engines) == 1
    assert owned_engines[0].ends == 1


def test_real_engine_is_released_when_cancelled_between_same_image_tasks(
    invoice_task: internal_OcrTask,
    owned_engines: list[Any],
) -> None:
    context = ExtractionScope(
        cancelled=lambda: bool(owned_engines and owned_engines[0].recognitions)
    )
    with pytest.raises(internal_ExtractionCancelled):
        tesseract.internal_recognize_group(
            (invoice_task, replace(invoice_task, rectangle=(30, 120, 650, 80))),
            raise_if_cancelled=context.raise_if_cancelled,
        )
    assert len(owned_engines) == 1
    assert owned_engines[0].recognitions == 1
    assert owned_engines[0].ends == 1

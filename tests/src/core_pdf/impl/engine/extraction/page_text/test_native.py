from io import BytesIO
from pathlib import Path
from typing import Any, cast

from core_pdf.impl.engine.extraction.document import PdfDocument
from core_pdf.impl.engine.extraction.page_text.engine import build_page_extraction_result

TESTS_DIR = Path(__file__).parents[6]
SAMPLE_PDF = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "global-AIDS-strategy-p74-75-p001.pdf"


def image_only_pdf() -> BytesIO:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] "
            b"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        ),
        (
            b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
            b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 3 >>\n"
            b"stream\n\xff\x00\x00\nendstream"
        ),
        b"<< /Length 24 >>\nstream\nq\n10 0 0 10 0 0 cm\n/Im0 Do\nQ\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode())
        data.extend(obj)
        data.extend(b"\nendobj\n")
    xref_offset = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return BytesIO(data)


def test_native_extraction_returns_pdf_text_without_external_services() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])
        text = page.extract_text()

        assert text.strip()
        assert page.extraction_cache is not None
        assert "native_text" in page.extraction_cache
        assert "page_extraction_snapshot" not in page.extraction_cache


def test_structured_page_result_reports_native_route() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])
        result = build_page_extraction_result(page)

        assert result.text.strip()
        assert result.base_route in {"native_fast", "native_layout"}
        assert result.resolved_lines


def test_image_only_page_does_not_attempt_text_extraction() -> None:
    with PdfDocument(image_only_pdf()) as document:
        page = cast(Any, document.pages[0])
        result = build_page_extraction_result(page)

        assert page.extract_text() == ""
        assert result.text == ""
        assert result.page_class == "image"
        assert result.resolved_lines == ()

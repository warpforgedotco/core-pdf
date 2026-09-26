import zlib

import pytest

from core_pdf import PdfDocument
from tests.src.core_pdf.pdf_bytes import serialize_pdf

MASK_SAMPLES = bytes((0, 255, 255, 255))


def image_pdf(mask_filter: bytes) -> bytes:
    mask = zlib.compress(MASK_SAMPLES) if mask_filter == b"/FlateDecode" else MASK_SAMPLES
    mask_entry = b" /Filter " + mask_filter if mask_filter else b""
    image = bytes((255, 0, 0)) * 4
    content = b"q 100 0 0 100 50 50 cm /Im1 Do Q"
    return serialize_pdf(
        {
            1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
            b"/Resources << /XObject << /Im1 5 0 R >> >> /Contents 4 0 R >>",
            4: b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            5: b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceRGB "
            b"/BitsPerComponent 8 /SMask 6 0 R /Length %d >>\nstream\n"
            % len(image)
            + image
            + b"\nendstream",
            6: b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray "
            b"/BitsPerComponent 8%s /Length %d >>\nstream\n"
            % (mask_entry, len(mask))
            + mask
            + b"\nendstream",
        }
    )


@pytest.mark.parametrize("mask_filter", [b"", b"/FlateDecode"])
def test_soft_mask_alpha_is_the_mean_of_the_decoded_mask(mask_filter: bytes) -> None:
    with PdfDocument(image_pdf(mask_filter)) as document:
        drawings = document.pages[0].get_page_program().drawings
    images = [drawing for drawing in drawings if drawing.kind == "image"]
    assert len(images) == 1
    assert images[0].soft_mask_alpha == pytest.approx(sum(MASK_SAMPLES) / (255 * 4))

from core_pdf import PdfDocument
from core_pdf.impl.capture_program import DEFAULT_CAPTURE
from tests.src.core_pdf.pdf_bytes import serialize_pdf, stream_object

GROUP_CONTENT = b"1 0 0 rg /GS1 gs 0 0 50 50 re f 0 g 0 0 200 200 re f"


def mask_group_drawn_after_use_pdf() -> bytes:
    group = b"<< /Type /XObject /Subtype /Form /BBox [0 0 200 200] "
    group += b"/Group << /S /Transparency >> /Resources << /ExtGState << /GS1 7 0 R >> >> "
    group += b"/Length %d >>\nstream\n" % len(GROUP_CONTENT) + GROUP_CONTENT + b"\nendstream"
    return serialize_pdf(
        {
            1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
            b"/Resources << /ExtGState << /GS1 7 0 R >> /XObject << /Fm0 6 0 R >> >> "
            b"/Contents 4 0 R >>",
            4: stream_object(b"q /GS1 gs 0 0 100 100 re f Q /Fm0 Do"),
            5: b"<< /Type /Mask /S /Alpha /G 6 0 R >>",
            6: group,
            7: b"<< /Type /ExtGState /SMask 5 0 R >>",
        }
    )


def test_a_mask_rejected_while_its_group_is_active_is_captured_once_inactive() -> None:
    with PdfDocument(mask_group_drawn_after_use_pdf()) as document:
        body = document.pages[0].get_page_program(options=DEFAULT_CAPTURE).body
    drawn_by_form = [d for d in body.drawings if d.xobject_depth == 1]
    assert drawn_by_form
    assert all(d.graphics_soft_mask is not None for d in drawn_by_form)

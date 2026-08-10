# SPDX-License-Identifier: AGPL-3.0-only
"""Annotation appearance streams carry text that belongs to the page.

The value a reader types into a form lives in the widget's appearance stream,
not in the page content, so a page interpreted without appearances silently
loses every filled-in field.
"""

from __future__ import annotations

import io

from core_pdf import PdfDocument


def assemble(objects: list[bytes]) -> bytes:
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


def stream_obj(data: bytes, extra: bytes = b"") -> bytes:
    return f"<< /Length {len(data)} {extra.decode()} >>\nstream\n".encode() + data + b"\nendstream"


def appearance(text: bytes) -> bytes:
    body = b"BT /F1 10 Tf 2 4 Td (" + text + b") Tj ET\n"
    return stream_obj(
        body,
        b"/Type /XObject /Subtype /Form /BBox [0 0 120 20] /Resources << /Font << /F1 9 0 R >> >>",
    )


def widget(rect: bytes, appearance_ref: int, extra: bytes = b"") -> bytes:
    return (
        b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (field) /Rect "
        + rect
        + b" /AP << /N "
        + str(appearance_ref).encode()
        + b" 0 R >> "
        + extra
        + b" >>"
    )


def form_pdf() -> bytes:
    """One visible widget, one hidden widget, and no catalog /AcroForm."""
    content = b"BT /F1 12 Tf 50 700 Td (PageContent) Tj ET\n"
    return assemble(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Annots [5 0 R 7 0 R] "
            b"/Resources << /Font << /F1 9 0 R >> >> /Contents 4 0 R >>",
            stream_obj(content),
            widget(b"[100 500 220 520]", 6, b"/V (VisibleValue)"),
            appearance(b"VisibleValue"),
            # /F bit 2 is Hidden.
            widget(b"[100 400 220 420]", 8, b"/V (HiddenValue) /F 2"),
            appearance(b"HiddenValue"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


def scratch_widget_pdf() -> bytes:
    """A widget with neither /FT nor /T -- not a control a reader draws."""
    content = b"BT /F1 12 Tf 50 700 Td (PageContent) Tj ET\n"
    return assemble(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Annots [5 0 R] "
            b"/Resources << /Font << /F1 9 0 R >> >> /Contents 4 0 R >>",
            stream_obj(content),
            b"<< /Type /Annot /Subtype /Widget /Rect [100 500 220 520] /AP << /N 6 0 R >> >>",
            appearance(b"ScratchValue"),
            b"<< >>",
            b"<< >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


def page_runs(data: bytes):
    with PdfDocument.open(io.BytesIO(data)) as document:
        page = document.pages[0]
        return page.text_diagnostics().runs, page.get_fields()


def test_widget_appearance_text_reaches_extraction() -> None:
    runs, _ = page_runs(form_pdf())
    text = "".join(run.text for run in runs)
    assert "PageContent" in text
    assert "VisibleValue" in text


def test_hidden_annotations_are_not_extracted() -> None:
    runs, _ = page_runs(form_pdf())
    text = "".join(run.text for run in runs)
    assert "HiddenValue" not in text


def placed_origin(data: bytes) -> tuple[float, float]:
    runs, _ = page_runs(data)
    placed = [run for run in runs if "VisibleValue" in run.text]
    assert placed, "appearance text was not captured"
    return placed[0].bbox[0], placed[0].bbox[1]


def test_appearance_text_is_placed_at_the_annotation_rect() -> None:
    x0, y0 = placed_origin(form_pdf())
    # /Rect is [100 500 220 520] and the BBox [0 0 120 20] is the same size, so
    # the appearance is translated by the rect origin without being scaled.
    # Drawn at x=2 inside the form, the text lands at 102 rather than at 2.
    assert round(x0, 2) == 102.0
    assert 500.0 < y0 < 520.0


def test_appearance_is_scaled_onto_a_rect_of_a_different_size() -> None:
    base_x, base_y = placed_origin(form_pdf())
    # The same appearance on a rect twice as wide and tall.
    doubled = form_pdf().replace(b"[100 500 220 520]", b"[100 500 340 540]")
    scaled_x, scaled_y = placed_origin(doubled)
    # 12.5.5 maps the BBox onto the rect, so every offset from the rect origin
    # doubles along with it.
    assert round(scaled_x - 100.0, 4) == round(2 * (base_x - 100.0), 4)
    assert round(scaled_y - 500.0, 4) == round(2 * (base_y - 500.0), 4)


def test_widget_without_field_type_or_name_is_skipped() -> None:
    runs, _ = page_runs(scratch_widget_pdf())
    assert "ScratchValue" not in "".join(run.text for run in runs)


def test_fields_are_found_without_a_catalog_acroform() -> None:
    _, fields = page_runs(form_pdf())
    values = {field.value_text for field in fields if field.value_text}
    assert values == {"VisibleValue", "HiddenValue"}

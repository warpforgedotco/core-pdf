from core_pdf.impl.engine.writing import serialize_pdf_object
from core_pdf.impl.primitives import PdfName, PdfReference, PdfString
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream


def test_serialize_pdf_object_covers_core_syntax_types() -> None:
    value = {
        PdfName.of("Type"): PdfName.of("Example"),
        PdfName.of("Ref"): PdfReference(12, 3),
        PdfName.of("Text"): PdfString(b"hello (pdf)"),
        PdfName.of("Items"): [1, True, None, 1.5],
    }

    assert serialize_pdf_object(value) == (
        b"<< /Type /Example /Ref 12 3 R /Text <68656C6C6F202870646629> /Items [1 true null 1.5] >>"
    )


def test_serialize_pdf_stream_sets_length_from_raw_bytes() -> None:
    stream = PdfStream({PdfName.of("Subtype"): PdfName.of("Data")}, b"abc")

    assert serialize_pdf_object(stream) == (
        b"<< /Subtype /Data /Length 3 >>\nstream\nabc\nendstream"
    )

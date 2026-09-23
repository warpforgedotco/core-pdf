import pytest

from core_pdf import PdfDocument


def build_document(xref_dictionary: bytes) -> tuple[bytes, int]:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 99 99] >>",
    }
    data = bytearray(b"%PDF-1.5\n")
    for number, body in objects.items():
        data += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_offset = len(data)
    data += b"4 0 obj\n" + xref_dictionary + b"\nstream\n\nendstream\nendobj\n"
    data += b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref_offset
    return bytes(data), xref_offset


def padded_xref_dictionary(entries: int) -> bytes:
    index = b" ".join(b"%d 1" % number for number in range(entries))
    return (
        b"<< /Index ["
        + index
        + b"] /Type /XRef /W [1 2 1] /Size 4 /Root 1 0 R /ID [<aa> <bb>] /Length 0 >>"
    )


@pytest.mark.parametrize("entries", [0, 700])
def test_xref_stream_survives_a_dictionary_longer_than_any_fixed_window(entries):
    # Nothing bounds how much dictionary may precede Type /XRef: a long Index
    # array pushes it arbitrarily deep into the object. A candidate filter that
    # only inspected a fixed prefix would skip the stream and lose the trailer
    # metadata it carries.
    dictionary = padded_xref_dictionary(entries)
    data, xref_offset = build_document(dictionary)
    with PdfDocument(data) as document:
        assert document.may_be_xref_stream(document.raw_data, xref_offset, len(document.raw_data))
        assert "ID" in document.infer_trailer_metadata()


def test_object_without_any_marker_is_ruled_out_without_parsing():
    data, _ = build_document(padded_xref_dictionary(0))
    with PdfDocument(data) as document:
        plain = data.index(b"1 0 obj")
        assert not document.may_be_xref_stream(document.raw_data, plain, plain + 40)


def test_hex_escaped_name_falls_through_to_parsing():
    # A name written with a # escape would not match the plain literals, so a
    # span carrying one must be parsed rather than ruled out.
    data, xref_offset = build_document(
        b"<< /Type /XRe#66 /W [1 2 1] /Size 4 /Root 1 0 R /Length 0 >>"
    )
    with PdfDocument(data) as document:
        assert document.may_be_xref_stream(document.raw_data, xref_offset, len(document.raw_data))

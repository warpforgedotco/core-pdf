"""Small PDF files built byte by byte for tests; imports nothing from core."""

from collections.abc import Mapping, Sequence

MULTI_PAGE_COUNT = 3


def serialize_pdf(
    objects: Mapping[int, bytes] | Sequence[bytes],
    version: bytes = b"1.4",
    *,
    header_end: bytes = b"\n",
) -> bytes:
    """Write objects 1..n with a classic xref table and a trailer whose Root is 1 0 R.

    A sequence numbers its objects from 1; a mapping must number them 1..n.
    header_end follows each ``N 0 obj`` keyword.
    """
    numbered: dict[int, bytes] = (
        dict(objects) if isinstance(objects, Mapping) else dict(enumerate(objects, 1))
    )
    size = len(numbered) + 1
    if sorted(numbered) != list(range(1, size)):
        raise ValueError("objects must be numbered 1..n")
    data = bytearray(b"%PDF-" + version + b"\n")
    offsets: dict[int, int] = {}
    for number in range(1, size):
        offsets[number] = len(data)
        data += b"%d 0 obj" % number + header_end + numbered[number] + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 %d\n0000000000 65535 f \n" % size
    data += b"".join(b"%010d 00000 n \n" % offsets[number] for number in range(1, size))
    data += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref)
    return bytes(data)


def stream_object(content: bytes) -> bytes:
    return b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"


def one_page_pdf(content: bytes, *, width: int = 200, height: int = 200) -> bytes:
    """One page whose content stream may show text in Helvetica as /F1."""
    return serialize_pdf(
        {
            1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] " % (width, height)
            + b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            4: stream_object(content),
            5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        }
    )


def multi_page_pdf(count: int = MULTI_PAGE_COUNT, *, tagged: bool = False) -> bytes:
    """count pages reading "Page 1".."Page n", optionally with an empty structure tree."""
    page_numbers = [3 + 2 * index for index in range(count)]
    tree_number = 4 + 2 * count
    catalog = b"<< /Type /Catalog /Pages 2 0 R"
    catalog += f" /StructTreeRoot {tree_number} 0 R >>".encode() if tagged else b" >>"
    objects: dict[int, bytes] = {
        1: catalog,
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
        objects[number + 1] = stream_object(content)
    if tagged:
        objects[tree_number] = b"<< /Type /StructTreeRoot /K [] >>"
    return serialize_pdf(objects)

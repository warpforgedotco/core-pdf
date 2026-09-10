import struct
from io import BytesIO

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .support import FIXTURES_ROOT
from .test_pymupdf_geometry import internal_geometry_expected

real_pypdf = pytest.importorskip("pypdf")
real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("family", range(16))
def test_embedded_font_family_classes(family: int) -> None:
    writer = real_pypdf.PdfWriter(
        clone_from=FIXTURES_ROOT / "PyMuPDF/tests/resources/battery-file-22.pdf"
    )
    page = writer.pages[0]
    for resource_name, font_ref in page["/Resources"]["/Font"].items():
        font = font_ref.get_object()
        if "Wingdings" not in str(font.get("/BaseFont")):
            continue
        stream = font["/DescendantFonts"][0].get_object()["/FontDescriptor"]["/FontFile2"]
        data = bytearray(stream.get_data())
        table_count = struct.unpack_from(">H", data, 4)[0]
        for position in range(12, 12 + table_count * 16, 16):
            tag, _, offset, _ = struct.unpack_from(">4sIII", data, position)
            if tag == b"OS/2":
                data[offset + 30] = family
        stream.set_data(bytes(data))
        content = real_pypdf.generic.DecodedStreamObject()
        content.set_data(f"BT {resource_name} 8 Tf 20 150 Td <008b> Tj ET".encode())
        page[real_pypdf.generic.NameObject("/Contents")] = writer._add_object(content)
        break
    output = BytesIO()
    writer.write(output)
    with (
        real_pymupdf.open(stream=output.getvalue()) as expected,
        compat_pymupdf.open(stream=output.getvalue()) as actual,
    ):
        assert actual[0].get_text("rawdict") == internal_geometry_expected(
            expected[0].get_text("rawdict")
        )

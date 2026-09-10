from io import BytesIO

import imagecodecs
import numpy
import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

real_pymupdf = pytest.importorskip("pymupdf")
real_pypdf = pytest.importorskip("pypdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("base", ["DeviceGray", "DeviceRGB", "DeviceCMYK", "DeviceN"])
@pytest.mark.parametrize("bits", [2, 8])
def test_indexed_image_palette_preserves_base_color_space(base: str, bits: int) -> None:
    g = real_pypdf.generic
    name = g.NameObject
    number = g.NumberObject
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    palette = bytes([0, 75, 128, 255])
    color_space = name("/" + base)
    if base == "DeviceRGB":
        palette = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 80, 90, 100])
    elif base in {"DeviceCMYK", "DeviceN"}:
        inks = [(0, 0, 0, 255), (128, 76, 0, 25), (31, 59, 87, 115), (0, 0, 0, 0)]
        palette = bytes(component for ink in inks for component in ink)
        if base == "DeviceN":
            function = g.DecodedStreamObject()
            function.set_data(b"{ pop }")
            function[name("/FunctionType")] = number(4)
            function[name("/Domain")] = g.ArrayObject([number(0), number(1)] * 5)
            function[name("/Range")] = g.ArrayObject([number(0), number(1)] * 4)
            color_space = g.ArrayObject(
                [
                    name("/DeviceN"),
                    g.ArrayObject([name("/" + ink) for ink in ("C", "M", "Y", "K", "Spot")]),
                    name("/DeviceCMYK"),
                    writer._add_object(function),
                ]
            )
            palette = bytes(component for ink in inks for component in (*ink, 127))
    image = g.DecodedStreamObject()
    image.set_data(bytes([0, 1, 2, 3]) if bits == 8 else b"\x1b")
    image.update(
        {
            name("/Type"): name("/XObject"),
            name("/Subtype"): name("/Image"),
            name("/Width"): number(4),
            name("/Height"): number(1),
            name("/BitsPerComponent"): number(bits),
            name("/ColorSpace"): g.ArrayObject(
                [name("/Indexed"), color_space, number(3), g.ByteStringObject(palette)]
            ),
        }
    )
    page[name("/Resources")] = g.DictionaryObject(
        {name("/XObject"): g.DictionaryObject({name("/I"): writer._add_object(image)})}
    )
    content = g.DecodedStreamObject()
    content.set_data(b"q 100 0 0 25 20 100 cm /I Do Q")
    page[name("/Contents")] = writer._add_object(content)
    output = BytesIO()
    writer.write(output)
    with (
        real_pymupdf.open(stream=output.getvalue()) as reference,
        compat_pymupdf.open(stream=output.getvalue()) as actual,
    ):
        expected = reference[0].get_text("dict")["blocks"][0]
        got = actual[0].get_text("dict")["blocks"][0]
        assert got["colorspace"] == expected["colorspace"]
        assert numpy.array_equal(
            imagecodecs.png_decode(got["image"]), imagecodecs.png_decode(expected["image"])
        )

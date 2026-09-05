# SPDX-License-Identifier: AGPL-3.0-only
"""Indirect image decoder inputs must behave like their direct equivalents.

All fourteen PDFs passed qpdf 12.3.2 --check before expectations were added.
Poppler 26.07.0 pdftoppm -r 72 paints the left half cyan for masked RGB and red
for stencils, with the right half white in both cases, for every representation.
"""

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.types import PdfReference
from tests.helpers.pdf_bytes import one_page_pdf, stream_obj


def internal_image_pdf(stencil: bool, indirect: str) -> bytes:
    decode = b"[1 0]" if stencil else b"[1 0 1 0 1 0]"
    values = {"Width": b"2", "Height": b"1", "BitsPerComponent": b"1", "Decode": decode}
    numbers = {"Width": 8, "Height": 9, "BitsPerComponent": 10, "Decode": 11}
    for key, number in numbers.items():
        if indirect in (key, "all"):
            values[key] = f"{number} 0 R".encode()
    if indirect == "decode-items":
        values["Decode"] = (
            b"[12 0 R 13 0 R]" if stencil else b"[12 0 R 13 0 R 12 0 R 13 0 R 12 0 R 13 0 R]"
        )
    metadata = b"/Type /XObject /Subtype /Image " + b" ".join(
        b"/" + key.encode() + b" " + value for key, value in values.items()
    )
    metadata += b" /ImageMask true" if stencil else b" /ColorSpace /DeviceRGB /SMask 7 0 R"
    metadata += b" /Custom 14 0 R"
    return one_page_pdf(
        b"1 0 0 rg 100 0 0 100 0 0 cm /Im Do",
        media_box=(0, 0, 100, 100),
        resources=b"<< /XObject << /Im 6 0 R >> >>",
        extra_objects=[
            stream_obj(b"\x80" if stencil else b"\x88", metadata),
            stream_obj(
                b"\xff\x00", b"/Width 2 /Height 1 /BitsPerComponent 8 /ColorSpace /DeviceGray"
            ),
            b"2",
            b"1",
            b"1",
            decode,
            b"1",
            b"0",
            b"(untouched)",
        ],
    )


@pytest.mark.parametrize("stencil", [False, True], ids=["masked-rgb", "stencil"])
@pytest.mark.parametrize(
    "indirect", ["none", "Width", "Height", "BitsPerComponent", "Decode", "decode-items", "all"]
)
def test_image_decoder_resolves_selected_inputs_without_changing_source_dictionary(
    stencil: bool, indirect: str
) -> None:
    with PdfDocument(internal_image_pdf(stencil, indirect)) as document:
        stream = document.resolve(PdfReference(6))
        assert isinstance(stream, PdfStream)
        original = dict(stream.dictionary)
        page = document.pages[0]
        drawing = page.get_page_program().drawings[0]
        source = drawing.image_source
        assert source is not None
        prepared = source.prepare()
        assert prepared is not None
        assert (prepared.raster.width, prepared.raster.height) == (2, 1)
        assert prepared.raster.array[0, :, -1].tolist() == [255, 0]
        assert prepared.is_stencil is stencil
        assert source.dictionary["Decode"] == ([1, 0] if stencil else [1, 0] * 3)
        assert source.dictionary["Custom"] == PdfReference(14)
        assert stream.dictionary == original
        raster = page.render().rasterize(background=(255, 255, 255, 255)).array()

    assert raster[50, 25].tolist() == ([255, 0, 0, 255] if stencil else [0, 255, 255, 255])
    assert raster[50, 75].tolist() == [255, 255, 255, 255]

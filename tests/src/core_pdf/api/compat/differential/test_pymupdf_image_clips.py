"""Conservative curve clips in structured image output."""

from pathlib import Path

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_geometry import internal_geometry_expected
from .test_pymupdf_structured_text import internal_output
from .test_pymupdf_text import real_pymupdf

pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("flags", [4, 199])
@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("matrix", [b"1 0 0 1 0 0", b".8 .3 -.2 .9 35 5"])
@pytest.mark.parametrize("form", [False, True])
def test_image_cubic_clip_bounds_and_graphics_restore(
    flags: int, inline: bool, matrix: bytes, form: bool
) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page(width=200, height=200)

        def stream(dictionary: str, content: bytes) -> int:
            xref = document.get_new_xref()
            document.update_object(xref, dictionary)
            document.update_stream(xref, content)
            return xref

        image = stream(
            "<< /Type /XObject /Subtype /Image /Width 2 /Height 2 "
            "/ColorSpace /DeviceGray /BitsPerComponent 8 >>",
            bytes((0, 80, 160, 255)),
        )
        paint = b"BI /W 2 /H 2 /BPC 8 /CS /G ID \x00\x50\xa0\xff EI" if inline else b"/Im Do"
        paint = b"q 140 0 0 140 10 10 cm " + paint + b" Q "
        # Both control points lie outside the curve's extrema. A trailing
        # moveto must not enlarge the clip. Q and the Form's implicit save
        # must each restore the outer clip before the next image is painted.
        clipped = (
            b"q "
            + matrix
            + b" cm 20 20 m 0 160 140 160 120 20 c h 180 190 m W n "
            + paint
            + b" Q "
            + paint
        )
        resources = f"<< /XObject << /Im {image} 0 R >> >>"
        if form:
            child = stream(
                f"<< /Type /XObject /Subtype /Form /BBox [15 15 145 130] /Resources {resources} >>",
                clipped,
            )
            resources = f"<< /XObject << /Im {image} 0 R /Fm {child} 0 R >> >>"
            clipped = b"/Fm Do "
        document.xref_set_key(page.xref, "Resources", resources)
        page.set_contents(stream("<< >>", b"5 5 160 170 re W n " + clipped + paint))
        source = document.tobytes()
    assert internal_output(
        compat_pymupdf, source, "dict", flags=flags
    ) == internal_geometry_expected(internal_output(real_pymupdf, source, "dict", flags=flags))


def test_clipped_images_from_fixture() -> None:
    source = Path("tests/fixtures/PyMuPDF/tests/resources/test_4942.pdf").read_bytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        for actual_page, expected_page in zip(actual, expected, strict=True):
            actual_images = [
                block for block in actual_page.get_text("dict")["blocks"] if block["type"] == 1
            ]
            expected_images = [
                block for block in expected_page.get_text("dict")["blocks"] if block["type"] == 1
            ]
            assert actual_images == internal_geometry_expected(expected_images)

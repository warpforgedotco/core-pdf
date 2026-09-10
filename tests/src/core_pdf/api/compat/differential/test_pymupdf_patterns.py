"""Structured images in direct, cached and nested tiling patterns."""

from pathlib import Path

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_geometry import internal_geometry_expected
from .test_pymupdf_structured_text import internal_output
from .test_pymupdf_text import real_pymupdf

pytestmark = pytest.mark.compat_differential


def internal_pattern_pdf(
    *,
    matrix: str = "1 0 0 1 0 0",
    paint: str = "10 10 20 20",
    bbox: str = "0 0 20 20",
    step: str = "20 20",
    operator: str = "f",
    content_matrix: str = "1 0 0 1 0 0",
    form: bool = False,
    nested: bool = False,
    inline: bool = False,
) -> bytes:
    with real_pymupdf.open() as document:
        page = document.new_page(width=200, height=200)

        def stream(dictionary: str, content: bytes) -> int:
            xref = document.get_new_xref()
            document.update_object(xref, dictionary)
            document.update_stream(xref, content)
            return xref

        image = stream(
            "<< /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 >>",
            b"\xff\x40\x00",
        )
        xstep, ystep = step.split()
        image_paint = b"BI /W 1 /H 1 /BPC 8 /CS /RGB ID \xff\x40\x00 EI" if inline else b"/Im Do"
        pattern = stream(
            "<< /Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 "
            f"/BBox [{bbox}] /XStep {xstep} /YStep {ystep} /Matrix [{matrix}] "
            f"/Resources << /XObject << /Im {image} 0 R >> >> >>",
            b"15 0 0 15 0 0 cm " + image_paint,
        )
        resources = f"<< /Pattern << /P {pattern} 0 R >> >>"
        content = (
            f"{content_matrix} cm /Pattern CS /P SCN /Pattern cs /P scn {paint} re {operator}"
        ).encode()
        wrapper_matrix = "1.2 .1 -.2 1.1 25 15"
        if nested:
            outer = stream(
                "<< /Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 "
                "/BBox [0 0 60 60] /XStep 60 /YStep 60 "
                f"/Matrix [{wrapper_matrix}] /Resources {resources} >>",
                content,
            )
            resources = f"<< /Pattern << /P {outer} 0 R >> >>"
            content = b"/Pattern cs /P scn 10 10 60 60 re f"
        if form:
            child = stream(
                "<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] "
                f"/Matrix [{wrapper_matrix}] /Resources {resources} >>",
                content,
            )
            resources = f"<< /XObject << /Fm {child} 0 R >> >>"
            content = b"/Fm Do"
        document.xref_set_key(page.xref, "Resources", resources)
        page.set_contents(stream("<< >>", content))
        return document.tobytes()


@pytest.mark.parametrize("flags", [4, 199])
@pytest.mark.parametrize("matrix", ["1 0 0 1 0 0", "1.2 .1 -.2 1.1 25 15"])
@pytest.mark.parametrize("paint", ["10 10 40 40", "0 0 20 20", "10 10 20 20", "0 100 40 40"])
@pytest.mark.parametrize("bbox,step", [("0 0 20 20", "20 20"), ("-5 -5 15 15", "-20 20")])
def test_pattern_image_cells_and_cache(
    flags: int, matrix: str, paint: str, bbox: str, step: str
) -> None:
    source = internal_pattern_pdf(matrix=matrix, paint=paint, bbox=bbox, step=step)
    assert internal_output(
        compat_pymupdf, source, "dict", flags=flags
    ) == internal_geometry_expected(internal_output(real_pymupdf, source, "dict", flags=flags))


@pytest.mark.parametrize("form", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("inline", [False, True])
def test_pattern_image_stream_anchors(form: bool, nested: bool, inline: bool) -> None:
    source = internal_pattern_pdf(form=form, nested=nested, inline=inline)
    assert internal_output(compat_pymupdf, source, "dict") == internal_geometry_expected(
        internal_output(real_pymupdf, source, "dict")
    )


@pytest.mark.parametrize("operator", ["f", "S", "B"])
@pytest.mark.parametrize("matrix", ["1 0 0 1 0 0", ".7 .2 -.3 .8 25 15"])
def test_pattern_paint_bounds(operator: str, matrix: str) -> None:
    source = internal_pattern_pdf(operator=operator, content_matrix=matrix)
    assert internal_output(compat_pymupdf, source, "dict") == internal_geometry_expected(
        internal_output(real_pymupdf, source, "dict")
    )


def test_pattern_image_fixture() -> None:
    source = Path("tests/fixtures/PyMuPDF/tests/resources/bug1945.pdf").read_bytes()
    assert internal_output(compat_pymupdf, source, "dict") == internal_geometry_expected(
        internal_output(real_pymupdf, source, "dict")
    )

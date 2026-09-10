from io import BytesIO

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .support import FIXTURES_ROOT
from .test_pymupdf_geometry import internal_geometry_expected

real_pypdf = pytest.importorskip("pypdf")
real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("bottom", [-10, -9.475, -9, 0])
@pytest.mark.parametrize("flags", [0, 64, 195, 199])
def test_annotation_content_clip_uses_empty_glyph_origin(bottom: float, flags: int) -> None:
    writer = real_pypdf.PdfWriter(clone_from=FIXTURES_ROOT / "PyMuPDF/tests/resources/1.pdf")
    for annotation_ref in writer.pages[0]["/Annots"]:
        annotation = annotation_ref.get_object()
        if annotation.get("/Contents") == "modified text field":
            appearance = annotation["/AP"]["/N"]
            appearance.set_data(
                appearance.get_data().replace(
                    b"-3 -9 94 12 re", f"-3 {bottom} 94 {3 - bottom} re".encode()
                )
            )
    output = BytesIO()
    writer.write(output)
    with (
        real_pymupdf.open(stream=output.getvalue()) as expected,
        compat_pymupdf.open(stream=output.getvalue()) as actual,
    ):
        expected_page, actual_page = expected[0], actual[0]
        for kind in ("text", "words", "blocks", "rawdict"):
            assert actual_page.get_text(kind, flags=flags) == internal_geometry_expected(
                expected_page.get_text(kind, flags=flags)
            )


@pytest.mark.parametrize("fontname", ["helv", "tiro"])
@pytest.mark.parametrize("bottom", [99, 100, 101])
@pytest.mark.parametrize("flags", [0, 64, 195])
def test_substitute_font_spaces_obey_content_clip(fontname: str, bottom: int, flags: int) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page(width=200, height=200)
        page.insert_text((20, 100), "Seed", fontname=fontname, fontsize=12)
        font = page.get_fonts()[0][4]
        document.update_stream(
            page.get_contents()[0],
            (
                f"q 20 {bottom} 100 20 re W n "
                f"BT /{font} 12 Tf 20 100 Td (AA BB ) Tj ET Q "
                f"q 20 40 100 20 re W n BT /{font} 2 Tf 20 50 Td ( ) Tj ET Q"
            ).encode(),
        )
        source = document.tobytes()
    with (
        real_pymupdf.open(stream=source) as expected,
        compat_pymupdf.open(stream=source) as actual,
    ):
        expected_page, actual_page = expected[0], actual[0]
        for kind in ("text", "words", "blocks", "rawdict"):
            assert actual_page.get_text(kind, flags=flags) == internal_geometry_expected(
                expected_page.get_text(kind, flags=flags)
            )


@pytest.mark.parametrize("scale", [(0.5, 2), (0.75, 1.5), (1.25, 0.8)])
def test_annotation_placement_rounds_source_boxes_before_scaling(
    scale: tuple[float, float],
) -> None:
    writer = real_pypdf.PdfWriter(clone_from=FIXTURES_ROOT / "PyMuPDF/tests/resources/1.pdf")
    for annotation_ref in writer.pages[0]["/Annots"]:
        annotation = annotation_ref.get_object()
        if annotation.get("/Contents") == "typewriter text":
            rect = annotation["/Rect"]
            annotation[real_pypdf.generic.NameObject("/Rect")] = real_pypdf.generic.ArrayObject(
                real_pypdf.generic.FloatObject(
                    (180.12, 600.34)[index % 2]
                    + (float(value) - float(rect[index % 2])) * scale[index % 2]
                )
                for index, value in enumerate(rect)
            )
    output = BytesIO()
    writer.write(output)
    with (
        real_pymupdf.open(stream=output.getvalue()) as expected,
        compat_pymupdf.open(stream=output.getvalue()) as actual,
    ):
        expected_page, actual_page = expected[0], actual[0]
        for kind in ("text", "words", "blocks", "rawdict"):
            assert actual_page.get_text(kind) == internal_geometry_expected(
                expected_page.get_text(kind)
            )

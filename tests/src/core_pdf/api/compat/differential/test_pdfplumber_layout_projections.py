"""Layout-object projections retain text, geometry, and disabled-layout behavior."""

from io import BytesIO

import pytest

from core_pdf.api.compat import pdfplumber as compat

reference = pytest.importorskip("pdfplumber")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("laparams", [None, {}, {"detect_vertical": True}])
@pytest.mark.parametrize(
    "property_name",
    ["textboxhorizontals", "textboxverticals", "textlinehorizontals", "textlineverticals"],
)
def test_layout_object_text_and_bounds_match_reference(text_pdf_bytes, laparams, property_name):
    snapshots = []
    for library in (reference, compat):
        with library.open(BytesIO(text_pdf_bytes), laparams=laparams) as pdf:
            page = pdf.pages[0]
            snapshots.append(
                [
                    (
                        item["text"],
                        tuple(round(item[key], 6) for key in ("x0", "top", "x1", "bottom")),
                    )
                    for item in getattr(page, property_name)
                ]
            )
    assert snapshots[1] == snapshots[0]


@pytest.mark.parametrize("box_name", ["artbox", "trimbox", "bleedbox"])
@pytest.mark.parametrize("present", [False, True])
def test_optional_page_boxes_match_reference(text_pdf_bytes, box_name, present):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject, RectangleObject

    writer = PdfWriter()
    page = writer.add_page(PdfReader(BytesIO(text_pdf_bytes)).pages[0])
    if present:
        page[
            NameObject(
                {"artbox": "/ArtBox", "trimbox": "/TrimBox", "bleedbox": "/BleedBox"}[box_name]
            )
        ] = RectangleObject((10, 20, 90, 160))
    stream = BytesIO()
    writer.write(stream)
    snapshots = []
    for library in (reference, compat):
        with library.open(BytesIO(stream.getvalue())) as pdf:
            snapshots.append(getattr(pdf.pages[0], box_name, None))
    assert snapshots[1] == snapshots[0]


@pytest.mark.parametrize("cropped", [False, True])
def test_annotation_and_hyperlink_geometry_match_reference(text_pdf_bytes, cropped):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DictionaryObject,
        NameObject,
        RectangleObject,
        TextStringObject,
    )

    writer = PdfWriter()
    page = writer.add_page(PdfReader(BytesIO(text_pdf_bytes)).pages[0])
    page[NameObject("/Annots")] = ArrayObject(
        [
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Annot"),
                    NameObject("/Subtype"): NameObject("/Link"),
                    NameObject("/Rect"): RectangleObject((20, 80, 100, 120)),
                    NameObject("/Contents"): TextStringObject("A link"),
                    NameObject("/A"): DictionaryObject(
                        {
                            NameObject("/S"): NameObject("/URI"),
                            NameObject("/URI"): TextStringObject("https://example.com/"),
                        }
                    ),
                }
            )
        ]
    )
    output = BytesIO()
    writer.write(output)
    snapshots = []
    for library in (reference, compat):
        with library.open(BytesIO(output.getvalue())) as pdf:
            page = pdf.pages[0]
            if cropped:
                page = page.crop((0, 0, 150, 150))
            annotations = [
                (item["contents"], tuple(item[key] for key in ("x0", "top", "x1", "bottom")))
                for item in page.annots
            ]
            links = [
                (item["uri"], tuple(item[key] for key in ("x0", "top", "x1", "bottom")))
                for item in page.hyperlinks
            ]
            snapshots.append((annotations, links))
    assert snapshots[1] == snapshots[0]

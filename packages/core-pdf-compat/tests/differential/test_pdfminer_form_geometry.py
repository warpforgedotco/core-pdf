from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    RectangleObject,
)

from core_pdf_compat import pdfminer as compat

reference = pytest.importorskip("pdfminer.high_level")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("media_origin", [(0, 0), (10, 20)])
@pytest.mark.parametrize("nested", [False, True])
def test_form_bounds_and_characters_match_reference_after_page_transform(
    rotation, media_origin, nested
):
    writer = PdfWriter()
    page = writer.add_blank_page(200, 300)
    left, bottom = media_origin
    page[NameObject("/MediaBox")] = RectangleObject((left, bottom, left + 200, bottom + 300))
    page[NameObject("/Rotate")] = NumberObject(rotation)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    form = DecodedStreamObject()
    form.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Form"),
            NameObject("/BBox"): RectangleObject((0, 0, 100, 80)),
            NameObject("/Resources"): DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            ),
        }
    )
    form.set_data(b"BT /F1 12 Tf 10 30 Td (AB) Tj ET")
    if nested:
        outer = DecodedStreamObject()
        outer.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Form"),
                NameObject("/BBox"): RectangleObject((0, 0, 150, 100)),
                NameObject("/Resources"): DictionaryObject(
                    {
                        NameObject("/XObject"): DictionaryObject(
                            {NameObject("/Inner"): writer._add_object(form)}
                        )
                    }
                ),
            }
        )
        outer.set_data(b"q 1 0 0 1 5 7 cm /Inner Do Q")
        form = outer
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject("/Form1"): writer._add_object(form)})}
    )
    content = DecodedStreamObject()
    content.set_data(b"q 1 0 0 1 30 40 cm /Form1 Do Q")
    page[NameObject("/Contents")] = writer._add_object(content)
    output = BytesIO()
    writer.write(output)

    def snapshot(item):
        kind = type(item).__name__
        return (
            kind,
            tuple(round(value, 6) for value in item.bbox),
            item.get_text() if kind == "LTChar" else None,
            tuple(snapshot(child) for child in item) if hasattr(item, "__iter__") else (),
        )

    snapshots = [
        snapshot(next(library.extract_pages(BytesIO(output.getvalue()))))
        for library in (reference, compat)
    ]
    assert snapshots[1] == snapshots[0]
    assert snapshots[0][3]

    def text(record):
        return record[2] if record[2] is not None else "".join(text(child) for child in record[3])

    assert text(snapshots[0]) == "AB"

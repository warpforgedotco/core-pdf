from io import BytesIO

import pytest

from core_pdf.api.compat import pikepdf as compat_pikepdf

real_pikepdf = pytest.importorskip("pikepdf")
real_pypdf = pytest.importorskip("pypdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize(
    ("media", "crop"),
    [
        ("missing", "missing"),
        ("null", "null"),
        ("valid", "missing"),
        ("valid", "null"),
        ("valid", "valid"),
    ],
)
@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("indirect", [False, True])
def test_recovered_page_boxes_match_reference(
    media: str, crop: str, inherited: bool, indirect: bool
) -> None:
    generic = real_pypdf.generic
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=300)
    del page["/MediaBox"]
    owner = writer.root_object["/Pages"] if inherited else page
    for name, kind, coordinates in (
        ("/MediaBox", media, (10.5, 20.25, 190.75, 280.5)),
        ("/CropBox", crop, (30.25, 40.5, 160.75, 220.25)),
    ):
        if kind == "missing":
            continue
        value = (
            generic.NullObject()
            if kind == "null"
            else generic.ArrayObject([generic.FloatObject(item) for item in coordinates])
        )
        owner[generic.NameObject(name)] = writer._add_object(value) if indirect else value
    stream = BytesIO()
    writer.write(stream)
    source = stream.getvalue()

    with (
        real_pikepdf.Pdf.open(BytesIO(source)) as expected,
        compat_pikepdf.Pdf.open(BytesIO(source)) as actual,
    ):
        assert len(actual.pages) == len(expected.pages) == 1
        assert tuple(actual.pages[0].mediabox) == tuple(expected.pages[0].mediabox)
        assert tuple(actual.pages[0].cropbox) == tuple(expected.pages[0].cropbox)

from io import BytesIO

import pytest

from core_pdf.api.compat import pikepdf as compat_pikepdf
from core_pdf.api.compat import pypdf as compat_pypdf

reference_pypdf = pytest.importorskip("pypdf")
reference_pikepdf = pytest.importorskip("pikepdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("box", [(0, 0, 200, 300), (-10, -20, 5, 8), (10, 20, -5, -8)])
@pytest.mark.parametrize(
    "attribute",
    [
        "left",
        "bottom",
        "right",
        "top",
        "lower_left",
        "lower_right",
        "upper_left",
        "upper_right",
        "width",
        "height",
    ],
)
def test_rectangle_accessors_match_reference(box, attribute):
    actual = compat_pypdf.Rectangle(*box)
    expected = reference_pypdf.generic.RectangleObject(box)
    assert getattr(actual, attribute) == getattr(expected, attribute)
    assert tuple(actual) == tuple(expected)


@pytest.mark.parametrize("operation", ["delete", "insert", "slice", "replace"])
def test_synthetic_page_collection_mutation_matches_reference(text_pdf_bytes, operation):
    def snapshot(module):
        with module.Pdf.open(BytesIO(text_pdf_bytes)) as source, module.Pdf.new() as target:
            target.pages.append(source.pages[0])
            target.pages.append(source.pages[0])
            if operation == "delete":
                del target.pages[0]
            elif operation == "insert":
                target.pages.insert(0, source.pages[0])
            elif operation == "slice":
                del target.pages[1:]
            else:
                target.pages[0] = source.pages[0]
            assert len(source.pages) == 1
            return [tuple(float(value) for value in page.mediabox) for page in target.pages]

    assert snapshot(compat_pikepdf) == snapshot(reference_pikepdf)

from io import BytesIO

import pytest

from .support import metadata

real_pikepdf = pytest.importorskip("pikepdf")
real_pypdf = pytest.importorskip("pypdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("alias_page_tree", [False, True])
@pytest.mark.parametrize("nested_pages", [False, True])
def test_info_page_tree_references_match_reference(
    alias_page_tree: bool, nested_pages: bool
) -> None:
    from core_pdf.api.compat import pikepdf as compat_pikepdf

    generic = real_pypdf.generic
    name = generic.NameObject
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=100, height=200)
    root = writer._pages.get_object()
    root[name("/MediaBox")] = page.pop("/MediaBox")
    root[name("/CropBox")] = generic.ArrayObject(
        [generic.NumberObject(value) for value in (1, 2, 90, 180)]
    )
    root[name("/Rotate")] = generic.NumberObject(90)
    root[name("/Resources")] = generic.DictionaryObject()
    if nested_pages:
        ancestor = generic.DictionaryObject(
            {
                name("/Type"): name("/Pages"),
                name("/Count"): generic.NumberObject(1),
                name("/Kids"): root["/Kids"],
                name("/Parent"): writer._pages,
            }
        )
        ancestor_ref = writer._add_object(ancestor)
        root[name("/Kids")] = generic.ArrayObject([ancestor_ref])
        page[name("/Parent")] = ancestor_ref
    # The writer copies the assigned Info dictionary into its own object. Keep
    # that independent copy, or redirect the trailer to the actual page tree.
    writer._info = writer._pages
    output = BytesIO()
    writer.write(output)
    data = output.getvalue()
    if alias_page_tree:
        data = data.replace(b"/Info 1 0 R", b"/Info 2 0 R")

    with (
        real_pikepdf.Pdf.open(BytesIO(data)) as expected,
        compat_pikepdf.Pdf.open(BytesIO(data)) as actual,
    ):
        assert [tuple(page.mediabox) for page in actual.pages] == [
            tuple(page.mediabox) for page in expected.pages
        ]
        assert metadata(actual.docinfo) == metadata(expected.docinfo)


def test_nested_info_arrays_and_dictionaries_match_reference() -> None:
    from core_pdf.api.compat import pikepdf as compat_pikepdf

    with real_pikepdf.Pdf.new() as writer:
        writer.add_blank_page(page_size=(100, 200))
        writer.docinfo["/Nested"] = real_pikepdf.Array(
            [real_pikepdf.Dictionary(Names=real_pikepdf.Array(["first", "second"])), [1, 2]]
        )
        output = BytesIO()
        writer.save(output)

    with (
        real_pikepdf.Pdf.open(BytesIO(output.getvalue())) as expected,
        compat_pikepdf.Pdf.open(BytesIO(output.getvalue())) as actual,
    ):
        assert metadata(actual.docinfo) == metadata(expected.docinfo)

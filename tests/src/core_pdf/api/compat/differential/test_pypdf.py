from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from .support import call_pair, differential_pdfs, pdf_id

real_pypdf = pytest.importorskip("pypdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("pdf_path", differential_pdfs(), ids=pdf_id)
def test_matches_real_library_on_fixture_corpus(pdf_path: Path) -> None:
    from core_pdf.api.compat import pypdf as compat_pypdf

    def snapshot(reader_type: Any) -> tuple[tuple[str, tuple[float, ...], int], ...]:
        with reader_type(pdf_path, strict=False) as reader:
            return tuple(
                (page.extract_text(), tuple(page.mediabox), page.rotation) for page in reader.pages
            )

    pair = call_pair(
        lambda: snapshot(real_pypdf.PdfReader), lambda: snapshot(compat_pypdf.PdfReader)
    )
    if pair is not None:
        assert pair[1] == pair[0]


@pytest.mark.parametrize("indirect_fields", [False, True], ids=["direct-fields", "indirect-fields"])
@pytest.mark.parametrize("indirect_kids", [False, True], ids=["direct-kids", "indirect-kids"])
@pytest.mark.parametrize("recover_xref", [False, True], ids=["valid-xref", "recovered-xref"])
def test_form_arrays_match_real_library(
    indirect_fields: bool, indirect_kids: bool, recover_xref: bool
) -> None:
    from core_pdf.api.compat import pypdf as compat_pypdf

    generic = real_pypdf.generic
    name = generic.NameObject
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    parent = generic.DictionaryObject({name("/T"): generic.TextStringObject("customer")})
    parent_ref = writer._add_object(parent)
    field = generic.DictionaryObject(
        {
            name("/FT"): name("/Tx"),
            name("/T"): generic.TextStringObject("name"),
            name("/V"): generic.TextStringObject("Alice"),
            name("/Parent"): parent_ref,
        }
    )
    field_ref = writer._add_object(field)
    child_fields = generic.ArrayObject([field_ref])
    parent[name("/Kids")] = writer._add_object(child_fields) if indirect_kids else child_fields
    widget = generic.DictionaryObject(
        {
            name("/Type"): name("/Annot"),
            name("/Subtype"): name("/Widget"),
            name("/Parent"): field_ref,
            name("/P"): page.indirect_reference,
            name("/Rect"): generic.ArrayObject(
                [generic.NumberObject(value) for value in (10, 10, 100, 30)]
            ),
        }
    )
    widget_ref = writer._add_object(widget)
    kids = generic.ArrayObject([widget_ref])
    field[name("/Kids")] = writer._add_object(kids) if indirect_kids else kids
    page[name("/Annots")] = generic.ArrayObject([widget_ref])
    fields = generic.ArrayObject([parent_ref])
    writer.root_object[name("/AcroForm")] = generic.DictionaryObject(
        {name("/Fields"): writer._add_object(fields) if indirect_fields else fields}
    )
    data = BytesIO()
    writer.write(data)
    pdf_bytes = data.getvalue()
    if recover_xref:
        pdf_bytes = pdf_bytes.replace(b"\nxref\n", b"\nxxxx\n", 1)
        pdf_bytes = pdf_bytes.rsplit(b"startxref", 1)[0] + b"startxref\n0\n%%EOF\n"

    with (
        real_pypdf.PdfReader(BytesIO(pdf_bytes), strict=False) as expected,
        compat_pypdf.PdfReader(BytesIO(pdf_bytes), strict=False) as actual,
    ):
        expected_fields = expected.get_form_text_fields(full_qualified_name=True)
        assert expected_fields == {"customer.name": "Alice"}
        assert actual.get_form_text_fields() == expected_fields

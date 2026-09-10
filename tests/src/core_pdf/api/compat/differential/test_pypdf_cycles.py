from io import BytesIO

import pytest

real_pypdf = pytest.importorskip("pypdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("cycle_length", [1, 2])
def test_cyclic_destination_dictionaries_are_rejected(cycle_length: int) -> None:
    from core_pdf.api.compat import pypdf as compat_pypdf

    generic = real_pypdf.generic
    name = generic.NameObject
    writer = real_pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    destinations = [generic.DictionaryObject() for _ in range(cycle_length)]
    references = [writer._add_object(destination) for destination in destinations]
    for index, destination in enumerate(destinations):
        destination[name("/D")] = references[(index + 1) % cycle_length]
    outline = writer.add_outline_item("cyclic", 0).get_object()
    outline.pop(name("/A"))
    outline[name("/Dest")] = references[0]
    data = BytesIO()
    writer.write(data)

    with (
        real_pypdf.PdfReader(BytesIO(data.getvalue()), strict=True) as expected,
        compat_pypdf.PdfReader(BytesIO(data.getvalue()), strict=True) as actual,
    ):
        with pytest.raises(real_pypdf.errors.PdfReadError, match="Unexpected destination"):
            _ = expected.outline
        # A recursion failure is not an intentional rejection of malformed input.
        with pytest.raises(ValueError, match="destination cycle detected"):
            _ = actual.outline


@pytest.mark.parametrize(
    ("value_shape", "recover_xref"),
    [
        ("self-cycle", True),
        ("mutual-cycle", True),
        ("shared-array", False),
        ("shared-array", True),
    ],
    ids=[
        "recovered-self-cycle",
        "recovered-mutual-cycle",
        "shared-array",
        "recovered-shared-array",
    ],
)
def test_form_value_array_recovery_matches_reference_fields(
    value_shape: str, recover_xref: bool
) -> None:
    from core_pdf.api.compat import pypdf as compat_pypdf

    generic = real_pypdf.generic
    name = generic.NameObject
    text = generic.TextStringObject
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    values = generic.ArrayObject([text("before")])
    values_ref = writer._add_object(values)
    if value_shape == "self-cycle":
        values.extend([values_ref, text("after")])
        expected_text = "before\nafter"
    elif value_shape == "mutual-cycle":
        nested = generic.ArrayObject([text("nested"), values_ref, text("tail")])
        values.extend([writer._add_object(nested), text("after")])
        expected_text = "before\nnested\ntail\nafter"
    else:
        values_ref = writer._add_object(
            generic.ArrayObject([values_ref, text("middle"), values_ref])
        )
        expected_text = "before\nmiddle\nbefore"

    fields = generic.ArrayObject()
    for field_name, field_type, value in (
        ("choices", "/Ch", values_ref),
        ("customer", "/Tx", text("Alice")),
    ):
        field = generic.DictionaryObject(
            {
                name("/T"): text(field_name),
                name("/FT"): name(field_type),
                name("/V"): value,
                name("/Subtype"): name("/Widget"),
                name("/P"): page.indirect_reference,
                name("/Rect"): generic.ArrayObject(
                    [generic.NumberObject(item) for item in (10, 10, 100, 30)]
                ),
            }
        )
        fields.append(writer._add_object(field))
    page[name("/Annots")] = fields
    writer.root_object[name("/AcroForm")] = generic.DictionaryObject({name("/Fields"): fields})
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
        expected_fields = expected.get_fields()
        assert expected_fields is not None
        actual_fields = actual.get_fields()
        assert {key: field.type for key, field in actual_fields.items()} == {
            key: str(field["/FT"]).lstrip("/") for key, field in expected_fields.items()
        }
        assert (
            actual.get_form_text_fields()
            == expected.get_form_text_fields()
            == {"customer": "Alice"}
        )
        # pypdf keeps choice values as raw arrays. Core also projects their text;
        # omit cyclic edges but retain siblings and repeated noncyclic arrays.
        assert actual_fields["choices"].value_text == expected_text

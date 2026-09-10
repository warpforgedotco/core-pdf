from pathlib import Path

import pytest

from core_pdf.api.compat.unstructured import partition_pdf

real_pypdf = pytest.importorskip("pypdf")
real_partition = pytest.importorskip("unstructured.partition.pdf").partition_pdf
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize(
    ("malformation", "element_count", "algorithm", "encryption_key"),
    [
        ("content-dictionary", 1, None, None),
        ("unsupported-filter", 0, None, None),
        *(
            ("encryption", 2, algorithm, key)
            for algorithm in ("AES-256-R5", "AES-256")
            for key in ("/O", "/OE")
        ),
    ],
)
def test_fast_partition_recovers_text_from_malformed_pdfs(
    tmp_path: Path,
    malformation: str,
    element_count: int,
    algorithm: str | None,
    encryption_key: str | None,
) -> None:
    generic = real_pypdf.generic
    name = generic.NameObject
    writer = real_pypdf.PdfWriter()
    font = generic.DictionaryObject(
        {
            name("/Type"): name("/Font"),
            name("/Subtype"): name("/Type1"),
            name("/BaseFont"): name("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    for index in range(2):
        page = writer.add_blank_page(width=200, height=200)
        page[name("/Resources")] = generic.DictionaryObject(
            {name("/Font"): generic.DictionaryObject({name("/F1"): font_reference})}
        )
        stream = generic.DecodedStreamObject()
        stream.set_data(
            b"<< /Broken ["
            if index == 1 and malformation == "content-dictionary"
            else b"BT /F1 12 Tf 20 100 Td (Visible text) Tj ET"
        )
        if index == 1 and malformation == "unsupported-filter":
            stream[name("/Filter")] = name("/UnsupportedFilter")
        page[name("/Contents")] = writer._add_object(stream)
    if malformation == "encryption":
        assert algorithm is not None
        assert encryption_key is not None
        writer.encrypt("", owner_password="owner", algorithm=algorithm)
        writer._encrypt_entry[name(encryption_key)] = generic.ByteStringObject(b"short")
    path = tmp_path / "malformed.pdf"
    writer.write(path)

    expected = real_partition(filename=str(path), strategy="fast")
    actual = partition_pdf(path)
    assert len(expected) == element_count
    assert [(item.category, item.text) for item in actual] == [
        (item.category, item.text) for item in expected
    ]

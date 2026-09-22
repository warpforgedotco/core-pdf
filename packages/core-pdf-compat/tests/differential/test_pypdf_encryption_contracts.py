from io import BytesIO

import pytest

from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf_compat.pypdf import PdfReader

reference = pytest.importorskip("pypdf")
pytestmark = pytest.mark.compat_differential


@pytest.fixture(params=["RC4-40", "RC4-128", "AES-128", "AES-256-R5", "AES-256"])
def encrypted_pdf(request):
    writer = reference.PdfWriter()
    writer.add_blank_page(width=120, height=240)
    writer.add_metadata({"/Title": "Encrypted test"})
    writer.encrypt("user-pass", "owner-pass", algorithm=request.param)
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


@pytest.mark.parametrize("password", ["user-pass", "owner-pass"])
@pytest.mark.parametrize("source_kind", [bytes, BytesIO])
def test_encrypted_reader_unlocks_and_materializes_matching_pages(
    encrypted_pdf, password, source_kind
):
    with PdfReader(source_kind(encrypted_pdf)) as actual:
        expected = reference.PdfReader(BytesIO(encrypted_pdf))
        assert actual.is_encrypted == expected.is_encrypted is True
        assert not actual.decrypt("wrong")
        assert not expected.decrypt("wrong")
        assert actual.decrypt(password)
        assert expected.decrypt(password)
        assert actual.is_encrypted == expected.is_encrypted is True
        assert actual.num_pages == len(expected.pages) == 1
        assert tuple(actual.pages[0].mediabox) == tuple(expected.pages[0].mediabox)
        assert actual.metadata["/Title"] == expected.metadata["/Title"]
        assert actual.get_page(0) is actual.pages[0]
        assert actual.get_page_number(actual.pages[0]) == 0
        assert actual.pages[0].extract_text() == expected.pages[0].extract_text() == ""


@pytest.mark.parametrize("operation", [len, iter, lambda pages: pages[0]])
def test_locked_page_access_fails_without_materializing(encrypted_pdf, operation):
    with PdfReader(encrypted_pdf) as reader:
        with pytest.raises(PdfUnsupportedError, match="decrypted"):
            operation(reader.pages)
        assert reader.outline == reader.outlines == []
        assert reader.get_fields() == reader.get_form_text_fields() == {}
        assert reader.get_form_xfa() is None


@pytest.mark.parametrize("password", ["user-pass", "owner-pass"])
def test_constructor_password_matches_reference(encrypted_pdf, password):
    with PdfReader(encrypted_pdf, password=password) as actual:
        expected = reference.PdfReader(BytesIO(encrypted_pdf), password=password)
        assert actual.num_pages == len(expected.pages) == 1
        assert actual.is_encrypted == expected.is_encrypted


def test_wrong_explicit_password_is_rejected(encrypted_pdf):
    with pytest.raises(PdfUnsupportedError):
        PdfReader(encrypted_pdf, password="wrong")
    with pytest.raises(reference.errors.WrongPasswordError):
        reference.PdfReader(BytesIO(encrypted_pdf), password="wrong")


def test_unencrypted_info_metadata_matches_reference_without_native_wrapper_fields():
    writer = reference.PdfWriter()
    writer.add_blank_page(width=120, height=240)
    writer.add_metadata({"/Title": "Plain test", "/Author": "A", "/Custom": "value"})
    stream = BytesIO()
    writer.write(stream)
    with PdfReader(stream.getvalue()) as actual:
        expected = reference.PdfReader(BytesIO(stream.getvalue()))
        assert actual.metadata == dict(expected.metadata)
        assert not actual.is_encrypted
        assert actual.decrypt("unused")

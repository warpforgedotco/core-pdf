from io import BytesIO

import pytest

from core_pdf.api.compat.pdfminer import extract_text
from core_pdf.impl.exceptions import PdfUnsupportedError

real_pdfminer = pytest.importorskip("pdfminer.high_level")
real_pdfdocument = pytest.importorskip("pdfminer.pdfdocument")
real_pymupdf = pytest.importorskip("pymupdf")
real_pypdf = pytest.importorskip("pypdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("algorithm", ["AES-256-R5", "AES-256"])
@pytest.mark.parametrize("owner_field", ["O", "OE"])
@pytest.mark.parametrize("user", ["", "user"])
def test_user_authentication_with_malformed_owner(
    algorithm: str, owner_field: str, user: str
) -> None:
    with real_pymupdf.open() as document:
        document.new_page().insert_text((50, 50), "User-authenticated text")
        source = document.tobytes()
    writer = real_pypdf.PdfWriter(clone_from=BytesIO(source))
    writer.encrypt(user, owner_password="owner", algorithm=algorithm)
    writer._encrypt_entry[real_pypdf.generic.NameObject("/" + owner_field)] = (
        real_pypdf.generic.ByteStringObject(b"short")
    )
    buffer = BytesIO()
    writer.write(buffer)
    source = buffer.getvalue()

    expected = real_pdfminer.extract_text(BytesIO(source), password=user)
    assert "User-authenticated text" in expected
    assert extract_text(BytesIO(source), password=user) == expected
    with pytest.raises(real_pdfdocument.PDFPasswordIncorrect):
        real_pdfminer.extract_text(BytesIO(source), password="wrong")
    with pytest.raises(PdfUnsupportedError, match="Incorrect password"):
        extract_text(BytesIO(source), password="wrong")

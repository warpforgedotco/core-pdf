import hashlib
import json
from pathlib import Path

import pytest

from core_pdf import PdfDocument

FIXTURES = Path(__file__).resolve().parents[3] / "tests/fixtures/security_interop/pdf_mac"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


@pytest.mark.parametrize("entry", MANIFEST["fixtures"], ids=lambda entry: entry["algorithm"])
@pytest.mark.parametrize("password_type", ["user_password", "owner_password"])
def test_authenticated_mac_document_extracts_expected_text(entry, password_type):
    data = (FIXTURES / entry["filename"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    with PdfDocument(data, password=entry[password_type]) as document:
        assert len(document.pages) == 1
        text = document.extract().text
        assert text == MANIFEST["expected"]["text"] + "\f"

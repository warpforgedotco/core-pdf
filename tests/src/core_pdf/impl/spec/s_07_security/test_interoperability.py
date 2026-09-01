from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfUnsupportedError

FIXTURE_DIRECTORY = Path(__file__).parents[5] / "fixtures" / "security_interop"
EXPECTED_TEXT = "Security Interoperability"
EXPECTED_INFO = {
    "Author": "core-pdf",
    "Subject": "qpdf interoperability",
    "Title": "Core PDF Security Fixture",
}
EXPECTED_XMP_MARKER = "security-xmp-marker"
EXPECTED_XMP = {
    "tag": "xmpmeta",
    "attributes": {},
    "children": [{"tag": "marker", "text": EXPECTED_XMP_MARKER}],
}
FIXTURES = (
    ("rc4-40-r2.pdf", "user-40", "owner-40"),
    ("rc4-128-r3.pdf", "user-128", "owner-128"),
    ("aes-128-r4-cleartext-metadata.pdf", "user-aes128", "owner-aes128"),
    ("aes-256-r5.pdf", "user-r5", "owner-r5"),
    ("aes-256-r6.pdf", "user-r6", "owner-r6"),
    ("aes-256-r6-blank-user.pdf", "", "owner-blank"),
)
PASSWORD_CASES = tuple(
    (filename, password)
    for filename, user_password, owner_password in FIXTURES
    for password in (user_password, owner_password)
)


@pytest.mark.parametrize(("filename", "password"), PASSWORD_CASES)
def test_qpdf_encrypted_fixture_opens_with_user_or_owner_password(
    filename: str,
    password: str,
) -> None:
    with PdfDocument.open(FIXTURE_DIRECTORY / filename, password=password) as document:
        assert document.pages[0].extract().text == EXPECTED_TEXT
        assert document.metadata["info"] == EXPECTED_INFO
        assert document.metadata["xmp"] == EXPECTED_XMP


def test_qpdf_encrypted_fixture_opens_with_default_blank_password() -> None:
    fixture = FIXTURE_DIRECTORY / "aes-256-r6-blank-user.pdf"

    with PdfDocument.open(fixture) as document:
        assert document.pages[0].extract().text == EXPECTED_TEXT


@pytest.mark.parametrize(
    "filename",
    [filename for filename, user_password, _ in FIXTURES if user_password],
)
def test_qpdf_encrypted_fixture_rejects_incorrect_password(filename: str) -> None:
    with pytest.raises(PdfUnsupportedError, match="Incorrect password"):
        PdfDocument.open(FIXTURE_DIRECTORY / filename, password="incorrect")


@pytest.mark.parametrize(
    ("filename", "markers"),
    [
        ("rc4-40-r2.pdf", (b"/R 2", b"/V 1", b"/Length 40")),
        ("rc4-128-r3.pdf", (b"/R 3", b"/V 2", b"/Length 128")),
        (
            "aes-128-r4-cleartext-metadata.pdf",
            (
                b"/R 4",
                b"/V 4",
                b"/CFM /AESV2",
                b"/StmF /StdCF",
                b"/StrF /StdCF",
                b"/EncryptMetadata false",
            ),
        ),
        ("aes-256-r5.pdf", (b"/R 5", b"/V 5", b"/CFM /AESV3")),
        ("aes-256-r6.pdf", (b"/R 6", b"/V 5", b"/CFM /AESV3")),
        ("aes-256-r6-blank-user.pdf", (b"/R 6", b"/V 5", b"/CFM /AESV3")),
    ],
)
def test_qpdf_fixture_has_expected_security_dictionary(
    filename: str,
    markers: tuple[bytes, ...],
) -> None:
    fixture_bytes = (FIXTURE_DIRECTORY / filename).read_bytes()

    assert all(marker in fixture_bytes for marker in markers)
    assert EXPECTED_TEXT.encode() not in fixture_bytes


def test_qpdf_cleartext_metadata_fixture_is_the_only_one_with_visible_xmp() -> None:
    marker = EXPECTED_XMP_MARKER.encode()

    for filename, _, _ in FIXTURES:
        fixture_bytes = (FIXTURE_DIRECTORY / filename).read_bytes()
        if filename == "aes-128-r4-cleartext-metadata.pdf":
            assert marker in fixture_bytes
        else:
            assert marker not in fixture_bytes


def test_qpdf_fixture_manifest_covers_and_authenticates_committed_pdfs() -> None:
    manifest = json.loads((FIXTURE_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["source"]
    fixture_records = manifest["fixtures"]
    records_by_filename = {record["filename"]: record for record in fixture_records}
    expected_filenames = {filename for filename, _, _ in FIXTURES} | {source["filename"]}

    assert manifest["generator"]["name"] == "qpdf"
    assert manifest["expected"] == {
        "info_title": EXPECTED_INFO["Title"],
        "text": EXPECTED_TEXT,
        "xmp_marker": EXPECTED_XMP_MARKER,
    }
    assert {path.name for path in FIXTURE_DIRECTORY.glob("*.pdf")} == expected_filenames
    assert records_by_filename.keys() == expected_filenames - {source["filename"]}
    for filename, user_password, owner_password in FIXTURES:
        assert records_by_filename[filename]["user_password"] == user_password
        assert records_by_filename[filename]["owner_password"] == owner_password
    for record in [source, *fixture_records]:
        fixture_bytes = (FIXTURE_DIRECTORY / record["filename"]).read_bytes()
        assert hashlib.sha256(fixture_bytes).hexdigest() == record["sha256"]

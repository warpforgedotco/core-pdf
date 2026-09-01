from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import cast

import pytest

from core_pdf import PdfDecryptionError, PdfDocument
from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.primitives import PdfName, PdfString
from core_pdf.impl.spec.s_07_syntax.types import PdfDict

FIXTURE_DIRECTORY = Path(__file__).parents[5] / "fixtures" / "security_interop"
AES_GCM_DIRECTORY = FIXTURE_DIRECTORY / "aes_gcm"
AES_GCM_FIXTURE = AES_GCM_DIRECTORY / "aes-256-r7-gcm.pdf"
AES_GCM_USER_PASSWORD = "user-gcm"
AES_GCM_OWNER_PASSWORD = "owner-gcm"
PDF_MAC_DIRECTORY = FIXTURE_DIRECTORY / "pdf_mac"
PDF_MAC_FIXTURES = (
    (
        "aes-256-r6-cbc-mac.pdf",
        "user-mac-cbc",
        "owner-mac-cbc",
        "AES-256-CBC",
        "AESV3",
        5,
        6,
    ),
    (
        "aes-256-r7-gcm-mac.pdf",
        "user-mac-gcm",
        "owner-mac-gcm",
        "AES-256-GCM",
        "AESV4",
        6,
        7,
    ),
)
PDF_MAC_PASSWORD_CASES = tuple(
    (filename, password)
    for filename, user_password, owner_password, *_ in PDF_MAC_FIXTURES
    for password in (user_password, owner_password)
)
PDF_MAC_TAMPER_CHECKS = (
    "covered-document-byte",
    "mac-byte",
    "kdf-salt",
    "byte-range",
    "truncated-file",
    "trailing-file-bytes",
)
PDF_MAC_BYTE_RANGE_PATTERN = re.compile(rb"/ByteRange\s*\[\s*0\s+(\d+)\s+(\d+)\s+(\d+)\s*\]")
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


def internal_hex_entry_bounds(data: bytes, key: str) -> tuple[int, int]:
    marker = f"/{key} <".encode()
    start = data.index(marker) + len(marker)
    return start, data.index(b">", start)


def internal_corrupt_hex_entry(data: bytes, key: str, *, from_end: bool = False) -> bytes:
    start, end = internal_hex_entry_bounds(data, key)
    index = end - 1 if from_end else start
    corrupted = bytearray(data)
    corrupted[index] = ord("0") if corrupted[index] != ord("0") else ord("1")
    return bytes(corrupted)


def internal_truncate_hex_entry(data: bytes, key: str) -> bytes:
    _, end = internal_hex_entry_bounds(data, key)
    corrupted = bytearray(data)
    corrupted[end - 2 : end] = b"  "
    return bytes(corrupted)


def internal_corrupt_page_stream_padding(data: bytes) -> bytes:
    length_marker = b"6 0 obj\n<< /Length "
    length_start = data.index(length_marker) + len(length_marker)
    length_end = data.index(b" ", length_start)
    length = int(data[length_start:length_end])
    stream_start = data.index(b"stream\n", length_end) + len(b"stream\n")
    assert length >= 32
    assert length % 16 == 0

    corrupted = bytearray(data)
    # Alter the previous CBC block so the final plaintext padding byte changes
    # deterministically while the ciphertext remains block-aligned.
    corrupted[stream_start + length - 17] ^= 1
    return bytes(corrupted)


def internal_corrupt_aes_gcm_page_stream(data: bytes, relative_index: int) -> bytes:
    object_start = data.index(b"6 0 obj\n")
    stream_marker = b">>\nstream\n"
    stream_start = data.index(stream_marker, object_start) + len(stream_marker)
    length_marker = b"/Length "
    length_start = data.index(length_marker, object_start, stream_start) + len(length_marker)
    length_end = data.index(b"\n", length_start, stream_start)
    length = int(data[length_start:length_end])
    assert length > 28
    assert data[stream_start + length :].startswith(b"\nendstream")

    corrupted = bytearray(data)
    absolute_index = stream_start + (
        length + relative_index if relative_index < 0 else relative_index
    )
    corrupted[absolute_index] ^= 1
    return bytes(corrupted)


def internal_tamper_pdf_mac(data: bytes, name: str) -> bytes:
    match name:
        case "covered-document-byte":
            second_comment = data.index(b"\n%", len(b"%PDF-")) + 2
            corrupted = bytearray(data)
            corrupted[second_comment] ^= 1
            return bytes(corrupted)
        case "mac-byte":
            return internal_corrupt_hex_entry(data, "MAC", from_end=True)
        case "kdf-salt":
            return internal_corrupt_hex_entry(data, "KDFSalt")
        case "byte-range":
            match = PDF_MAC_BYTE_RANGE_PATTERN.search(data)
            assert match is not None
            old_length = match.group(3)
            new_length = str(int(old_length) - 1).encode("ascii")
            assert len(new_length) == len(old_length)
            return data[: match.start(3)] + new_length + data[match.end(3) :]
        case "truncated-file":
            return data[:-1]
        case "trailing-file-bytes":
            return data + b"% PDF MAC coverage tamper\n"
        case _:
            raise AssertionError(f"unknown PDF MAC tamper case: {name}")


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


@pytest.mark.parametrize("password", [AES_GCM_USER_PASSWORD, AES_GCM_OWNER_PASSWORD])
def test_pyhanko_aes_gcm_fixture_opens_with_user_or_owner_password(password: str) -> None:
    with PdfDocument.open(AES_GCM_FIXTURE, password=password) as document:
        assert document.pages[0].extract().text == EXPECTED_TEXT
        assert document.metadata["info"] == EXPECTED_INFO
        assert document.metadata["xmp"] == EXPECTED_XMP

        extensions = cast(PdfDict, document.catalog()["Extensions"])
        declarations = cast(list[object], extensions["ISO_"])
        extension = cast(PdfDict, declarations[0])
        assert extension["BaseVersion"] == PdfName.of("2.0")
        assert extension["ExtensionLevel"] == 32003
        revision = extension["ExtensionRevision"]
        assert isinstance(revision, PdfString)
        assert revision.data == b":2023"


def test_pyhanko_aes_gcm_fixture_rejects_incorrect_password() -> None:
    with pytest.raises(PdfUnsupportedError, match="Incorrect password"):
        PdfDocument.open(AES_GCM_FIXTURE, password="incorrect")


@pytest.mark.parametrize(("filename", "password"), PDF_MAC_PASSWORD_CASES)
def test_pyhanko_pdf_mac_fixture_opens_after_integrity_validation(
    filename: str,
    password: str,
) -> None:
    with PdfDocument.open(PDF_MAC_DIRECTORY / filename, password=password) as document:
        assert document.pages[0].extract().text == EXPECTED_TEXT
        assert document.metadata["info"] == EXPECTED_INFO
        assert document.metadata["xmp"] == EXPECTED_XMP

        extensions = cast(PdfDict, document.catalog()["Extensions"])
        declarations = cast(list[object], extensions["ISO_"])
        declared_revisions = {
            (
                cast(PdfDict, declaration)["ExtensionLevel"],
                cast(PdfString, cast(PdfDict, declaration)["ExtensionRevision"]).data,
            )
            for declaration in declarations
        }
        expected_revisions = {(32004, b":2024")}
        if "r7-gcm" in filename:
            expected_revisions.add((32003, b":2023"))
        assert declared_revisions == expected_revisions


@pytest.mark.parametrize("filename", [fixture[0] for fixture in PDF_MAC_FIXTURES])
def test_pyhanko_pdf_mac_fixture_rejects_incorrect_password(filename: str) -> None:
    with pytest.raises(PdfUnsupportedError, match="Incorrect password"):
        PdfDocument.open(PDF_MAC_DIRECTORY / filename, password="incorrect")


@pytest.mark.parametrize(
    ("filename", "password"),
    [(fixture[0], fixture[1]) for fixture in PDF_MAC_FIXTURES],
)
@pytest.mark.parametrize("tamper_name", PDF_MAC_TAMPER_CHECKS)
def test_pyhanko_pdf_mac_tamper_cases_remain_fail_closed(
    filename: str,
    password: str,
    tamper_name: str,
) -> None:
    pristine = (PDF_MAC_DIRECTORY / filename).read_bytes()
    tampered = internal_tamper_pdf_mac(pristine, tamper_name)
    assert tampered != pristine

    with pytest.raises(PdfDecryptionError, match="Invalid PDF MAC"):
        PdfDocument.open(tampered, password=password)


@pytest.mark.parametrize(
    ("filename", "password"),
    [(fixture[0], fixture[1]) for fixture in PDF_MAC_FIXTURES],
)
def test_pyhanko_pdf_mac_permission_signal_is_bound_by_encrypted_permissions(
    filename: str,
    password: str,
) -> None:
    pristine = (PDF_MAC_DIRECTORY / filename).read_bytes()
    # Keep the serialized width fixed so this mutation isolates permission bit
    # 13. ISO/TS 32004:2024, Table 3 relies on ISO 32000-2:2020's encrypted
    # Perms entry to make changing the signal from zero to one tamper-evident.
    tampered = pristine.replace(b"/P -4100", b"/P -0004", 1)
    assert tampered != pristine

    with pytest.raises(PdfDecryptionError, match="Invalid encryption permissions"):
        PdfDocument.open(tampered, password=password)


@pytest.mark.parametrize(
    ("entry", "replacement", "exception", "message"),
    [
        (b"/AuthCode", b"/BadCode ", PdfDecryptionError, "Invalid PDF MAC"),
        (
            b"/KDFSalt",
            b"/BadSalt",
            PdfUnsupportedError,
            "Invalid encryption dictionary",
        ),
    ],
)
def test_pyhanko_pdf_mac_required_entries_cannot_be_removed(
    entry: bytes,
    replacement: bytes,
    exception: type[Exception],
    message: str,
) -> None:
    pristine = (PDF_MAC_DIRECTORY / "aes-256-r6-cbc-mac.pdf").read_bytes()
    assert len(entry) == len(replacement)
    tampered = pristine.replace(entry, replacement, 1)
    assert tampered != pristine

    with pytest.raises(exception, match=message):
        PdfDocument.open(tampered, password="user-mac-cbc")


@pytest.mark.parametrize(
    "relative_index",
    [0, 12, -1],
    ids=["initialization-vector", "ciphertext", "authentication-tag"],
)
def test_pyhanko_aes_gcm_fixture_authenticates_page_stream(relative_index: int) -> None:
    fixture_bytes = AES_GCM_FIXTURE.read_bytes()
    corrupted = internal_corrupt_aes_gcm_page_stream(fixture_bytes, relative_index)

    with PdfDocument.open(corrupted, password=AES_GCM_USER_PASSWORD) as document:
        with pytest.raises(PdfDecryptionError, match="Invalid encrypted object ciphertext"):
            document.pages[0].extract()


@pytest.mark.parametrize(
    "filename",
    [filename for filename, user_password, _ in FIXTURES if user_password],
)
def test_qpdf_encrypted_fixture_rejects_incorrect_password(filename: str) -> None:
    with pytest.raises(PdfUnsupportedError, match="Incorrect password"):
        PdfDocument.open(FIXTURE_DIRECTORY / filename, password="incorrect")


@pytest.mark.parametrize(
    ("filename", "password"),
    [
        ("aes-256-r5.pdf", "user-r5"),
        ("aes-256-r6.pdf", "user-r6"),
    ],
)
def test_qpdf_modern_fixture_rejects_corrupted_permissions(
    filename: str,
    password: str,
) -> None:
    fixture_bytes = (FIXTURE_DIRECTORY / filename).read_bytes()
    corrupted = internal_corrupt_hex_entry(fixture_bytes, "Perms")

    with pytest.raises(PdfDecryptionError, match="Invalid encryption permissions"):
        PdfDocument.open(corrupted, password=password)


def test_qpdf_modern_fixture_rejects_permissions_mismatched_with_dictionary() -> None:
    fixture_bytes = (FIXTURE_DIRECTORY / "aes-256-r6.pdf").read_bytes()
    corrupted = fixture_bytes.replace(b"/P -4 ", b"/P -8 ", 1)
    assert corrupted != fixture_bytes

    with pytest.raises(PdfDecryptionError, match="Invalid encryption permissions"):
        PdfDocument.open(corrupted, password="user-r6")


@pytest.mark.parametrize("entry_name", ["O", "U", "OE", "UE", "Perms"])
def test_qpdf_modern_fixture_rejects_truncated_encryption_entry(entry_name: str) -> None:
    fixture_bytes = (FIXTURE_DIRECTORY / "aes-256-r6.pdf").read_bytes()
    corrupted = internal_truncate_hex_entry(fixture_bytes, entry_name)

    with pytest.raises(PdfUnsupportedError, match="Invalid encryption dictionary"):
        PdfDocument.open(corrupted, password="user-r6")


def test_qpdf_modern_fixture_rejects_invalid_stream_padding() -> None:
    fixture_bytes = (FIXTURE_DIRECTORY / "aes-256-r6.pdf").read_bytes()
    corrupted = internal_corrupt_page_stream_padding(fixture_bytes)

    with PdfDocument.open(corrupted, password="user-r6") as document:
        with pytest.raises(PdfDecryptionError, match="Invalid encrypted object ciphertext"):
            document.pages[0].extract()


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        (b"/V 5", b"/V 6", "Invalid encryption dictionary"),
        (b"/R 6", b"/R 4", "Invalid encryption dictionary"),
        (b"/P -4", b"/P -1", "Invalid encryption dictionary"),
        (b"/P -4 ", b"/P -68", "Invalid encryption dictionary"),
        (b"/StmF /StdCF", b"/StmF /BadCF", "Invalid encryption dictionary"),
        (b"/StrF /StdCF", b"/StrF /BadCF", "Invalid encryption dictionary"),
        (b"/AuthEvent /DocOpen", b"/AuthEvent /EFOpen ", "Invalid encryption dictionary"),
        (b"/CFM /AESV3", b"/CFM /AESV2", "Invalid encryption dictionary"),
        (b"/Length 32", b"/Length 16", "Invalid encryption dictionary"),
        (b"/Length 256", b"/Length 128", "Invalid encryption dictionary"),
    ],
)
def test_qpdf_modern_fixture_rejects_inconsistent_security_configuration(
    original: bytes,
    replacement: bytes,
    message: str,
) -> None:
    fixture_bytes = (FIXTURE_DIRECTORY / "aes-256-r6.pdf").read_bytes()
    assert len(original) == len(replacement)
    assert original in fixture_bytes
    corrupted = fixture_bytes.replace(original, replacement, 1)

    with pytest.raises(PdfUnsupportedError, match=message):
        PdfDocument.open(corrupted, password="user-r6")


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


def test_pyhanko_aes_gcm_fixture_and_manifest_are_pinned() -> None:
    manifest = json.loads((AES_GCM_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    fixture = manifest["fixture"]
    source = manifest["source"]

    assert manifest["specification"] == "ISO/TS 32003:2023"
    assert manifest["generator"] == {
        "commit": "00362ec2772b2d39e5d9ba2c0287efb4077421d8",
        "license": "MIT",
        "name": "pyHanko",
        "version": "0.37.0",
        "website": "https://github.com/MatthiasValvekens/pyHanko",
    }
    assert fixture == {
        "algorithm": "AES-256-GCM",
        "crypt_filter_method": "AESV4",
        "filename": AES_GCM_FIXTURE.name,
        "owner_password": AES_GCM_OWNER_PASSWORD,
        "pdf_mac": False,
        "revision": 7,
        "sha256": hashlib.sha256(AES_GCM_FIXTURE.read_bytes()).hexdigest(),
        "user_password": AES_GCM_USER_PASSWORD,
        "version": 6,
    }
    assert source["filename"] == "../source.pdf"
    source_bytes = (AES_GCM_DIRECTORY / source["filename"]).read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == source["sha256"]
    assert {path.name for path in AES_GCM_DIRECTORY.glob("*.pdf")} == {AES_GCM_FIXTURE.name}


def test_pyhanko_aes_gcm_fixture_has_expected_security_dictionary() -> None:
    fixture_bytes = AES_GCM_FIXTURE.read_bytes()
    markers = (
        b"/V 6",
        b"/R 7",
        b"/CFM /AESV4",
        b"/Length 256",
        b"/Length 32",
        b"/StmF /StdCF",
        b"/StrF /StdCF",
        b"/ExtensionLevel 32003",
    )

    assert all(marker in fixture_bytes for marker in markers)
    assert b"/KDFSalt" not in fixture_bytes
    assert b"/AuthCode" not in fixture_bytes
    assert EXPECTED_TEXT.encode() not in fixture_bytes


def test_pyhanko_pdf_mac_fixtures_and_manifest_are_pinned() -> None:
    manifest = json.loads((PDF_MAC_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    records = {record["filename"]: record for record in manifest["fixtures"]}
    source = manifest["source"]

    assert manifest["specification"] == "ISO/TS 32004:2024"
    assert manifest["expected"] == {"text": EXPECTED_TEXT}
    assert manifest["generator"] == {
        "commit": "00362ec2772b2d39e5d9ba2c0287efb4077421d8",
        "license": "MIT",
        "name": "pyHanko",
        "version": "0.37.0",
        "website": "https://github.com/MatthiasValvekens/pyHanko",
    }
    assert manifest["regenerate"] == [
        "uv",
        "run",
        "--with",
        "pyhanko==0.37.0",
        "python",
        "scripts/generate_pdf_mac_interop_fixtures.py",
    ]
    assert source["filename"] == "../source.pdf"
    source_bytes = (PDF_MAC_DIRECTORY / source["filename"]).read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == source["sha256"]

    expected_filenames = {fixture[0] for fixture in PDF_MAC_FIXTURES}
    assert records.keys() == expected_filenames
    assert {path.name for path in PDF_MAC_DIRECTORY.glob("*.pdf")} == expected_filenames
    for (
        filename,
        user_password,
        owner_password,
        algorithm,
        crypt_filter_method,
        version,
        revision,
    ) in PDF_MAC_FIXTURES:
        fixture_bytes = (PDF_MAC_DIRECTORY / filename).read_bytes()
        assert records[filename] == {
            "algorithm": algorithm,
            "crypt_filter_method": crypt_filter_method,
            "filename": filename,
            "kdf_salt_bytes": 32,
            "mac_algorithm": "HMAC-SHA-256",
            "mac_digest_algorithm": "SHA-256",
            "mac_key_wrap_algorithm": "AES-256-KW",
            "mac_location": "Standalone",
            "owner_password": owner_password,
            "pdf_mac": True,
            "revision": revision,
            "sha256": hashlib.sha256(fixture_bytes).hexdigest(),
            "tamper_checks": list(PDF_MAC_TAMPER_CHECKS),
            "user_password": user_password,
            "version": version,
        }


@pytest.mark.parametrize(
    ("filename", "crypt_filter_method", "version", "revision"),
    [(fixture[0], fixture[4], fixture[5], fixture[6]) for fixture in PDF_MAC_FIXTURES],
)
def test_pyhanko_pdf_mac_fixture_has_expected_standalone_structure(
    filename: str,
    crypt_filter_method: str,
    version: int,
    revision: int,
) -> None:
    fixture_bytes = (PDF_MAC_DIRECTORY / filename).read_bytes()

    # ISO/TS 32004:2024, Tables 1-3, 5-6 and 6.5.1 require the extension
    # declaration, a 32-byte KDFSalt, bit 13 clear, a direct AuthCode
    # dictionary, a standalone DER token, and exact whole-file coverage.
    assert re.search(rb"trailer\s*<<\s*/AuthCode\s*<<", fixture_bytes)
    assert b"/MACLocation /Standalone" in fixture_bytes
    assert b"/SigObjRef" not in fixture_bytes
    assert b"/P -4100" in fixture_bytes
    permissions = (-4100) & 0xFFFFFFFF
    assert permissions & (1 << 12) == 0
    assert f"/V {version}".encode() in fixture_bytes
    assert f"/R {revision}".encode() in fixture_bytes
    assert f"/CFM /{crypt_filter_method}".encode() in fixture_bytes
    assert b"/ExtensionLevel 32004" in fixture_bytes
    if crypt_filter_method == "AESV4":
        assert b"/ExtensionLevel 32003" in fixture_bytes
    else:
        assert b"/ExtensionLevel 32003" not in fixture_bytes

    kdf_start, kdf_end = internal_hex_entry_bounds(fixture_bytes, "KDFSalt")
    assert len(bytes.fromhex(fixture_bytes[kdf_start:kdf_end].decode("ascii"))) == 32

    byte_range = PDF_MAC_BYTE_RANGE_PATTERN.search(fixture_bytes)
    assert byte_range is not None
    first_length, second_start, second_length = map(int, byte_range.groups())
    mac_start, mac_end = internal_hex_entry_bounds(fixture_bytes, "MAC")
    assert first_length == mac_start - 1
    assert second_start == mac_end + 1
    assert second_start + second_length == len(fixture_bytes)
    assert (
        fixture_bytes[first_length:second_start] == b"<" + fixture_bytes[mac_start:mac_end] + b">"
    )

    encoded_mac = fixture_bytes[mac_start:mac_end]
    assert encoded_mac == encoded_mac.upper()
    assert not any(chr(byte).isspace() for byte in encoded_mac)
    assert bytes.fromhex(encoded_mac.decode("ascii")).startswith(b"\x30")
    assert EXPECTED_TEXT.encode() not in fixture_bytes

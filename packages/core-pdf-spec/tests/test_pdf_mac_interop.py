"""ISO/TS 32004:2024 integrity checks using independently generated fixtures."""

import hashlib
import json
import re
from pathlib import Path

import pytest
from asn1crypto import algos, cms, core

from core_pdf_spec.exceptions import PdfDecryptionError, PdfUnsupportedError
from core_pdf_spec.s_07_security import pdf_mac as mac
from core_pdf_spec.s_07_security.document import initialize_document_security
from core_pdf_spec.s_07_security.standard import create_standard_security_handler
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.xref import XRefScanner
from core_pdf_spec.types import PdfName, PdfString

FIXTURES = Path(__file__).resolve().parents[3] / "tests/fixtures/security_interop/pdf_mac"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


@pytest.fixture(params=MANIFEST["fixtures"], ids=lambda item: item["algorithm"])
def fixture(request):
    entry = request.param
    raw = (FIXTURES / entry["filename"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    start = re.search(rb"startxref\s+(\d+)", raw)
    assert start is not None
    section = XRefScanner.parse_section_at(raw, int(start[1]))
    return raw, section, entry


@pytest.mark.parametrize("password_type", ["user_password", "owner_password"])
def test_real_mac_authenticates_before_installing_decipher(fixture, password_type):
    raw, section, entry = fixture
    resolver = ObjectResolver(raw, section.entries)
    try:
        assert resolver.decipher is None
        decipher = initialize_document_security(
            raw, section.trailer, resolver, entry[password_type]
        )
        assert decipher is not None
        assert resolver.decipher == decipher
    finally:
        resolver.close()


@pytest.mark.parametrize("change", MANIFEST["fixtures"][0]["tamper_checks"])
def test_tampered_file_never_installs_decipher(fixture, change):
    raw, section, entry = fixture
    auth = section.trailer["AuthCode"]
    if change == "trailing-file-bytes":
        damaged = raw + b"\n"
    elif change == "truncated-file":
        damaged = raw[:-1]
    else:
        damaged = bytearray(raw)
        if change == "covered-document-byte":
            # Change only the binary header comment; parsing still succeeds.
            offset = raw.index(b"%\xc2") + 1
        elif change == "mac-byte":
            offset = auth["ByteRange"][2] - 2
        elif change == "kdf-salt":
            offset = raw.index(b"/KDFSalt <") + len(b"/KDFSalt <")
        else:
            match = re.search(rb"/ByteRange\s*\[\s*(\d+)", raw)
            assert match is not None
            offset = match.start(1)
        damaged[offset] = ord("1") if damaged[offset] != ord("1") else ord("2")
        damaged = bytes(damaged)
    # Parse the modified trailer too: AuthCode must reflect the actual bytes.
    updated = XRefScanner.parse_section_at(damaged, section.offset)
    resolver = ObjectResolver(damaged, updated.entries)
    try:
        with pytest.raises(PdfDecryptionError, match="Invalid PDF MAC"):
            initialize_document_security(damaged, updated.trailer, resolver, entry["user_password"])
        assert resolver.decipher is None
    finally:
        resolver.close()


@pytest.mark.parametrize("location", ["AttachedToSig", "Unknown"])
def test_unsupported_mac_locations_remain_distinct(location):
    with pytest.raises(PdfUnsupportedError):
        mac.internal_validate_standalone_pdf_mac(
            b"", {"MACLocation": PdfName.of(location)}, b"", b""
        )


@pytest.mark.parametrize(
    "byte_range",
    [None, [], [0, 1, 4], [True, 1, 4, 1], [0, -1, 4, 1], [1, 1, 4, 1], [0, 1, 9, 0], [0, 1, 4, 0]],
)
def test_byte_range_must_cover_exactly_the_file(byte_range):
    with pytest.raises(ValueError, match="ByteRange"):
        mac.internal_extract_standalone_token(
            b"a<00>b", {"ByteRange": byte_range, "MAC": PdfString(b"\0", is_literal=False)}
        )


@pytest.mark.parametrize(
    "algorithm", ["sha256", "sha384", "sha512", "sha3_256", "sha3_384", "sha3_512"]
)
def test_standard_digest_algorithms_have_positive_controls(algorithm):
    # ISO/TS 32004:2024 Table 8; SHA-3 requires absent parameters.
    identifier = algos.DigestAlgorithm({"algorithm": algorithm})
    digest = mac.internal_digest(b"contract", mac.internal_digest_algorithm(identifier))
    assert digest == hashlib.new(algorithm, b"contract").digest()


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "recipients",
        "mac",
        "attrs",
        "digest",
        "content",
        "unauthenticated",
        "kdf",
        "wrapped-key",
    ],
)
def test_malformed_authenticated_data_is_rejected(fixture, change):
    raw, section, entry = fixture
    resolver = ObjectResolver(raw, section.entries)
    try:
        encryption = resolver.resolve_dict(section.trailer["Encrypt"])
        assert encryption is not None
        handler = create_standard_security_handler(
            section.trailer["ID"],
            encryption,
            entry["user_password"],
        )
        assert handler.config.kdf_salt is not None
        byte_range, token = mac.internal_extract_standalone_token(raw, section.trailer["AuthCode"])
        auth = cms.ContentInfo.load(token)["content"]
        if change == "version":
            auth["version"] = "v1"
        elif change == "recipients":
            auth["recipient_infos"] = []
        elif change == "mac":
            auth["mac"] = b"short"
        elif change == "attrs":
            auth["auth_attrs"] = None
        elif change == "digest":
            auth["digest_algorithm"] = {"algorithm": "sha1"}
        elif change == "content":
            auth["encap_content_info"]["content_type"] = "data"
        elif change == "unauthenticated":
            auth["unauth_attrs"] = []
        elif change == "kdf":
            auth["recipient_infos"][0].chosen["key_derivation_algorithm"] = None
        else:
            auth["recipient_infos"][0].chosen["encrypted_key"] = b"short"
        error = PdfUnsupportedError if change == "digest" else ValueError
        with pytest.raises(error):
            mac.internal_validate_authenticated_data(
                raw, byte_range, auth, handler.file_key, handler.config.kdf_salt
            )
    finally:
        resolver.close()


def test_der_rejects_trailing_data_and_noncanonical_encoding():
    for data in (b"\x02\x01\x01\x00", b"\x02\x81\x01\x01"):
        with pytest.raises(ValueError):
            mac.internal_parse_der(data, core.Integer)
    assert mac.internal_parse_der(b"\x02\x01\x01", core.Integer).native == 1

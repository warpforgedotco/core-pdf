# SPDX-License-Identifier: AGPL-3.0-only
"""ISO/TS 32004:2024 token primitives: digests, DER strictness, and algorithm gates."""

import hashlib

import pytest
from asn1crypto import algos, cms, core

from core_pdf_crypto.errors import UnsupportedAlgorithmError
from core_pdf_crypto.pdf_mac import (
    PdfMacIntegrityInfo,
    digest,
    digest_algorithm,
    digest_byte_range,
    parse_der,
    validate_pdf_mac_token,
)


@pytest.mark.parametrize(
    "algorithm", ["sha256", "sha384", "sha512", "sha3_256", "sha3_384", "sha3_512"]
)
def test_standard_digest_algorithms_have_positive_controls(algorithm: str) -> None:
    # ISO/TS 32004:2024 Table 8; SHA-3 requires absent parameters.
    identifier = algos.DigestAlgorithm({"algorithm": algorithm})
    assert (
        digest(b"contract", digest_algorithm(identifier))
        == hashlib.new(algorithm, b"contract").digest()
    )


def test_digest_algorithm_outside_table_8_is_unsupported_not_invalid() -> None:
    identifier = algos.DigestAlgorithm({"algorithm": "sha1"})
    with pytest.raises(UnsupportedAlgorithmError, match="Unsupported PDF MAC digest algorithm"):
        digest_algorithm(identifier)


def test_sha3_identifiers_require_absent_parameters() -> None:
    identifier = algos.DigestAlgorithm({"algorithm": "sha3_256", "parameters": core.Null()})
    with pytest.raises(ValueError, match="parameters"):
        digest_algorithm(identifier)


def test_der_rejects_trailing_data_and_noncanonical_encoding() -> None:
    for data in (b"\x02\x01\x01\x00", b"\x02\x81\x01\x01"):
        with pytest.raises(ValueError):
            parse_der(data, core.Integer)
    assert parse_der(b"\x02\x01\x01", core.Integer).native == 1


def test_byte_range_digest_covers_both_ranges_in_order() -> None:
    raw = b"header<00>trailer"
    byte_range = (0, 6, 10, 7)
    expected = hashlib.sha256(b"header" + b"trailer").digest()
    assert (
        digest_byte_range(
            raw, byte_range, digest_algorithm(algos.DigestAlgorithm({"algorithm": "sha256"}))
        )
        == expected
    )
    assert (
        digest_byte_range(
            memoryview(raw),
            byte_range,
            digest_algorithm(algos.DigestAlgorithm({"algorithm": "sha256"})),
        )
        == expected
    )


def test_integrity_info_round_trips_through_der() -> None:
    info = PdfMacIntegrityInfo({"version": 0, "data_digest": b"\x01" * 32})
    assert parse_der(info.dump(force=True), PdfMacIntegrityInfo)["version"].native == 0


def test_token_must_be_authenticated_data() -> None:
    token = cms.ContentInfo({"content_type": "data", "content": b""}).dump(force=True)
    with pytest.raises(ValueError, match="not CMS AuthenticatedData"):
        validate_pdf_mac_token(b"", (0, 0, 0, 0), token, b"\0" * 32, b"\0" * 32)

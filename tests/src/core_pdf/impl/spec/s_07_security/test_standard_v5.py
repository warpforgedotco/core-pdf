from __future__ import annotations

from dataclasses import dataclass

import pytest

from core_pdf.impl.primitives import PdfName
from core_pdf.impl.spec.s_07_security.errors import PDFEncryptionError, PDFPasswordIncorrect
from core_pdf.impl.spec.s_07_security.standard_v5 import PdfStandardSecurityHandlerV5
from core_pdf.impl.spec.s_07_syntax.types import PdfDict


@dataclass(frozen=True)
class SecurityVector:
    revision: int
    user_password: str
    owner_password: str
    o_value: bytes
    u_value: bytes
    oe_value: bytes
    ue_value: bytes


FILE_KEY = bytes.fromhex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
ENCRYPTED_PAYLOAD = bytes.fromhex(
    "00112233445566778899aabbccddeeff"
    "0377a0375b250ad44920b35cb142fb9a8"
    "3f77b5258eee937c1f3c55c4534ce53"
)

# These fixed values were calculated independently from PDF Algorithms 2.A/2.B
# using OpenSSL AES-CBC. The R6 passwords deliberately exercise SASLprep mapping.
SECURITY_VECTORS = (
    SecurityVector(
        revision=5,
        user_password="user-password",
        owner_password="owner-password",
        u_value=bytes.fromhex(
            "15fdefb63c597efbac9aeb86e9b90e16"
            "6f88e7464392138cd32339a40e91211f"
            "102132435465768798a9bacbdcedfe0f"
        ),
        ue_value=bytes.fromhex("aaa3f45c44b236ee325c507f87e99cefa7d4eccfff33c5014916f6fbfa80eb84"),
        o_value=bytes.fromhex(
            "cba9ff0c663f73233941f4cdeacf208f"
            "60376885e9b2888e95cba80d63839671"
            "ffeeddccbbaa99887766554433221100"
        ),
        oe_value=bytes.fromhex("1cf9b07830c7340d0d4b3c1efdb863bb206955f0e2b13d7e9bba13ed827f84db"),
    ),
    SecurityVector(
        revision=6,
        user_password="user\u00a0name",
        owner_password="own\u00ader",
        u_value=bytes.fromhex(
            "25df86f2f67ed36307fcc0b79499b790"
            "6735e38dd74991a85050fcccca7389bc"
            "102132435465768798a9bacbdcedfe0f"
        ),
        ue_value=bytes.fromhex("bb5ba7ff7e02f44a1309a20e128a7d5e2ea12260fc47f200854e1ee48eec242b"),
        o_value=bytes.fromhex(
            "7876c763c3031db2b9e6e4cf8d990044"
            "eb533515913f5571f2b891c12a1a6ac0"
            "ffeeddccbbaa99887766554433221100"
        ),
        oe_value=bytes.fromhex("228c50ab0187f54804ca03e39e2994c797e53b52469635f7e02ebe2893ea81f7"),
    ),
)


def encryption_dictionary(vector: SecurityVector) -> PdfDict:
    return {
        "V": 5,
        "R": vector.revision,
        "P": -4,
        "O": vector.o_value,
        "U": vector.u_value,
        "OE": vector.oe_value,
        "UE": vector.ue_value,
        "CF": {"StdCF": {"CFM": PdfName.of("AESV3")}},
        "StmF": PdfName.of("StdCF"),
        "StrF": PdfName.of("StdCF"),
    }


@pytest.mark.parametrize("vector", SECURITY_VECTORS, ids=lambda vector: f"R{vector.revision}")
def test_authenticates_user_and_owner_passwords_from_known_vectors(
    vector: SecurityVector,
) -> None:
    handler = PdfStandardSecurityHandlerV5(
        [],
        encryption_dictionary(vector),
        vector.user_password,
    )

    assert handler.key == FILE_KEY
    assert handler.authenticate(vector.owner_password) == FILE_KEY


@pytest.mark.parametrize("vector", SECURITY_VECTORS, ids=lambda vector: f"R{vector.revision}")
def test_rejects_incorrect_passwords(vector: SecurityVector) -> None:
    handler = PdfStandardSecurityHandlerV5(
        [],
        encryption_dictionary(vector),
        vector.user_password,
    )

    assert handler.authenticate("incorrect-password") is None
    with pytest.raises(PDFPasswordIncorrect, match="Incorrect password"):
        PdfStandardSecurityHandlerV5(
            [],
            encryption_dictionary(vector),
            "incorrect-password",
        )


@pytest.mark.parametrize("vector", SECURITY_VECTORS, ids=lambda vector: f"R{vector.revision}")
def test_decrypts_aesv3_payload_with_authenticated_file_key(vector: SecurityVector) -> None:
    handler = PdfStandardSecurityHandlerV5(
        [],
        encryption_dictionary(vector),
        vector.owner_password,
    )

    assert handler.decrypt(42, 7, ENCRYPTED_PAYLOAD) == b"authenticated payload"


def test_rejects_non_aesv3_crypt_filter() -> None:
    params = encryption_dictionary(SECURITY_VECTORS[0])
    params["CF"] = {"StdCF": {"CFM": PdfName.of("AESV2")}}

    with pytest.raises(PDFEncryptionError, match="Unknown crypt filter method"):
        PdfStandardSecurityHandlerV5([], params, SECURITY_VECTORS[0].user_password)


def test_r6_normalization_preserves_empty_password_and_truncates_utf8_bytes() -> None:
    handler = object.__new__(PdfStandardSecurityHandlerV5)
    handler.r = 6

    assert handler.normalize_password("") == b""
    assert handler.normalize_password("\u00e9" * 64) == ("\u00e9" * 64).encode()[:127]

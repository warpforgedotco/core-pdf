from __future__ import annotations

import pytest

from core_pdf.impl.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf.impl.primitives import PdfName
from core_pdf.impl.spec.s_07_security import create_standard_decipher
from core_pdf.impl.spec.s_07_security.ciphers import (
    internal_aes_cbc_decrypt,
    internal_aes_cbc_encrypt,
    internal_rc4_crypt,
)
from core_pdf.impl.spec.s_07_security.standard import (
    internal_CryptMethod,
    internal_resolve_crypt_method,
    internal_saslprep,
    internal_stream_crypt_filter_name,
)
from core_pdf.impl.spec.s_07_syntax.types import PdfDict


def test_aes_cbc_known_vector() -> None:
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    initialization_vector = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    plaintext = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")
    expected = bytes.fromhex("7649abac8119b246cee98e9b12e9197d")

    encrypted = internal_aes_cbc_encrypt(
        key,
        initialization_vector,
        plaintext,
        use_padding=False,
    )

    assert encrypted == expected
    assert (
        internal_aes_cbc_decrypt(
            key,
            initialization_vector,
            encrypted,
            use_padding=False,
        )
        == plaintext
    )


def test_aes_cbc_pkcs7_padding_round_trip() -> None:
    key = bytes(32)
    initialization_vector = bytes(range(16))
    plaintext = b"not aligned to an AES block"
    encrypted = internal_aes_cbc_encrypt(
        key,
        initialization_vector,
        plaintext,
        use_padding=True,
    )

    assert (
        internal_aes_cbc_decrypt(
            key,
            initialization_vector,
            encrypted,
            use_padding=True,
        )
        == plaintext
    )


def test_rc4_known_vector() -> None:
    key = bytes.fromhex("0102030405")
    plaintext = bytes(16)
    encrypted = internal_rc4_crypt(key, plaintext)

    assert encrypted.hex() == "b2396305f03dc027ccc3524a0a1118a8"
    assert internal_rc4_crypt(key, encrypted) == plaintext


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({}, "Invalid encryption dictionary"),
        ({"Filter": PdfName.of("PubSec")}, "Public-key encryption"),
        ({"Filter": PdfName.of("Custom")}, "Unsupported encryption filter"),
        (
            {"Filter": PdfName.of("Standard"), "V": 99},
            "Unsupported standard encryption algorithm",
        ),
    ],
)
def test_standard_security_factory_rejects_unsupported_handler(
    params: PdfDict,
    message: str,
) -> None:
    with pytest.raises(PdfUnsupportedError, match=message):
        create_standard_decipher([b"document-id"], params)


def test_standard_security_factory_rejects_mismatched_revision() -> None:
    params: PdfDict = {
        "Filter": PdfName.of("Standard"),
        "V": 4,
        "R": 3,
        "P": -4,
        "O": bytes(32),
        "U": bytes(32),
    }

    with pytest.raises(PdfUnsupportedError, match="Invalid encryption dictionary"):
        create_standard_decipher([b"document-id"], params)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("user\u00a0name", "user name"),
        ("I\u00adX", "IX"),
        ("\u00aa", "a"),
        ("\u2168", "IX"),
        ("\u0627\u0628", "\u0627\u0628"),
        ("", ""),
        ("\u00ad", ""),
    ],
)
def test_saslprep_maps_and_normalizes_valid_passwords(value: str, expected: str) -> None:
    assert internal_saslprep(value) == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("\u0007", "prohibited character"),
        ("\ue000", "prohibited character"),
        ("\u0627a\u0628", "prohibited character"),
        ("\u0627\u06280", "bidirectional"),
    ],
)
def test_saslprep_rejects_prohibited_and_bidirectionally_invalid_passwords(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        internal_saslprep(value)


def test_security_handler_uses_explicit_named_crypt_filter() -> None:
    attrs: PdfDict = {
        "Filter": [PdfName.of("Crypt"), PdfName.of("FlateDecode")],
        "DecodeParms": [{"Name": PdfName.of("Special")}, None],
    }

    assert internal_stream_crypt_filter_name(attrs, "Default") == "Special"
    methods: dict[str, internal_CryptMethod] = {"Special": "AESV2"}
    assert internal_resolve_crypt_method("Special", methods) == "AESV2"


def test_security_handler_defaults_explicit_crypt_to_identity() -> None:
    attrs: PdfDict = {"Filter": PdfName.of("Crypt")}

    assert internal_stream_crypt_filter_name(attrs, "Default") == "Identity"


def test_security_handler_rejects_late_crypt_filter() -> None:
    attrs: PdfDict = {"Filter": [PdfName.of("FlateDecode"), PdfName.of("Crypt")]}

    with pytest.raises(PdfParseError, match="first"):
        internal_stream_crypt_filter_name(attrs, "Default")


def test_security_handler_rejects_unknown_named_crypt_filter() -> None:
    with pytest.raises(PdfUnsupportedError, match="Undefined crypt filter"):
        internal_resolve_crypt_method("Unknown", {})

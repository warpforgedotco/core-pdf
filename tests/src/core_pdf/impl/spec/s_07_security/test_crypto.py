from __future__ import annotations

from pathlib import Path

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.spec.s_07_security.ciphers import (
    internal_aes_cbc_decrypt,
    internal_aes_cbc_encrypt,
    internal_rc4_crypt,
)
from core_pdf.impl.spec.s_07_security.saslprep import saslprep

ENCRYPTION_FIXTURES = (
    Path(__file__).parents[5] / "fixtures" / "pdfminer.six" / "samples" / "encryption"
)


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
    ("filename", "password", "expected"),
    [
        ("rc4-40.pdf", "foo", "Secret!"),
        ("rc4-128.pdf", "foo", "Secret!"),
        ("aes-128.pdf", "foo", "Secret!"),
        ("aes-256.pdf", "foo", "Secret!"),
        ("aes-256-r6.pdf", "usersecret", "Hello World"),
    ],
)
def test_encrypted_pdf_fixture_opens(filename: str, password: str, expected: str) -> None:
    with PdfDocument.open(ENCRYPTION_FIXTURES / filename, password=password) as document:
        assert document.pages[0].extract().text == expected


def test_encrypted_pdf_rejects_incorrect_password() -> None:
    with pytest.raises(PdfUnsupportedError, match="Incorrect password"):
        PdfDocument.open(ENCRYPTION_FIXTURES / "aes-256-r6.pdf", password="wrong")


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
    assert saslprep(value) == expected


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
        saslprep(value)

from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_07_security.aes import AES
from core_pdf.impl.spec.s_07_security.rc4 import CryptRC4
from core_pdf.impl.spec.s_07_security.saslprep import saslprep


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (
            "000102030405060708090a0b0c0d0e0f",
            "69c4e0d86a7b0430d8cdb78070b4c55a",
        ),
        (
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
            "8ea2b7ca516745bfeafc49904b496089",
        ),
    ],
)
def test_aes_block_known_vectors(key: str, expected: str) -> None:
    cipher = AES(bytes.fromhex(key))
    plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")
    encrypted = cipher.encrypt_block(plaintext)

    assert encrypted.hex() == expected
    assert cipher.decrypt_block(encrypted) == plaintext


def test_rc4_known_vector() -> None:
    cipher = CryptRC4(b"Key")
    encrypted = cipher.encrypt(b"Plaintext")

    assert encrypted.hex() == "bbf316e8d940af0ad3"
    assert cipher.decrypt(encrypted) == b"Plaintext"


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

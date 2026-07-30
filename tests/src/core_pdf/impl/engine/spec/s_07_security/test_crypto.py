from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_security.aes import AES
from core_pdf.impl.engine.spec.s_07_security.rc4 import CryptRC4
from core_pdf.impl.engine.spec.s_07_security.saslprep import saslprep


def test_aes_128_block_vector() -> None:
    cipher = AES(bytes.fromhex("000102030405060708090a0b0c0d0e0f"))
    plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")
    encrypted = cipher.encrypt_block(plaintext)

    assert encrypted.hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"
    assert cipher.decrypt_block(encrypted) == plaintext


def test_rc4_known_vector() -> None:
    cipher = CryptRC4(b"Key")
    encrypted = cipher.encrypt(b"Plaintext")

    assert encrypted.hex() == "bbf316e8d940af0ad3"
    assert cipher.decrypt(encrypted) == b"Plaintext"


def test_saslprep_maps_nonbreaking_space() -> None:
    assert saslprep("user\u00a0name") == "user name"

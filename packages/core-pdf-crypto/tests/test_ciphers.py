# SPDX-License-Identifier: AGPL-3.0-only
"""Cipher primitives authenticate, pad, and frame exactly as their standards require."""

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core_pdf_crypto.ciphers import (
    AES_GCM_IV_BYTES,
    AES_GCM_TAG_BYTES,
    aes_cbc_decrypt,
    aes_cbc_encrypt,
    aes_ecb_decrypt,
    aes_gcm_decrypt,
    rc4_crypt,
)
from core_pdf_crypto.errors import DecryptionError

internal_KEY = bytes(range(32))
internal_IV = bytes(range(16))


def test_aes_gcm_decrypts_the_iso_32003_serialized_form() -> None:
    # ISO/TS 32003:2023, 5.2: <12-byte IV><ciphertext><16-byte tag>, nil AAD.
    nonce = b"\x07" * AES_GCM_IV_BYTES
    ciphertext_and_tag = AESGCM(internal_KEY).encrypt(nonce, b"plain text", None)
    assert len(ciphertext_and_tag) == len(b"plain text") + AES_GCM_TAG_BYTES
    assert aes_gcm_decrypt(internal_KEY, nonce + ciphertext_and_tag) == b"plain text"


def test_aes_gcm_rejects_tampered_tag_and_short_input() -> None:
    nonce = b"\x07" * AES_GCM_IV_BYTES
    data = nonce + AESGCM(internal_KEY).encrypt(nonce, b"plain text", None)
    damaged = data[:-1] + bytes([data[-1] ^ 1])
    with pytest.raises(DecryptionError):
        aes_gcm_decrypt(internal_KEY, damaged)
    with pytest.raises(DecryptionError):
        aes_gcm_decrypt(internal_KEY, nonce)


def test_aes_gcm_requires_a_256_bit_key() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        aes_gcm_decrypt(internal_KEY[:16], b"\0" * 28)


def test_aes_cbc_round_trips_with_and_without_padding() -> None:
    padded = aes_cbc_encrypt(internal_KEY[:16], internal_IV, b"seven b", use_padding=True)
    assert len(padded) == 16
    assert aes_cbc_decrypt(internal_KEY[:16], internal_IV, padded, use_padding=True) == b"seven b"
    block = aes_cbc_encrypt(internal_KEY, internal_IV, b"x" * 32, use_padding=False)
    assert aes_cbc_decrypt(internal_KEY, internal_IV, block, use_padding=False) == b"x" * 32


def test_aes_cbc_reports_bad_padding_as_decryption_error() -> None:
    with pytest.raises(DecryptionError):
        aes_cbc_decrypt(internal_KEY[:16], internal_IV, b"\0" * 16, use_padding=True)


def test_aes_ecb_rejects_partial_blocks() -> None:
    with pytest.raises(DecryptionError):
        aes_ecb_decrypt(internal_KEY, b"\0" * 15)


def test_rc4_is_an_involution_with_a_known_vector() -> None:
    # RFC 6229, 40-bit key 0102030405: the first keystream bytes.
    key = bytes.fromhex("0102030405")
    keystream = bytes.fromhex("b2396305f03dc027ccc3524a0a1118a8")
    assert rc4_crypt(key, bytes(16)) == keystream
    assert rc4_crypt(key, rc4_crypt(key, b"Attack at dawn")) == b"Attack at dawn"

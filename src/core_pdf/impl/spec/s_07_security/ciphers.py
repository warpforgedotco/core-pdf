# SPDX-License-Identifier: AGPL-3.0-only
"""Vetted cipher operations used by the PDF standard security handlers."""

from __future__ import annotations

from cryptography.hazmat.decrepit.ciphers.algorithms import ARC4
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from core_pdf.impl.exceptions import internal_InvalidCipherPaddingError


def internal_aes_algorithm(key: bytes) -> algorithms.AES:
    if len(key) not in (16, 32):
        raise ValueError(f"AES key must be 16 or 32 bytes, got {len(key)}")
    return algorithms.AES(key)


def internal_aes_cbc_encrypt(
    key: bytes,
    initialization_vector: bytes,
    plaintext: bytes,
    *,
    use_padding: bool,
) -> bytes:
    algorithm = internal_aes_algorithm(key)
    if use_padding:
        padder = padding.PKCS7(algorithm.block_size).padder()
        plaintext = padder.update(plaintext) + padder.finalize()
    encryptor = Cipher(algorithm, modes.CBC(initialization_vector)).encryptor()
    return encryptor.update(plaintext) + encryptor.finalize()


def internal_aes_cbc_decrypt(
    key: bytes,
    initialization_vector: bytes,
    ciphertext: bytes,
    *,
    use_padding: bool,
) -> bytes:
    algorithm = internal_aes_algorithm(key)
    decryptor = Cipher(algorithm, modes.CBC(initialization_vector)).decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    if use_padding:
        unpadder = padding.PKCS7(algorithm.block_size).unpadder()
        try:
            plaintext = unpadder.update(plaintext) + unpadder.finalize()
        except ValueError as exc:
            raise internal_InvalidCipherPaddingError from exc
    return plaintext


def internal_rc4_crypt(key: bytes, data: bytes) -> bytes:
    cryptor = Cipher(ARC4(key), mode=None).encryptor()
    return cryptor.update(data) + cryptor.finalize()


__all__ = ()

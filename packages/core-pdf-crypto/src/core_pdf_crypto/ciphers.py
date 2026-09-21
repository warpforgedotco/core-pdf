# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.decrepit.ciphers.algorithms import ARC4
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core_pdf_crypto.errors import DecryptionError

AES_GCM_KEY_BYTES = 32
AES_GCM_IV_BYTES = 12
AES_GCM_TAG_BYTES = 16
AES_GCM_MAX_PLAINTEXT_BYTES = (1 << 39) - 256


def internal_aes_algorithm(key: bytes) -> algorithms.AES:
    if len(key) not in (16, 32):
        raise ValueError(f"AES key must be 16 or 32 bytes, got {len(key)}")
    return algorithms.AES(key)


def aes_cbc_encrypt(
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


def aes_cbc_decrypt(
    key: bytes,
    initialization_vector: bytes,
    ciphertext: bytes,
    *,
    use_padding: bool,
) -> bytes:
    algorithm = internal_aes_algorithm(key)
    try:
        decryptor = Cipher(algorithm, modes.CBC(initialization_vector)).decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        if use_padding:
            unpadder = padding.PKCS7(algorithm.block_size).unpadder()
            plaintext = unpadder.update(plaintext) + unpadder.finalize()
    except ValueError as exc:
        raise DecryptionError("Invalid encrypted object ciphertext") from exc
    return plaintext


def aes_ecb_decrypt(key: bytes, ciphertext: bytes) -> bytes:
    algorithm = internal_aes_algorithm(key)
    try:
        decryptor = Cipher(algorithm, modes.ECB()).decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    except ValueError as exc:
        raise DecryptionError("Invalid encrypted object ciphertext") from exc


def aes_gcm_decrypt(key: bytes, data: bytes) -> bytes:
    if len(key) != AES_GCM_KEY_BYTES:
        raise ValueError(f"AESV4 key must be {AES_GCM_KEY_BYTES} bytes, got {len(key)}")
    minimum_length = AES_GCM_IV_BYTES + AES_GCM_TAG_BYTES
    maximum_length = AES_GCM_MAX_PLAINTEXT_BYTES + minimum_length
    if not minimum_length <= len(data) <= maximum_length:
        raise DecryptionError("Invalid encrypted object ciphertext")

    initialization_vector = data[:AES_GCM_IV_BYTES]
    ciphertext_and_tag = data[AES_GCM_IV_BYTES:]
    try:
        return AESGCM(key).decrypt(initialization_vector, ciphertext_and_tag, None)
    except (InvalidTag, OverflowError, ValueError) as exc:
        raise DecryptionError("Invalid encrypted object ciphertext") from exc


def rc4_crypt(key: bytes, data: bytes) -> bytes:
    cryptor = Cipher(ARC4(key), mode=None).encryptor()
    return cryptor.update(data) + cryptor.finalize()


__all__ = (
    "AES_GCM_KEY_BYTES",
    "AES_GCM_IV_BYTES",
    "AES_GCM_TAG_BYTES",
    "AES_GCM_MAX_PLAINTEXT_BYTES",
    "aes_cbc_encrypt",
    "aes_cbc_decrypt",
    "aes_ecb_decrypt",
    "aes_gcm_decrypt",
    "rc4_crypt",
)

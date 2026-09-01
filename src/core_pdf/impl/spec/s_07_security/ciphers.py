# SPDX-License-Identifier: AGPL-3.0-only
"""Vetted cipher operations used by the PDF standard security handlers."""

from __future__ import annotations

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.decrepit.ciphers.algorithms import ARC4
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core_pdf.impl.exceptions import PdfDecryptionError

internal_AES_GCM_KEY_BYTES = 32
internal_AES_GCM_IV_BYTES = 12
internal_AES_GCM_TAG_BYTES = 16
internal_AES_GCM_MAX_PLAINTEXT_BYTES = (1 << 39) - 256


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
    # Object strings and streams use a 16-byte IV and PKCS#7-style padding:
    # - ISO 32000-1:2008, 7.6.2 (AESV2 / revision 4)
    # - Adobe Supplement to ISO 32000, BaseVersion 1.7, ExtensionLevel 3,
    #   June 2008, 3.5.1, Algorithm 3.1a (AESV3 / revision 5)
    # - ISO 32000-2:2020, 7.6.3.1 and 7.6.3.3 (AESV3 / revision 6)
    # The R5/R6 password algorithms explicitly pass use_padding=False; see
    # that Adobe supplement's 3.5.2, Algorithm 3.2a and ISO 32000-2:2020,
    # 7.6.4.3.3, Algorithm 2.A.
    algorithm = internal_aes_algorithm(key)
    try:
        decryptor = Cipher(algorithm, modes.CBC(initialization_vector)).decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        if use_padding:
            unpadder = padding.PKCS7(algorithm.block_size).unpadder()
            plaintext = unpadder.update(plaintext) + unpadder.finalize()
    except ValueError as exc:
        raise PdfDecryptionError("Invalid encrypted object ciphertext") from exc
    return plaintext


def internal_aes_ecb_decrypt(key: bytes, ciphertext: bytes) -> bytes:
    """Decrypt the fixed permissions block required by R5 and R6.

    Sources: Adobe Supplement to ISO 32000, BaseVersion 1.7,
    ExtensionLevel 3, June 2008, 3.5.2, Algorithm 3.13; and
    ISO 32000-2:2020, 7.6.4.4.12, Algorithm 13.
    """
    algorithm = internal_aes_algorithm(key)
    try:
        decryptor = Cipher(algorithm, modes.ECB()).decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    except ValueError as exc:
        raise PdfDecryptionError("Invalid encrypted object ciphertext") from exc


def internal_aes_gcm_decrypt(key: bytes, data: bytes) -> bytes:
    """Decrypt one AESV4 string or stream and authenticate it before returning.

    ISO/TS 32003:2023, 5.2 specifies a 32-byte key, 12-byte IV, nil AAD,
    16-byte authentication tag, no PDF-level padding, and the serialized form
    ``<IV><ciphertext><tag>``. It limits each plaintext object to 2^39 - 256
    bytes. AES-GCM itself is defined by NIST SP 800-38D (November 2007).
    """
    if len(key) != internal_AES_GCM_KEY_BYTES:
        raise ValueError(f"AESV4 key must be {internal_AES_GCM_KEY_BYTES} bytes, got {len(key)}")
    minimum_length = internal_AES_GCM_IV_BYTES + internal_AES_GCM_TAG_BYTES
    maximum_length = internal_AES_GCM_MAX_PLAINTEXT_BYTES + minimum_length
    if not minimum_length <= len(data) <= maximum_length:
        raise PdfDecryptionError("Invalid encrypted object ciphertext")

    initialization_vector = data[:internal_AES_GCM_IV_BYTES]
    ciphertext_and_tag = data[internal_AES_GCM_IV_BYTES:]
    try:
        return AESGCM(key).decrypt(initialization_vector, ciphertext_and_tag, None)
    except (InvalidTag, OverflowError, ValueError) as exc:
        raise PdfDecryptionError("Invalid encrypted object ciphertext") from exc


def internal_rc4_crypt(key: bytes, data: bytes) -> bytes:
    cryptor = Cipher(ARC4(key), mode=None).encryptor()
    return cryptor.update(data) + cryptor.finalize()


__all__ = ()

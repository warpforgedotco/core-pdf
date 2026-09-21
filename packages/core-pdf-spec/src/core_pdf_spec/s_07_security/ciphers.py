# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf_crypto import ciphers
from core_pdf_crypto.errors import DecryptionError
from core_pdf_spec.exceptions import PdfDecryptionError


def internal_aes_cbc_encrypt(
    key: bytes,
    initialization_vector: bytes,
    plaintext: bytes,
    *,
    use_padding: bool,
) -> bytes:
    return ciphers.aes_cbc_encrypt(key, initialization_vector, plaintext, use_padding=use_padding)


def internal_aes_cbc_decrypt(
    key: bytes,
    initialization_vector: bytes,
    ciphertext: bytes,
    *,
    use_padding: bool,
) -> bytes:
    try:
        return ciphers.aes_cbc_decrypt(
            key, initialization_vector, ciphertext, use_padding=use_padding
        )
    except DecryptionError as exc:
        raise PdfDecryptionError(str(exc)) from exc


def internal_aes_ecb_decrypt(key: bytes, ciphertext: bytes) -> bytes:
    try:
        return ciphers.aes_ecb_decrypt(key, ciphertext)
    except DecryptionError as exc:
        raise PdfDecryptionError(str(exc)) from exc


def internal_aes_gcm_decrypt(key: bytes, data: bytes) -> bytes:
    try:
        return ciphers.aes_gcm_decrypt(key, data)
    except DecryptionError as exc:
        raise PdfDecryptionError(str(exc)) from exc


def internal_rc4_crypt(key: bytes, data: bytes) -> bytes:
    return ciphers.rc4_crypt(key, data)


__all__ = ()

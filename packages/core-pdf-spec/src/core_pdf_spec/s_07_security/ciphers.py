# SPDX-License-Identifier: AGPL-3.0-only
"""PDF security-handler cipher calls, mapping kernel failures to PDF errors.

The cipher operations live in ``core_pdf_crypto.ciphers``. These wrappers keep
the signatures the standard security handler uses and translate
``DecryptionError`` into ``PdfDecryptionError``.
"""

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
    # Object strings and streams use a 16-byte IV and PKCS#7-style padding:
    # - ISO 32000-1:2008, 7.6.2 (AESV2 / revision 4)
    # - Adobe Supplement to ISO 32000, BaseVersion 1.7, ExtensionLevel 3,
    #   June 2008, 3.5.1, Algorithm 3.1a (AESV3 / revision 5)
    # - ISO 32000-2:2020, 7.6.3.1 and 7.6.3.3 (AESV3 / revision 6)
    # The R5/R6 password algorithms explicitly pass use_padding=False; see
    # that Adobe supplement's 3.5.2, Algorithm 3.2a and ISO 32000-2:2020,
    # 7.6.4.3.3, Algorithm 2.A.
    try:
        return ciphers.aes_cbc_decrypt(
            key, initialization_vector, ciphertext, use_padding=use_padding
        )
    except DecryptionError as exc:
        raise PdfDecryptionError(str(exc)) from exc


def internal_aes_ecb_decrypt(key: bytes, ciphertext: bytes) -> bytes:
    """Decrypt the fixed permissions block required by R5 and R6.

    Sources: Adobe Supplement to ISO 32000, BaseVersion 1.7,
    ExtensionLevel 3, June 2008, 3.5.2, Algorithm 3.13; and
    ISO 32000-2:2020, 7.6.4.4.12, Algorithm 13.
    """
    try:
        return ciphers.aes_ecb_decrypt(key, ciphertext)
    except DecryptionError as exc:
        raise PdfDecryptionError(str(exc)) from exc


def internal_aes_gcm_decrypt(key: bytes, data: bytes) -> bytes:
    """Decrypt one AESV4 string or stream (ISO/TS 32003:2023, 5.2)."""
    try:
        return ciphers.aes_gcm_decrypt(key, data)
    except DecryptionError as exc:
        raise PdfDecryptionError(str(exc)) from exc


def internal_rc4_crypt(key: bytes, data: bytes) -> bytes:
    return ciphers.rc4_crypt(key, data)


__all__ = ()

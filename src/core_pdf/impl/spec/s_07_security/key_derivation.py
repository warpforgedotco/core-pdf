# SPDX-License-Identifier: AGPL-3.0-only
"""Standard Security (RC4/MD5) key-derivation rituals shared by the read and
write paths (PDF 32000 §7.6.3.3-7.6.3.4)."""

from __future__ import annotations

from hashlib import md5

from core_pdf.impl.spec.s_07_security.crypto_constants import PDF_PADDING
from core_pdf.impl.spec.s_07_security.rc4 import CryptRC4


def pad_password(password: bytes) -> bytes:
    return (password + PDF_PADDING)[:32]


def md5_50_rounds(digest: bytes, keep: int = 16) -> bytes:
    """Apply the revision-3 50-round MD5 hardening, keeping ``keep`` bytes per round."""
    for _ in range(50):
        digest = md5(digest[:keep]).digest()
    return digest


def rc4_xor_cascade(key: bytes, data: bytes, indexes: range) -> bytes:
    """Run the O/U-entry RC4 cascade with the key XORed by each index.

    RC4 is symmetric, so the same cascade encrypts (ascending indexes) and
    decrypts (descending indexes).
    """
    for index in indexes:
        data = CryptRC4(bytes(byte ^ index for byte in key)).encrypt(data)
    return data


__all__ = ("md5_50_rounds", "pad_password", "rc4_xor_cascade")

# SPDX-License-Identifier: AGPL-3.0-only
"""Standard Security (RC4/MD5) reader key derivation (PDF 32000 §7.6.3.3-7.6.3.4)."""

from __future__ import annotations

from hashlib import md5

from core_pdf.impl.spec.s_07_security.ciphers import internal_rc4_crypt

PDF_PASSWORD_PADDING: bytes = (
    b"\x28\xbf\x4e\x5e\x4e\x75\x8a\x41\x64\x00\x4e\x56\xff\xfa\x01\x08"
    b"\x2e\x2e\x00\xb6\xd0\x68\x3e\x80\x2f\x0c\xa9\xfe\x64\x53\x69\x7a"
)


def pad_password(password: bytes) -> bytes:
    return (password + PDF_PASSWORD_PADDING)[:32]


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
        data = internal_rc4_crypt(bytes(byte ^ index for byte in key), data)
    return data


__all__ = ()

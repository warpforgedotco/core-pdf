# SPDX-License-Identifier: AGPL-3.0-only
"""Error family for cipher and PDF MAC kernels."""

from __future__ import annotations


class CryptoError(Exception):
    """Base class for cipher and MAC validation failures."""


class DecryptionError(CryptoError):
    """Ciphertext could not be decrypted or authenticated."""


class UnsupportedAlgorithmError(CryptoError):
    """A well-formed structure names an algorithm this package does not implement."""


__all__ = ("CryptoError", "DecryptionError", "UnsupportedAlgorithmError")

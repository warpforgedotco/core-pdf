# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations


class CryptoError(Exception):
    pass


class DecryptionError(CryptoError):
    pass


class UnsupportedAlgorithmError(CryptoError):
    pass


__all__ = ("CryptoError", "DecryptionError", "UnsupportedAlgorithmError")

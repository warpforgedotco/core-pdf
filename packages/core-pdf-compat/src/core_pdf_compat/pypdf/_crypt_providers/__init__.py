"""pypdf's crypt provider shapes, over core_pdf.impl.crypto's ciphers."""

from __future__ import annotations

__lazy_modules__ = ["core_pdf.impl.crypto", "core_pdf.impl.crypto.ciphers", "importlib.metadata"]

import logging
import secrets
from importlib.metadata import version

from core_pdf.impl.crypto import ciphers
from core_pdf.impl.crypto.ciphers import rc4_crypt
from core_pdf_compat.pdfminer.arcfour import Arcfour
from core_pdf_compat.pypdf.errors import PdfStreamError

logger = logging.getLogger(__name__)
BLOCK_BYTES = 16
PADDING_ERROR = "Invalid padding bytes."
LENGTH_ERROR = "The length of the provided data is not a multiple of the block length."


def __getattr__(name: str) -> object:
    # Looking cryptography's version up scans the installed distributions, so it waits until
    # something reads crypt_provider, and is kept from then on.
    if name == "crypt_provider":
        provider = globals()["crypt_provider"] = ("cryptography", version("cryptography"))
        return provider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class CryptBase:
    def encrypt(self, data: bytes) -> bytes:
        return data

    def decrypt(self, data: bytes, *, strict: bool = True) -> bytes:
        del strict
        return data


class CryptIdentity(CryptBase):
    pass


_RC4_STATE = {"supported": True}


def _rc4(key: bytes, data: bytes) -> bytes:
    """RC4 through cryptography, or in Python once OpenSSL has refused it, as pypdf 7.0.0
    falls back (#3853), warning the first time.
    """
    if _RC4_STATE["supported"]:
        try:
            return rc4_crypt(key, data)
        except Exception as error:
            # cryptography's UnsupportedAlgorithm, without importing it here.
            if type(error).__name__ != "UnsupportedAlgorithm":
                raise
            logger.warning(
                "RC4 is not supported by the current OpenSSL build; "
                "falling back to the pure-Python RC4 implementation."
            )
            _RC4_STATE["supported"] = False
    return Arcfour(key).process(data)


class CryptRC4(CryptBase):
    def __init__(self, key: bytes) -> None:
        self.key = key

    def __repr__(self) -> str:
        return "CryptRC4()"

    def encrypt(self, data: bytes) -> bytes:
        return _rc4(self.key, data)

    def decrypt(self, data: bytes, *, strict: bool = True) -> bytes:
        del strict
        return _rc4(self.key, data)


def padding_is_valid(data: bytes) -> bool:
    count = data[-1]
    return 1 <= count <= BLOCK_BYTES and data.endswith(bytes((count,)) * count)


class CryptAES(CryptBase):
    def __init__(self, key: bytes) -> None:
        self.key = key

    def __repr__(self) -> str:
        return "CryptAES()"

    def encrypt(self, data: bytes) -> bytes:
        iv = secrets.token_bytes(BLOCK_BYTES)
        return iv + ciphers.aes_cbc_encrypt(self.key, iv, data, use_padding=True)

    def decrypt(self, data: bytes, *, strict: bool = True) -> bytes:
        iv, data = data[:BLOCK_BYTES], data[BLOCK_BYTES:]
        if not data:
            return data
        if len(data) % BLOCK_BYTES:
            if strict:
                raise PdfStreamError(LENGTH_ERROR)
            logger.warning("Adding missing padding.")
            missing = BLOCK_BYTES - len(data) % BLOCK_BYTES
            data += bytes((missing,)) * missing
        padded = ciphers.aes_cbc_decrypt(self.key, iv, data, use_padding=False)
        if padding_is_valid(padded):
            return padded[: -padded[-1]]
        if strict:
            raise PdfStreamError(PADDING_ERROR)
        logger.warning("Ignoring padding error: %s", PADDING_ERROR)
        return padded[: -padded[-1]]


# The provider's module-level helpers, without padding, as pypdf's are.
def rc4_encrypt(key: bytes, data: bytes) -> bytes:
    return _rc4(key, data)


def rc4_decrypt(key: bytes, data: bytes) -> bytes:
    return _rc4(key, data)


def aes_ecb_encrypt(key: bytes, data: bytes) -> bytes:
    return ciphers.aes_ecb_encrypt(key, data)


def aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    return ciphers.aes_ecb_decrypt(key, data)


def aes_cbc_encrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    return ciphers.aes_cbc_encrypt(key, iv, data, use_padding=False)


def aes_cbc_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    return ciphers.aes_cbc_decrypt(key, iv, data, use_padding=False)


__all__ = (
    "CryptAES",
    "CryptBase",
    "CryptIdentity",
    "CryptRC4",
    "aes_cbc_decrypt",
    "aes_cbc_encrypt",
    "aes_ecb_decrypt",
    "aes_ecb_encrypt",
    "crypt_provider",
    "rc4_decrypt",
    "rc4_encrypt",
)

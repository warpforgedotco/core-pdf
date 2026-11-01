"""pypdf's crypt provider shapes, over core-pdf-crypto's ciphers."""

from __future__ import annotations

import logging
import secrets
from importlib.metadata import version

from core_pdf_compat.pypdf.errors import PdfStreamError
from core_pdf_crypto.ciphers import aes_cbc_decrypt, aes_cbc_encrypt, rc4_crypt

logger = logging.getLogger(__name__)
crypt_provider = ("cryptography", version("cryptography"))
BLOCK_BYTES = 16
PADDING_ERROR = "Invalid padding bytes."
LENGTH_ERROR = "The length of the provided data is not a multiple of the block length."


class CryptBase:
    def encrypt(self, data: bytes) -> bytes:
        return data

    def decrypt(self, data: bytes, *, strict: bool = True) -> bytes:
        del strict
        return data


class CryptIdentity(CryptBase):
    pass


class CryptRC4(CryptBase):
    def __init__(self, key: bytes) -> None:
        self.key = key

    def __repr__(self) -> str:
        return "CryptRC4()"

    def encrypt(self, data: bytes) -> bytes:
        return rc4_crypt(self.key, data)

    def decrypt(self, data: bytes, *, strict: bool = True) -> bytes:
        del strict
        return rc4_crypt(self.key, data)


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
        return iv + aes_cbc_encrypt(self.key, iv, data, use_padding=True)

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
        padded = aes_cbc_decrypt(self.key, iv, data, use_padding=False)
        if padding_is_valid(padded):
            return padded[: -padded[-1]]
        if strict:
            raise PdfStreamError(PADDING_ERROR)
        logger.warning("Ignoring padding error: %s", PADDING_ERROR)
        return padded[: -padded[-1]]


__all__ = ("CryptAES", "CryptBase", "CryptIdentity", "CryptRC4", "crypt_provider")

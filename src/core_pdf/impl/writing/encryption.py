# SPDX-License-Identifier: AGPL-3.0-only
"""PDF Standard Security Revision 3 writing support."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from hashlib import md5
from typing import Any

from core_pdf.impl.primitives import PdfName, PdfString
from core_pdf.impl.spec.s_07_security.crypto_constants import PDF_PADDING
from core_pdf.impl.spec.s_07_security.key_derivation import (
    md5_50_rounds,
    pad_password,
    rc4_xor_cascade,
)
from core_pdf.impl.spec.s_07_security.rc4 import CryptRC4
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream


@dataclass(frozen=True, slots=True)
class StandardPdfEncryption:
    """Password encryption settings for PDF Standard Security Revision 3."""

    user_password: str
    owner_password: str | None = None
    permissions: int = 0xFFFFFFFC
    encrypt_metadata: bool = True

    def context(self, file_id: bytes) -> StandardPdfEncryptionContext:
        return StandardPdfEncryptionContext.create(self, file_id)


@dataclass(frozen=True, slots=True)
class StandardPdfEncryptionContext:
    key: bytes
    owner_entry: bytes
    user_entry: bytes
    permissions: int
    encrypt_metadata: bool

    @classmethod
    def from_security_handler(cls, handler: Any) -> StandardPdfEncryptionContext:
        """Reuse an authenticated parser security handler for an update."""
        if getattr(handler, "r", None) != 3:
            raise ValueError("encrypted incremental updates require Standard Security revision 3")
        try:
            key = handler.key
            owner_entry = handler.o
            user_entry = handler.u
            permissions = handler.p
            encrypt_metadata = handler.encrypt_metadata
        except AttributeError as exc:
            raise ValueError("missing Standard Security handler state") from exc
        if not all(isinstance(value, bytes) for value in (key, owner_entry, user_entry)):
            raise ValueError("invalid Standard Security handler state")
        return cls(
            key=key,
            owner_entry=owner_entry,
            user_entry=user_entry,
            permissions=permissions,
            encrypt_metadata=encrypt_metadata,
        )

    @classmethod
    def create(
        cls,
        settings: StandardPdfEncryption,
        file_id: bytes,
    ) -> StandardPdfEncryptionContext:
        if not file_id:
            raise ValueError("PDF encryption requires a non-empty file ID")
        owner_password = settings.owner_password or settings.user_password
        owner_key = internal_password_key(owner_password)
        owner_entry = internal_owner_entry(
            owner_key, internal_password_bytes(settings.user_password)
        )
        key = internal_file_key(
            internal_password_bytes(settings.user_password),
            owner_entry,
            settings.permissions,
            file_id,
            settings.encrypt_metadata,
        )
        user_entry = internal_user_entry(key, file_id)
        return cls(
            key=key,
            owner_entry=owner_entry,
            user_entry=user_entry,
            permissions=settings.permissions,
            encrypt_metadata=settings.encrypt_metadata,
        )

    def encryption_dictionary(self) -> dict[PdfName, object]:
        return {
            PdfName.of("Filter"): PdfName.of("Standard"),
            PdfName.of("V"): 2,
            PdfName.of("Length"): 128,
            PdfName.of("R"): 3,
            PdfName.of("O"): PdfString(self.owner_entry),
            PdfName.of("U"): PdfString(self.user_entry),
            PdfName.of("P"): self.permissions,
            PdfName.of("EncryptMetadata"): self.encrypt_metadata,
        }

    def encrypt_object(
        self,
        value: object,
        object_number: int,
        generation_number: int = 0,
    ) -> object:
        if isinstance(value, PdfString):
            return PdfString(self.encrypt_bytes(value.data, object_number, generation_number))
        if isinstance(value, (bytes, bytearray, memoryview)):
            return self.encrypt_bytes(bytes(value), object_number, generation_number)
        if isinstance(value, PdfStream):
            dictionary = {
                key: self.encrypt_object(item, object_number, generation_number)
                for key, item in value.dictionary.items()
            }
            return value.replace(
                dictionary=dictionary,
                raw_data=self.encrypt_bytes(
                    bytes(value.raw_data), object_number, generation_number
                ),
                decoded_data=None,
            )
        if isinstance(value, list):
            return [self.encrypt_object(item, object_number, generation_number) for item in value]
        if isinstance(value, tuple):
            return tuple(
                self.encrypt_object(item, object_number, generation_number) for item in value
            )
        if isinstance(value, dict):
            return {
                key: self.encrypt_object(item, object_number, generation_number)
                for key, item in value.items()
            }
        return value

    def encrypt_bytes(self, data: bytes, object_number: int, generation_number: int) -> bytes:
        object_key = md5(
            self.key
            + struct.pack("<L", object_number)[:3]
            + struct.pack("<L", generation_number)[:2]
        ).digest()[: min(len(self.key) + 5, 16)]
        return CryptRC4(object_key).encrypt(data)


def internal_password_bytes(password: str) -> bytes:
    return password.encode("latin-1", errors="strict")


def internal_password_key(password: str) -> bytes:
    digest = md5(pad_password(internal_password_bytes(password))).digest()
    return md5_50_rounds(digest)[:16]


def internal_owner_entry(owner_key: bytes, user_password: bytes) -> bytes:
    encrypted = CryptRC4(owner_key).encrypt(pad_password(user_password))
    return rc4_xor_cascade(owner_key, encrypted, range(1, 20))


def internal_file_key(
    user_password: bytes,
    owner_entry: bytes,
    permissions: int,
    file_id: bytes,
    encrypt_metadata: bool,
) -> bytes:
    digest = md5()
    digest.update(pad_password(user_password))
    digest.update(owner_entry)
    digest.update(struct.pack("<L", permissions & 0xFFFFFFFF))
    digest.update(file_id)
    if not encrypt_metadata:
        digest.update(b"\xff\xff\xff\xff")
    return md5_50_rounds(digest.digest())[:16]


def internal_user_entry(key: bytes, file_id: bytes) -> bytes:
    value = md5(PDF_PADDING + file_id).digest()
    value = CryptRC4(key).encrypt(value)
    return rc4_xor_cascade(key, value, range(1, 20)) + b"\0" * 16


__all__ = ("StandardPdfEncryption", "StandardPdfEncryptionContext")

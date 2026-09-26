# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from types import TracebackType

from core_pdf_crypto import ciphers
from core_pdf_crypto.errors import DecryptionError
from core_pdf_spec.exceptions import PdfDecryptionError


class DecryptionErrorsAsPdf:
    """Re-raise core-pdf-crypto's DecryptionError as PdfDecryptionError."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if isinstance(exc, DecryptionError):
            raise PdfDecryptionError(str(exc)) from exc


DECRYPTION_ERRORS_AS_PDF = DecryptionErrorsAsPdf()


def aes_cbc_decrypt(
    key: bytes,
    initialization_vector: bytes,
    ciphertext: bytes,
    *,
    use_padding: bool,
) -> bytes:
    with DECRYPTION_ERRORS_AS_PDF:
        return ciphers.aes_cbc_decrypt(
            key, initialization_vector, ciphertext, use_padding=use_padding
        )


def aes_ecb_decrypt(key: bytes, ciphertext: bytes) -> bytes:
    with DECRYPTION_ERRORS_AS_PDF:
        return ciphers.aes_ecb_decrypt(key, ciphertext)


def aes_gcm_decrypt(key: bytes, data: bytes) -> bytes:
    with DECRYPTION_ERRORS_AS_PDF:
        return ciphers.aes_gcm_decrypt(key, data)


__all__ = ()

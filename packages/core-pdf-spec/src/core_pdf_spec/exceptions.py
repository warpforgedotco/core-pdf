# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations


class PdfError(Exception):
    pass


class PdfParseError(PdfError):
    pass


class PdfDecryptionError(PdfError):
    pass


class PdfInvalidPermissionsError(PdfDecryptionError):
    """The password is right, but /Perms does not match /P, so the file was altered."""

    def __init__(self, message: str, password_kind: str) -> None:
        super().__init__(message)
        self.password_kind = password_kind


class PdfUnsupportedError(PdfError):
    pass


class PdfPasswordError(PdfUnsupportedError): ...


__all__ = (
    "PdfDecryptionError",
    "PdfError",
    "PdfInvalidPermissionsError",
    "PdfParseError",
    "PdfPasswordError",
    "PdfUnsupportedError",
)

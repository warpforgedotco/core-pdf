"""pypdf's message for a missing AES library; core-pdf always has one (core_pdf.impl.crypto)."""

_DEPENDENCY_ERROR_STR = "cryptography>=3.1 is required for AES algorithm"

__all__ = ("_DEPENDENCY_ERROR_STR",)

# SPDX-License-Identifier: AGPL-3.0-only
"""PDF 7.6 Standard Security support."""

from core_pdf.impl.spec.s_07_security.pdf_mac import (
    validate_pdf_mac_extension,
    validate_pdf_mac_if_present,
)
from core_pdf.impl.spec.s_07_security.standard import (
    create_standard_decipher,
    create_standard_security_handler,
)

__all__ = (
    "create_standard_decipher",
    "create_standard_security_handler",
    "validate_pdf_mac_extension",
    "validate_pdf_mac_if_present",
)

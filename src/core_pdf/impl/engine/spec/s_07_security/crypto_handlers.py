# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_security.standard import PdfStandardSecurityHandler
from core_pdf.impl.engine.spec.s_07_security.standard_v4 import (
    PdfStandardSecurityHandlerV4,
)
from core_pdf.impl.engine.spec.s_07_security.standard_v5 import (
    PdfStandardSecurityHandlerV5,
)

SECURITY_HANDLER_REGISTRY = {
    1: PdfStandardSecurityHandler,
    2: PdfStandardSecurityHandler,
    3: PdfStandardSecurityHandler,
    4: PdfStandardSecurityHandlerV4,
    5: PdfStandardSecurityHandlerV5,
    6: PdfStandardSecurityHandlerV5,
}

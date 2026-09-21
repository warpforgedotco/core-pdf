# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf_spec.exceptions import (
    PdfDecryptionError as PdfDecryptionError,
)
from core_pdf_spec.exceptions import (
    PdfError as PdfError,
)
from core_pdf_spec.exceptions import (
    PdfParseError as PdfParseError,
)
from core_pdf_spec.exceptions import (
    PdfUnsupportedError as PdfUnsupportedError,
)


class PdfSourceError(PdfError):
    pass


class PdfContractError(PdfError, TypeError):
    pass


class PdfRasterTooLargeError(PdfError, ValueError):
    pass


class PdfDocumentClosedError(PdfError, ValueError):
    pass

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations


class PdfError(Exception):
    pass


class PdfSourceError(PdfError):
    pass


class PdfParseError(PdfError):
    pass


class PdfUnsupportedError(PdfError):
    pass

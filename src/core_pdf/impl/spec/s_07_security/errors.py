# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations


class PDFEncryptionError(Exception):
    pass


class PDFPasswordIncorrect(PDFEncryptionError):
    pass

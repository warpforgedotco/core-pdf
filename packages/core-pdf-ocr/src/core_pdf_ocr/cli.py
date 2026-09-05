# SPDX-License-Identifier: AGPL-3.0-only
"""Command-line extraction with OCR support."""

from __future__ import annotations

from collections.abc import Sequence

from core_pdf.cli import run
from core_pdf_ocr import PdfDocument


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv, document_class=PdfDocument, program_name="core-pdf-ocr")


if __name__ == "__main__":
    raise SystemExit(main())

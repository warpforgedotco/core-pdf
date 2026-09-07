# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate OCR extraction with the shared SCORE-Bench scorer."""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    from core_pdf_ocr import PdfDocument
    from scripts.score_unstructured_bench import ScoreBench

    benchmark = ScoreBench.from_cli(argv)
    benchmark.document_class = PdfDocument
    return benchmark.run()


if __name__ == "__main__":
    raise SystemExit(main())

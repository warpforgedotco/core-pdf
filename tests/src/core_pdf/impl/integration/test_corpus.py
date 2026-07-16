# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from core_pdf.impl.engine.extraction.document import PdfDocument

TESTS_DIR = Path(__file__).parents[4]
SAMPLES_DIR = TESTS_DIR / "fixtures" / "pdfminer.six" / "samples"
GROUND_TRUTH_DIR = TESTS_DIR / "fixtures" / "pdfminer.six_ground_truth"
OCR_DISABLED_SAMPLES = {
    "contrib/issue-00352-asw-oct96-p41.pdf",
    "contrib/issue-1061-colour-space-stack.pdf",
    "contrib/pdf-with-jbig2.pdf",
    "contrib/pr-00530-ml-lines.pdf",
    "encryption/encrypted_doc_no_id.pdf",
    "nonfree/175.pdf",
    "nonfree/dmca.pdf",
    "nonfree/f1040nr.pdf",
    "nonfree/i1040nr.pdf",
    "nonfree/naacl06-shinyama.pdf",
    "nonfree/nlp2004slides.pdf",
}
# These compatibility cases depend on OCR or OCR-specific reconciliation. Keep
# them disabled while OCR is globally shut off in core-pdf.
PDF_SAMPLES = tuple(
    path
    for path in sorted(SAMPLES_DIR.rglob("*.pdf"))
    if path.relative_to(SAMPLES_DIR).as_posix() not in OCR_DISABLED_SAMPLES
)

SAMPLE_PASSWORDS = {
    "encryption/aes-128-m.pdf": "foo",
    "encryption/aes-128.pdf": "foo",
    "encryption/aes-256-m.pdf": "foo",
    "encryption/aes-256-r6.pdf": "usersecret",
    "encryption/aes-256.pdf": "foo",
    "encryption/base.pdf": "foo",
    "encryption/rc4-128.pdf": "foo",
    "encryption/rc4-40.pdf": "foo",
}


def sample_id(path: Path) -> str:
    return path.relative_to(SAMPLES_DIR).as_posix()


def ground_truth_for(pdf_path: Path) -> str:
    path = GROUND_TRUTH_DIR / pdf_path.relative_to(SAMPLES_DIR).with_suffix(".txt")
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("pdf_path", PDF_SAMPLES, ids=sample_id)
def test_extract_text_handles_pdfminer_sample_corpus(pdf_path: Path) -> None:
    sample = sample_id(pdf_path)
    password = SAMPLE_PASSWORDS.get(sample, "")
    expected = ground_truth_for(pdf_path)

    with PdfDocument.open(pdf_path, password=password) as document:
        result = cast(Any, document).extract_text()

    assert result == expected, f"Failed for sample: {sample}"

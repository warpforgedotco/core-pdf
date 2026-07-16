# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from pathlib import Path

import pytest
from pdfminer.high_level import extract_text as pdfminer_extract_text

from core_pdf.integrations.pdfminer.high_level import extract_text as core_pdf_extract_text

TESTS_DIR = Path(__file__).parents[4]
SAMPLES_DIR = TESTS_DIR / "fixtures" / "pdfminer.six" / "samples"
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


@pytest.mark.parametrize("pdf_path", PDF_SAMPLES, ids=sample_id)
def test_extract_text_handles_pdfminer_sample_corpus(pdf_path: Path) -> None:
    sample = sample_id(pdf_path)
    password = SAMPLE_PASSWORDS.get(sample, "")
    expected = pdfminer_extract_text(pdf_path, password=password)

    result = core_pdf_extract_text(pdf_path, password=password)

    assert result == expected, f"Failed for sample: {sample}"

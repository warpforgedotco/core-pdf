# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
from pdfminer.high_level import extract_text as pdfminer_six_extract_text

from core_pdf.impl.engine.extraction.document import PdfDocument
from core_pdf.impl.types import PdfSource

TESTS_DIR = Path(__file__).parents[4]
SAMPLES_DIR = TESTS_DIR / "fixtures" / "pdfminer.six" / "samples"
EXPECTED_TEXT_DIR = Path(__file__).with_name("expected_text")
PDF_SAMPLES = tuple(sorted(SAMPLES_DIR.rglob("*.pdf")))

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

EXPECTED_PDFMINER_FAILURES = {
    "contrib/issue-1249-evil-xrefs.pdf": "RecursionError",
}

EXPECTED_CORE_FAILURES: dict[str, str] = {}


def stored_text(name: str) -> str:
    return (EXPECTED_TEXT_DIR / name).read_text(encoding="utf-8")


def comparable_pdfminer_text(text: str) -> str:
    return "\f".join(re.sub(r"[ \t\r\n]+", " ", page).strip() for page in text.split("\f"))


EXPECTED_PDFMINER_TEXT = {
    "acroform/AcroForm_TEST.pdf": stored_text("acroform.pdfminer.txt"),
    "acroform/AcroForm_TEST_compiled.pdf": stored_text("acroform.pdfminer.txt"),
    "contrib/2b.pdf": stored_text("2b.pdfminer.txt"),
    "contrib/issue-00369-excel.pdf": stored_text("issue-00369-excel.pdfminer.txt"),
    "contrib/issue-00352-asw-oct96-p41.pdf": stored_text("issue-00352-asw-oct96-p41.pdfminer.txt"),
    "contrib/issue-00352-hash-twos-complement.pdf": stored_text(
        "issue-00352-hash-twos-complement.pdfminer.txt"
    ),
    "contrib/issue-1008-inline-ascii85.pdf": stored_text("issue-1008-inline-ascii85.pdfminer.txt"),
    "contrib/issue-1059-cmap-decode.pdf": stored_text("issue-1059-cmap-decode.pdfminer.txt"),
    "contrib/issue-1061-colour-space-stack.pdf": stored_text(
        "issue-1061-colour-space-stack.pdfminer.txt"
    ),
    "contrib/issue-1062-filters.pdf": stored_text("issue-1062-filters.pdfminer.txt"),
    "contrib/issue-1082-annotations.pdf": stored_text("issue-1082-annotations.pdfminer.txt"),
    "contrib/issue-1113-evil-xobjects.pdf": stored_text("issue-1113-evil-xobjects.pdfminer.txt"),
    "contrib/issue-449-horizontal.pdf": stored_text("issue-449-horizontal.pdfminer.txt"),
    "contrib/issue-449-vertical.pdf": stored_text("issue-449-vertical.pdfminer.txt"),
    "contrib/issue-598-cmap-other-fonts.pdf": stored_text(
        "issue-598-cmap-other-fonts.pdfminer.txt"
    ),
    "contrib/issue-625-identity-cmap.pdf": stored_text("issue-625-identity-cmap.pdfminer.txt"),
    "contrib/issue-791-non-unicode-cmap.pdf": stored_text(
        "issue-791-non-unicode-cmap.pdfminer.txt"
    ),
    "contrib/issue-886-xref-stream-widths.pdf": stored_text(
        "issue-886-xref-stream-widths.pdfminer.txt"
    ),
    "contrib/issue_495_pdfobjref.pdf": stored_text("issue_495_pdfobjref.pdfminer.txt"),
    "contrib/issue_566_test_1.pdf": stored_text("issue_566_test_1.pdfminer.txt"),
    "contrib/issue_566_test_2.pdf": stored_text("issue_566_test_2.pdfminer.txt"),
    "contrib/matplotlib.pdf": stored_text("matplotlib.pdfminer.txt"),
    "contrib/pagelabels.pdf": stored_text("pagelabels.pdfminer.txt"),
    "contrib/pr-00530-ml-lines.pdf": stored_text("pr-00530-ml-lines.pdfminer.txt"),
    "encryption/aes-128-m.pdf": stored_text("encryption-aes-128-m.pdfminer.txt"),
    "encryption/aes-128.pdf": stored_text("encryption-aes-128.pdfminer.txt"),
    "encryption/aes-256-m.pdf": stored_text("encryption-aes-256-m.pdfminer.txt"),
    "encryption/aes-256-r6.pdf": stored_text("encryption-aes-256-r6.pdfminer.txt"),
    "encryption/aes-256.pdf": stored_text("encryption-aes-256.pdfminer.txt"),
    "encryption/base.pdf": stored_text("encryption-base.pdfminer.txt"),
    "encryption/rc4-128.pdf": stored_text("encryption-rc4-128.pdfminer.txt"),
    "encryption/rc4-40.pdf": stored_text("encryption-rc4-40.pdfminer.txt"),
    "font-size-test.pdf": stored_text("font-size-test.pdfminer.txt"),
}

EXPECTED_CORE_TEXT = {
    "acroform/AcroForm_TEST.pdf": stored_text("acroform.core.txt"),
    "acroform/AcroForm_TEST_compiled.pdf": stored_text("acroform.core.txt"),
    "contrib/2b.pdf": stored_text("2b.core.txt"),
    "contrib/issue-00369-excel.pdf": stored_text("issue-00369-excel.core.txt"),
    "contrib/issue-00352-asw-oct96-p41.pdf": stored_text("issue-00352-asw-oct96-p41.core.txt"),
    "contrib/issue-00352-hash-twos-complement.pdf": stored_text(
        "issue-00352-hash-twos-complement.core.txt"
    ),
    "contrib/issue-1008-inline-ascii85.pdf": stored_text("issue-1008-inline-ascii85.core.txt"),
    "contrib/issue-1059-cmap-decode.pdf": stored_text("issue-1059-cmap-decode.core.txt"),
    "contrib/issue-1061-colour-space-stack.pdf": stored_text(
        "issue-1061-colour-space-stack.core.txt"
    ),
    "contrib/issue-1062-filters.pdf": stored_text("issue-1062-filters.core.txt"),
    "contrib/issue-1082-annotations.pdf": stored_text("issue-1082-annotations.core.txt"),
    "contrib/issue-1113-evil-xobjects.pdf": stored_text("issue-1113-evil-xobjects.core.txt"),
    "contrib/issue-1249-evil-xrefs.pdf": stored_text("issue-1249-evil-xrefs.core.txt"),
    "contrib/issue-449-horizontal.pdf": stored_text("issue-449-horizontal.core.txt"),
    "contrib/issue-449-vertical.pdf": stored_text("issue-449-vertical.core.txt"),
    "contrib/issue-598-cmap-other-fonts.pdf": stored_text("issue-598-cmap-other-fonts.core.txt"),
    "contrib/issue-625-identity-cmap.pdf": stored_text("issue-625-identity-cmap.core.txt"),
    "contrib/issue-791-non-unicode-cmap.pdf": stored_text("issue-791-non-unicode-cmap.core.txt"),
    "contrib/issue-886-xref-stream-widths.pdf": stored_text(
        "issue-886-xref-stream-widths.core.txt"
    ),
    "contrib/issue_495_pdfobjref.pdf": stored_text("issue_495_pdfobjref.core.txt"),
    "contrib/issue_566_test_1.pdf": stored_text("issue_566_test_1.core.txt"),
    "contrib/issue_566_test_2.pdf": stored_text("issue_566_test_2.core.txt"),
    "contrib/matplotlib.pdf": stored_text("matplotlib.core.txt"),
    "contrib/pagelabels.pdf": stored_text("pagelabels.core.txt"),
    "contrib/pr-00530-ml-lines.pdf": stored_text("pr-00530-ml-lines.core.txt"),
    "encryption/aes-128-m.pdf": stored_text("encryption-aes-128-m.core.txt"),
    "encryption/aes-128.pdf": stored_text("encryption-aes-128.core.txt"),
    "encryption/aes-256-m.pdf": stored_text("encryption-aes-256-m.core.txt"),
    "encryption/aes-256-r6.pdf": stored_text("encryption-aes-256-r6.core.txt"),
    "encryption/aes-256.pdf": stored_text("encryption-aes-256.core.txt"),
    "encryption/base.pdf": stored_text("encryption-base.core.txt"),
    "encryption/rc4-128.pdf": stored_text("encryption-rc4-128.core.txt"),
    "encryption/rc4-40.pdf": stored_text("encryption-rc4-40.core.txt"),
    "font-size-test.pdf": stored_text("font-size-test.core.txt"),
    "jo.pdf": stored_text("jo.core.txt"),
    "nonfree/175.pdf": stored_text("nonfree-175.core.txt"),
    "nonfree/cmp_itext_logo.pdf": stored_text("cmp_itext_logo.core.txt"),
    "nonfree/dmca.pdf": stored_text("dmca.core.txt"),
    "nonfree/f1040nr.pdf": stored_text("f1040nr.core.txt"),
    "nonfree/i1040nr.pdf": stored_text("i1040nr.core.txt"),
    "nonfree/kampo.pdf": stored_text("kampo.core.txt"),
    "nonfree/naacl06-shinyama.pdf": stored_text("naacl06-shinyama.core.txt"),
    "nonfree/nlp2004slides.pdf": stored_text("nlp2004slides.core.txt"),
    "sampleOneByteIdentityEncode.pdf": stored_text("sampleOneByteIdentityEncode.core.txt"),
    "simple1.pdf": stored_text("simple1.core.txt"),
    "simple3.pdf": stored_text("simple3.core.txt"),
    "simple4.pdf": stored_text("simple4.core.txt"),
    "simple5.pdf": stored_text("simple5.core.txt"),
    "zen_of_python_corrupted.pdf": stored_text("zen_of_python_corrupted.core.txt"),
}


def core_extract_text(source: PdfSource, *, password: str = "") -> str:
    with PdfDocument.open(source, password=password) as document:
        pages = []
        for page in cast(Any, document).pages:
            pages.append(
                "\n".join(
                    str(line.get("text") or "").replace("\r", "\n") for line in page.extract_lines()
                )
            )
        return "\f".join(pages) + "\f"


def sample_id(path: Path) -> str:
    return path.relative_to(SAMPLES_DIR).as_posix()


def pdfminer_extract_text(pdf_path: Path, *, password: str = "") -> str:
    try:
        text = pdfminer_six_extract_text(pdf_path, password=password, caching=False)
    except Exception as exc:
        raise PdfminerExtractionError(type(exc).__name__) from exc
    if not isinstance(text, str):
        raise TypeError("invalid pdfminer text result")
    return text


class PdfminerExtractionError(Exception):
    def __init__(self, exc_type: str) -> None:
        super().__init__(exc_type)
        self.exc_type = exc_type


@pytest.mark.parametrize("pdf_path", PDF_SAMPLES, ids=sample_id)
def test_extract_text_handles_pdfminer_sample_corpus(pdf_path: Path) -> None:
    sample = sample_id(pdf_path)
    password = SAMPLE_PASSWORDS.get(sample, "")

    try:
        expected_pdfminer_six = pdfminer_extract_text(pdf_path, password=password)
    except PdfminerExtractionError as exc:
        if exc.exc_type != EXPECTED_PDFMINER_FAILURES.get(sample):
            raise AssertionError(f"Unexpected pdfminer.six failure for sample: {sample}") from exc
        expected_pdfminer_six = None
    else:
        assert sample not in EXPECTED_PDFMINER_FAILURES, (
            f"pdfminer.six unexpectedly succeeded for sample: {sample}"
        )
        if sample in EXPECTED_PDFMINER_TEXT:
            assert comparable_pdfminer_text(expected_pdfminer_six) == comparable_pdfminer_text(
                EXPECTED_PDFMINER_TEXT[sample]
            ), f"pdfminer.six extraction changed for sample: {sample}"

    try:
        result = core_extract_text(pdf_path, password=password)
    except Exception as exc:
        if type(exc).__name__ != EXPECTED_CORE_FAILURES.get(sample):
            raise AssertionError(f"Unexpected core-pdf failure for sample: {sample}") from exc
        return
    assert sample not in EXPECTED_CORE_FAILURES, (
        f"core-pdf unexpectedly succeeded for sample: {sample}"
    )

    expected_core_pdf = EXPECTED_CORE_TEXT.get(sample, expected_pdfminer_six)
    if expected_core_pdf is None:
        raise AssertionError(f"Missing core-pdf expectation for sample: {sample}")

    assert result == expected_core_pdf, f"Failed for sample: {sample}"

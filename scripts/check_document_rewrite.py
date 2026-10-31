"""Check that rewriting each upstream PDF keeps its pages, text and an edit.

uv run --all-packages python scripts/check_document_rewrite.py
"""

import collections
import glob
import sys
from typing import Any

from core_pdf import EditableDocument, PdfDocument
from core_pdf.impl.exceptions import PdfError, PdfUnsupportedError
from core_pdf.impl.types import PdfString

PATTERNS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/*.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/*.pdf",
    "tests/upstream/pypdf/resources/*.pdf",
)
MAX_PAGES = 5


def page_texts(document: Any) -> list[str]:
    return [
        "\n".join(block.text for block in page.extract().blocks)
        for page in document.pages[:MAX_PAGES]
    ]


def check(path: str, counts: collections.Counter[str]) -> None:
    with PdfDocument.open(path) as source:
        expected_pages = len(source.pages)
        expected_text = page_texts(source)
        editor = EditableDocument(source)
        info = editor.add_object({"Title": PdfString(b"Rewritten")})
        editor.trailer["Info"] = info
        output = editor.save()
    with PdfDocument.open(output) as result:
        if len(result.pages) != expected_pages:
            counts["page_mismatch"] += 1
            print(f"pages differ: {path}")
        elif page_texts(result) != expected_text:
            counts["text_mismatch"] += 1
            print(f"text differs: {path}")
        elif result.get_metadata()["info"].get("Title") != "Rewritten":
            counts["edit_lost"] += 1
            print(f"edit lost: {path}")
        else:
            counts["identical"] += 1


def main() -> int:
    counts: collections.Counter[str] = collections.Counter()
    for pattern in PATTERNS:
        for path in sorted(glob.glob(pattern)):  # noqa: PTH207
            try:
                check(path, counts)
            except PdfUnsupportedError:
                counts["unsupported"] += 1
            except (PdfError, RecursionError, ValueError, KeyError, OSError) as error:
                counts["errors"] += 1
                print(f"error: {path}: {type(error).__name__}: {str(error)[:70]}")
    print(dict(counts))
    return 1 if set(counts) - {"identical", "unsupported", "errors"} else 0


if __name__ == "__main__":
    sys.exit(main())

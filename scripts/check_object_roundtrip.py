"""Check that every reachable object in the upstream PDFs survives serialize and reparse.

uv run --all-packages python scripts/check_object_roundtrip.py
"""

import collections
import glob
import sys
from typing import Any

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.serialize import serialize_object
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.types import PdfReference

PATTERNS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/*.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/*.pdf",
    "tests/upstream/pypdf/resources/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)


def normalized(value: Any) -> Any:
    if isinstance(value, PdfStream):
        data = bytes(value.raw_data)
        return normalized(dict(value.dictionary)) | {"Length": len(data), "<data>": data}
    if isinstance(value, dict):
        return {str(getattr(key, "value", key)): normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    if isinstance(value, str):
        return str(value)
    if hasattr(value, "value_bytes"):
        return value.value
    return value


def reparsed(value: Any) -> Any:
    data = serialize_object(value)
    if isinstance(value, PdfStream):
        return PdfLexer(b"1 0 obj\n" + data + b"\nendobj\n").parse_indirect_object()
    return PdfLexer(data + b" ").parse_object()


def check_document(path: str, counts: collections.Counter[str]) -> None:
    with PdfDocument.open(path) as document:
        seen: set[int] = set()
        pending: list[Any] = [document.trailer_dict]
        while pending:
            value = pending.pop()
            if isinstance(value, PdfReference):
                if value.object_number in seen:
                    continue
                seen.add(value.object_number)
                resolved = document.resolver.resolve(value)
                counts["objects"] += 1
                if normalized(reparsed(resolved)) != normalized(resolved):
                    counts["mismatches"] += 1
                    print(f"mismatch: {path} object {value.object_number}")
                pending.append(resolved)
            elif isinstance(value, PdfStream):
                pending.extend(value.dictionary.values())
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)


def main() -> int:
    counts: collections.Counter[str] = collections.Counter()
    for pattern in PATTERNS:
        for path in glob.glob(pattern, recursive=True):  # noqa: PTH207
            try:
                check_document(path, counts)
            except PdfError, RecursionError, ValueError, OSError:
                # Some fixtures are deliberately corrupt, encrypted or empty.
                counts["unreadable"] += 1
    print(dict(counts))
    return 1 if counts["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())

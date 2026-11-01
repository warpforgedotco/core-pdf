"""Check that every reachable object in the upstream PDFs survives serialize and reparse.

uv run --all-packages python scripts/check_object_roundtrip.py

With --xref-stream, each file is saved with an xref stream instead, and every object is compared
with the one read back from the saved file. --incremental saves an xref-stream update.
"""

import collections
import glob
import sys
from typing import Any

from core_pdf import EditableDocument, PdfDocument
from core_pdf.impl.exceptions import PdfError, PdfUnsupportedError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.serialize import serialize_object
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.types import PdfReference

XREF_STREAM = "--xref-stream" in sys.argv
INCREMENTAL = "--incremental" in sys.argv
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


def saved_with_xref_stream(document: Any) -> bytes:
    editor = EditableDocument(document)
    if INCREMENTAL:
        if "Root" in editor.trailer:
            editor.edit_object(editor.trailer["Root"])
        return editor.save_incremental(xref_stream=True)
    return editor.save(xref_stream=True)


def check_document(path: str, counts: collections.Counter[str]) -> None:
    with PdfDocument.open(path) as document:
        if XREF_STREAM:
            with PdfDocument.open(saved_with_xref_stream(document)) as saved:
                walk(path, document, saved, counts)
        else:
            walk(path, document, None, counts)


def walk(path: str, document: Any, saved: Any, counts: collections.Counter[str]) -> None:
    start = document.trailer_dict
    if saved is not None:
        # A save keeps what these reach, and drops any other reference in the old trailer.
        start = {key: start.get(key) for key in ("Root", "Info", "ID")}
    seen: set[int] = set()
    pending: list[Any] = [start]
    while pending:
        value = pending.pop()
        if isinstance(value, PdfReference):
            if value.object_number in seen:
                continue
            seen.add(value.object_number)
            resolved = document.resolver.resolve(value)
            counts["objects"] += 1
            other = reparsed(resolved) if saved is None else saved.resolver.resolve(value)
            if normalized(other) != normalized(resolved):
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
            except PdfUnsupportedError:
                counts["unsupported" if XREF_STREAM else "unreadable"] += 1
            except PdfError, RecursionError, ValueError, OSError:
                # Some fixtures are deliberately corrupt, encrypted or empty.
                counts["unreadable"] += 1
    print(dict(counts))
    return 1 if counts["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())

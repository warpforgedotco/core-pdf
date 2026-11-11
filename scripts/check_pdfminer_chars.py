# SPDX-License-Identifier: AGPL-3.0-only

"""Check that the compiled LTChars are exactly what pdfminer's Python projection makes.

core_pdf_compat.pdfminer's project_page makes the chars of plain horizontal text with
core-pdf-cythonized's pdfminer_chars, and the Python loop beside it makes the rest. This
lays out every page of the upstream test PDFs both ways, the second with the kernel leaving
every glyph to the loop, and compares the whole layout tree: each item's type, its box to
the bit, and a char's text, font name and size. It fails on the first page that differs, or
if the kernel made no char.

uv run --all-packages python scripts/check_pdfminer_chars.py [pages per file]
"""

from __future__ import annotations

import glob
import logging
import struct
import sys
import warnings
from typing import Any

from core_pdf_compat.pdfminer import _projection
from core_pdf_compat.pdfminer.high_level import extract_pages

CORPUS = (
    "tests/upstream/pdfminer.six/samples/**/*.pdf",
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)


def exact(value: object) -> object:
    # Packed, so even NaNs whose payloads differ compare unequal.
    if type(value) is float:
        return struct.pack(">d", value)
    if isinstance(value, (tuple, list)):
        return tuple(exact(item) for item in value)
    return (type(value).__name__, repr(value))


def tree(item: Any) -> object:
    node: list[object] = [type(item).__name__, exact(getattr(item, "bbox", None))]
    if type(item).__name__ == "LTChar":
        node += [item.get_text(), item.fontname, exact(item.size)]
    if hasattr(type(item), "__iter__"):
        node.append(tuple(tree(child) for child in item))
    return tuple(node)


def layouts(path: str, pages: int) -> list[object]:
    try:
        return [tree(page) for page in extract_pages(path, maxpages=pages)]
    except Exception as error:  # noqa: BLE001
        return [("raised", type(error).__name__, str(error))]


def main(argv: list[str]) -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    pages_per_file = int(argv[0]) if argv else 10
    paths = sorted({path for pattern in CORPUS for path in glob.glob(pattern, recursive=True)})
    compiled = _projection.pdfminer_chars
    written = 0
    marker = _projection.no_char()

    def counted(glyphs: Any, *args: Any) -> list[Any]:
        nonlocal written
        chars = compiled(glyphs, *args)
        written += sum(1 for char in chars if char is not None and char is not marker)
        return chars

    def leaving(glyphs: Any, *_args: Any) -> list[Any]:
        return [None] * len(glyphs)

    differences = []
    pages = 0
    for path in paths:
        _projection.pdfminer_chars = counted  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        try:
            as_compiled = layouts(path, pages_per_file)
        finally:
            _projection.pdfminer_chars = leaving  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        try:
            by_python = layouts(path, pages_per_file)
        finally:
            _projection.pdfminer_chars = compiled
        pages += len(as_compiled)
        if as_compiled != by_python:
            differences.append(path)
    print(
        f"{pages} pages of {len(paths)} files, {written} chars made by the kernel:",
        f"{len(differences)} differ",
    )
    for difference in differences[:10]:
        print(difference)
    return 1 if differences or not written else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

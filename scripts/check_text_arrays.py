# SPDX-License-Identifier: AGPL-3.0-only

"""Check that text shown by the compiled TJ loop captures exactly what the interpreter's does.

A capture state shows a TJ array, and a Tj string as an array of it, with
core-pdf-cythonized's show_text_array, which inlines the content interpreter's
append_tj_array, append_text and append_decoded_text and the state's show_text and
text_boundary, and hands the state's own methods any string it cannot show exactly. This
captures every page of the upstream test PDFs both ways, the second with the interpreter's
methods, and compares the captured runs, glyphs, drawings and
text boundaries by repr, which writes each float's shortest round-tripping digits and an
int as an int, object addresses left out. It fails on the first page that differs, or if
nothing took the loop.

uv run --all-packages python scripts/check_text_arrays.py [pages per file]
"""

from __future__ import annotations

import contextlib
import glob
import logging
import os
import re
import sys
import warnings
from collections.abc import Generator
from typing import Any

from core_pdf import PdfDocument
from core_pdf.impl.capture import text_painting
from core_pdf.impl.capture.recording import TextState

CORPUS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/**/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)
MAX_BYTES = 8_000_000
# The address in the repr of an object without fields of its own, such as a captured path.
ADDRESS = re.compile(r" at 0x[0-9a-f]+")


@contextlib.contextmanager
def interpreter_loop() -> Generator[None]:
    """Show every TJ array and Tj string with the content interpreter's methods."""
    compiled = TextState.show_compiled_text_array
    TextState.show_compiled_text_array = declined  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        TextState.show_compiled_text_array = compiled  # type: ignore[method-assign]


def declined(state: TextState, array: Any) -> bool:
    return False


def captured(path: str, page_count: int) -> list[str]:
    """Each page's captured runs, glyphs, drawings and text boundaries, or its error."""
    pages: list[str] = []
    try:
        with PdfDocument(path) as document:
            for page in document.pages[:page_count]:
                try:
                    program = page.get_page_program()
                except Exception as error:  # noqa: BLE001
                    pages.append(f"error {type(error).__name__}: {error}")
                    continue
                state = (program.runs, program.glyphs, program.drawings, program.text_boundaries)
                pages.append(ADDRESS.sub("", repr(state)))
    except Exception as error:  # noqa: BLE001
        pages.append(f"error {type(error).__name__}: {error}")
    return pages


def main() -> int:
    page_count = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    taken = 0
    compiled = text_painting.show_text_array

    def counting(state: Any, array: Any, decoder: Any, context: tuple[Any, ...]) -> bool:
        nonlocal taken
        shown = compiled(state, array, decoder, context)
        taken += shown
        return shown

    paths = sorted(
        {
            path
            for pattern in CORPUS
            for path in glob.glob(pattern, recursive=True)
            if os.path.getsize(path) <= MAX_BYTES
        }
    )
    pages = 0
    for path in paths:
        text_painting.show_text_array = counting  # ty: ignore[invalid-assignment]
        try:
            loop = captured(path, page_count)
        finally:
            text_painting.show_text_array = compiled
        with interpreter_loop():
            reference = captured(path, page_count)
        for number, (got, expected) in enumerate(zip(loop, reference, strict=True), 1):
            if got != expected:
                print(f"{path} page {number}: the compiled TJ loop captured something else")
                return 1
        pages += len(loop)
    if not taken:
        print("no text took the compiled loop")
        return 1
    print(f"{pages} pages of {len(paths)} files, {taken} shown by the loop: 0 differ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

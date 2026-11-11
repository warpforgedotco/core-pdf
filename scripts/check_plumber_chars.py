# SPDX-License-Identifier: AGPL-3.0-only

"""Check that the compiled char dicts are exactly what pdfplumber's Python char loop writes.

core_pdf_compat.pdfplumber writes a page's chars with core-pdf-cythonized's
plumber_char_dicts, and with the Python loop beside it for a page the kernel declines (one
with vertical text, or a number that is not a float). This reads the chars of every page of
the upstream test PDFs both ways, the second with the kernel declining, and compares every
dict: its keys in order, and each value's type and, for floats, its bits. It fails on the
first page that differs, or if no page took the kernel.

uv run --all-packages python scripts/check_plumber_chars.py [pages per file]
"""

from __future__ import annotations

import contextlib
import glob
import logging
import struct
import sys
import warnings
from collections.abc import Generator
from typing import Any

from core_pdf_compat import pdfplumber

CORPUS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/**/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)


def exact(value: object) -> object:
    """A key equal only for values of the same type and the same bits, NaN payloads too."""
    if type(value) is float:
        return ("float", struct.pack(">d", value))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(exact(item) for item in value))
    return (type(value).__name__, repr(value))


def page_chars(path: str, pages: int) -> list[object]:
    out: list[object] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:pages]:
                try:
                    out.append(
                        tuple(
                            tuple((key, exact(value)) for key, value in char.items())
                            for char in page.chars
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    out.append(("raised", type(error).__name__, str(error)))
    except Exception as error:  # noqa: BLE001
        out.append(("open", type(error).__name__))
    return out


@contextlib.contextmanager
def declining() -> Generator[None]:
    compiled = pdfplumber.plumber_char_dicts
    pdfplumber.plumber_char_dicts = lambda *_args: None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        pdfplumber.plumber_char_dicts = compiled


def main(argv: list[str]) -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    pages_per_file = int(argv[0]) if argv else 10
    paths = sorted({path for pattern in CORPUS for path in glob.glob(pattern, recursive=True)})
    taken = 0
    compiled = pdfplumber.plumber_char_dicts

    def counted(*args: Any) -> Any:
        nonlocal taken
        result = compiled(*args)
        taken += result is not None
        return result

    differences = []
    pages = 0
    for path in paths:
        pdfplumber.plumber_char_dicts = counted  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        try:
            as_compiled = page_chars(path, pages_per_file)
        finally:
            pdfplumber.plumber_char_dicts = compiled
        with declining():
            by_python = page_chars(path, pages_per_file)
        pages += len(as_compiled)
        differences.extend(
            f"{path} page {number + 1}"
            for number, (left, right) in enumerate(zip(as_compiled, by_python, strict=True))
            if left != right
        )
    print(
        f"{pages} pages of {len(paths)} files, {taken} written by the kernel:",
        f"{len(differences)} differ",
    )
    for difference in differences[:10]:
        print(difference)
    return 1 if differences or not taken else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

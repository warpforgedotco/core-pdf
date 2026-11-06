"""Check that the pypdf facade writes the same bytes as pypdf, on the upstream PDFs.

uv run --all-packages --with "pypdf[full]==7.0.0" python scripts/check_pypdf_bytes.py

Each file goes through the real pypdf and the facade the same way, and the outputs are
compared byte for byte:

incremental: PdfWriter(incremental=True), a new /Title, write.
clone: PdfWriter(clone_from=...), write.
append: PdfWriter(), append(reader), write.

Every difference fails the check unless DEPARTURES lists it, with the reason the facade does
not follow pypdf there. A listed file that matches fails it too, so the list stays true.
"""

import collections
import glob
import io
import logging
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pypdf as real  # ty: ignore[unresolved-import]

from core_pdf_compat import pypdf as facade

PATTERNS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/*.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/*.pdf",
    "tests/upstream/pypdf/resources/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)
DEPARTURES: dict[tuple[str, str], str] = {}


def incremental(library: Any, data: bytes) -> bytes:
    writer = library.PdfWriter(io.BytesIO(data), incremental=True)
    writer.add_metadata({"/Title": "x"})
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def clone(library: Any, data: bytes) -> bytes:
    writer = library.PdfWriter(clone_from=io.BytesIO(data))
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def append(library: Any, data: bytes) -> bytes:
    writer = library.PdfWriter()
    writer.append(library.PdfReader(io.BytesIO(data)))
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


MODES: dict[str, Callable[[Any, bytes], bytes]] = {
    "incremental": incremental,
    "clone": clone,
    "append": append,
}


def first_difference(expected: bytes, actual: bytes) -> str:
    at = next(
        (i for i, (x, y) in enumerate(zip(expected, actual, strict=False)) if x != y),
        min(len(expected), len(actual)),
    )
    return f"at byte {at}: pypdf {expected[at : at + 40]!r}, facade {actual[at : at + 40]!r}"


def main() -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    counts: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    paths = sorted({path for pattern in PATTERNS for path in glob.glob(pattern, recursive=True)})
    for path in paths:
        data = open(path, "rb").read()  # noqa: SIM115, PTH123
        for name, write in MODES.items():
            try:
                expected = write(real, data)
            except Exception:  # noqa: BLE001
                counts[f"{name}: pypdf cannot write"] += 1
                continue
            try:
                actual = write(facade, data)
            except Exception as error:  # noqa: BLE001
                counts[f"{name}: FACADE FAILS"] += 1
                problems.append(f"{name} {path}: {error!r}"[:200])
                continue
            listed = DEPARTURES.get((name, Path(path).name))
            if expected == actual:
                counts[f"{name}: identical"] += 1
                if listed is not None:
                    counts[f"{name}: LISTED BUT IDENTICAL"] += 1
                    problems.append(f"{name} {path}: listed as a departure but identical")
            elif listed is not None:
                counts[f"{name}: differ, listed departure"] += 1
            else:
                counts[f"{name}: DIFFER"] += 1
                problems.append(f"{name} {path} {first_difference(expected, actual)}"[:300])
    for key, count in sorted(counts.items()):
        print(f"{count:>6}  {key}")
    for problem in problems:
        print(problem)
    failed = any(key.split(": ", 1)[1].isupper() for key in counts)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

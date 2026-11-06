r"""Check that the unstructured facade partitions PDFs as unstructured does, on the upstream PDFs.

uv run --all-packages --with "unstructured[pdf]==0.27.36" \
    python scripts/check_unstructured_output.py

Each file goes through the real unstructured and the facade with
partition_pdf(filename, strategy="fast"), and the elements are compared one by one: class
name, category, text, and the metadata fields page_number, coordinates (points rounded to
COORDINATE_DIGITS, system, layout size), languages, links, element_id, parent_id, filename,
file_directory and filetype. A file that raises must raise the same exception type on both.
The first difference of each file is printed.

Files with more than MAX_PAGES (30) pages are skipped to keep the run to about two minutes;
the count of skipped files is printed. There are seven, the largest DA-619p.pdf, and each
matched when last run without the cap.

The element lists are compared, not metadata.to_dict(), whose layout the facade does not
follow (it keeps bbox, and coordinates as a tuple with width and height).

Every difference fails the check unless DEPARTURES lists it, with the reason the facade does
not follow unstructured there. A listed file that matches fails it too, so the list stays true.
"""

import collections
import glob
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

from unstructured.partition.pdf import (  # ty: ignore[unresolved-import]
    partition_pdf as real_partition_pdf,
)

from core_pdf import PdfDocument
from core_pdf_compat.unstructured import partition_pdf as facade_partition_pdf

PATTERNS = (
    "tests/upstream/unstructured/example-docs/**/*.pdf",
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/*.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/*.pdf",
    "tests/upstream/pypdf/resources/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)
MAX_PAGES = 30
# Six digits hide float noise such as 271.39500000000004 against 271.395.
COORDINATE_DIGITS = 6
# Each of these would come from the pdfminer facade's layout, handing the unstructured facade
# different text boxes or pages than pdfminer.six 20260107 hands unstructured; none does now.
DEPARTURES: dict[str, str] = {}


def page_count(path: str) -> int:
    try:
        with PdfDocument.open(path) as document:
            return len(document.pages)
    except Exception:  # noqa: BLE001
        return 0


def coordinates(metadata: Any) -> object:
    value = metadata.coordinates
    if value is None:
        return None
    system = value.system
    # unstructured keeps the layout size on a CoordinateSystem, the facade on the tuple.
    width = getattr(system, "width", None) if not isinstance(system, str) else value.width
    height = getattr(system, "height", None) if not isinstance(system, str) else value.height
    name = system if isinstance(system, str) else type(system).__name__
    points = tuple(
        (round(x, COORDINATE_DIGITS), round(y, COORDINATE_DIGITS)) for x, y in value.points
    )
    return (points, name, width, height)


def describe(element: Any) -> dict[str, object]:
    metadata = element.metadata
    return {
        "class": type(element).__name__,
        "category": element.category,
        "text": element.text,
        "page_number": metadata.page_number,
        "coordinates": coordinates(metadata),
        "languages": metadata.languages,
        "links": metadata.links,
        "element_id": element.id,
        "parent_id": metadata.parent_id,
        "filename": metadata.filename,
        "file_directory": metadata.file_directory,
        "filetype": metadata.filetype,
    }


def partition(function: Any, path: str) -> list[dict[str, object]] | str:
    try:
        return [describe(element) for element in function(path, strategy="fast")]
    except Exception as error:  # noqa: BLE001
        return f"raises {type(error).__name__}"


def first_difference(
    expected: list[dict[str, object]] | str, actual: list[dict[str, object]] | str
) -> str:
    if isinstance(expected, str) or isinstance(actual, str):
        summary = lambda value: value if isinstance(value, str) else f"{len(value)} elements"  # noqa: E731
        return f"unstructured {summary(expected)}, facade {summary(actual)}"
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        for key, value in left.items():
            if value != right[key]:
                return (
                    f"element {index} {key}: unstructured {value!r}, facade {right[key]!r}"
                    f" (text {left['text']!r:.60})"
                )
    return f"unstructured {len(expected)} elements, facade {len(actual)}"


def main() -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    counts: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    paths = sorted({path for pattern in PATTERNS for path in glob.glob(pattern, recursive=True)})
    for path in paths:
        if page_count(path) > MAX_PAGES:
            counts[f"skipped, over {MAX_PAGES} pages"] += 1
            continue
        expected = partition(real_partition_pdf, path)
        actual = partition(facade_partition_pdf, path)
        listed = DEPARTURES.get(Path(path).name)
        if expected == actual:
            counts["identical"] += 1
            if listed is not None:
                counts["LISTED BUT IDENTICAL"] += 1
                problems.append(f"{path}: listed as a departure but identical")
        elif listed is not None:
            counts["differ, listed departure"] += 1
            print(f"listed {path}: {first_difference(expected, actual)}"[:300])
        else:
            counts["DIFFER"] += 1
            problems.append(f"{path}: {first_difference(expected, actual)}"[:400])
    for key, count in sorted(counts.items()):
        print(f"{count:>6}  {key}")
    for problem in problems:
        print(problem)
    failed = any(key.isupper() for key in counts)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

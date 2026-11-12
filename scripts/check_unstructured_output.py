r"""Check that the unstructured facade partitions PDFs as unstructured does, on the upstream PDFs.

unstructured[pdf] needs onnxruntime, which has no CPython 3.15 wheels, and the workspace needs
3.15, so the real library and the facade run in separate interpreters, in three steps:

uv run --all-packages python scripts/check_unstructured_output.py --list files.txt
uv run --python 3.14 --isolated --no-project --with "unstructured[pdf]==0.27.37" \
    python scripts/check_unstructured_output.py --record files.txt real.json
uv run --all-packages --extra unstructured python scripts/check_unstructured_output.py \
    --compare files.txt real.json

--list writes the files to compare, --record runs the real unstructured on them and keeps its
elements, and --compare runs the facade and compares. --record imports nothing from the
workspace.

Each file goes through partition_pdf(filename, strategy="fast"), and the elements are
compared one by one: class name, category, text, and the metadata fields page_number,
coordinates (points rounded to COORDINATE_DIGITS, system, layout size), languages, links,
element_id, parent_id, filename, file_directory and filetype. A file that raises must raise
the same exception type and message on both. The first difference of each file is printed.
Both sides go through the same JSON round trip, so a tuple and a list of the same items match.

The files are every PDF under tests/upstream but pypdf's downloads in pdf_cache, which CI
does not fetch for this check; symbolic links are left out, as each names a file covered
under its own name or none. Files with more than MAX_PAGES (30) pages are skipped to keep
the run short; the count of skipped files is printed.

The element lists are compared, not metadata.to_dict(), whose layout the facade does not
follow (it keeps bbox, and coordinates as a tuple with width and height).

Every difference fails the check unless DEPARTURES lists it, with the reason the facade does
not follow unstructured there. A listed file that matches fails it too, so the list stays true.
"""

import argparse
import collections
import glob
import json
import logging
import os
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Every PDF in the upstream checkouts but pypdf's downloads, which CI does not fetch for this
# check. Symbolic links are left out: each points at a file covered under its own name, or at
# none.
PATTERN = "tests/upstream/**/*.pdf"
DOWNLOADS = "tests/upstream/pypdf/tests/pdf_cache/"
MAX_PAGES = 30
# Six digits hide float noise such as 271.39500000000004 against 271.395.
COORDINATE_DIGITS = 6
# Each of these would come from the pdfminer facade's layout, handing the unstructured facade
# different text boxes or pages than pdfminer.six 20260107 hands unstructured; none does now.
DEPARTURES: dict[str, str] = {}


def page_count(path: str) -> int:
    from core_pdf import PdfDocument  # noqa: PLC0415  # the workspace is not in --record's

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


Result = list[dict[str, Any]] | str


def partition(function: Callable[..., Any], path: str) -> Result:
    try:
        elements: Result = [describe(element) for element in function(path, strategy="fast")]
    except Exception as error:  # noqa: BLE001
        elements = f"raises {type(error).__name__}: {error}"
    # The real side's elements come back from JSON; both go through it alike.
    return json.loads(json.dumps(elements, default=repr))


def first_difference(expected: Result, actual: Result) -> str:
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


def list_files(out: Path) -> None:
    counts: collections.Counter[str] = collections.Counter()
    paths = sorted(
        path
        for path in glob.glob(PATTERN, recursive=True)
        if not path.startswith(DOWNLOADS) and not os.path.islink(path)
    )
    kept = []
    for path in paths:
        if page_count(path) > MAX_PAGES:
            counts[f"skipped, over {MAX_PAGES} pages"] += 1
        else:
            kept.append(path)
    out.write_text("".join(f"{path}\n" for path in kept))
    for key, count in sorted(counts.items()):
        print(f"{count:>6}  {key}")
    print(f"{len(kept):>6}  to compare, listed in {out}")


def record(files: Path, out: Path) -> None:
    from unstructured.partition.pdf import partition_pdf  # noqa: PLC0415  # ty: ignore[unresolved-import]

    results = {path: partition(partition_pdf, path) for path in files.read_text().splitlines()}
    out.write_text(json.dumps(results))
    print(f"{len(results):>6}  files partitioned by unstructured, written to {out}")


def compare(files: Path, recorded: Path) -> int:
    from core_pdf_compat.unstructured import partition_pdf  # noqa: PLC0415

    expected_by_path = json.loads(recorded.read_text())
    counts: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    for path in files.read_text().splitlines():
        if path not in expected_by_path:
            counts["NOT RECORDED"] += 1
            problems.append(f"{path}: missing from {recorded}")
            continue
        expected = expected_by_path[path]
        actual = partition(partition_pdf, path)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    steps = parser.add_mutually_exclusive_group(required=True)
    steps.add_argument("--list", type=Path, metavar="FILES", help="write the files to compare")
    steps.add_argument("--record", type=Path, nargs=2, metavar=("FILES", "OUT"))
    steps.add_argument("--compare", type=Path, nargs=2, metavar=("FILES", "RECORDED"))
    arguments = parser.parse_args()
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    if arguments.list:
        list_files(arguments.list)
        return 0
    if arguments.record:
        record(*arguments.record)
        return 0
    return compare(*arguments.compare)


if __name__ == "__main__":
    sys.exit(main())

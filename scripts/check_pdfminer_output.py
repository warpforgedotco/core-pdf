"""Check that the pdfminer.six facade extracts what pdfminer.six does, on the upstream PDFs.

uv run --all-packages --with "pdfminer.six==20260107" python scripts/check_pdfminer_output.py

Each file goes through the real pdfminer and the facade the same way, with default LAParams
and at most MAX_PAGES pages, and two things are compared:

text: pdfminer.high_level.extract_text, page by page (split on the form feed).
layout: the tree extract_pages yields, one record per object, depth first: the class name
    and bbox of every LTTextBox, LTTextLine, LTFigure, LTImage, LTRect, LTLine and LTCurve,
    and for every LTChar its text, fontname, size and bbox. LTAnno records carry their text.
    Numbers match when they are within TOLERANCE of each other.

When one side raises, the other must raise an exception of the same class name, at the same
page. The first difference in a file is printed.

Every difference fails the check unless DEPARTURES lists it, with the reason the facade does
not follow pdfminer there. A listed file that matches fails it too, so the list stays true.
Pass file names to check only those.
"""

import collections
import glob
import logging
import math
import signal
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pdfminer.high_level as real_high_level  # ty: ignore[unresolved-import]
import pdfminer.layout as real_layout  # ty: ignore[unresolved-import]

import core_pdf_compat.pdfminer as facade_high_level
from core_pdf_compat.pdfminer import layout as facade_layout

PATTERNS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/*.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/*.pdf",
    "tests/upstream/pypdf/resources/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)
MAX_PAGES = 5
TOLERANCE = 1e-3
SECONDS_PER_FILE = 120
# (check, file name) -> why the facade does not follow pdfminer there.
DEPARTURES: dict[tuple[str, str], str] = {}

Record = tuple[Any, ...]


class Timeout(Exception):
    pass


def records(item: Any, depth: int, layout: Any) -> Iterator[Record]:
    name = type(item).__name__
    if isinstance(item, layout.LTChar):
        yield (depth, name, item.get_text(), item.fontname, item.size, *item.bbox)
        return
    if isinstance(item, layout.LTAnno):
        yield (depth, name, item.get_text())
        return
    yield (depth, name, *getattr(item, "bbox", ()))
    if hasattr(type(item), "__iter__"):
        for child in item:
            yield from records(child, depth + 1, layout)


def layout_pages(module: Any, layout: Any, path: str) -> list[list[Record] | str]:
    """Each page's records, and the exception class name last if extraction raised."""
    pages: list[list[Record] | str] = []
    try:
        for page in module.extract_pages(path, maxpages=MAX_PAGES):
            pages.append(list(records(page, 0, layout)))
    except Timeout:
        raise
    except Exception as error:  # noqa: BLE001
        pages.append(type(error).__name__)
    return pages


def text_pages(module: Any, path: str) -> list[str]:
    try:
        text = module.extract_text(path, maxpages=MAX_PAGES)
    except Timeout:
        raise
    except Exception as error:  # noqa: BLE001
        return [f"<raised {type(error).__name__}>"]
    return text.split("\f")


def same_value(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float | int) and isinstance(actual, float | int):
        return math.isclose(expected, actual, rel_tol=0.0, abs_tol=TOLERANCE)
    return bool(expected == actual)


def same_record(expected: Record, actual: Record) -> bool:
    return len(expected) == len(actual) and all(map(same_value, expected, actual))


def first_layout_difference(expected: list[Record] | str, actual: list[Record] | str) -> str | None:
    if isinstance(expected, str) or isinstance(actual, str):
        return None if expected == actual else f"pdfminer {short(expected)}, facade {short(actual)}"
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if not same_record(left, right):
            return f"record {index}: pdfminer {short(left)}, facade {short(right)}"
    if len(expected) != len(actual):
        at = min(len(expected), len(actual))
        extra = expected[at] if len(expected) > at else actual[at]
        side = "pdfminer" if len(expected) > at else "facade"
        return f"{len(expected)} vs {len(actual)} records; {side} has more, first {short(extra)}"
    return None


def first_text_difference(expected: str, actual: str) -> str | None:
    if expected == actual:
        return None
    at = next(
        (i for i, (x, y) in enumerate(zip(expected, actual, strict=False)) if x != y),
        min(len(expected), len(actual)),
    )
    start = max(at - 10, 0)
    return f"at char {at}: pdfminer {expected[start : at + 30]!r}, facade {actual[start : at + 30]!r}"


def short(value: Any) -> str:
    if isinstance(value, tuple):
        value = tuple(round(item, 3) if isinstance(item, float) else item for item in value)
    return repr(value)[:160]


def compare_pages(
    expected: list[Any], actual: list[Any], differ: Callable[[Any, Any], str | None]
) -> tuple[int, int, str | None]:
    """Identical pages, different pages, and the first difference."""
    same = different = 0
    first: str | None = None
    for index in range(max(len(expected), len(actual))):
        left = expected[index] if index < len(expected) else "<no page>"
        right = actual[index] if index < len(actual) else "<no page>"
        difference = differ(left, right)
        if difference is None:
            same += 1
        else:
            different += 1
            first = first or f"page {index + 1} {difference}"
    return same, different, first


def on_alarm(signum: int, frame: Any) -> None:
    raise Timeout


def main(names: list[str]) -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    signal.signal(signal.SIGALRM, on_alarm)
    counts: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    paths = sorted({path for pattern in PATTERNS for path in glob.glob(pattern, recursive=True)})
    if names:
        paths = [path for path in paths if Path(path).name in names]
    for path in paths:
        signal.alarm(SECONDS_PER_FILE)
        try:
            results = {
                "text": (
                    text_pages(real_high_level, path),
                    text_pages(facade_high_level, path),
                    first_text_difference,
                ),
                "layout": (
                    layout_pages(real_high_level, real_layout, path),
                    layout_pages(facade_high_level, facade_layout, path),
                    first_layout_difference,
                ),
            }
        except Timeout:
            counts["files: TIMED OUT"] += 1
            problems.append(f"{path}: took over {SECONDS_PER_FILE}s")
            continue
        finally:
            signal.alarm(0)
        for check, (expected, actual, differ) in results.items():
            same, different, first = compare_pages(expected, actual, differ)
            counts[f"{check} pages: identical"] += same
            listed = DEPARTURES.get((check, Path(path).name))
            if first is None:
                counts[f"{check} files: identical"] += 1
                if listed is not None:
                    counts[f"{check} files: LISTED BUT IDENTICAL"] += 1
                    problems.append(f"{check} {path}: listed as a departure but identical")
            elif listed is not None:
                counts[f"{check} files: differ, listed departure"] += 1
                counts[f"{check} pages: differ, listed departure"] += different
            else:
                counts[f"{check} files: DIFFER"] += 1
                counts[f"{check} pages: DIFFER"] += different
                problems.append(f"{check} {path} {first}"[:400])
    print(f"at most {MAX_PAGES} pages per file, {len(paths)} files")
    for key, count in sorted(counts.items()):
        print(f"{count:>6}  {key}")
    for problem in problems:
        print(problem)
    failed = any(key.split(": ", 1)[1].isupper() for key in counts)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

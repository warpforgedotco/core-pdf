"""Check that the x-ray facade finds the same bad redactions as x-ray, on the upstream PDFs.

uv run --all-packages --group test --with x-ray==0.4.0 --with pymupdf==1.24.14 \
    python scripts/check_xray_output.py [--limit N]

x-ray is pinned to the version checked out under tests/upstream/x-ray, and PyMuPDF to the
version that x-ray pins (PyMuPDF==1.24.14), since x-ray cannot be installed with any other.
On PyMuPDF 2.1, x-ray 0.4.0 finds none of the bad redactions its own tests expect: MuPDF 1.28
rounds a clipped pixmap outward, so the clip under a redaction takes in a white edge and
filter_redactions_by_pixmap drops it. The facade follows x-ray as released, on its PyMuPDF.

inspect() is a coroutine in both, run to completion with asyncio.run. Each file goes through
xray.inspect and the facade's inspect, given a path and given the bytes,
and the results are compared exactly: the pages with bad redactions, and on each page every
redaction's bbox and text, in order. A file x-ray cannot read must fail in the facade too.
PyMuPDF raises its own classes (fitz.FileDataError and the like), which the facade cannot
raise without PyMuPDF, so a failure matches when every built-in class the PyMuPDF error
derives from (RuntimeError, ValueError, ...) also catches the facade's.

The helpers the facade ports from x-ray are compared too, on what x-ray met in the file:
text_utils' is_repeated_chars, is_ok_words and looks_like_a_date on the text under each
rectangle x-ray considered, check_if_all_dates on the result, and pdf_utils.intersects on the
spans and rectangles of each page (given PyMuPDF's Rects, as it requires). The other
pdf_utils helpers take PyMuPDF pages, which the facade has none of.

Files run in sorted order, at most --limit of them (default 250, all the samples, a few
minutes). Every difference fails the check unless DEPARTURES lists it, with the reason the
facade does not follow x-ray there. A listed file that matches fails it too, so the list stays
true.
"""

import argparse
import asyncio
import collections
import contextlib
import glob
import logging
import multiprocessing
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fitz  # ty: ignore[unresolved-import]
import xray as real  # ty: ignore[unresolved-import]
from xray import pdf_utils as real_pdf_utils  # ty: ignore[unresolved-import]
from xray import text_utils as real_text_utils  # ty: ignore[unresolved-import]

from core_pdf_compat import xray as facade
from core_pdf_compat.xray import pdf_utils as facade_pdf_utils
from core_pdf_compat.xray import text_utils as facade_text_utils

PATTERNS = (
    "tests/upstream/x-ray/tests/assets/*.pdf",
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/*.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/*.pdf",
    "tests/upstream/pypdf/resources/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)
DEPARTURES: dict[tuple[str, str], str] = {}
# Sources that are not sample files, for the errors PyMuPDF raises before reading a PDF.
ODD_SOURCES: tuple[object, ...] = (
    b"",
    b"not a pdf",
    b"%PDF-1.4 junk",
    "tests/upstream/no-such-file.pdf",
    "tests/upstream",
)
SPANS_PER_PAGE = 400
TIMEOUT = 300


def by_path(library: Any, path: Path) -> object:
    return asyncio.run(library.inspect(str(path)))


def by_bytes(library: Any, path: Path) -> object:
    return asyncio.run(library.inspect(path.read_bytes()))


def by_source(library: Any, source: object) -> object:
    return asyncio.run(library.inspect(source))


MODES: dict[str, Callable[[Any, Path], object]] = {"path": by_path, "bytes": by_bytes}


def builtin_bases(error: BaseException) -> tuple[str, ...]:
    """The built-in classes an error derives from, beyond Exception itself."""
    return tuple(
        kind.__name__
        for kind in type(error).__mro__
        if kind.__module__ == "builtins" and kind not in {BaseException, Exception, object}
    )


def outcome(run: Callable[[Any, Any], object], library: Any, path: Any) -> tuple[Any, ...]:
    """("ok", result) or ("error", repr, built-in bases), which survive a pipe."""
    try:
        return ("ok", run(library, path))
    except Exception as error:  # noqa: BLE001
        return ("error", repr(error), builtin_bases(error))


def first_difference(expected: Any, actual: Any) -> str:
    if not isinstance(actual, dict) or list(expected) != list(actual):
        pages = list(actual) if isinstance(actual, dict) else actual
        return f"pages: x-ray {list(expected)}, facade {pages!r}"
    for page, redactions in expected.items():
        found = actual[page]
        if len(redactions) != len(found):
            return f"page {page}: x-ray {len(redactions)} redactions, facade {len(found)}"
        for index, (want, got) in enumerate(zip(redactions, found, strict=True)):
            for key in ("bbox", "text"):
                if want.get(key) != got.get(key):
                    where = f"page {page} redaction {index} {key}"
                    return f"{where}: x-ray {want.get(key)!r}, facade {got.get(key)!r}"
            if list(want) != list(got):
                return f"page {page} redaction {index} keys: x-ray {list(want)}, facade {list(got)}"
    return "same"


def helper_differences(path: Path, result: object) -> list[str]:
    """Run the ported helpers on what x-ray meets in the file, with both implementations."""
    problems: list[str] = []
    texts: list[str] = []
    try:
        document = fitz.Document(str(path))
    except Exception:  # noqa: BLE001
        return problems
    if document.needs_pass:
        document.close()
        return problems
    with document:
        for number, page in enumerate(document, 1):
            try:
                rectangles = real_pdf_utils.get_good_rectangles(page)
                spans = page.get_texttrace()[:SPANS_PER_PAGE] if rectangles else []
                chars = real_pdf_utils.get_intersecting_chars(page, rectangles)
                redactions = real_pdf_utils.group_chars_by_rect(chars, rectangles)
            except Exception:  # noqa: BLE001
                continue
            texts += [redaction["text"] for redaction in redactions]
            for span in spans:
                span_rect = fitz.Rect(span["bbox"])
                span_rect.seqno = span["seqno"]
                span_rect.fill = span["color"]
                for threshold in (0.0, 0.8):
                    want = real_pdf_utils.intersects(span_rect, rectangles, threshold)
                    got = facade_pdf_utils.intersects(span_rect, rectangles, threshold)
                    if want != got:
                        where = f"intersects page {number} span {span['bbox']}"
                        problems.append(f"{where}: x-ray {want}, facade {got}")
    for text in texts:
        for name in ("is_repeated_chars", "is_ok_words", "looks_like_a_date"):
            want = getattr(real_text_utils, name)(text)
            got = getattr(facade_text_utils, name)(text)
            if want != got:
                problems.append(f"{name}({text!r}): x-ray {want}, facade {got}")
    if isinstance(result, dict):
        want = real_text_utils.check_if_all_dates(result)
        got = facade_text_utils.check_if_all_dates(result)
        if want != got:
            problems.append(f"check_if_all_dates: x-ray {want!r}, facade {got!r}")
    return problems


def reference(path: Path, connection: Any) -> None:
    """Run x-ray on one file in a child process, since PyMuPDF can crash on a sample."""
    results = {name: outcome(run, real, path) for name, run in MODES.items()}
    found = results["path"][1] if results["path"][0] == "ok" else None
    connection.send((results, helper_differences(path, found)))
    connection.close()


def run_reference(path: Path) -> tuple[dict[str, tuple[Any, ...]], list[str]] | None:
    """x-ray's outcomes for the file, or None when PyMuPDF crashes or hangs on it."""
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=reference, args=(path, sender))
    process.start()
    sender.close()
    answer = None
    with contextlib.suppress(EOFError):
        answer = receiver.recv() if receiver.poll(TIMEOUT) else None
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join()
    return answer


def compare(expected: tuple[Any, ...], actual: tuple[Any, ...]) -> tuple[bool, str, str]:
    """Whether the outcomes match, the count's name, and the first difference."""
    if expected[0] == "ok" and actual[0] == "ok":
        same = expected[1] == actual[1] and type(expected[1]) is type(actual[1])
        return same, "identical", "" if same else first_difference(expected[1], actual[1])
    if expected[0] == "ok":
        return False, "same error", f"facade raises {actual[1]}"
    if actual[0] == "ok":
        return False, "same error", f"x-ray raises {expected[1]}, facade reads it"
    # The facade's stand-ins for PyMuPDF's errors share their names, messages and built-in
    # classes, so the reprs match and every built-in class catching PyMuPDF's catches them.
    same = expected[1] == actual[1] and set(expected[2]) <= set(actual[2])
    return same, "same error", f"x-ray raises {expected[1]}, facade {actual[1]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=250)
    limit = parser.parse_args().limit
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    fitz.TOOLS.mupdf_display_errors(False)  # noqa: FBT003
    fitz.TOOLS.mupdf_display_warnings(False)  # noqa: FBT003
    counts: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    paths = sorted({path for pattern in PATTERNS for path in glob.glob(pattern, recursive=True)})
    for name_path in paths[:limit]:
        path = Path(name_path)
        answer = run_reference(path)
        if answer is None:
            counts["x-ray crashes or hangs, not compared"] += 1
            problems.append(f"not compared {path}: PyMuPDF crashes or hangs on it")
            continue
        results, helpers = answer
        for name, run in MODES.items():
            listed = DEPARTURES.get((name, path.name))
            same, kind, difference = compare(results[name], outcome(run, facade, path))
            if same:
                counts[f"{name}: {kind}"] += 1
                if listed is not None:
                    counts[f"{name}: LISTED BUT IDENTICAL"] += 1
                    problems.append(f"{name} {path}: listed as a departure but identical")
            elif listed is not None:
                counts[f"{name}: differ, listed departure"] += 1
            else:
                counts[f"{name}: DIFFER"] += 1
                problems.append(f"{name} {path} {difference}"[:400])
        counts["helpers: identical" if not helpers else "helpers: DIFFER"] += 1
        problems += [f"helpers {path} {problem}"[:400] for problem in helpers[:1]]
    for source in ODD_SOURCES:
        same, _, difference = compare(
            outcome(by_source, real, source), outcome(by_source, facade, source)
        )
        counts["odd sources: same error" if same else "odd sources: DIFFER"] += 1
        if not same:
            problems.append(f"odd source {source!r} {difference}")
    print(f"{min(limit, len(paths))} of {len(paths)} files")
    for key, count in sorted(counts.items()):
        print(f"{count:>6}  {key}")
    for problem in problems:
        print(problem)
    failed = any(key.split(": ", 1)[1].isupper() for key in counts if ": " in key)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

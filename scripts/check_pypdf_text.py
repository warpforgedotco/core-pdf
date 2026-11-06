"""Check that the pypdf facade extracts the same text as pypdf, page by page, on the upstream PDFs.

uv run --all-packages --with "pypdf[full]==6.21.4" python scripts/check_pypdf_text.py

Each file that real pypdf opens (an encrypted one is opened with the empty password) is read
through both libraries, and the first MAX_PAGES pages of it are compared in both of pypdf's
extraction modes: page.extract_text() and page.extract_text(extraction_mode="layout"). The
text must be equal, and a page that raises must raise the same exception class in both. The
files are the upstream test PDFs, plus pypdf's downloaded test files under pdf_cache when
they are present (pypdf's own suite downloads them).

Every difference fails the check unless it is a departure, with the reason the facade does
not follow pypdf 6.15.0 there:

- VERSION_DEPARTURES are behaviours the facade takes from the pypdf the upstream test suite is
  pinned at (tests/upstream/pypdf, newer than the 6.15.0 release), whose tests it passes. A
  differing page is re-extracted with real pypdf patched to that behaviour, and counts as the
  departure only if the facade then matches it exactly.
- DEPARTURES lists (mode, file name) pairs the facade does not match, with the reason. A listed
  file that matches fails the check too, so the list stays true.

Pass file paths to check only those files and print every differing page.
"""

import collections
import glob
import io
import logging
import os
import sys
import time
import warnings
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any

import pypdf as real  # ty: ignore[unresolved-import]
from pypdf._text_extraction._layout_mode._text_state_params import (  # ty: ignore[unresolved-import]
    TextStateParams,
)

from core_pdf_compat import pypdf as facade

PATTERNS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/*.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/*.pdf",
    "tests/upstream/pypdf/resources/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
    "tests/upstream/pypdf/tests/pdf_cache/*.pdf",
)
# Pages compared per file: enough to cover every page of almost every test file while the
# few long documents (hundreds of pages) keep the run to a few minutes.
MAX_PAGES = 30
MODES: dict[str, dict[str, Any]] = {"plain": {}, "layout": {"extraction_mode": "layout"}}
# (mode, file name) -> why the facade's text differs from pypdf's there.
DEPARTURES: dict[tuple[str, str], str] = {}


@contextmanager
def per_glyph_character_spacing() -> Generator[None]:
    """Layout mode applying Tc to every character, as the pinned upstream pypdf does.

    Follows TextStateParams.word_tx of pypdf, Copyright (c) 2006, Mathieu Fenniak, under its
    BSD-style license, with its one change after 6.15.0: `len(word) * Tc` for `Tc`.
    """
    original = TextStateParams.word_tx

    def word_tx(self: Any, word: bytes | str, td_offset: float = 0.0) -> float:
        if isinstance(word, bytes):
            word = self._decoded_value
        width = sum(
            self.font.space_width
            if char == self.font.space_char
            else self.font.get_text_width(char)
            for char in word
        )
        return (
            self.font_size * ((width - td_offset) / 1000.0)
            + len(word) * self.Tc
            + word.count(self.font.space_char) * self.Tw
        ) * (self.Tz / 100.0)

    TextStateParams.word_tx = word_tx
    try:
        yield
    finally:
        TextStateParams.word_tx = original


# mode -> {departure: (why, a context in which real pypdf behaves as the facade does)}.
VERSION_DEPARTURES: dict[str, dict[str, tuple[str, Callable[[], AbstractContextManager[None]]]]] = {
    "layout": {
        "per-glyph Tc": (
            "the upstream tests (test_layout_mode_character_spacing_per_glyph) apply the "
            "character spacing to each character; pypdf 6.15.0 adds it once per string",
            per_glyph_character_spacing,
        ),
    },
}


def open_reader(library: Any, data: bytes) -> Any:
    reader = library.PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        reader.decrypt("")
    return reader


def extract(reader: Any, index: int, options: dict[str, Any]) -> Any:
    try:
        return reader.pages[index].extract_text(**options)
    except Exception as error:  # noqa: BLE001
        return ("error", type(error).__name__)


def explaining_departure(
    reader: Any, index: int, options: dict[str, Any], mode: str, actual: Any
) -> str | None:
    """The version departure under which real pypdf gives the facade's text, if one does."""
    for name, (_, patched) in VERSION_DEPARTURES.get(mode, {}).items():
        with patched():
            if extract(reader, index, options) == actual:
                return name
    return None


def first_difference(expected: Any, actual: Any) -> str:
    if not (isinstance(expected, str) and isinstance(actual, str)):
        return f"pypdf {str(expected)[:100]!r}, facade {str(actual)[:100]!r}"
    at = next(
        (i for i, (x, y) in enumerate(zip(expected, actual, strict=False)) if x != y),
        min(len(expected), len(actual)),
    )
    start = max(0, at - 20)
    return f"at char {at}: pypdf {expected[start : at + 40]!r}, facade {actual[start : at + 40]!r}"


def compare(
    path: str, counts: collections.Counter[str], problems: list[str], *, every: bool
) -> None:
    data = Path(path).read_bytes()
    try:
        expected_reader = open_reader(real, data)
        pages = min(len(expected_reader.pages), MAX_PAGES)
    except Exception:  # noqa: BLE001
        counts["pypdf cannot open"] += 1
        return
    try:
        actual_reader = open_reader(facade, data)
        actual_pages = len(actual_reader.pages)
    except Exception as error:  # noqa: BLE001
        counts["FACADE CANNOT OPEN"] += 1
        problems.append(f"{path}: {error!r}"[:200])
        return
    if actual_pages != len(expected_reader.pages):
        counts["PAGE COUNT DIFFERS"] += 1
        problems.append(f"{path}: pypdf {len(expected_reader.pages)} pages, facade {actual_pages}")
        return
    for mode, options in MODES.items():
        differing = []
        explained = collections.Counter[str]()
        for index in range(pages):
            expected = extract(expected_reader, index, options)
            actual = extract(actual_reader, index, options)
            if expected == actual:
                counts[f"{mode}: pages identical"] += 1
                continue
            departure = explaining_departure(expected_reader, index, options, mode, actual)
            if departure is not None:
                explained[departure] += 1
            else:
                differing.append((index, expected, actual))
        for departure, count in explained.items():
            counts[f"{mode}: pages differ, departure: {departure}"] += count
        if explained and not differing:
            counts[f"{mode}: files differ only by departures"] += 1
        listed = DEPARTURES.get((mode, Path(path).name))
        if not differing:
            if not explained:
                counts[f"{mode}: files identical"] += 1
            if listed is not None:
                counts[f"{mode}: LISTED BUT IDENTICAL"] += 1
                problems.append(f"{mode} {path}: listed as a departure but identical")
            continue
        if listed is not None:
            counts[f"{mode}: files differ, listed departure"] += 1
            counts[f"{mode}: pages differ, listed departure"] += len(differing)
            continue
        counts[f"{mode}: FILES DIFFER"] += 1
        counts[f"{mode}: PAGES DIFFER"] += len(differing)
        for index, expected, actual in differing if every else differing[:1]:
            problems.append(
                f"{mode} {path} page {index}: {first_difference(expected, actual)}"[:400]
            )


def main(argv: list[str]) -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    counts: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    paths = argv or sorted(
        {path for pattern in PATTERNS for path in glob.glob(pattern, recursive=True)}
    )
    for path in paths:
        started = time.perf_counter()
        compare(path, counts, problems, every=bool(argv))
        if os.environ.get("CHECK_TIMES"):
            print(f"{time.perf_counter() - started:8.2f}s {path}", file=sys.stderr)
    for key, count in sorted(counts.items()):
        print(f"{count:>6}  {key}")
    for mode, departures in VERSION_DEPARTURES.items():
        for name, (why, _) in departures.items():
            print(f"{mode} departure {name}: {why}")
    for problem in problems:
        print(problem)
    failed = any(key.split(": ", 1)[-1].isupper() for key in counts)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

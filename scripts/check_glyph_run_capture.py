# SPDX-License-Identifier: AGPL-3.0-only

"""Check that the pdfminer and pdfplumber captures show a TJ array as one glyph run exactly
as they show it one glyph at a time.

The captures place each glyph of a horizontal TJ array themselves and show the array in one
TextState.show_glyph_run call, which captures every glyph as show_text captures it alone.
This captures the first pages of the upstream test PDFs both ways, the second with
show_glyph_run declining, so that each run is shown a glyph at a time, and compares every
glyph observation, glyph matrix, layout event, drawing and pdfplumber char mark; it fails
on the first page that differs.

uv run --all-packages python scripts/check_glyph_run_capture.py [pages per file]
"""

from __future__ import annotations

import glob
import logging
import re
import sys
import warnings
from typing import Any

from core_pdf import PdfDocument
from core_pdf.impl.capture_recording import TextState
from core_pdf_compat.pdfminer._capture import pdfminer_page_program
from core_pdf_compat.pdfplumber._capture import plumber_page_capture

CORPUS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/**/*.pdf",
)
ADDRESS = re.compile(r" at 0x[0-9a-f]+|0x[0-9a-f]{6,}")
PLAIN = (int, float, str, bytes, tuple, type(None), bool)


def observation(glyph: Any, fields: list[str]) -> tuple[Any, ...]:
    values = []
    for name in fields:
        value = getattr(glyph, name, None)
        values.append(value if isinstance(value, PLAIN) else ADDRESS.sub("@", repr(value)))
    return tuple(values)


def program(captured: Any) -> tuple[Any, ...]:
    glyphs = captured.glyphs
    fields = (
        [
            name
            for cls in type(glyphs[0]).__mro__
            for name in getattr(cls, "__slots__", ())
            if not name.startswith("__")
        ]
        if glyphs
        else []
    )
    events = []
    for event in captured.layout_events:
        if event[0] == "glyphs":
            # A run records its glyphs as one range; one glyph at a time records each alone.
            events.extend(("glyph", index) for index in range(event[1], event[2]))
        elif event[0] == "image":
            # An inline image is named by its object's id, which each capture makes anew.
            events.append(ADDRESS.sub("@", repr(event[2:])))
        else:
            events.append(ADDRESS.sub("@", repr(event)))
    return (
        tuple(observation(glyph, fields) for glyph in glyphs),
        tuple(sorted(captured.glyph_matrices.items())),
        tuple(events),
        len(captured.drawings),
    )


def capture(page: Any) -> tuple[Any, ...]:
    out: list[Any] = []
    for name, run in (
        ("pdfminer", lambda: program(pdfminer_page_program(page))),
        ("pdfplumber", lambda: plumber(page)),
    ):
        try:
            out.append(run())
        except Exception as error:  # noqa: BLE001
            out.append((name, "error", type(error).__name__, str(error)[:120]))
    return tuple(out)


def plumber(page: Any) -> tuple[Any, ...]:
    captured, paths, marks, _ = plumber_page_capture(page)
    return program(captured), ADDRESS.sub("@", repr(marks)), len(paths)


def declined(*_args: Any, **_kwargs: Any) -> None:
    return None


def main(argv: list[str]) -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    pages_per_file = int(argv[0]) if argv else 10
    paths = sorted({path for pattern in CORPUS for path in glob.glob(pattern, recursive=True)})
    if not paths:
        print("no upstream test PDFs: run git submodule update --init --recursive --depth 1")
        return 1
    shown_as_runs = 0
    show_glyph_run = TextState.show_glyph_run

    def counted(self: TextState, *args: Any, **kwargs: Any) -> Any:
        nonlocal shown_as_runs
        shown = show_glyph_run(self, *args, **kwargs)
        shown_as_runs += shown is not None
        return shown

    pages = 0
    failures = []
    for path in paths:
        try:
            document = PdfDocument.open(path)
            file_pages = list(document.pages)[:pages_per_file]
        except Exception:  # noqa: BLE001, S112
            continue
        for number, page in enumerate(file_pages):
            pages += 1
            TextState.show_glyph_run = counted  # type: ignore[method-assign]
            try:
                as_runs = capture(page)
            finally:
                TextState.show_glyph_run = show_glyph_run  # type: ignore[method-assign]
            # Declined, every run is shown a glyph at a time with show_text.
            TextState.show_glyph_run = declined  # type: ignore[method-assign]
            try:
                by_glyph = capture(page)
            finally:
                TextState.show_glyph_run = show_glyph_run  # type: ignore[method-assign]
            if as_runs != by_glyph:
                failures.append(f"{path} page {number + 1}")
    print(
        f"{pages} pages of {len(paths)} files, {shown_as_runs} TJ arrays shown as runs:",
        f"{len(failures)} differ",
    )
    for failure in failures[:10]:
        print(failure)
    return 1 if failures or not shown_as_runs else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

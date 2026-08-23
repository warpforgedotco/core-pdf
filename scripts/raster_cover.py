#!/usr/bin/env python
"""Recompute the covering subset used by ``tests/test_rendering_golden.py``.

Traces which lines of ``engine/render/`` each corpus document executes while its
first page is rasterized, then greedily picks the smallest set of documents that
between them reach every line the whole corpus reaches.  That subset is what the
always-on golden layer renders, so it stays fast without losing reach.

Run it after the corpus changes, or after a rasterizer refactor moves enough
code that the old subset may no longer cover::

    uv run python scripts/raster_cover.py

Prints a ready-to-paste ``COVERING_SUBSET`` tuple.
"""

from __future__ import annotations

import pathlib
import sys
import time
from types import FrameType
from typing import Any

from core_pdf import PdfDocument
from core_pdf.impl.engine.render import display, kernels, page, raster_image, target

CORPUS = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "SCORE-Bench" / "src"
TARGET_FILES = frozenset(
    module.__file__ for module in (display, kernels, page, raster_image, target)
)
RenderLine = tuple[str, int]


def trace_document(pdf: pathlib.Path) -> set[RenderLine]:
    """Return the rendering source lines executed while rasterizing page 1."""
    hits: set[RenderLine] = set()

    def tracer(frame: FrameType, event: str, arg: Any) -> Any:
        filename = frame.f_code.co_filename
        if filename not in TARGET_FILES:
            return None
        if event == "line":
            hits.add((filename, frame.f_lineno))
        return tracer

    sys.settrace(tracer)
    try:
        with PdfDocument.open(pdf) as document:
            document.pages[0].render().rasterize(
                scale=1.0, background=(255, 255, 255, 255), cache=False
            )
    except Exception as error:  # a document that fails to render still tells us nothing
        print(f"  ! {pdf.name}: {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        sys.settrace(None)
    return hits


def greedy_cover(per_document: dict[str, set[RenderLine]]) -> list[str]:
    universe: set[RenderLine] = set().union(*per_document.values()) if per_document else set()
    chosen: list[str] = []
    covered: set[RenderLine] = set()
    while covered != universe:
        best = max(per_document, key=lambda name: len(per_document[name] - covered))
        gain = len(per_document[best] - covered)
        if gain == 0:
            break
        chosen.append(best)
        covered |= per_document[best]
    return chosen


def main() -> int:
    if not CORPUS.is_dir():
        print(f"corpus not found at {CORPUS}", file=sys.stderr)
        return 1
    pdfs = sorted(CORPUS.glob("*.pdf"))
    started = time.time()
    per_document = {pdf.name: trace_document(pdf) for pdf in pdfs}
    universe = set().union(*per_document.values()) if per_document else set()
    chosen = greedy_cover(per_document)
    print(
        f"traced {len(pdfs)} documents in {time.time() - started:.0f}s; "
        f"{len(universe)} lines reachable; {len(chosen)} documents cover them all\n"
    )
    print("COVERING_SUBSET = (")
    for name in chosen:
        print(f'    "{name}",')
    print(")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

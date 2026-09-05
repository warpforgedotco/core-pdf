#!/usr/bin/env python
"""Recompute the covering subset used by ``tests/test_rendering_golden.py``.

Traces which lines of ``render/`` each corpus document executes while its
first page is rasterized, then greedily picks the smallest additional set of
documents that, together with the always-on tolerant snapshot pages, reaches
every line the whole corpus reaches. That subset keeps the exact-digest golden
layer fast without duplicating the tolerant cases.

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
from core_pdf.impl._impl.render import (
    blend,
    clipping,
    image_affine_target,
    image_axis_target,
    kernels,
    model,
    page,
    path_fill_target,
    path_shape_target,
    path_stroke_target,
    paths,
    patterns,
    target,
)

internal_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(internal_ROOT) not in sys.path:
    sys.path.insert(0, str(internal_ROOT))

from scripts.raster_golden import load_snapshot  # noqa: E402

CORPUS = internal_ROOT / "tests" / "fixtures" / "SCORE-Bench" / "src"
TARGET_FILES = frozenset(
    module.__file__
    for module in (
        blend,
        clipping,
        image_affine_target,
        image_axis_target,
        kernels,
        model,
        page,
        path_fill_target,
        path_shape_target,
        path_stroke_target,
        paths,
        patterns,
        target,
    )
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
            document.pages[0].render().rasterize(scale=1.0, background=(255, 255, 255, 255))
    except Exception as error:  # a document that fails to render still tells us nothing
        print(f"  ! {pdf.name}: {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        sys.settrace(None)
    return hits


def greedy_cover(
    per_document: dict[str, set[RenderLine]],
    *,
    precovered_names: frozenset[str] = frozenset(),
) -> list[str]:
    """Choose additions after seeding coverage from fixed always-on documents."""
    missing = precovered_names.difference(per_document)
    if missing:
        raise ValueError(f"precovered documents are absent from the corpus: {sorted(missing)}")
    universe: set[RenderLine] = set().union(*per_document.values()) if per_document else set()
    chosen: list[str] = []
    covered: set[RenderLine] = set().union(*(per_document[name] for name in precovered_names))
    candidates = {
        name: lines for name, lines in per_document.items() if name not in precovered_names
    }
    while covered != universe:
        best = max(candidates, key=lambda name: len(candidates[name] - covered))
        gain = len(candidates[best] - covered)
        if gain == 0:
            break
        chosen.append(best)
        covered |= candidates.pop(best)
    return chosen


def main() -> int:
    if not CORPUS.is_dir():
        print(f"corpus not found at {CORPUS}", file=sys.stderr)
        return 1
    pdfs = sorted(CORPUS.glob("*.pdf"))
    started = time.time()
    per_document = {pdf.name: trace_document(pdf) for pdf in pdfs}
    universe = set().union(*per_document.values()) if per_document else set()
    tolerant_names = frozenset(load_snapshot().portable)
    chosen = greedy_cover(per_document, precovered_names=tolerant_names)
    print(
        f"traced {len(pdfs)} documents in {time.time() - started:.0f}s; "
        f"{len(universe)} lines reachable; {len(tolerant_names)} tolerant documents plus "
        f"{len(chosen)} exact documents cover them all\n"
    )
    print("COVERING_SUBSET = (")
    for name in chosen:
        print(f'    "{name}",')
    print(")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

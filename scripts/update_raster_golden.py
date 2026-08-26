#!/usr/bin/env python
"""Regenerate the first-page raster snapshot without running pytest.

Each corpus PDF is rendered exactly once, then the complete digest mapping is
written to ``tests/snapshots/raster/first_page_scale1.json``::

    uv run python scripts/update_raster_golden.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

from core_pdf import PdfDocument

internal_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = internal_ROOT / "tests" / "fixtures" / "SCORE-Bench" / "src"
SNAPSHOT = internal_ROOT / "tests" / "snapshots" / "raster" / "first_page_scale1.json"

# Fixed so the digest is reproducible; do not vary these without regenerating.
RASTER_SCALE = 1.0
RASTER_BACKGROUND = (255, 255, 255, 255)


def raster_digest(pdf: pathlib.Path) -> str:
    """Render page 1 and hash the RGBA buffer."""
    with PdfDocument.open(pdf) as document:
        rendered = document.pages[0].render()
        raster = rendered.rasterize(
            scale=RASTER_SCALE,
            background=RASTER_BACKGROUND,
            cache=False,
        )
        return hashlib.sha256(raster.pixels.tobytes()).hexdigest()


def corpus_pdfs() -> list[pathlib.Path]:
    """Return every PDF included in the raster golden corpus."""
    if not CORPUS.is_dir():
        return []
    return sorted(CORPUS.glob("*.pdf"))


def main() -> int:
    pdfs = corpus_pdfs()
    if not pdfs:
        print(f"corpus not found or empty at {CORPUS}", file=sys.stderr)
        return 1

    started = time.monotonic()
    digests: dict[str, str] = {}
    for index, pdf in enumerate(pdfs, start=1):
        digests[pdf.name] = raster_digest(pdf)
        if index % 10 == 0 or index == len(pdfs):
            print(f"rendered {index}/{len(pdfs)} documents", flush=True)

    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(digests, indent=1, sort_keys=True) + "\n")
    print(f"updated {SNAPSHOT} in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

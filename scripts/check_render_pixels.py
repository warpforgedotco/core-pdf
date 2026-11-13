# SPDX-License-Identifier: AGPL-3.0-only

"""Record or compare the pixels core renders for the upstream test PDFs.

Rendering has no reference implementation to compare with, so a change to it is checked
against core itself: record the pages' pixels on the commit before the change, then
compare on the commit with it. Each of the first pages of every file is rendered
(PdfPage.render) and rasterized (RenderedPage.rasterize) at the default scale, and keeps
the SHA-256 of its width, height and RGBA bytes, or the error it raised.

uv run --all-packages python scripts/check_render_pixels.py --record before.json [pages]
uv run --all-packages python scripts/check_render_pixels.py --compare before.json [pages]

--compare fails on any page whose pixels or error differ from the recording, and lists the
first of them.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import signal
import sys
import warnings
from pathlib import Path
from types import FrameType

from core_pdf import PdfDocument

CORPUS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/**/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
    "tests/upstream/unstructured/example-docs/pdf/*.pdf",
)
MAX_BYTES = 8_000_000
# A page that takes longer is recorded as such rather than holding up the run.
PAGE_SECONDS = 60


class PageTimeout(Exception):
    pass


def on_alarm(signum: int, frame: FrameType | None) -> None:
    raise PageTimeout


def page_pixels(path: str, pages: int) -> list[str]:
    """Each page's pixel digest, or the error it raised."""
    digests: list[str] = []
    try:
        with PdfDocument(path) as document:
            for page in document.pages[:pages]:
                signal.alarm(PAGE_SECONDS)
                try:
                    image = page.render().rasterize()
                    digest = hashlib.sha256(f"{image.width}x{image.height}:".encode())
                    digest.update(bytes(image.pixels))
                    digests.append(digest.hexdigest())
                except PageTimeout:
                    digests.append("timeout")
                except Exception as error:  # noqa: BLE001
                    digests.append(f"error {type(error).__name__}: {error}")
                finally:
                    signal.alarm(0)
    except Exception as error:  # noqa: BLE001
        digests.append(f"open {type(error).__name__}: {error}")
    return digests


def render_corpus(pages: int) -> dict[str, list[str]]:
    paths = sorted(
        {
            path
            for pattern in CORPUS
            for path in glob.glob(pattern, recursive=True)
            if os.path.getsize(path) <= MAX_BYTES
        }
    )
    return {path: page_pixels(path, pages) for path in paths}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in {"--record", "--compare"}:
        print(__doc__, file=sys.stderr)
        return 2
    mode, recording = argv[0], Path(argv[1])
    pages = int(argv[2]) if len(argv) > 2 else 2
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    signal.signal(signal.SIGALRM, on_alarm)
    rendered = render_corpus(pages)
    page_count = sum(len(digests) for digests in rendered.values())
    if mode == "--record":
        recording.write_text(json.dumps(rendered, indent=0, sort_keys=True))
        print(f"{page_count} pages of {len(rendered)} files recorded in {recording}")
        return 0
    recorded: dict[str, list[str]] = json.loads(recording.read_text())
    differences: list[str] = []
    for path in sorted(set(rendered) | set(recorded)):
        now, before = rendered.get(path, []), recorded.get(path, [])
        differences.extend(
            f"{path} page {index + 1}"
            for index in range(max(len(now), len(before)))
            if (now[index] if index < len(now) else None)
            != (before[index] if index < len(before) else None)
        )
    print(f"{page_count} pages of {len(rendered)} files: {len(differences)} differ")
    for difference in differences[:10]:
        print(difference)
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

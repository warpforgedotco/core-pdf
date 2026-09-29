#!/usr/bin/env python3
"""Precompute the CID-to-Unicode votes of every packaged CID collection.

Writes src/core_pdf/impl/data/cid_unicode/<Registry>-<Ordering>.bin.gz, which
core_pdf.impl.fonts_unicode.CIDUnicodeMap reads instead of casting the votes
live. Rerun it after changing CID_COLLECTION_UNICODE_SOURCES or
CID_COLLECTION_UNICODE_OVERRIDES, the voting rules, or the Adobe CMap data;
tests/src/core_pdf/test_cid_unicode_tables.py fails until it is.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/core_pdf/impl/data/cid_unicode"

from core_pdf.impl import fonts_unicode  # noqa: E402
from core_pdf.impl.fonts_cmap import (  # noqa: E402
    CID_COLLECTION_UNICODE_OVERRIDES,
    CID_COLLECTION_UNICODE_SOURCES,
)


def collection_cids(registry: str, ordering: str) -> set[int]:
    cids = set(CID_COLLECTION_UNICODE_OVERRIDES.get((registry, ordering), {}))
    for sources in CID_COLLECTION_UNICODE_SOURCES[(registry, ordering)].values():
        for cmap_name, _codec, _weight in sources:
            cmap = fonts_unicode.compact_cmap(cmap_name)
            if cmap is not None:
                cids.update(cmap)
    return cids


def collection_votes(registry: str, ordering: str, *, vertical: bool) -> dict[int, str]:
    mapping = fonts_unicode.CIDUnicodeMap(registry, ordering, vertical)
    votes: dict[int, str] = {}
    for cid in sorted(collection_cids(registry, ordering)):
        text = mapping.vote_from_sources(cid)
        if text is None:
            continue
        if len(text) != 1:
            raise ValueError(f"{registry}-{ordering} CID {cid} votes for {text!r}, not one scalar")
        votes[cid] = text
    return votes


def build_table(registry: str, ordering: str) -> bytes:
    return fonts_unicode.encode_cid_unicode_table(
        collection_votes(registry, ordering, vertical=False),
        collection_votes(registry, ordering, vertical=True),
    )


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for registry, ordering in sorted(CID_COLLECTION_UNICODE_SOURCES):
        path = OUTPUT / fonts_unicode.cid_unicode_table_name(registry, ordering)
        path.write_bytes(build_table(registry, ordering))
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size} bytes", file=sys.stderr)


if __name__ == "__main__":
    build()

#!/usr/bin/env python3
"""Build the mmap-friendly runtime word-rank index from vendored sources."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDLIST_DIR = ROOT / "src/core_pdf/impl/layout/data/wordlists"
OUTPUT = WORDLIST_DIR / "english_word_ranks.bin"
MAGIC = b"CPWRANK1"
HEADER = struct.Struct("<8sI")
UINT32 = struct.Struct("<I")


def load_ranks() -> dict[str, int]:
    ranks: dict[str, int] = {}
    with gzip.open(WORDLIST_DIR / "norvig_count_1w.txt.gz", "rt", encoding="utf-8") as handle:
        for rank, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            word = parts[0].casefold()
            try:
                int(parts[1])
            except ValueError:
                continue
            if word and word.isalpha():
                ranks[word] = rank
    with gzip.open(WORDLIST_DIR / "wordninja_words.txt.gz", "rt", encoding="utf-8") as handle:
        for rank, line in enumerate(handle, start=1):
            word = line.strip().casefold()
            if word and word.isalpha() and word not in ranks:
                ranks[word] = rank
    return ranks


def build() -> None:
    entries = sorted(load_ranks().items())
    offsets = bytearray()
    records = bytearray()
    for word, rank in entries:
        offsets.extend(UINT32.pack(len(records)))
        records.extend(word.encode("utf-8"))
        records.append(0)
        records.extend(UINT32.pack(rank))
    offsets.extend(UINT32.pack(len(records)))
    temporary = OUTPUT.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        handle.write(HEADER.pack(MAGIC, len(entries)))
        handle.write(offsets)
        handle.write(records)
    temporary.replace(OUTPUT)


if __name__ == "__main__":
    build()

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import gzip
import mmap
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

WORDLIST_PACKAGE = "core_pdf.impl.engine.layout.data.wordlists"
NORVIG_COUNTS = "norvig_count_1w.txt.gz"
WORDNINJA_WORDS = "wordninja_words.txt.gz"
WORD_RANK_INDEX = "english_word_ranks.bin"
WORD_RANK_MAGIC = b"CPWRANK1"
WORD_RANK_HEADER = struct.Struct("<8sI")
UINT32 = struct.Struct("<I")


@dataclass(frozen=True, slots=True)
class WordFrequency:
    count: int
    rank: int


class WordRankIndex(Mapping[str, int]):
    """Read-only binary-search index backed directly by packaged byte buffer."""

    __slots__ = ("internal_cm", "internal_count", "internal_data_start", "internal_mmap")

    def __init__(self, path_or_bytes: str | bytes) -> None:
        if isinstance(path_or_bytes, bytes):
            mapped: mmap.mmap | bytes = path_or_bytes
        else:
            with open(path_or_bytes, "rb") as handle:
                try:
                    mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                except (OSError, ValueError):
                    mapped = handle.read()
        if len(mapped) < WORD_RANK_HEADER.size:
            if isinstance(mapped, mmap.mmap):
                mapped.close()
            raise ValueError("word-rank index is truncated")
        magic, count = WORD_RANK_HEADER.unpack_from(mapped)
        data_start = WORD_RANK_HEADER.size + (count + 1) * UINT32.size
        if magic != WORD_RANK_MAGIC or data_start > len(mapped):
            if isinstance(mapped, mmap.mmap):
                mapped.close()
            raise ValueError("word-rank index has an unsupported format")
        final_offset = UINT32.unpack_from(
            mapped,
            WORD_RANK_HEADER.size + count * UINT32.size,
        )[0]
        if data_start + final_offset != len(mapped):
            if isinstance(mapped, mmap.mmap):
                mapped.close()
            raise ValueError("word-rank index has invalid offsets")
        self.internal_mmap = mapped
        self.internal_count = count
        self.internal_data_start = data_start

    def internal_offset(self, index: int) -> int:
        return UINT32.unpack_from(
            self.internal_mmap,
            WORD_RANK_HEADER.size + index * UINT32.size,
        )[0]

    def internal_entry(self, index: int) -> tuple[bytes, int]:
        start = self.internal_offset(index)
        stop = self.internal_offset(index + 1)
        absolute_stop = self.internal_data_start + stop
        word = self.internal_mmap[self.internal_data_start + start : absolute_stop - 5]
        rank = UINT32.unpack_from(self.internal_mmap, absolute_stop - UINT32.size)[0]
        return word, rank

    def lookup(self, normalized: str) -> int | None:
        target = normalized.encode("utf-8")
        low = 0
        high = self.internal_count
        while low < high:
            middle = (low + high) // 2
            word, rank = self.internal_entry(middle)
            if word < target:
                low = middle + 1
            elif word > target:
                high = middle
            else:
                return rank
        return None

    def __getitem__(self, word: str) -> int:
        rank = self.lookup(word)
        if rank is None:
            raise KeyError(word)
        return rank

    def __iter__(self) -> Iterator[str]:
        for index in range(self.internal_count):
            yield self.internal_entry(index)[0].decode("utf-8")

    def __len__(self) -> int:
        return self.internal_count


@lru_cache(maxsize=1)
def english_word_frequencies() -> dict[str, WordFrequency]:
    frequencies: dict[str, WordFrequency] = {}
    load_norvig_counts(frequencies)
    load_wordninja_ranks(frequencies)
    return frequencies


@lru_cache(maxsize=1)
def english_word_ranks() -> Mapping[str, int]:
    """Open the packaged rank index without inflating source word lists."""
    import os
    import sys

    # When running under Nuitka compiled / onefile mode, locate file directly in unpacked dist tree
    if "__compiled__" in globals() or "__compiled__" in sys.modules:
        base_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(base_dir, "data", "wordlists", WORD_RANK_INDEX),
            os.path.join(
                sys.prefix,
                "core_pdf",
                "impl",
                "engine",
                "layout",
                "data",
                "wordlists",
                WORD_RANK_INDEX,
            ),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                try:
                    return WordRankIndex(candidate)
                except Exception:
                    # Path exists but is not a usable index (truncated, wrong layout,
                    # unreadable). Fall through to the next candidate.
                    pass

    res = files(WORDLIST_PACKAGE).joinpath(WORD_RANK_INDEX)
    try:
        return WordRankIndex(res.read_bytes())
    except Exception:
        # Not readable as package data -- e.g. inside a zipimport or a compiled
        # bundle. Fall through to the as_file() path below.
        pass
    from importlib.resources import as_file

    try:
        with as_file(res) as path_obj:
            return WordRankIndex(path_obj.read_bytes())
    except Exception:
        return {word: freq.rank for word, freq in english_word_frequencies().items()}


def load_norvig_counts(frequencies: dict[str, WordFrequency]) -> None:
    res = files(WORDLIST_PACKAGE).joinpath(NORVIG_COUNTS)
    try:
        raw_bytes = res.read_bytes()
        lines = gzip.decompress(raw_bytes).decode("utf-8").splitlines()
    except (TypeError, ValueError, OSError, AttributeError):
        from importlib.resources import as_file

        with as_file(res) as path_obj, gzip.open(str(path_obj), "rt", encoding="utf-8") as handle:
            lines = handle.readlines()

    for rank, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        word, raw_count = parts
        word = word.casefold()
        if not word or not word.isalpha():
            continue
        try:
            count = int(raw_count)
        except ValueError:
            continue
        frequencies[word] = WordFrequency(count, rank)


def load_wordninja_ranks(frequencies: dict[str, WordFrequency]) -> None:
    res = files(WORDLIST_PACKAGE).joinpath(WORDNINJA_WORDS)
    try:
        raw_bytes = res.read_bytes()
        lines = gzip.decompress(raw_bytes).decode("utf-8").splitlines()
    except (TypeError, ValueError, OSError, AttributeError):
        from importlib.resources import as_file

        with as_file(res) as path_obj, gzip.open(str(path_obj), "rt", encoding="utf-8") as handle:
            lines = handle.readlines()

    for rank, line in enumerate(lines, start=1):
        word = line.strip().casefold()
        if not word or not word.isalpha() or word in frequencies:
            continue
        frequencies[word] = WordFrequency(0, rank)


@lru_cache(maxsize=262_144)
def normalized_word_rank(normalized: str) -> int | None:
    if not normalized or not normalized.isalpha():
        return None
    ranks = english_word_ranks()
    if isinstance(ranks, WordRankIndex):
        return ranks.lookup(normalized)
    return ranks.get(normalized)


@lru_cache(maxsize=65536)
def word_rank(word: str) -> int | None:
    return normalized_word_rank(word.casefold())

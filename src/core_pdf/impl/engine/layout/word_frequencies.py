# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import gzip
import mmap
import struct
from bisect import bisect_left, bisect_right
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
    """Read-only binary-search index backed directly by the packaged mmap."""

    __slots__ = ("internal_count", "internal_data_start", "internal_mmap")

    def __init__(self, path: str) -> None:
        with open(path, "rb") as handle:
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        if len(mapped) < WORD_RANK_HEADER.size:
            mapped.close()
            raise ValueError("word-rank index is truncated")
        magic, count = WORD_RANK_HEADER.unpack_from(mapped)
        data_start = WORD_RANK_HEADER.size + (count + 1) * UINT32.size
        if magic != WORD_RANK_MAGIC or data_start > len(mapped):
            mapped.close()
            raise ValueError("word-rank index has an unsupported format")
        final_offset = UINT32.unpack_from(
            mapped,
            WORD_RANK_HEADER.size + count * UINT32.size,
        )[0]
        if data_start + final_offset != len(mapped):
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
def english_word_ranks() -> WordRankIndex:
    """Open the packaged rank index without inflating source word lists."""
    path = str(files(WORDLIST_PACKAGE).joinpath(WORD_RANK_INDEX))
    return WordRankIndex(path)


@lru_cache(maxsize=1)
def english_word_frequency_items_sorted() -> tuple[tuple[str, WordFrequency], ...]:
    return tuple(sorted(english_word_frequencies().items()))


@lru_cache(maxsize=1)
def english_word_frequency_words_sorted() -> tuple[str, ...]:
    return tuple(word for word, internal_frequency in english_word_frequency_items_sorted())


@lru_cache(maxsize=1)
def english_word_frequency_items_by_rank() -> tuple[tuple[str, WordFrequency], ...]:
    return tuple(
        sorted(
            english_word_frequencies().items(),
            key=lambda item: item[1].rank,
        )
    )


def load_norvig_counts(frequencies: dict[str, WordFrequency]) -> None:
    path = str(files(WORDLIST_PACKAGE).joinpath(NORVIG_COUNTS))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for rank, line in enumerate(handle, start=1):
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
    path = str(files(WORDLIST_PACKAGE).joinpath(WORDNINJA_WORDS))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for rank, line in enumerate(handle, start=1):
            word = line.strip().casefold()
            if not word or not word.isalpha() or word in frequencies:
                continue
            frequencies[word] = WordFrequency(0, rank)


@lru_cache(maxsize=262_144)
def normalized_word_frequency(normalized: str) -> WordFrequency | None:
    if not normalized or not normalized.isalpha():
        return None
    return english_word_frequencies().get(normalized)


def word_frequency(word: str) -> WordFrequency | None:
    return normalized_word_frequency(word.casefold())


@lru_cache(maxsize=262_144)
def normalized_word_rank(normalized: str) -> int | None:
    if not normalized or not normalized.isalpha():
        return None
    return english_word_ranks().lookup(normalized)


def english_word_frequency_prefix_items(
    prefix: str,
) -> tuple[tuple[str, WordFrequency], ...]:
    normalized = prefix.casefold()
    if not normalized:
        return ()
    words = english_word_frequency_words_sorted()
    start = bisect_left(words, normalized)
    stop = bisect_right(words, f"{normalized}\uffff")
    return english_word_frequency_items_sorted()[start:stop]


def word_count(word: str) -> int:
    frequency = word_frequency(word)
    return frequency.count if frequency is not None else 0


@lru_cache(maxsize=65536)
def word_rank(word: str) -> int | None:
    return normalized_word_rank(word.casefold())


def is_common_word(word: str, *, max_rank: int = 75_000) -> bool:
    rank = word_rank(word)
    return rank is not None and rank <= max_rank

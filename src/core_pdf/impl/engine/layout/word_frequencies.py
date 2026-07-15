# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import gzip
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

WORDLIST_PACKAGE = "core_pdf.impl.engine.layout.data.wordlists"
NORVIG_COUNTS = "norvig_count_1w.txt.gz"
WORDNINJA_WORDS = "wordninja_words.txt.gz"


@dataclass(frozen=True, slots=True)
class WordFrequency:
    count: int
    rank: int


@lru_cache(maxsize=1)
def english_word_frequencies() -> dict[str, WordFrequency]:
    frequencies: dict[str, WordFrequency] = {}
    load_norvig_counts(frequencies)
    load_wordninja_ranks(frequencies)
    return frequencies


@lru_cache(maxsize=1)
def english_word_frequency_items_sorted() -> tuple[tuple[str, WordFrequency], ...]:
    return tuple(sorted(english_word_frequencies().items()))


@lru_cache(maxsize=1)
def english_word_frequency_words_sorted() -> tuple[str, ...]:
    return tuple(word for word, _frequency in english_word_frequency_items_sorted())


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
    frequency = normalized_word_frequency(normalized)
    return frequency.rank if frequency is not None else None


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


def word_rank(word: str) -> int | None:
    return normalized_word_rank(word.casefold())


def is_common_word(word: str, *, max_rank: int = 75_000) -> bool:
    rank = word_rank(word)
    return rank is not None and rank <= max_rank

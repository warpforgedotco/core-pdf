import random

from core_pdf.impl.layout.word_frequencies import (
    english_word_frequencies,
    english_word_ranks,
    word_rank,
)


def test_rank_only_index_matches_full_frequency_index() -> None:
    frequencies = english_word_frequencies()
    ranks = english_word_ranks()

    assert len(ranks) == len(frequencies)
    # A deterministic sample proves the packed index agrees with the source
    # table without probing every one of the hundred-thousand-odd words.
    sample = random.Random(0).sample(sorted(frequencies), 500)
    assert all(ranks[word] == frequencies[word].rank for word in sample)
    assert word_rank("the") == ranks["the"]
    assert word_rank("not-a-word") is None

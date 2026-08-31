from core_pdf.impl.layout.word_frequencies import (
    english_word_frequencies,
    english_word_ranks,
    word_rank,
)


def test_rank_only_index_matches_full_frequency_index() -> None:
    frequencies = english_word_frequencies()
    ranks = english_word_ranks()

    assert ranks == {word: frequency.rank for word, frequency in frequencies.items()}
    assert word_rank("the") == ranks["the"]
    assert word_rank("not-a-word") is None

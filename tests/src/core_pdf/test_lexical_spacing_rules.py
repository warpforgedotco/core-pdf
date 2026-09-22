import pytest

from core_pdf.impl.layout import text_rules as rules


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ("", "word", False),
        ("word", "", False),
        ("T", "he", True),
        ("T", "his", True),
        ("hello", "world", True),
        ("é", "界", True),
        ("word!", "next", False),
        ("word", "1", False),
    ],
)
def test_estimated_spacing_requires_alphabetic_boundary_characters(previous, current, expected):
    assert rules.should_use_estimated_word_spacing(previous, current) is expected


@pytest.mark.parametrize(
    ("text", "leading", "trailing"),
    [
        ("", "", ""),
        ("123", "", ""),
        ("hello", "hello", "hello"),
        ("éclair!世界", "éclair", "世界"),
        (" word ", "", ""),
    ],
)
def test_alpha_tokens_stop_at_nonletters(text, leading, trailing):
    assert rules.leading_alpha_token(text) == leading
    assert rules.trailing_alpha_token(text) == trailing


@pytest.mark.parametrize(
    ("previous", "current", "rank", "expected"),
    [
        ("word", "is", None, False),
        ("123", "next", None, False),
        ("word", "next", None, True),
        ("of", "next", 250, True),
        ("of", "next", 251, False),
        ("xy", "next", None, False),
    ],
)
def test_phrase_spacing_uses_frequency_only_for_short_previous_tokens(
    monkeypatch, previous, current, rank, expected
):
    monkeypatch.setattr(rules, "word_rank", lambda word: rank)
    assert rules.should_insert_phrase_continuation_space(previous, current) is expected


@pytest.mark.parametrize(
    ("previous", "current", "gap", "prev_visible", "visible", "short", "expected"),
    [
        ("hel", "lo", 0, False, True, False, False),
        ("hel", "lo", 0, True, False, False, False),
        (" ", "lo", 0, True, True, False, False),
        ("hel", " ", 0, True, True, False, False),
        ("he", "llo", 0, True, True, False, False),
        ("hel", "123", 0, True, True, False, False),
        ("HEL", "lo", 0, True, True, False, False),
        ("hel", "Lo", 0, True, True, False, False),
        ("hel", "lo", -1.01, True, True, False, False),
        ("hel", "lo", -1, True, True, False, True),
        ("hel", "lo", 7.25, True, True, False, True),
        ("hel", "lo", 7.26, True, True, False, False),
        ("h", "e", 1.8, True, True, True, True),
        ("h", "e", 1.81, True, True, True, False),
    ],
)
def test_split_word_join_requires_visibility_case_and_close_geometry(
    monkeypatch, previous, current, gap, prev_visible, visible, short, expected
):
    monkeypatch.setattr(rules, "word_rank", {"hello": 10}.get)
    assert (
        rules.should_join_plausible_split_word(
            previous,
            current,
            x_gap=gap,
            height=10,
            space_width=5,
            prev_visible=prev_visible,
            visible=visible,
            allow_short_prefix=short,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("previous", "current", "joined", "tail", "head", "expected"),
    [
        ("some", "thing", None, None, None, False),
        ("some", "thing", 150001, None, None, False),
        ("some", "thing", 150000, None, None, True),
        ("some", "thing", 75000, 1, None, True),
        ("some", "thing", 75001, 1, None, False),
        ("some", "thing", 9, 10, 11, True),
        ("some", "thing", 10, 10, 11, False),
        ("some", "one", 150000, 1, 2, True),
        ("another", "thing", 50, 100, None, True),
        ("another", "thing", 100, 50, None, False),
    ],
)
def test_split_word_join_compares_whole_word_and_fragment_frequencies(
    monkeypatch, previous, current, joined, tail, head, expected
):
    ranks = {previous + current: joined, previous: tail, current: head}
    monkeypatch.setattr(rules, "word_rank", ranks.get)
    assert (
        rules.should_join_plausible_split_word(
            previous,
            current,
            x_gap=0,
            height=10,
            space_width=5,
            prev_visible=True,
            visible=True,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("previous", "current", "gap", "expected"),
    [
        ("", "12", 0, False),
        ("12", "", 0, False),
        ("ab", "12", 0, False),
        ("12", "ab", 0, False),
        ("12", "34", 0.5, True),
        ("12", "34", 0.51, False),
    ],
)
def test_digit_fragments_join_only_at_digit_boundaries_with_small_gaps(
    previous, current, gap, expected
):
    assert (
        rules.digit_fragments_are_tightly_joined(
            previous, current, x_gap=gap, height=10, space_width=5
        )
        is expected
    )


@pytest.mark.parametrize(
    ("previous", "current", "gap", "expected"),
    [
        ("", "word", 0, False),
        ("word", "", 0, False),
        ("a", "word", 0, False),
        ("word", "a", 0, False),
        ("word!", "next", 0, False),
        ("word", "123", 0, False),
        ("the", "word", -0.26, False),
        ("the", "word", -0.25, True),
        ("word", "the", 0, True),
        ("ABC", "word", 0, True),
        ("ABCDEFGHI", "word", 0, False),
        ("word", "next", 0, False),
        ("ABC", "WORD", 0, False),
    ],
)
def test_tight_word_space_uses_frequent_words_or_short_uppercase_prefixes(
    monkeypatch, previous, current, gap, expected
):
    monkeypatch.setattr(rules, "word_rank", {"the": 1}.get)
    assert (
        rules.should_insert_tight_word_space(
            prev_text=previous, text=current, x_gap=gap, height=10, space_width=5
        )
        is expected
    )


@pytest.mark.parametrize(
    ("previous", "current", "gap", "prev_visible", "visible", "expected"),
    [
        ("word", "Next", 0, True, False, False),
        ("word", "Next", 0, False, True, False),
        ("a", "word", 0, False, False, False),
        ("word!", "next", 0, False, False, False),
        ("word", "!next", 0, False, False, False),
        ("ABC", "DEF", -1.76, False, False, False),
        ("ABC", "DEF", -1.75, False, False, True),
        ("12", "34", 0, False, False, False),
        ("12", "word", 0, False, False, True),
        ("word", "12", 0, False, False, True),
        ("word", "Next", 0, False, False, True),
        ("word", "next", 0, False, False, False),
    ],
)
def test_hidden_overlap_spacing_preserves_digits_and_uses_case_transitions(
    previous, current, gap, prev_visible, visible, expected
):
    assert (
        rules.should_insert_hidden_text_overlap_space(
            prev_text=previous,
            text=current,
            x_gap=gap,
            height=10,
            space_width=5,
            prev_visible=prev_visible,
            visible=visible,
        )
        is expected
    )

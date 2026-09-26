import pytest

from core_pdf.impl import layout_text_rules as rules
from core_pdf.impl.runs import TextRun


def make_run(
    text: str = "a",
    x: float = 0,
    *,
    width: float = 5,
    height: float = 10,
    space: float = 5,
    order: int = 0,
    stream_order: int = 0,
) -> TextRun:
    return TextRun(text, x, 0, x + width, height, x, 0, height, space, order, stream_order, 0)


@pytest.mark.parametrize(
    ("runs", "expected"),
    [
        ([], True),
        ([make_run()], True),
        ([make_run(x=0), make_run(x=10)], True),
        ([make_run(x=10), make_run(x=0)], False),
        ([make_run(width=1, order=1), make_run(width=1, order=0)], False),
        ([make_run(width=1, order=0), make_run(width=1, order=1)], True),
        ([make_run(width=10), make_run(x=7.5)], True),
        ([make_run(width=10), make_run(x=7.4)], False),
        ([make_run(width=20, height=20), make_run(x=3, height=10)], False),
    ],
)
def test_left_to_right_requires_ordered_positions_and_limited_overlap(runs, expected):
    assert rules.runs_are_left_to_right(runs) is expected


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        ([], False),
        ([0], False),
        ([20, 10, 0], True),
        ([0, 10, 20], False),
        ([0, 10, 0], False),
        ([20, 20, 0], True),
        ([0, 0], False),
    ],
)
def test_right_to_left_uses_majority_of_stream_position_changes(positions, expected):
    runs = [make_run(x=x, order=i) for i, x in enumerate(positions)]
    assert rules.runs_are_right_to_left(list(reversed(runs))) is expected


def test_right_to_left_ignores_empty_runs_and_breaks_order_ties_by_stream_order():
    blank = make_run(" ", x=100)
    assert not rules.runs_are_right_to_left([blank, make_run()])
    runs = [make_run(x=0, stream_order=1), blank, make_run(x=10, stream_order=0)]
    assert rules.runs_are_right_to_left(runs)


@pytest.mark.parametrize(
    ("x", "space", "expected"), [(5.5, 1, False), (5.4, 1, True), (5.4, 10, False), (-1, 1, True)]
)
def test_interleaved_overlap_respects_glyph_width_and_space_tolerance(x, space, expected):
    runs = [
        make_run(width=10, space=space),
        make_run(" "),
        make_run(x=x, width=10, space=space),
    ]
    assert rules.has_interleaved_horizontal_overlap(runs) is expected
    assert not rules.has_interleaved_horizontal_overlap([])


def test_positive_gaps_ignore_spaces_touching_and_overlapping_runs():
    runs = [
        make_run(x=0),
        make_run(" ", x=200),
        make_run(x=7),
        make_run(x=12),
        make_run(x=14),
        make_run(x=22),
    ]
    assert rules.positive_run_gaps(runs) == [2, 3]
    assert rules.positive_run_gaps([]) == []


@pytest.mark.parametrize(
    ("words", "step", "expected"),
    [
        (["a"] * 5, 8, False),
        (["ab"] * 6, 8, False),
        (["a"] * 5 + ["lengthy"], 8, False),
        (["a"] * 6, 5, False),
        (["a"] * 6, 8, True),
        (["a"] * 6, 29, True),
        (["a"] * 6, 30, False),
    ],
)
def test_tracked_glyph_detection_requires_short_text_and_consistent_small_gaps(
    words, step, expected
):
    runs = [make_run(word, i * step) for i, word in enumerate(words)]
    assert rules.is_tracked_glyph_run_line(runs) is expected


@pytest.mark.parametrize(
    ("words", "spaces", "expected"),
    [
        (["a"] * 8, 1, False),
        (["a"] * 7, 2, False),
        (["ab"] * 8, 2, False),
        (["a"] * 7 + ["lengthy"], 2, False),
        (["a"] * 8, 2, True),
    ],
)
def test_explicit_spaces_control_only_lines_mostly_made_of_single_glyphs(words, spaces, expected):
    assert (
        rules.explicit_spaces_should_control_glyph_gaps(
            [make_run(word) for word in words], explicit_space_count=spaces
        )
        is expected
    )


@pytest.mark.parametrize(
    ("gaps", "height", "expected"),
    [
        ([1] * 4, 10, None),
        ([1] * 5, 10, None),
        ([1, 1, 1, 1, 10], 10, 2.2),
        ([1, 1, 1, 1, 10], 30, 6.6),
        ([0.1, 0.1, 0.1, 0.1, 10], 1, 1.5),
    ],
)
def test_tracked_word_threshold_requires_a_distinct_larger_gap(gaps, height, expected):
    runs = [make_run(height=height)]
    for gap in gaps:
        runs.append(make_run(x=runs[-1].x1 + gap, height=height))
    result = rules.tracked_glyph_word_gap_threshold(runs)
    assert result is None if expected is None else result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("runs", "expected"),
    [
        ([], 12),
        ([make_run(height=0, space=0)], 12),
        ([make_run(height=20, space=1)], 31),
        ([make_run(height=1, space=10)], 50),
        ([make_run(), make_run(x=25)], 80),
    ],
)
def test_column_gap_threshold_combines_observed_gaps_height_and_spaces(runs, expected):
    assert rules.column_gap_threshold_for_runs(runs) == expected


@pytest.mark.parametrize(("space", "expected"), [(0, 0.5), (1, 0.48), (5, 2.4)])
def test_suspect_zero_width_runs_use_position_deltas_even_without_space_metrics(space, expected):
    step = 0.5 if space <= 1 else 3
    runs = [make_run(x=i * step, width=0, space=space) for i in range(5)]
    assert rules.estimated_char_width_for_suspect_line(runs) == pytest.approx(expected)


def test_character_width_estimation_ignores_hidden_blank_and_non_alphabetic_runs():
    runs = [make_run(x=i * 3, width=5, space=5) for i in range(5)]
    hidden = make_run(x=500)
    hidden.visible = False
    runs[2:2] = [hidden, make_run(" ", x=600), make_run("123", x=700)]
    assert rules.estimated_char_width_for_suspect_line(runs) == pytest.approx(2.4)


def test_character_width_estimation_requires_enough_suspect_runs_and_consistent_deltas():
    assert rules.estimated_char_width_for_suspect_line([make_run()] * 4) is None
    assert (
        rules.estimated_char_width_for_suspect_line(
            [make_run(x=i * 6, width=2, space=5) for i in range(5)]
        )
        is None
    )
    assert (
        rules.estimated_char_width_for_suspect_line(
            [make_run(x=i * 50, width=0, space=5) for i in range(5)]
        )
        is None
    )

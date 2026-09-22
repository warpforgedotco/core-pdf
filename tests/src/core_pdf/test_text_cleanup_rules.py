import pytest

from core_pdf.impl.layout import text_rules as rules
from core_pdf.impl.model.runs import TextRun


def internal_run(text, x=0, y=0, width=10, height=10, font=10, rotation=0):
    return TextRun(
        text, x, y, x + width, y + height, x, y, font, 2, 0, 0, 0, rotation_angle=rotation
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("abc123", "abc123"),
        ("1.2Label", "1.2 Label"),
        ("12Label", "12Label"),
        ("1.2label", "1.2label"),
        ("time:12", "time: 12"),
        (" m2 rest", " m 2 rest"),
        (" g2", " g2"),
        (" l2x", " l2x"),
    ],
)
def test_numeric_label_splitting_preserves_plain_identifiers(text, expected):
    assert rules.split_glued_numeric_label_boundaries(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello", "hello"),
        ("a\ue000b\U000f0000c\x00d\u200be", "abcde"),
        ("a\ufffdb\u00adc»,«•·●", "abc,"),
        ("\t\n\r", "\t\n\r"),
        (".................---------", "...---"),
    ],
)
def test_cleanup_removes_nontext_marks_but_preserves_layout_whitespace(text, expected):
    assert rules.strip_private_use_chars(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"), [("", ""), ("a b", "a b"), (" a       b  \t c ", " a b \t c ")]
)
def test_space_collapse_preserves_single_spaces_and_tabs(text, expected):
    assert rules.collapse_repeated_spaces(text) == expected


@pytest.mark.parametrize(("rotation", "expected"), [(0, 30), (90, 20), (180, 30), (270, 20)])
def test_baseline_midpoint_uses_rotated_axis(rotation, expected):
    assert rules.baseline_midpoint((10, 20, 30, 40), rotation) == expected


@pytest.mark.parametrize(("text", "expected"), [("", False), ("abc", False), ("a√b", True)])
def test_formula_detection_uses_recognized_markers(text, expected):
    assert rules.formula_like_runs([internal_run(" "), internal_run(text)]) is expected


@pytest.mark.parametrize("intervening", [False, True])
def test_stacked_fraction_reordering_preserves_every_run_once(intervening):
    denominator = internal_run("√x")
    numerator = internal_run("2", y=3)
    middle = internal_run("+", x=20)
    runs = [denominator, middle, numerator] if intervening else [denominator, numerator]
    original = list(runs)
    reordered = rules.reorder_stacked_formula_numerators(runs)
    assert reordered == (
        [numerator, denominator, middle] if intervening else [numerator, denominator]
    )
    assert runs == original
    assert len({id(run) for run in reordered}) == len(runs)


@pytest.mark.parametrize("text", ["word", "123", "t", "2"])
def test_unmatched_numerators_and_ordinary_text_retain_order(text):
    runs = [internal_run("ordinary"), internal_run(text, y=3)]
    assert rules.reorder_stacked_formula_numerators(runs) == runs


@pytest.mark.parametrize(
    ("denominator", "numerator", "expected"),
    [
        (internal_run(""), internal_run("2", y=3), False),
        (internal_run("√"), internal_run(""), False),
        (internal_run("3"), internal_run("2", y=3), False),
        (internal_run("x"), internal_run("2", y=3), False),
        (internal_run("√"), internal_run("a", y=3), False),
        (internal_run("√", rotation=90), internal_run("2", y=3), False),
        (internal_run("√"), internal_run("2", y=3, rotation=90), False),
        (internal_run("√", height=0), internal_run("2", y=3), False),
        (internal_run("√"), internal_run("2", y=3, height=0), False),
        (internal_run("√"), internal_run("2", y=3, height=6.9), False),
        (internal_run("√"), internal_run("2", y=3, height=13.1), False),
        (internal_run("√"), internal_run("2", y=3, height=7), True),
        (internal_run("√"), internal_run("2", y=3, height=13), True),
        (internal_run("√"), internal_run("2", x=10, y=3), False),
        (internal_run("√"), internal_run("2", y=1.9), False),
        (internal_run("√"), internal_run("2", y=2), True),
    ],
)
def test_stacked_denominator_requires_formula_text_and_compatible_geometry(
    denominator, numerator, expected
):
    assert rules.stacked_formula_denominator(denominator, numerator) is expected


@pytest.mark.parametrize(
    ("following", "expected"),
    [(None, False), ("", False), (" ", False), ("word", False), (")", True), (",", True)],
)
def test_time_symbol_numerator_requires_following_punctuation(following, expected):
    assert (
        rules.stacked_formula_denominator(
            internal_run("T"),
            internal_run("t", y=3),
            following=internal_run(following) if following is not None else None,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [("²", True), (" ₂³ ", True), ("2", False), ("", False), (" ", False), ("²a", False)],
)
def test_script_digits_require_only_script_characters(text, expected):
    assert rules.script_digit_text(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"), [("™", True), (" ® ", True), ("", False), ("a", False), ("™™", False)]
)
def test_inline_marker_is_a_single_supported_symbol(text, expected):
    assert rules.inline_marker_text(text) is expected


@pytest.mark.parametrize(
    ("previous", "current", "gap", "expected"),
    [
        ("12k", "V", 0.3, True),
        ("12k", "V", 0.51, False),
        ("12", "V", 0, False),
        ("k", "V", 0, False),
        ("", "V", 0, False),
        ("12k", "A", 0, False),
    ],
)
def test_voltage_suffix_join_requires_numeric_prefix_and_close_unit(
    previous, current, gap, expected
):
    assert (
        rules.compact_unit_suffix_should_join(
            previous, current, x_gap=gap, height=10, space_width=2
        )
        is expected
    )


@pytest.mark.parametrize(
    ("text", "expected"), [("", False), ("H ", True), ("h", False), ("H2", False)]
)
def test_chemical_prefix_requires_trailing_uppercase_letter(text, expected):
    assert rules.chemical_subscript_prefix_text(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("page 12", True), (" Page12 ", True), ("page", False), ("body page 12", False)],
)
def test_footer_text_requires_complete_page_label(text, expected):
    assert rules.is_tiny_page_footer(text) is expected


@pytest.mark.parametrize(
    (
        "page_text",
        "digit_text",
        "page_width",
        "digit_width",
        "gap",
        "page_font",
        "digit_font",
        "body_font",
        "digit_x",
        "expected",
    ),
    [
        ("page", "1", 10, 5, 40, 10, 10, 10, 60, {1, 2}),
        ("other", "1", 10, 5, 40, 5, 5, 10, 60, set()),
        ("page", "x", 10, 5, 40, 5, 5, 10, 60, set()),
        ("page", "1", 0, 5, 40, 5, 5, 10, 60, set()),
        ("page", "1", 10, 0, 40, 5, 5, 10, 60, set()),
        ("page", "1", 18, 15, 10, 7, 5, 10, 60, set()),
        ("page", "1", 18, 15, 10, 5, 7, 10, 60, set()),
        ("page", "1", 21, 15, 10, 5, 5, 10, 60, set()),
        ("page", "1", 18, 17, 10, 5, 5, 10, 60, set()),
        ("page", "1", 18, 15, 10, 6, 6, 8, 60, set()),
        ("page", "1", 18, 15, 10, 5, 5, 10, 60, {1, 2}),
        ("page", "1", 18, 15, 10, 5, 5, 0, 60, {1, 2}),
        ("page", "1", 10, 5, 40, 5, 5, 10, 0, set()),
    ],
)
def test_footer_geometry_distinguishes_tiny_labels_from_body_text(
    page_text,
    digit_text,
    page_width,
    digit_width,
    gap,
    page_font,
    digit_font,
    body_font,
    digit_x,
    expected,
):
    runs = [
        internal_run("body", font=body_font),
        internal_run(page_text, x=10 + gap, width=page_width, font=page_font),
        internal_run(digit_text, x=digit_x, width=digit_width, font=digit_font),
        internal_run(" "),
    ]
    assert rules.trailing_tiny_page_label_run_indexes(runs) == expected
    assert rules.trailing_tiny_page_label_run_indexes(runs[:2]) == set()

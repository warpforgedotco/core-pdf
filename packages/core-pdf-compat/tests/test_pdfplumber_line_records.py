import pytest

from core_pdf_compat import pdfplumber as compat


@pytest.mark.parametrize("return_chars", [False, True])
@pytest.mark.parametrize(
    ("characters", "expected"),
    [
        ([("A", 0), ("B", 5)], "AB"),
        ([("A", 0), ("B", 8)], "AB"),
        ([("A", 0), ("B", 8.1)], "A B"),
        ([("A", 0), (" ", 5), ("B", 6)], "A B"),
        ([(" ", 0), ("A", 1), (" ", 6)], "A"),
        ([(".", 0), ("!", 20)], ".!"),
        ([("ﬁ", 0), ("A", 5)], "fiA"),
    ],
)
def test_default_line_text_and_character_projection(characters, expected, return_chars):
    chars = [
        {"text": text, "x0": x, "x1": x + 5, "top": 10, "bottom": 20, "doctop": 110, "size": 10}
        for text, x in characters
    ]
    lines = compat._lines(reversed(chars), return_chars=return_chars)
    assert len(lines) == 1
    assert lines[0]["text"] == expected
    assert (lines[0]["x0"], lines[0]["top"], lines[0]["x1"], lines[0]["bottom"]) == (
        chars[0]["x0"],
        10,
        chars[-1]["x1"],
        20,
    )
    assert lines[0]["doctop"] == 110
    if return_chars:
        assert all(
            actual is original for actual, original in zip(lines[0]["chars"], chars, strict=True)
        )
    else:
        assert "chars" not in lines[0]


@pytest.mark.parametrize(
    ("size", "tops", "expected"),
    [
        (10, (0, 3), ["AB"]),
        (10, (0, 3.1), ["A", "B"]),
        (1, (0, 25), ["AB"]),
        (1, (0, 25.1), ["A", "B"]),
    ],
)
def test_line_grouping_thresholds_and_blank_only_groups(size, tops, expected):
    chars = [
        {
            "text": text,
            "x0": x,
            "x1": x + 5,
            "top": top,
            "bottom": top + 10,
            "doctop": top,
            "size": size,
        }
        for text, x, top in [("A", 0, tops[0]), ("B", 5, tops[1]), (" ", 0, 100)]
    ]
    assert [line["text"] for line in compat._lines(chars)] == expected
    assert compat._lines([]) == []
    assert compat._lines([chars[-1]]) == []

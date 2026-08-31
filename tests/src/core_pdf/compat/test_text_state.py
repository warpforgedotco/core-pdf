import pytest

from core_pdf import PdfDocument
from core_pdf.api.compat._text_state import (
    internal_append_directional_text,
    internal_ensure_line_break,
    internal_flush_text,
    internal_orientation,
    internal_positioned_text,
)
from core_pdf.api.compat.pypdf._text import extract_legacy_text

IDENTITY = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        ([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0),
        ([-1.0, 0.0, 0.0, -1.0, 0.0, 0.0], 180),
        ([0.0, 1.0, -1.0, 0.0, 0.0, 0.0], 90),
        ([0.0, -1.0, 1.0, 0.0, 0.0, 0.0], 270),
    ],
)
def test_orientation_matches_legacy_text_projection(matrix: list[float], expected: int) -> None:
    assert internal_orientation(matrix) == expected


def test_directional_transition_reverses_rtl_and_discards_finished_runs() -> None:
    text, rtl = internal_append_directional_text("", False, "A")
    text, rtl = internal_append_directional_text(text, rtl, "א")
    text, rtl = internal_append_directional_text(text, rtl, "ב")
    text, rtl = internal_append_directional_text(text, rtl, " ")

    assert (text, rtl) == (" בא", True)
    assert internal_append_directional_text(text, rtl, "Z") == ("Z", False)


def test_flush_and_line_break_share_one_append_only_output_transition() -> None:
    parts: list[str] = []

    text, last = internal_flush_text(parts, "abc", "")
    last = internal_ensure_line_break(parts, last)

    assert (parts, text, last) == (["abc", "\n"], "", "\n")
    assert internal_ensure_line_break(parts, last) == "\n"
    assert parts == ["abc", "\n"]


def test_position_transition_inserts_horizontal_space() -> None:
    parts: list[str] = []

    text, last = internal_positioned_text(
        parts,
        "A",
        "",
        previous_text_matrix=IDENTITY,
        previous_current_matrix=IDENTITY,
        text_matrix=[1.0, 0.0, 0.0, 1.0, 4.0, 0.0],
        current_matrix=IDENTITY,
        line_height=12.0,
        font_size=12.0,
        space_width=250.0,
        string_width=0.0,
    )

    assert (parts, text, last) == ([], "A ", "")


def test_position_transition_flushes_a_vertical_line_move() -> None:
    parts: list[str] = []

    text, last = internal_positioned_text(
        parts,
        "A",
        "",
        previous_text_matrix=IDENTITY,
        previous_current_matrix=IDENTITY,
        text_matrix=[1.0, 0.0, 0.0, 1.0, 0.0, 12.0],
        current_matrix=IDENTITY,
        line_height=12.0,
        font_size=12.0,
        space_width=250.0,
        string_width=0.0,
    )

    assert (parts, text, last) == (["A\n"], "", "\n")


def test_pypdf_projection_preserves_mixed_direction_text() -> None:
    with PdfDocument.open("tests/fixtures/pypdf/resources/hello-world.pdf") as document:
        text = extract_legacy_text(document.pages[0])

    assert "Hello World" in text
    assert "مرحبا بالعالم" in text

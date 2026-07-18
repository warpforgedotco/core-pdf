from core_layout.impl.layout.models import LayoutBox, LayoutLine, TextRun
from core_layout.impl.layout.text_lines import (
    inline_marker_text,
    reconstruct_layout_line_text,
    script_digit_text,
)

from core_pdf.impl.engine.extraction.common.render import text_boxes_in_reading_order


def rotated_run(text: str, y0: float, y1: float) -> TextRun:
    return TextRun(
        text,
        100.0,
        y0,
        110.0,
        y1,
        100.0,
        y0,
        8.0,
        3.0,
        0,
        0,
        0,
        rotation_angle=90,
    )


def test_rotated_table_cells_use_geometric_spaces() -> None:
    fields = (
        "IN2015_C01",
        "016",
        "GAB",
        "OR02, Area25",
        "−36.069",
        "132.637",
        "4,607",
        "31/10/15",
    )
    runs: list[TextRun] = []
    y = 10.0
    for field in fields:
        width = max(8.0, len(field) * 4.0)
        runs.append(rotated_run(field, y, y + width))
        y += width + 12.0

    assert reconstruct_layout_line_text(runs).text == " ".join(fields)


def test_ninety_degree_boxes_are_ordered_by_row_axis() -> None:
    boxes = [
        LayoutBox(x0=300.0, x1=310.0, y1=700.0, mid_y=500.0),
        LayoutBox(x0=100.0, x1=110.0, y1=600.0, mid_y=300.0),
        LayoutBox(x0=200.0, x1=210.0, y1=650.0, mid_y=700.0),
    ]

    assert [box.x0 for box in text_boxes_in_reading_order(boxes, 90)] == [
        100.0,
        200.0,
        300.0,
    ]


def test_rotated_character_runs_preserve_intentional_joining() -> None:
    text = "Ø10H9Ø10H9"
    runs = []
    y = 10.0
    for index, char in enumerate(text):
        if index == 5:
            y += 100.0
        runs.append(rotated_run(char, y, y + 5.0))
        y += 5.0

    assert reconstruct_layout_line_text(runs).text == text


def test_layout_line_reuses_reconstructed_text_until_a_run_changes() -> None:
    run = rotated_run("A", 10.0, 15.0)
    line = LayoutLine(runs=[run])

    first = line.reconstructed_text()

    assert line.reconstructed_text() is first
    run.set_text("B")
    second = line.reconstructed_text()
    assert second is not first
    assert second.text == "B"


def test_layout_lines_share_reconstruction_for_the_same_run_state() -> None:
    run = rotated_run("Shared", 10.0, 30.0)

    first = LayoutLine(runs=[run]).reconstructed_text()
    second = LayoutLine(runs=[run]).reconstructed_text()

    assert second is first


def test_cached_word_reconstruction_preserves_public_mutability_isolation() -> None:
    run = rotated_run("Shared", 10.0, 30.0)
    line = LayoutLine(runs=[run])

    first_text, first_words = line.text_and_words()
    second_text, second_words = line.text_and_words()

    assert first_text == second_text == "Shared"
    assert first_words is not second_words
    assert first_words[0] is not second_words[0]
    first_words[0].text = "changed"
    assert second_words[0].text == "Shared"

    run.set_text("Updated")
    assert line.cached_text_and_words()[0] == "Updated"


def test_layout_line_reconstruction_cache_tracks_geometry_and_run_order() -> None:
    first_run = rotated_run("A", 10.0, 15.0)
    second_run = rotated_run("B", 15.0, 20.0)
    line = LayoutLine(runs=[first_run, second_run])

    first = line.reconstructed_text()
    second_run.y0 = 40.0
    after_geometry_change = line.reconstructed_text()
    assert after_geometry_change is not first

    second_run.coords[TextRun.Y0] = 50.0
    after_direct_coordinate_change = line.reconstructed_text()
    assert after_direct_coordinate_change is not after_geometry_change

    line.runs.reverse()
    assert line.reconstructed_text() is not after_direct_coordinate_change


def test_inline_marker_and_script_digit_fast_paths_preserve_padded_text() -> None:
    assert inline_marker_text("™")
    assert inline_marker_text("  ™ ")
    assert not inline_marker_text("T")
    assert script_digit_text("²")
    assert script_digit_text("  ²³ ")
    assert not script_digit_text("2")

from core_pdf.impl.engine.extraction.common.render import text_boxes_in_reading_order
from core_pdf.impl.engine.layout.models import LayoutBox, TextRun
from core_pdf.impl.engine.layout.text_lines import reconstruct_layout_line_text


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

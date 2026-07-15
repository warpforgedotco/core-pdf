from core_pdf.impl.engine.extraction.page_text.native import (
    native_text_runs_inside_page_bounds,
)
from core_pdf.impl.engine.layout.models import TextRun


def text_run(text: str, x0: float, y0: float, x1: float, y1: float) -> TextRun:
    return TextRun(
        text,
        x0,
        y0,
        x1,
        y1,
        x0,
        y0,
        10.0,
        4.0,
        0,
        0,
        0,
    )


def test_page_bounds_keep_long_text_with_bad_horizontal_font_metrics() -> None:
    run = text_run(
        "A valid line whose reported width exceeds the page",
        60.0,
        100.0,
        1_200.0,
        112.0,
    )

    assert native_text_runs_inside_page_bounds(
        [run],
        (0.0, 0.0, 612.0, 792.0),
    ) == [run]


def test_page_bounds_still_drop_text_starting_outside_page() -> None:
    run = text_run(
        "A line entirely beyond the declared page",
        620.0,
        100.0,
        1_200.0,
        112.0,
    )

    assert not native_text_runs_inside_page_bounds(
        [run],
        (0.0, 0.0, 612.0, 792.0),
    )

import pytest

from core_pdf.impl.layout.reconstruction import (
    GlyphLineBuilder,
    reconstruct_layout_line_text,
    reconstruct_rotated_table_line,
)
from core_pdf.impl.model.runs import TextRun


def internal_run(text: str, position: float = 0, *, size: float = 10, rotation: int = 0) -> TextRun:
    return TextRun(
        text,
        position,
        position,
        position + 5,
        position + 5,
        position,
        position,
        size,
        2,
        0,
        0,
        0,
        rotation_angle=rotation,
    )


@pytest.mark.parametrize("count", [255, 256, 257, 321])
def test_long_line_retains_all_text_when_duplicate_history_is_trimmed(count):
    runs = [internal_run("z", i * 5) for i in range(count)]
    result = GlyphLineBuilder(runs).build()
    assert result.text == "z" * count
    assert len(result.segments) == count


@pytest.mark.parametrize("size", [5, 5.01, 10])
@pytest.mark.parametrize("fragmented", [False, True])
def test_page_footer_suppression_respects_font_size(size, fragmented):
    runs = (
        [internal_run("page", size=size), internal_run("1", 5, size=size)]
        if fragmented
        else [internal_run("page1", size=size)]
    )
    result = reconstruct_layout_line_text(runs)
    assert result.text == ("" if size <= 5 else "page1")
    if size <= 5:
        assert result.segments == ()


@pytest.mark.parametrize("rotation", [90, 270])
@pytest.mark.parametrize("gap", [0, 6])
def test_rotated_cells_use_their_reading_axis_for_spacing(rotation, gap):
    runs = [internal_run("ab", i * (5 + gap), rotation=rotation) for i in range(8)]
    if rotation == 270:
        runs.reverse()
    result = reconstruct_rotated_table_line(runs)
    assert result.text == (" " if gap else "").join(["ab"] * 8)
    assert len(result.segments) == 8


@pytest.mark.parametrize("text", ["", "\x00", "\ue000"])
def test_empty_rotated_table_text_has_no_segments(text):
    result = reconstruct_rotated_table_line([internal_run(text, rotation=90)])
    assert result.text == ""
    assert result.segments == ()

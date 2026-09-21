import pytest

from core_pdf.impl._impl.layout.reconstruction import reconstruct_layout_line_text
from core_pdf.impl._impl.model.runs import TextRun


def run(text, x=0, y=0, width=10, rotation=0):
    return TextRun(text, x, y, x + width, y + 10, x, y, 10, 2, 0, 0, 0, rotation_angle=rotation)


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_empty_records_cannot_bypass_formula_ordering(position, count):
    denominator, numerator = run("√x"), run("2", y=3)
    runs = [denominator, numerator]
    runs[position:position] = [run("", x=100, rotation=90) for _ in range(count)]
    original = list(runs)
    result = reconstruct_layout_line_text(runs)
    assert result.text == "2√x"
    assert all(a is b for a, b in zip(runs, original, strict=True))
    assert len(result.segments) == 2


@pytest.mark.parametrize(("dx", "dy"), [(0, 0), (100, -25), (-50, 400)])
def test_translating_formula_preserves_text_and_translates_segments(dx, dy):
    runs = [run("√x"), run("2", y=3)]
    translated = [
        r.replace(
            x0=r.x0 + dx, x1=r.x1 + dx, y0=r.y0 + dy, y1=r.y1 + dy, tx=r.tx + dx, ty=r.ty + dy
        )
        for r in runs
    ]
    base = reconstruct_layout_line_text(runs)
    actual = reconstruct_layout_line_text(translated)
    assert actual.text == base.text
    for original, moved in zip(base.segments, actual.segments, strict=True):
        assert moved.advance_bbox == pytest.approx(
            tuple(
                coordinate + delta
                for coordinate, delta in zip(original.advance_bbox, (dx, dy, dx, dy), strict=True)
            )
        )


@pytest.mark.parametrize("empty_count", [0, 1, 3])
def test_fragmenting_contiguous_text_preserves_explicit_spaces(empty_count):
    complete = [run("hello world", width=55)]
    fragmented = [run("hello", width=25), run(" ", x=25, width=5), run("world", x=30, width=25)]
    fragmented.extend(run("") for _ in range(empty_count))
    assert (
        reconstruct_layout_line_text(fragmented).text == reconstruct_layout_line_text(complete).text
    )


def test_all_empty_line_has_no_segments():
    result = reconstruct_layout_line_text([run(""), run("")])
    assert result.text == ""
    assert result.segments == ()

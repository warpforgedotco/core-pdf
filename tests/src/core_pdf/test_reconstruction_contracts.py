import pytest

from core_pdf.impl._impl.layout.reconstruction import reconstruct_layout_line_text
from core_pdf.impl._impl.model.runs import TextRun


def run(text, x=0, y=0, width=10, rotation=0):
    return TextRun(text, x, y, x + width, y + 10, x, y, 10, 2, 0, 0, 0, rotation_angle=rotation)


def test_replace_rejects_unknown_field_names():
    with pytest.raises(TypeError, match="font_nam"):
        run("x").replace(font_nam="Helvetica")


def test_replace_preserves_the_runtime_class():
    class Subclass(TextRun):
        __slots__ = ()

    original = Subclass("x", 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 2.0, 0, 0, 0)
    assert type(original.replace(font_name="Helvetica")) is Subclass


def test_replace_preserves_every_field_it_was_not_asked_to_change():
    values = {
        "text": "x",
        "x0": 3.0,
        "y0": 7.0,
        "x1": 15.0,
        "y1": 17.0,
        "tx": 3.5,
        "ty": 7.5,
        "font_size": 11.0,
        "space_width": 2.5,
        "order": 4,
        "stream_order": 5,
        "xobject_depth": 1,
        "font_name": "Courier",
        "is_vertical": True,
        "rotation_angle": 90,
        "visible": False,
        "inside_active_clip": False,
        "line_break_before": True,
        "seqno": 7,
        "fill_color": (0.1, 0.2, 0.3),
        "advance_bbox": (1.0, 2.0, 3.0, 4.0),
        "ink_bbox": (5.0, 6.0, 7.0, 8.0),
        "baseline": (0.0, 1.0, 2.0, 3.0),
        "provenance": (("k", "v"),),
        "confidence": 0.5,
        "glyph_clusters": (),
    }
    assert set(values) == set(TextRun.__fields__)
    original = TextRun(**values)
    replaced = original.replace(font_name="Helvetica")
    assert replaced is not original
    assert replaced.font_name == "Helvetica"
    for name, value in values.items():
        if name == "font_name":
            continue
        assert getattr(replaced, name) == value, name


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

from typing import Any

import pytest

from core_pdf.impl.model.glyphs import GlyphCluster, GlyphObservation
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.spec.s_07_content.text_runs import RunAccumulator
from core_pdf.impl.types import Rectangle


def internal_run(
    text: str,
    bbox: Rectangle = (0.0, 0.0, 4.0, 10.0),
    *,
    seqno: int = 0,
    rotation: int = 0,
    font_name: str = "F1",
) -> TextRun:
    glyph = GlyphObservation(text, bbox, bbox, seqno)
    cluster = GlyphCluster(seqno, text, (glyph,), bbox, bbox, None, 1.0)
    return TextRun(
        text,
        *bbox,
        tx=bbox[0],
        ty=bbox[1],
        font_size=10.0,
        space_width=4.0,
        order=seqno,
        stream_order=0,
        xobject_depth=0,
        font_name=font_name,
        rotation_angle=rotation,
        seqno=seqno,
        glyph_clusters=(cluster,),
    )


def test_accumulator_defers_joining_text_and_clusters_until_flush() -> None:
    output: list[TextRun] = []
    accumulator = RunAccumulator(output)
    first = internal_run("A")
    second = internal_run("B", (4.0, 0.0, 8.0, 10.0), seqno=1)

    accumulator.append(first)
    accumulator.append(second)

    assert not output
    assert first.text == "A"
    assert len(first.glyph_clusters) == 1
    accumulator.flush()
    assert output == [first]
    assert first.text == "AB"
    assert first.advance_bbox == first.ink_bbox == (0.0, 0.0, 8.0, 10.0)
    assert [cluster.cluster_id for cluster in first.glyph_clusters] == [0, 1]
    assert accumulator.pending is None
    accumulator.flush()
    assert output == [first]


def test_right_to_left_text_preserves_glyph_clusters_in_emission_order() -> None:
    output: list[TextRun] = []
    accumulator = RunAccumulator(output)
    for seqno, (text, x) in enumerate([("ג", 8.0), ("ב", 4.0), ("א", 0.0)]):
        accumulator.append(internal_run(text, (x, 0.0, x + 4.0, 10.0), seqno=seqno))
    accumulator.flush()

    assert len(output) == 1
    run = output[0]
    assert run.text == "אבג"
    assert run.advance_bbox == run.ink_bbox == (0.0, 0.0, 12.0, 10.0)
    assert [cluster.text for cluster in run.glyph_clusters] == ["ג", "ב", "א"]
    assert [cluster.cluster_id for cluster in run.glyph_clusters] == [0, 1, 2]


@pytest.mark.parametrize(
    ("rotation", "first_box", "second_box", "expected"),
    [
        (0, (0, 0, 4, 10), (4, 0, 8, 10), ["AB"]),
        (90, (0, 0, 10, 4), (0, 4, 10, 8), ["AB"]),
        (180, (4, 0, 8, 10), (0, 0, 4, 10), ["AB"]),
        (180, (0, 0, 4, 10), (4, 0, 8, 10), ["BA"]),
        (270, (4, 0, 8, 10), (0, 0, 4, 10), ["AB"]),
        (270, (0, 4, 10, 8), (0, 0, 10, 4), ["A", "B"]),
    ],
)
def test_rotation_preserves_existing_merge_direction(
    rotation: int, first_box: Rectangle, second_box: Rectangle, expected: list[str]
) -> None:
    output: list[TextRun] = []
    accumulator = RunAccumulator(output)
    accumulator.append(internal_run("A", first_box, rotation=rotation))
    accumulator.append(internal_run("B", second_box, rotation=rotation, seqno=1))
    accumulator.flush()

    assert [run.text for run in output] == expected
    assert [cluster.cluster_id for run in output for cluster in run.glyph_clusters] == [0, 1]


@pytest.mark.parametrize(
    "changed",
    [
        {"font_size": 11.0},
        {"fill_color": (1.0, 0.0, 0.0)},
        {"visible": False},
        {"line_break_before": True},
        {"rotation_angle": 90},
        {"y0": 6.0, "y1": 16.0},
    ],
)
def test_style_and_line_changes_finish_the_pending_run(changed: dict[str, Any]) -> None:
    output: list[TextRun] = []
    accumulator = RunAccumulator(output)
    first = internal_run("A")
    second = internal_run("B", (4.0, 0.0, 8.0, 10.0)).replace(**changed)
    accumulator.append(first)
    accumulator.append(second)

    assert output == [first]
    assert output[0].text == "A"
    accumulator.flush()
    assert output == [first, second]


@pytest.mark.parametrize(
    ("left", "right", "gap", "expected"),
    [
        ("A", "B", 0.5, ["AB"]),
        ("A", "B", 1.0, ["A B"]),
        ("A", ",", 1.0, ["A,"]),
        ("(", "B", 1.0, ["(B"]),
        ("A ", "B", 1.0, ["A B"]),
        ("A", " B", 1.0, ["A B"]),
        ("A", "B", -2.0, ["AB"]),
        ("A", "B", -2.1, ["A", "B"]),
        ("A", "B", 2.0, ["A", "B"]),
    ],
)
def test_spacing_distinguishes_letters_words_and_separate_runs(
    left: str, right: str, gap: float, expected: list[str]
) -> None:
    output: list[TextRun] = []
    accumulator = RunAccumulator(output)
    accumulator.append(internal_run(left))
    accumulator.append(internal_run(right, (4.0 + gap, 0.0, 8.0 + gap, 10.0)))
    accumulator.flush()

    assert [run.text for run in output] == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("inter", "face", ["interface"]),
        ("name", "_suffix", ["name_suffix"]),
        ("(", "!", ["(", "!"]),
        ("word ", "next", ["word next"]),
        (" ", "next", [" ", "next"]),
    ],
)
def test_font_changes_preserve_word_boundary_rules(
    left: str, right: str, expected: list[str]
) -> None:
    output: list[TextRun] = []
    accumulator = RunAccumulator(output)
    accumulator.append(internal_run(left, font_name="F1"))
    accumulator.append(internal_run(right, (4.0, 0.0, 8.0, 10.0), font_name="F2"))
    accumulator.flush()

    assert [run.text for run in output] == expected

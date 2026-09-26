from core_pdf.impl.capture_text_runs import RunAccumulator
from core_pdf.impl.glyphs import GlyphCluster
from core_pdf.impl.runs import TextRun


def run(text: str, x0: float, x1: float, *, rotation: int = 0) -> TextRun:
    box = (x0, 0, x1, 10)
    return TextRun(
        text,
        *box,
        x0,
        0,
        10,
        2,
        0,
        0,
        0,
        rotation_angle=rotation,
        glyph_clusters=(GlyphCluster(0, text, (), box, box, None, None),),
    )


def merged(*runs: TextRun) -> TextRun:
    output: list[TextRun] = []
    accumulator = RunAccumulator(output)
    for item in runs:
        accumulator.append(item)
    accumulator.flush()
    assert len(output) == 1
    return output[0]


def cluster_text(result: TextRun) -> str:
    return "".join(cluster.text for cluster in result.glyph_clusters)


def test_run_prepended_on_the_left_puts_its_clusters_first() -> None:
    result = merged(run("b", 10, 15), run("a", 5, 10))
    assert result.text == "ab"
    assert cluster_text(result) == "ab"


def test_mixed_directions_keep_clusters_in_text_order() -> None:
    result = merged(run("c", 10, 15), run("d", 15, 20), run("b", 5, 10), run("a", 0, 5))
    assert result.text == "abcd"
    assert cluster_text(result) == "abcd"


def test_right_to_left_rotation_appends_on_the_left_side() -> None:
    result = merged(run("b", 10, 15, rotation=180), run("a", 15, 20, rotation=180))
    assert result.text == "ab"
    assert cluster_text(result) == "ab"

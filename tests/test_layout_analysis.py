from types import SimpleNamespace

from core_pdf.impl.engine.layout.analysis import (
    chart_table_html,
    layout_prediction,
    page_layout_items,
)
from core_pdf.impl.engine.layout.chart_geometry import detect_chart_regions
from core_pdf.impl.engine.layout.chart_model import positioned_tokens
from core_pdf.impl.engine.structured import BlockKind


def chart_marks() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(x0=x, y0=20.0, x1=x + 8.0, y1=20.0 + height, line_width=1.0)
        for x, height in zip((15.0, 25.0, 35.0, 45.0, 55.0, 65.0), (30, 42, 24, 50, 35, 46))
    )


def test_layout_prediction_clips_off_page_boxes() -> None:
    element = SimpleNamespace(
        bbox=(-10.0, -20.0, 110.0, 120.0),
        text="running header",
        kind=BlockKind.PARAGRAPH,
        lines=(),
        confidence=0.9,
    )

    prediction = layout_prediction(element, 100.0, 100.0)

    assert prediction is not None
    assert prediction["bbox"] == [0.0, 0.0, 100.0, 100.0]
    assert prediction["label"] == "page-header"


def test_chart_regions_cluster_repeated_vector_marks() -> None:
    regions = detect_chart_regions(
        lines=chart_marks(),
        drawings=(),
        page_width=100.0,
        page_height=100.0,
    )

    assert len(regions) == 1
    assert regions[0].bbox == (15.0, 20.0, 73.0, 70.0)
    assert regions[0].geometry_count == 6


def test_chart_regions_reject_sparse_page_geometry() -> None:
    lines = tuple(
        SimpleNamespace(x0=float(index * 20), y0=10.0, x1=float(index * 20 + 4), y1=14.0)
        for index in range(6)
    )

    assert (
        detect_chart_regions(
            lines=lines,
            drawings=(),
            page_width=100.0,
            page_height=100.0,
        )
        == ()
    )


def test_chart_regions_fall_back_to_a_large_figure_bbox() -> None:
    figure = SimpleNamespace(bbox=(10.0, 20.0, 90.0, 80.0))

    regions = detect_chart_regions(
        lines=(),
        drawings=(),
        figures=(figure,),
        page_width=100.0,
        page_height=100.0,
    )

    assert len(regions) == 1
    assert regions[0].bbox == (10.0, 20.0, 90.0, 80.0)
    assert regions[0].geometry_count == 0


def test_chart_regions_ignore_pages_without_geometry() -> None:
    assert (
        detect_chart_regions(
            lines=(),
            drawings=(),
            page_width=100.0,
            page_height=100.0,
        )
        == ()
    )


def test_chart_table_uses_only_text_associated_with_geometry() -> None:
    labels = tuple(
        SimpleNamespace(text=label, x0=x, x1=x + 8.0, y0=5.0, y1=15.0)
        for x, label in zip((15.0, 25.0, 35.0, 45.0, 55.0, 65.0), ("A", "B", "C", "D", "E", "F"))
    )
    values = tuple(
        SimpleNamespace(text=str(value), x0=x, x1=x + 8.0, y0=height + 20.0, y1=height + 28.0)
        for x, height, value in zip(
            (15.0, 25.0, 35.0, 45.0, 55.0, 65.0),
            (30, 42, 24, 50, 35, 46),
            (10, 20, 5, 30, 15, 25),
        )
    )
    outside = SimpleNamespace(text="Unrelated sentence", x0=0.0, x1=90.0, y0=75.0, y1=85.0)
    page = SimpleNamespace(
        width=100.0,
        height=100.0,
        get_page_program=lambda: SimpleNamespace(
            products=SimpleNamespace(
                lines=chart_marks(), drawings=(), runs=labels + values + (outside,)
            )
        ),
    )

    table = chart_table_html(SimpleNamespace(pages=(page,)))

    assert table.count("<tr>") == 7
    assert "<td>A</td><td>10</td>" in table
    assert "Unrelated sentence" not in table


def test_chart_table_uses_structured_figure_and_block_geometry() -> None:
    labels = tuple(
        SimpleNamespace(text=label, bbox=(x, 25.0, x + 8.0, 35.0))
        for x, label in zip((15.0, 25.0, 35.0), ("A", "B", "C"))
    )
    values = tuple(
        SimpleNamespace(text=str(value), bbox=(x, 45.0, x + 8.0, 55.0))
        for x, value in zip((15.0, 25.0, 35.0), (10, 20, 30))
    )
    source_page = SimpleNamespace(
        width=100.0,
        height=100.0,
        get_page_program=lambda: SimpleNamespace(
            products=SimpleNamespace(lines=(), drawings=(), runs=())
        ),
    )
    structured_page = SimpleNamespace(
        figures=(SimpleNamespace(bbox=(10.0, 20.0, 90.0, 80.0)),),
        elements=labels + values,
    )

    table = chart_table_html(
        SimpleNamespace(pages=(source_page,)),
        SimpleNamespace(pages=(structured_page,)),
    )

    assert "<td>A</td><td>10</td>" in table
    assert "<td>C</td><td>30</td>" in table


def test_positioned_tokens_split_multiline_blocks_and_deduplicate() -> None:
    runs = (
        SimpleNamespace(text="10\n20", bbox=(10.0, 10.0, 20.0, 30.0)),
        SimpleNamespace(text="10\n20", bbox=(10.0, 10.0, 20.0, 30.0)),
    )

    tokens = positioned_tokens(runs)

    assert [(token.text, token.numeric) for token in tokens] == [("10", True), ("20", True)]
    assert tokens[0].bbox == (10.0, 10.0, 20.0, 20.0)


def test_page_layout_items_omits_tables() -> None:
    page = SimpleNamespace(width=100.0, height=100.0, elements=())

    assert page_layout_items(page) == []

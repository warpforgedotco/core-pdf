from core_pdf.impl.engine.extraction.page_text.engine import (
    ResolvedLineRecord,
    build_text_blocks,
    render_page_blocks,
)


def line(
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    break_before: int = 1,
) -> ResolvedLineRecord:
    return ResolvedLineRecord(
        text=text,
        break_before=break_before,
        kind="native_line",
        source="native_text",
        bbox=(x0, y0, x1, y1),
        advance_bbox=None,
        ink_bbox=None,
        confidence=1.0,
        baseline=None,
        contributing_sources=("native_text",),
    )


def test_blocks_keep_indented_lines_in_one_column() -> None:
    blocks = build_text_blocks(
        (
            line("A paragraph starts", 100.0, 700.0, 400.0, 712.0),
            line("with an indented continuation", 150.0, 680.0, 450.0, 692.0),
        ),
        rotation=0,
    )

    assert len(blocks) == 1
    assert blocks[0].column_index == 0
    assert blocks[0].bbox == (100.0, 680.0, 450.0, 712.0)


def test_blocks_separate_paragraphs_but_preserve_column() -> None:
    blocks = build_text_blocks(
        (
            line("First paragraph", 100.0, 700.0, 300.0, 712.0),
            line("Second paragraph", 130.0, 660.0, 330.0, 672.0, break_before=2),
        ),
        rotation=90,
    )

    assert [block.column_index for block in blocks] == [0, 0]
    assert [block.rotation for block in blocks] == [90, 90]
    assert render_page_blocks(blocks) == "First paragraph\n\nSecond paragraph"


def test_blocks_keep_distinct_narrow_columns_separate() -> None:
    blocks = build_text_blocks(
        (
            line("left", 100.0, 700.0, 150.0, 712.0),
            line("right", 400.0, 700.0, 450.0, 712.0),
        ),
        rotation=0,
    )

    assert [block.column_index for block in blocks] == [0, 1]

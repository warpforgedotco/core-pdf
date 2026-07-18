from core_pdf.impl.engine.extraction.document_ir import page_result_to_document_page
from core_pdf.impl.engine.extraction.page_text.engine import (
    DocumentExtractionResult,
    DocumentExtractionSummary,
    PageExtractionResult,
    ResolvedLineRecord,
    TextBlock,
    build_text_blocks,
    related_page_records,
    render_page_blocks,
)
from core_pdf.impl.engine.spec.s_07_document.metadata_types import MetadataRecord

EMPTY_METADATA: MetadataRecord = {"info": {}, "xmp": None}


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
    assert blocks[0].kind == "paragraph"
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
    assert [block.kind for block in blocks] == ["paragraph", "paragraph"]
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


def test_blocks_classify_conservative_uppercase_headings() -> None:
    blocks = build_text_blocks(
        (line("A SHORT HEADING", 100.0, 700.0, 260.0, 712.0),),
        rotation=0,
    )

    assert blocks[0].kind == "heading"


def test_table_blocks_do_not_promote_cells_to_headings() -> None:
    blocks = build_text_blocks(
        (line("TOTAL 2,046 3,169", 100.0, 700.0, 300.0, 712.0),),
        rotation=90,
        page_class="table",
    )

    assert blocks[0].kind == "paragraph"


def test_blocks_classify_lists_without_rewriting_their_text() -> None:
    blocks = build_text_blocks(
        (
            line("1. First item", 100.0, 700.0, 250.0, 712.0),
            line("- Second item", 100.0, 680.0, 250.0, 692.0),
        ),
        rotation=0,
    )

    assert len(blocks) == 1
    assert blocks[0].kind == "list"
    assert len(blocks[0].lines) == 2
    assert render_page_blocks(blocks) == "1. First item\n- Second item"


def test_document_markdown_preserves_page_boundaries() -> None:
    first = line("first page", 0.0, 0.0, 100.0, 10.0)
    second = line("second page", 0.0, 0.0, 100.0, 10.0)
    pages = (
        PageExtractionResult(
            1,
            None,
            0.9,
            "native_text",
            "native_fast",
            (first,),
            (TextBlock(1, first.bbox, 0, 0, (first,)),),
        ),
        PageExtractionResult(
            2,
            None,
            0.9,
            "native_text",
            "native_fast",
            (second,),
            (TextBlock(1, second.bbox, 0, 0, (second,)),),
        ),
    )
    result = DocumentExtractionResult(
        metadata=EMPTY_METADATA,
        pages=pages,
        summary=DocumentExtractionSummary(2, 0, {"native_text": 2}, {"native_fast": 2}),
    )

    assert result.to_markdown() == "first page\fsecond page\f"


def test_empty_page_markdown_is_a_page_break() -> None:
    page = PageExtractionResult(1, None, 0.25, "empty", "skip", (), ())
    result = DocumentExtractionResult(
        metadata=EMPTY_METADATA,
        pages=(page,),
        summary=DocumentExtractionSummary(1, 1, {"empty": 1}, {"skip": 1}),
    )

    assert result.to_markdown() == "\f"


def test_pdf_related_records_adapt_into_core_document_page() -> None:
    result = PageExtractionResult(
        page_number=1,
        page_label=None,
        confidence=0.8,
        page_class="mixed",
        base_route="native_layout",
        resolved_lines=(),
        blocks=(),
        width=612.0,
        height=792.0,
        rotation=0,
        tables=({"rows": [["Name", "Value"]], "bbox": (10.0, 20.0, 100.0, 40.0)},),
        figures=({"kind": "image", "bbox": (10.0, 50.0, 100.0, 140.0), "width": 90},),
        links=({"bbox": (10.0, 150.0, 100.0, 165.0), "url": "https://example.test"},),
        annotations=({"subtype": "Text", "bbox": (10.0, 170.0, 20.0, 180.0)},),
        form_fields=({"name": "name", "field_type": "Tx", "value_text": "Ada"},),
    )

    page = page_result_to_document_page(result)

    assert page.tables[0].rows[0][0].text == "Name"
    assert page.figures[0].kind == "image"
    assert page.links[0].url == "https://example.test"
    assert page.annotations[0].subtype == "Text"
    assert page.form_fields[0].value_text == "Ada"


def test_related_extraction_failures_are_reported_as_diagnostics() -> None:
    class FailingPage:
        page_number = 4

        def get_links(self) -> list[object]:
            raise ValueError("malformed link annotation")

    records, diagnostics = related_page_records(FailingPage())

    assert records["links"] == ()
    assert diagnostics == (
        {
            "code": "related_extraction_failed",
            "message": "Failed to extract get_links: malformed link annotation",
            "severity": "warning",
            "page_number": 4,
        },
    )

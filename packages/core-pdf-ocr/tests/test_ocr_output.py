from core_pdf.impl._impl.output.model import Block, BlockKind, Page, TextLine


def test_text_view_exposes_lines_and_words() -> None:
    page = Page(
        page_number=1,
        blocks=(
            Block(
                order=0,
                kind=BlockKind.PARAGRAPH,
                lines=(
                    TextLine(text="one two", bbox=(0.0, 0.0, 70.0, 10.0), source="native"),
                    TextLine(text="three", bbox=(0.0, 20.0, 30.0, 30.0), source="ocr"),
                ),
            ),
        ),
    )

    assert [line.text for line in page.text_view.lines] == ["one two", "three"]
    assert [word.text for word in page.text_view.words] == ["one", "two", "three"]
    assert [word.block_index for word in page.text_view.words] == [0, 0, 0]
    assert [word.page_number for word in page.text_view.words] == [1, 1, 1]
    assert [word.bbox for word in page.text_view.words] == [None, None, None]

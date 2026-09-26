import pytest

from core_pdf.impl import extract_table_cleanup as cleanup
from core_pdf.impl.output_model import Table, TableAssociatedText, TableCell


def make_table(texts, *, geometry=True, source="stream"):
    rows = tuple(
        tuple(
            TableCell(
                i,
                j,
                text,
                bbox=(j * 10, 100 - (i + 1) * 10, (j + 1) * 10, 100 - i * 10) if geometry else None,
            )
            for j, text in enumerate(row)
        )
        for i, row in enumerate(texts)
    )
    return Table(
        3,
        rows,
        confidence=0.5,
        title=TableAssociatedText("Title"),
        caption=TableAssociatedText("Caption"),
        metadata={"source": source},
    )


@pytest.mark.parametrize("boundary", [2, 3, 4])
@pytest.mark.parametrize("geometry", [False, True])
def test_section_split_retains_cells_and_assigns_title_caption_to_ends(boundary, geometry):
    texts = [["Name", "Value"]] + [[str(i), str(i + 10)] for i in range(1, 6)]
    texts[boundary] = ["Next", "Section"]
    original = make_table(texts, geometry=geometry)
    first, second = cleanup.split_semantic_table(original)
    assert first.rows == original.rows[:boundary]
    assert second.rows == original.rows[boundary:]
    assert all(a is b for a, b in zip(first.rows + second.rows, original.rows, strict=True))
    assert first.title is original.title
    assert second.title is None
    assert first.caption is None
    assert second.caption is original.caption
    assert (first.order, second.order) == (3, 4)
    assert first.confidence == second.confidence == 0.5
    assert first.metadata == second.metadata == original.metadata
    if geometry:
        assert first.bbox == (0, 100 - boundary * 10, 20, 100)
        assert second.bbox == (0, 40, 20, 100 - boundary * 10)
    else:
        assert first.bbox is None
        assert second.bbox is None


@pytest.mark.parametrize("row_count", [0, 2, 5, 6, 10])
@pytest.mark.parametrize("numeric", [False, True])
def test_tables_without_section_headers_keep_identity(row_count, numeric):
    text = ["1", "2"] if numeric else ["ordinary", "words"]
    original = make_table([text] * row_count)
    (result,) = cleanup.split_semantic_table(original)
    assert result is original


def test_repeated_identical_headers_in_long_table_do_not_create_sections():
    texts = [[str(i), str(i + 10)] for i in range(10)]
    texts[2] = texts[6] = ["Name", "Value"]
    original = make_table(texts)
    (result,) = cleanup.split_semantic_table(original)
    assert result is original


@pytest.mark.parametrize(
    ("texts", "spans", "expected"),
    [
        ([], [], False),
        ([" "], [1], False),
        (["Title"], [1], False),
        (["Title"], [2], True),
        (["Name", "Value"], [1, 1], True),
        (["Name", "10"], [1, 1], False),
    ],
)
def test_semantic_headers_require_labels_or_a_spanning_title(texts, spans, expected):
    row = tuple(
        TableCell(0, i, text, column_span=span)
        for i, (text, span) in enumerate(zip(texts, spans, strict=True))
    )
    assert cleanup.semantic_header_row(row) is expected


@pytest.mark.parametrize(
    ("rows", "columns", "text", "expected"),
    [
        (12, 8, "word", True),
        (11, 8, "word", False),
        (12, 7, "word", False),
        (12, 8, "1234", False),
        (12, 8, "abcdefghijklmnop", False),
        (4, 2, "This is a complete sentence longer than a cell value", True),
        (4, 2, "value", False),
        (0, 0, "", True),
    ],
)
def test_stream_prose_detection_distinguishes_word_grids_and_numeric_tables(
    rows, columns, text, expected
):
    table = make_table([[text] * columns for _ in range(rows)])
    assert cleanup.stream_table_reads_like_prose(table) is expected


@pytest.mark.parametrize("source", ["stream", "grid"])
@pytest.mark.parametrize("columns", [7, 8])
@pytest.mark.parametrize("text", ["a b c d", "1234"])
def test_character_spaced_prose_requires_wide_stream_grid(source, columns, text):
    table = make_table([[text] * columns] * 4, source=source)
    assert cleanup.table_character_spaced_prose(table) is (
        source == "stream" and columns == 8 and text == "a b c d"
    )

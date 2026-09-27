import hashlib
import json
import sys
from pathlib import Path

from core_pdf import PdfDocument
from core_pdf.impl.output_model import (
    Annotation,
    Block,
    BlockKind,
    Diagnostic,
    Document,
    Figure,
    FormField,
    Link,
    Page,
    Table,
    TableAssociatedText,
    TableCell,
    TableRowBand,
    TextLine,
    TextSpan,
)

REPOSITORY = Path(__file__).resolve().parents[3]
FIXTURES = REPOSITORY / "tests" / "fixtures"
SNAPSHOT = Path(__file__).with_name("output_format_snapshots.json")
PAGE_LIMIT = 3
OUTPUT_FORMAT_FIXTURES = (
    "SCORE-Bench/src/VBA-29-0975-ARE-p002.pdf",
    "x-ray/tests/assets/rect_ordering_6.19.pdf",
    "pdfplumber/tests/pdfs/150109DSP-Milw-505-90D.pdf",
    "pdfminer.six/samples/acroform/AcroForm_TEST.pdf",
    "pypdf/sample-files/024-annotations/annotated_pdf.pdf",
    "PyMuPDF/tests/resources/merge-form1.pdf",
    "pdfminer.six/samples/simple1.pdf",
    "synthetic",
)


SYNTHETIC = "synthetic"
STYLES = ("bold", "italic", "underline", "strikeout", "mark", "superscript", "subscript")


def styled_line(text: str, *, underline: bool = False) -> TextLine:
    spans = (
        TextSpan(f"{text}<&>"),
        *(TextSpan(f" {name}", **{name: True}) for name in STYLES),
        TextSpan(" all", **dict.fromkeys(STYLES, True)),
    )
    combined = "".join(span.text for span in spans)
    return TextLine(combined, bbox=(1.0, 2.0, 3.0, 4.0), spans=spans, underline=underline)


def synthetic_document() -> Document:
    blocks = (
        Block(0, BlockKind.HEADING, (styled_line("Title"),), level=1),
        Block(1, BlockKind.HEADING, (TextLine("Untitled level"), TextLine("second")), level=None),
        Block(
            2,
            BlockKind.LIST,
            (
                TextLine("- dash item", bold=True),
                TextLine("1. numbered", underline=True, strikeout=True),
                TextLine("• bullet", spans=(TextSpan("• bu", italic=True), TextSpan("llet"))),
                TextLine("no prefix", mark=True, superscript=True),
                TextLine("a) lettered", subscript=True),
            ),
        ),
        Block(
            3,
            BlockKind.PARAGRAPH,
            (
                styled_line("Body", underline=True),
                TextLine("tail <b>", break_before=2, italic=True, bold=True),
            ),
        ),
    )
    header = (TableCell(0, 0, "H<1>"), TableCell(0, 1, "H2", column_span=2))
    body = (TableCell(1, 0, "a&b", row_span=2), TableCell(1, 1, "c"))
    tables = (
        Table(4, rows=(header, body)),
        Table(
            5,
            rows=(header, body, (TableCell(2, 0, "caption row"),)),
            title=TableAssociatedText("Table <1>", kind="title"),
            row_bands=(
                TableRowBand(0, kind="header"),
                TableRowBand(1, kind="body"),
                TableRowBand(2, kind="caption"),
            ),
        ),
        Table(
            6,
            rows=(header, body),
            row_bands=(TableRowBand(0, kind="header"), TableRowBand(1, kind="body")),
        ),
        Table(7),
    )
    figures = (Figure(8, (0.0, 0.0, 1.0, 1.0), kind="chart", metadata={"source": "vector"}),)
    page = Page(
        1,
        page_label="i",
        width=100.0,
        height=200.0,
        blocks=blocks,
        tables=tables,
        figures=figures,
        links=(Link((0.0, 0.0, 1.0, 1.0), "https://example.com", "uri", "link"),),
        annotations=(Annotation("Text", None, "note", {"page": 1, "path": ["a", 2]}),),
        form_fields=(FormField("field", "Tx", "value", options=("x", "y")),),
        header="Header",
        footer="Footer",
        diagnostics=(Diagnostic("code", "message", page_number=1),),
    )
    second = Page(2, blocks=(Block(0, BlockKind.PARAGRAPH, (TextLine("second page"),)),))
    return Document((page, second), metadata={"title": "Synthetic", "list": [1, "a", None]})


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def output_format_digests(fixture: str) -> dict[str, str]:
    if fixture == SYNTHETIC:
        document_output = synthetic_document()
    else:
        with PdfDocument(FIXTURES / fixture) as document:
            count = document.page_count()
            document_output = document.extract(pages=range(1, min(count, PAGE_LIMIT) + 1))
    return {
        "json": digest(document_output.to_json()),
        "json-compact": digest(document_output.to_json(indent=None, sort_keys=False)),
        "markdown": digest(document_output.to_markdown()),
        "html": digest(document_output.to_html()),
        "csv": digest(document_output.to_csv()),
        "csv-first-page": digest(document_output.to_csv(pages=[1])),
        "tei": digest(document_output.to_tei()),
        "tei-first-page": digest(document_output.to_tei(pages=[1])),
        "page-markdown": digest("\n".join(page.to_markdown() for page in document_output.pages)),
        "page-html": digest("\n".join(page.to_html() for page in document_output.pages)),
    }


def recorded() -> dict[str, dict[str, str]]:
    return json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}


def update() -> None:
    digests = {fixture: output_format_digests(fixture) for fixture in OUTPUT_FORMAT_FIXTURES}
    SNAPSHOT.write_text(json.dumps(digests, indent=1, sort_keys=True) + "\n")
    print(f"recorded {len(digests)} output format snapshots", file=sys.stderr)


if __name__ == "__main__":
    update()

import hashlib
import json
import sys
from pathlib import Path

from core_pdf import PdfDocument

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
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def output_format_digests(fixture: str) -> dict[str, str]:
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

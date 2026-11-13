"""CodSpeed benchmarks for core and the core_pdf_compat facades, on upstream test PDFs.

    uv run --all-packages --group benchmark pytest tests/benchmarks --codspeed

CI runs them under CodSpeed's CPU simulation (.github/workflows/codspeed.yml), which counts
instructions, so a result moves only when the code does. The calls are the ones
scripts/benchmark_facades.py times against the real libraries, on files picked from its run:
each takes tens of milliseconds, and most are where a facade was slowest next to its library.
Reading a file's bytes happens outside the timed call.
"""

import asyncio
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "tests" / "upstream"

PLUMBER = "pdfplumber/tests/pdfs"
FILES = {
    "pr-138": f"{PLUMBER}/pr-138-example.pdf",
    "federal-register": f"{PLUMBER}/federal-register-2020-17221.pdf",
    "issue-598": f"{PLUMBER}/issue-598-example.pdf",
    "issue-982": f"{PLUMBER}/issue-982-example.pdf",
    "issue-914-xmp": "pypdf/resources/issue-914-xmp-data.pdf",
    "fontsampler": "pypdf/resources/fontsampler.pdf",
    "rect-ordering": "x-ray/tests/assets/rect_ordering_3.20.pdf",
    "whitespace-redactions": "x-ray/tests/assets/whitespace_redactions.pdf",
    # Stale xref offsets: opening it falls back to a scan of the whole file.
    "da-1p": "unstructured/example-docs/pdf/DA-1p.pdf",
}
PAGES = 5


def data(name: str) -> bytes:
    path = UPSTREAM / FILES[name]
    if not path.exists():
        pytest.skip(f"{path} is missing; run git submodule update --init --recursive --depth 1")
    return path.read_bytes()


def path_of(name: str) -> str:
    data(name)
    return str(UPSTREAM / FILES[name])


def edited(editor: Any) -> Any:
    """The edit scripts/check_document_rewrite.py makes: a new Info dictionary."""
    from core_pdf.impl.base.types import PdfString  # noqa: PLC0415

    editor.trailer["Info"] = editor.add_object({"Title": PdfString(b"Rewritten")})
    return editor


@pytest.mark.parametrize("name", ["pr-138", "issue-598", "issue-914-xmp"])
def test_core_rewrite(benchmark: Any, name: str) -> None:
    from core_pdf import EditableDocument, PdfDocument  # noqa: PLC0415

    source = data(name)

    def run() -> None:
        with PdfDocument.open(source) as document:
            edited(EditableDocument(document)).save()

    benchmark(run)


@pytest.mark.parametrize("name", ["pr-138", "issue-914-xmp", "issue-982"])
def test_core_incremental(benchmark: Any, name: str) -> None:
    from core_pdf import EditableDocument, PdfDocument  # noqa: PLC0415

    source = data(name)

    def run() -> None:
        with PdfDocument.open(source) as document:
            edited(EditableDocument(document)).save_incremental()

    benchmark(run)


def pypdf_text(source: bytes, options: dict[str, Any]) -> None:
    from core_pdf_compat import pypdf  # noqa: PLC0415

    reader = pypdf.PdfReader(io.BytesIO(source))
    for page in reader.pages[:PAGES]:
        page.extract_text(**options)


def pypdf_write(make: Callable[[bytes], Any]) -> Callable[[bytes], None]:
    def run(source: bytes) -> None:
        make(source).write(io.BytesIO())

    return run


def pypdf_incremental(source: bytes) -> Any:
    from core_pdf_compat import pypdf  # noqa: PLC0415

    writer = pypdf.PdfWriter(io.BytesIO(source), incremental=True)
    writer.add_metadata({"/Title": "x"})
    return writer


def pypdf_clone(source: bytes) -> Any:
    from core_pdf_compat import pypdf  # noqa: PLC0415

    return pypdf.PdfWriter(clone_from=io.BytesIO(source))


def pypdf_append(source: bytes) -> Any:
    from core_pdf_compat import pypdf  # noqa: PLC0415

    writer = pypdf.PdfWriter()
    writer.append(pypdf.PdfReader(io.BytesIO(source)))
    return writer


def pypdf_reads(source: bytes) -> None:
    from core_pdf_compat import pypdf  # noqa: PLC0415

    reader = pypdf.PdfReader(io.BytesIO(source))
    numbers = [(n, g) for g, entries in reader.xref.items() for n in entries]
    numbers += [(n, 0) for n in reader.xref_objStm]
    for number, generation in numbers:
        value = reader.get_object(pypdf.generic.IndirectObject(number, generation, reader))
        if value is not None:
            value.write_to_stream(io.BytesIO())
        if isinstance(value, pypdf.generic.StreamObject):
            value.get_data()


PYPDF: dict[str, tuple[Callable[[bytes], None], tuple[str, ...]]] = {
    "text": (lambda source: pypdf_text(source, {}), ("pr-138", "federal-register")),
    "layout": (
        lambda source: pypdf_text(source, {"extraction_mode": "layout"}),
        ("pr-138", "federal-register"),
    ),
    "incremental": (pypdf_write(pypdf_incremental), ("issue-914-xmp", "issue-982")),
    "clone": (pypdf_write(pypdf_clone), ("pr-138", "issue-598")),
    "append": (pypdf_write(pypdf_append), ("pr-138", "fontsampler")),
    "reads": (pypdf_reads, ("pr-138", "issue-598")),
}


@pytest.mark.parametrize(
    ("operation", "name"),
    [(operation, name) for operation, (_, names) in PYPDF.items() for name in names],
)
def test_pypdf(benchmark: Any, operation: str, name: str) -> None:
    run, _ = PYPDF[operation]
    source = data(name)
    benchmark(run, source)


@pytest.mark.parametrize("name", ["rect-ordering", "whitespace-redactions", "pr-138", "da-1p"])
def test_pdfminer_extract_text(benchmark: Any, name: str) -> None:
    import core_pdf_compat.pdfminer as high_level  # noqa: PLC0415

    path = path_of(name)
    benchmark(lambda: high_level.extract_text(path, maxpages=PAGES))


@pytest.mark.parametrize("name", ["rect-ordering", "pr-138"])
def test_pdfminer_extract_pages(benchmark: Any, name: str) -> None:
    import core_pdf_compat.pdfminer as high_level  # noqa: PLC0415

    def walk(item: Any) -> None:
        if hasattr(item, "get_text"):
            item.get_text()
        if hasattr(type(item), "__iter__"):
            for child in item:
                walk(child)

    path = path_of(name)

    def run() -> None:
        for page in high_level.extract_pages(path, maxpages=PAGES):
            walk(page)

    benchmark(run)


PLUMBER_CHECKS: dict[str, Callable[[Any], object]] = {
    "chars": lambda page: page.chars,
    "words": lambda page: page.extract_words(),
    "text": lambda page: page.extract_text(),
    "layout": lambda page: page.extract_text(layout=True),
    "tables": lambda page: page.extract_tables(),
}


# Every check on one file; on the larger file only chars, as each check parses the pages first.
@pytest.mark.parametrize(
    ("name", "check"),
    [*(("pr-138", check) for check in PLUMBER_CHECKS), ("federal-register", "chars")],
)
def test_pdfplumber(benchmark: Any, check: str, name: str) -> None:
    from core_pdf_compat import pdfplumber  # noqa: PLC0415

    path = path_of(name)
    measure = PLUMBER_CHECKS[check]

    def run() -> None:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:PAGES]:
                measure(page)

    benchmark(run)


@pytest.mark.parametrize("name", ["pr-138"])
def test_unstructured_partition_fast(benchmark: Any, name: str) -> None:
    from core_pdf_compat.unstructured import partition_pdf  # noqa: PLC0415

    path = path_of(name)
    benchmark(lambda: [element.text for element in partition_pdf(path, strategy="fast")])


@pytest.mark.parametrize("name", ["pr-138", "federal-register"])
def test_llamaindex_pages(benchmark: Any, name: str) -> None:
    from core_pdf_compat.llamaindex import PDFReader  # noqa: PLC0415

    path = Path(path_of(name))
    benchmark(lambda: PDFReader().load_data(path))


@pytest.mark.parametrize("name", ["rect-ordering", "whitespace-redactions"])
def test_xray_inspect(benchmark: Any, name: str) -> None:
    from core_pdf_compat import xray  # noqa: PLC0415

    source = data(name)

    benchmark(lambda: asyncio.run(xray.inspect(source)))

"""Check that a save tells the truth about removed data, on the upstream PDFs.

uv run --all-packages python scripts/check_save_statements.py

Each file that opens and is not encrypted gets a last page that holds a plain marker in six
objects: its content stream, an image it draws, an annotation's text, and a font, its
descriptor and its embedded font program. Then that page is
removed and the file saved. The marker is what a caller believes is gone.

full: PdfWriter and EditableDocument.save. The marker must leave the file, and the save must
say it was a full rewrite.

incremental: PdfWriter(incremental=True) and EditableDocument.save_incremental. The marker must
stay, because the source's bytes stay, and the save must say so: SavedPdf.incremental and
SavedPdf.removed naming all six objects, and for the writer a last_save that names them and a
logged warning.

encrypted: PdfWriter.encrypt, with the page kept. The marker must not appear, and SavedPdf must
name the algorithm.

Last, the writer modules must hold no PDF structure as bytes, as only the serializer writes it.
"""

import ast
import collections
import glob
import logging
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from core_pdf import EditableDocument, PdfDocument
from core_pdf.impl.exceptions import PdfError
from core_pdf_compat.pypdf import PdfReader, PdfWriter
from core_pdf_compat.pypdf.errors import PyPdfError
from core_pdf_compat.pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)
from core_pdf_spec.exceptions import PdfParseError

PATTERNS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/*.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/*.pdf",
    "tests/upstream/pypdf/resources/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)
MARKER = b"corepdfremovedmarker7f3a91"
ALGORITHMS = ("RC4-40", "RC4-128", "AES-128", "AES-256-R5", "AES-256")
WRITER_MODULES = "packages/core-pdf-compat/src/core_pdf_compat/pypdf/_writer*.py"
PDF_STRUCTURE = re.compile(rb"endobj|endstream|trailer|%%EOF|startxref|\bxref\b|\bobj\b")


class Warnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.removed = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.removed += "removed objects stay readable" in record.getMessage()


def name(value: str) -> NameObject:
    return NameObject(value)


def with_marked_page(path: str) -> tuple[bytes, set[int]] | None:
    """The file with a marked last page, and the numbers of the six marked objects."""
    if PdfReader(path).is_encrypted:
        return None
    writer = PdfWriter(clone_from=path)
    page = writer.add_blank_page(200, 200)
    image = DecodedStreamObject()
    image.set_data(MARKER)
    image.update(
        {
            name("/Type"): name("/XObject"),
            name("/Subtype"): name("/Image"),
            name("/Width"): NumberObject(len(MARKER)),
            name("/Height"): NumberObject(1),
            name("/ColorSpace"): name("/DeviceGray"),
            name("/BitsPerComponent"): NumberObject(8),
        }
    )
    xobjects = DictionaryObject({name("/Marked"): writer._add_object(image)})
    program = DecodedStreamObject()
    program.set_data(b"%!PS-AdobeFont-1.0: " + MARKER)
    descriptor = DictionaryObject(
        {
            name("/Type"): name("/FontDescriptor"),
            # Names are never encrypted, so the marker goes in the font program only.
            name("/FontName"): name("/MarkedFont"),
            name("/FontFile"): writer._add_object(program),
        }
    )
    font = DictionaryObject(
        {
            name("/Type"): name("/Font"),
            name("/Subtype"): name("/Type1"),
            name("/BaseFont"): name("/MarkedFont"),
            name("/FontDescriptor"): writer._add_object(descriptor),
        }
    )
    fonts = DictionaryObject({name("/FM"): writer._add_object(font)})
    page[name("/Resources")] = DictionaryObject({name("/XObject"): xobjects, name("/Font"): fonts})
    content = DecodedStreamObject()
    content.set_data(b"q 100 0 0 10 0 0 cm /Marked Do Q BT /FM 12 Tf (" + MARKER + b") Tj ET")
    page[name("/Contents")] = writer._add_object(content)
    annotation = DictionaryObject(
        {
            name("/Type"): name("/Annot"),
            name("/Subtype"): name("/Text"),
            name("/Rect"): ArrayObject([NumberObject(n) for n in (0, 0, 20, 20)]),
            name("/Contents"): TextStringObject(MARKER.decode()),
        }
    )
    page[name("/Annots")] = ArrayObject([writer._add_object(annotation)])
    output = BytesIO()
    writer.write(output)
    last = PdfReader(output).pages[-1]
    font = last["/Resources"]["/Font"]["/FM"]
    marked = {
        last.raw_get("/Contents").idnum,
        last["/Resources"]["/XObject"].raw_get("/Marked").idnum,
        last["/Annots"][0].idnum,
        last["/Resources"]["/Font"].raw_get("/FM").idnum,
        font.raw_get("/FontDescriptor").idnum,
        font["/FontDescriptor"].raw_get("/FontFile").idnum,
    }
    return output.getvalue(), marked


def check_writer(source: bytes, marked: set[int], seen: Warnings, report: Any) -> None:
    writer = PdfWriter(clone_from=BytesIO(source))
    del writer.pages[-1]
    full = writer.write_bytes()
    report("writer full: marker left in the file", MARKER in full)
    report("writer full: claims incremental", full.incremental)
    saved = writer.last_save
    report(
        "writer full: last_save is wrong", saved is None or saved.kind != "full" or saved.removed
    )

    writer = PdfWriter(BytesIO(source), incremental=True)
    del writer.pages[-1]
    before = seen.removed
    update = writer.write_bytes()
    report("writer incremental: source bytes changed", not update.startswith(source))
    report("writer incremental: claims a full rewrite", not update.incremental)
    saved = writer.last_save
    report("writer incremental: last_save is wrong", saved is None or saved.kind != "incremental")
    report(
        "writer incremental: removed not in last_save", saved is None or marked - set(saved.removed)
    )
    report("writer incremental: removed not in SavedPdf", marked - set(update.removed))
    report("writer incremental: no warning", seen.removed == before)


def without_last_page(editor: EditableDocument) -> None:
    root = editor.get_object(editor.trailer["Root"])
    pages = root["Pages"]
    node = editor.edit_object(pages)
    node["Kids"] = list(node["Kids"])[:-1]
    node["Count"] = node["Count"] - 1


def check_core(source: bytes, marked: set[int], report: Any) -> None:
    with PdfDocument.open(source) as document:
        editor = EditableDocument(document)
        without_last_page(editor)
        full = editor.save()
        update = editor.save_incremental()
    report("core full: marker left in the file", MARKER in full)
    report("core full: claims incremental", full.incremental)
    report("core full: lists removed objects", full.removed)
    report("core incremental: source bytes changed", not update.startswith(source))
    report("core incremental: claims a full rewrite", not update.incremental)
    report("core incremental: removed not in SavedPdf", marked - set(update.removed))


def check_encrypted(source: bytes, algorithm: str, report: Any) -> None:
    writer = PdfWriter(clone_from=BytesIO(source))
    writer.encrypt("user", "owner", algorithm=algorithm)
    output = writer.write_bytes()
    report("encrypted: marker readable in the file", MARKER in output)
    report("encrypted: says it is not", not output.encrypted)


def check_writer_modules(report: Any) -> None:
    for path in glob.glob(WRITER_MODULES):
        for node in ast.walk(ast.parse(Path(path).read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
                report(f"PDF structure in {path}", bool(PDF_STRUCTURE.search(node.value)))


def main() -> int:
    seen = Warnings()
    logging.getLogger().addHandler(seen)
    counts: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    paths = sorted({path for pattern in PATTERNS for path in glob.glob(pattern, recursive=True)})
    path = ""

    def report(claim: str, failed: Any) -> None:
        if failed:
            counts[f"FAILED {claim}"] += 1
            if len(problems) < 40:
                problems.append(f"{path}: {claim}")

    check_writer_modules(report)
    for index, path in enumerate(paths):
        try:
            marked = with_marked_page(path)
            if marked is None:
                counts["encrypted, skipped"] += 1
                continue
            source, numbers = marked
            check_writer(source, numbers, seen, report)
            check_core(source, numbers, report)
            check_encrypted(source, ALGORITHMS[index % len(ALGORITHMS)], report)
            counts["files checked"] += 1
        except (
            PdfError,
            PdfParseError,
            PyPdfError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            AttributeError,
        ) as error:
            counts["files that could not run"] += 1
            if len(problems) < 40:
                problems.append(f"{path}: {error!r}"[:200])
    for name, count in sorted(counts.items()):
        print(f"{count:>8}  {name}")
    for problem in problems:
        print(problem)
    failed = any(name.startswith("FAILED") for name in counts)
    return 1 if failed or not counts["files checked"] else 0


if __name__ == "__main__":
    sys.exit(main())

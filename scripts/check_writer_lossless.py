"""Check that PdfWriter leaves the objects nobody changed alone, on the upstream PDFs.

uv run --all-packages python scripts/check_writer_lossless.py

Three checks, on every file that opens and is not encrypted:

clone: a full rewrite with no edits. Each source object is compared with its copy in the
output, reparsed by the lexer, with references mapped to the new numbers. Cloning rebuilds
the page tree, so its nodes are left out, and so are the entries of a page that depend on
it. An object with no references and an unchanged number of
bytes in its body is also compared byte for byte, against the source file.

incremental: an update that only changes /Info. The output must start with the source's
bytes, and every other object must read the same through the new xref.

The rewrite renumbers objects and normalizes syntax, so only values are promised there. An
incremental update is the way to keep bytes.
"""

import collections
import glob
import logging
import sys
from io import BytesIO
from typing import Any

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfError
from core_pdf.impl.types import PdfName, PdfReference
from core_pdf_compat.pypdf import PdfReader, PdfWriter
from core_pdf_compat.pypdf.errors import PyPdfError
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.stream import PdfStream

PATTERNS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/*.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/*.pdf",
    "tests/upstream/pypdf/resources/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)
# Entries of a page that a clone rewrites on purpose: it flattens the tree and what the page
# inherits. What /Resources refers to is compared as objects of its own.
PAGE_KEYS = {"Parent", "Resources", "MediaBox", "CropBox", "Rotate"}


def plain(value: Any, renumber: dict[int, int] | None) -> Any:
    """A comparable form, with source references mapped to output numbers."""
    if isinstance(value, PdfReference):
        number = value.object_number
        if renumber is None:
            return ("ref", number)
        return ("ref", renumber[number]) if number in renumber else ("unmapped", number)
    if isinstance(value, PdfStream):
        return {
            **plain({k: v for k, v in value.dictionary.items() if str(k) != "Length"}, renumber),
            "<data>": bytes(value.raw_data),
        }
    if isinstance(value, dict):
        return {str(getattr(k, "value", k)): plain(v, renumber) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item, renumber) for item in value]
    if isinstance(value, (PdfName, str)):
        # The lexer gives a few well-known names as plain strings.
        return ("name", str(getattr(value, "value", value)))
    if hasattr(value, "data"):
        return ("string", bytes(value.data))
    return value


def matches(before: Any, after: Any, source: Any, written: Any) -> bool:
    """Compare two plain values. pypdf copies a stream in an array anew each time, so such a
    reference has no entry in the translation and is compared by the stream it reaches.
    """
    if isinstance(before, tuple) and before[0] == "unmapped":
        if not (isinstance(after, tuple) and after[0] == "ref"):
            return False
        original = source.resolver.resolve(PdfReference(before[1], 0))
        copy = written.resolver.resolve(PdfReference(after[1], 0))
        if isinstance(original, PdfStream) and isinstance(copy, PdfStream):
            return bytes(original.raw_data) == bytes(copy.raw_data)
        return original is None and copy is None
    both_references = isinstance(before, tuple) and isinstance(after, tuple)
    if both_references and before != after and before[0] == after[0] == "ref":
        first = written.resolver.resolve(PdfReference(before[1], 0))
        second = written.resolver.resolve(PdfReference(after[1], 0))
        if isinstance(first, PdfStream) and isinstance(second, PdfStream):
            return bytes(first.raw_data) == bytes(second.raw_data)
    if isinstance(before, dict) and isinstance(after, dict):
        return before.keys() == after.keys() and all(
            matches(before[key], after[key], source, written) for key in before
        )
    if isinstance(before, list) and isinstance(after, list):
        return len(before) == len(after) and all(
            matches(*pair, source, written) for pair in zip(before, after, strict=True)
        )
    return before == after


def object_type(value: Any) -> str | None:
    kind = value.get("Type") if isinstance(value, dict) else None
    return None if kind is None else str(getattr(kind, "value", kind))


def without_page_keys(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in PAGE_KEYS}


def raw_body(document: Any, number: int) -> bytes | None:
    """The source bytes of an object's body, or None if it is in an object stream."""
    for key, entry in document.xref.items():
        if key >> 16 == number and entry.in_use and entry.object_stream is None:
            data = document.raw_data
            start = bytes(data[entry.offset : entry.offset + 32]).find(b"obj")
            end = data.find(b"endobj", entry.offset)
            return None if start < 0 or end < 0 else bytes(data[entry.offset + start + 3 : end])
    return None


def check_clone(path: str, counts: collections.Counter[str], problems: list[str]) -> None:
    reader = PdfReader(path)
    if reader.is_encrypted:
        counts["encrypted, skipped"] += 1
        return
    source = reader._pdf
    assert source is not None
    writer = PdfWriter(clone_from=reader)
    output = BytesIO()
    writer.write(output)
    renumber = {k: v for k, v in writer._translation(reader).items() if isinstance(k, int)}
    with PdfDocument.open(output.getvalue()) as written:
        for number, new_number in sorted(renumber.items()):
            original = source.resolver.resolve(PdfReference(number, 0))
            kind = object_type(original)
            if original is None or kind == "Pages":
                counts["left out (page tree nodes and nulls)"] += 1
                continue
            copy = written.resolver.resolve(PdfReference(new_number, 0))
            counts["objects compared by value"] += 1
            before, after = plain(original, renumber), plain(copy, None)
            if kind == "Page":
                before, after = without_page_keys(before), without_page_keys(after)
            if not matches(before, after, source, written):
                counts["objects that differ"] += 1
                if len(problems) < 20:
                    problems.append(f"{path} object {number}: {str(original)[:160]}")
            elif isinstance(original, PdfStream):
                counts["streams with identical data"] += 1
            body = raw_body(source, number)
            if body is None or isinstance(original, PdfStream):
                continue
            if b" R" not in body:
                counts["bodies compared byte for byte"] += 1
                written_body = raw_body(written, new_number)
                if written_body is not None and written_body.strip() == body.strip():
                    counts["bodies byte identical"] += 1
                else:
                    counts["bodies normalized (same value)"] += 1


def check_incremental(path: str, counts: collections.Counter[str], problems: list[str]) -> None:
    with open(path, "rb") as handle:
        data = handle.read()
    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted:
        return
    writer = PdfWriter(reader, incremental=True)
    writer.add_metadata({"/Keywords": "core-pdf lossless check"})
    changed = {reference.idnum for reference in writer.list_objects_in_increment()}
    output = BytesIO()
    writer.write(output)
    saved = output.getvalue()
    counts["incremental updates"] += 1
    if not saved.startswith(data):
        counts["incremental updates that changed the source bytes"] += 1
        problems.append(f"{path}: the source's bytes changed")
    with PdfDocument.open(data) as before, PdfDocument.open(saved) as after:
        seen: set[int] = set()
        pending: list[Any] = [before.trailer_dict]
        while pending:
            value = pending.pop()
            if isinstance(value, PdfReference):
                number = value.object_number
                if number in seen:
                    continue
                seen.add(number)
                old = before.resolver.resolve(value)
                pending.append(old)
                if number not in changed:
                    counts["objects read through the update"] += 1
                    if plain(old, None) != plain(after.resolver.resolve(value), None):
                        counts["objects changed by the update"] += 1
                        problems.append(f"{path} object {number} changed")
            elif isinstance(value, PdfStream):
                pending.extend(value.dictionary.values())
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)


def main() -> int:
    logging.disable(logging.CRITICAL)
    counts: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    paths = sorted({path for pattern in PATTERNS for path in glob.glob(pattern, recursive=True)})
    for path in paths:
        for check in (check_clone, check_incremental):
            try:
                check(path, counts, problems)
            except (
                PdfError,
                PdfParseError,
                PyPdfError,
                ValueError,
                KeyError,
                TypeError,
                IndexError,
            ) as error:
                counts[f"{check.__name__} could not run"] += 1
                if len(problems) < 40:
                    problems.append(f"{path} {check.__name__}: {error!r}"[:200])
    for name, count in sorted(counts.items()):
        print(f"{count:>8}  {name}")
    for problem in problems:
        print(problem)
    return 1 if counts["objects that differ"] or counts["objects changed by the update"] else 0


if __name__ == "__main__":
    sys.exit(main())

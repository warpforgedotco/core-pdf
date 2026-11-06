"""Check that the llama_index facade returns the same Documents as llama-index-readers-file.

uv run --all-packages --with llama-index-readers-file==0.7.0 \
    --with llama-index-core==0.14.25 --with "pypdf[full]==6.21.4" \
    python scripts/check_llamaindex_output.py [--limit N]

Each upstream sample PDF goes through llama_index's PDFReader and the facade's PDFReader the
same way, and the Documents are compared: how many there are and their order, each one's text,
its metadata (keys in order, and values such as page_label and file_name), and its
get_content() under every MetadataMode. A file the real reader cannot read must fail in the
facade with the same exception class.

pages: PDFReader().load_data(path), one Document per page.
full: PDFReader(return_full_document=True).load_data(path), one Document for the file.
extra: PDFReader().load_data(path, extra_info), with extra_info overriding file_name.

pypdf is pinned to the version the pypdf facade follows, since llama_index's text is pypdf's
extract_text(). The suite runner leaves it unpinned; a newer pypdf changes text extraction,
which is the pypdf facade's concern rather than this one's.

llama_index wraps load_data in tenacity's @retry, so a file it cannot read raises
tenacity.RetryError after three attempts. The facade cannot raise tenacity's class without
depending on it, so the check compares the class of the error the last attempt raised.

Files run in sorted order, at most --limit of them (default 200, all the samples, about a minute
or two). Every difference fails the check unless DEPARTURES lists it, with the reason the
facade does not follow llama_index there. A listed file that matches fails it too, so the list
stays true.
"""

import argparse
import collections
import glob
import logging
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tenacity  # ty: ignore[unresolved-import]
from llama_index.core.schema import MetadataMode as RealMetadataMode  # ty: ignore[unresolved-import]
from llama_index.readers.file import PDFReader as RealPDFReader  # ty: ignore[unresolved-import]

from core_pdf_compat.llamaindex import MetadataMode as FacadeMetadataMode
from core_pdf_compat.llamaindex import PDFReader as FacadePDFReader

PATTERNS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/*.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/*.pdf",
    "tests/upstream/pypdf/resources/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)
EXTRA_INFO = {"source": "check", "file_name": "renamed.pdf"}
DEPARTURES: dict[tuple[str, str], str] = {}


def pages(reader: Any, path: Path) -> list[Any]:
    return reader().load_data(path)


def full(reader: Any, path: Path) -> list[Any]:
    return reader(return_full_document=True).load_data(path)


def extra(reader: Any, path: Path) -> list[Any]:
    return reader().load_data(path, dict(EXTRA_INFO))


MODES: dict[str, Callable[[Any, Path], list[Any]]] = {
    "pages": pages,
    "full": full,
    "extra": extra,
}


def summary(documents: list[Any], modes: Any) -> list[tuple[object, ...]]:
    return [
        (
            document.text,
            list(document.metadata.items()),
            *(document.get_content(metadata_mode=mode) for mode in modes),
        )
        for document in documents
    ]


FIELDS = ("text", "metadata", "content NONE", "content LLM", "content EMBED", "content ALL")
REAL_MODES = (
    RealMetadataMode.NONE,
    RealMetadataMode.LLM,
    RealMetadataMode.EMBED,
    RealMetadataMode.ALL,
)
FACADE_MODES = (
    FacadeMetadataMode.NONE,
    FacadeMetadataMode.LLM,
    FacadeMetadataMode.EMBED,
    FacadeMetadataMode.ALL,
)


def first_difference(expected: list[tuple[object, ...]], actual: list[tuple[object, ...]]) -> str:
    if len(expected) != len(actual):
        return f"llama_index {len(expected)} documents, facade {len(actual)}"
    for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
        for field, x, y in zip(FIELDS, want, got, strict=True):
            if x != y:
                if isinstance(x, str) and isinstance(y, str):
                    at = next(
                        (i for i, (a, b) in enumerate(zip(x, y, strict=False)) if a != b),
                        min(len(x), len(y)),
                    )
                    x, y = x[max(0, at - 10) : at + 30], y[max(0, at - 10) : at + 30]
                    field = f"{field} at char {at}"
                return f"document {index} {field}: llama_index {x!r}, facade {y!r}"
    return "same"


def real_error(error: BaseException) -> BaseException:
    if isinstance(error, tenacity.RetryError):
        inner = error.last_attempt.exception()
        if inner is not None:
            return inner
    return error


def error_name(error: BaseException | None) -> str:
    """The class's dotted name, the facade's read as the library it stands for."""
    if error is None:
        return "no error"
    kind = type(error)
    return f"{kind.__module__.removeprefix('core_pdf_compat.')}.{kind.__qualname__}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    limit = parser.parse_args().limit
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    counts: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    paths = sorted({path for pattern in PATTERNS for path in glob.glob(pattern, recursive=True)})
    for name_path in paths[:limit]:
        path = Path(name_path)
        for name, load in MODES.items():
            listed = DEPARTURES.get((name, path.name))
            expected_error: BaseException | None = None
            actual_error: BaseException | None = None
            expected: list[tuple[object, ...]] = []
            actual: list[tuple[object, ...]] = []
            try:
                expected = summary(load(RealPDFReader, path), REAL_MODES)
            except Exception as error:  # noqa: BLE001
                expected_error = real_error(error)
            try:
                actual = summary(load(FacadePDFReader, path), FACADE_MODES)
            except Exception as error:  # noqa: BLE001
                actual_error = error
            if expected_error is not None or actual_error is not None:
                same = error_name(expected_error) == error_name(actual_error)
                difference = (
                    f"llama_index raises {expected_error!r}, facade {actual_error!r}"
                    if actual_error is not None
                    else f"llama_index raises {expected_error!r}, facade reads it"
                )
                if actual_error is not None and expected_error is None:
                    difference = f"facade raises {actual_error!r}"
                kind = "same error"
            else:
                same = expected == actual
                difference = "" if same else first_difference(expected, actual)
                kind = "identical"
            if same:
                counts[f"{name}: {kind}"] += 1
                if listed is not None:
                    counts[f"{name}: LISTED BUT IDENTICAL"] += 1
                    problems.append(f"{name} {path}: listed as a departure but identical")
            elif listed is not None:
                counts[f"{name}: differ, listed departure"] += 1
            else:
                counts[f"{name}: DIFFER"] += 1
                problems.append(f"{name} {path} {difference}"[:400])
    print(f"{min(limit, len(paths))} of {len(paths)} files")
    for key, count in sorted(counts.items()):
        print(f"{count:>6}  {key}")
    for problem in problems:
        print(problem)
    failed = any(key.split(": ", 1)[1].isupper() for key in counts)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

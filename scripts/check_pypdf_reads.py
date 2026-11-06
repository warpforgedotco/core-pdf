"""Check that the pypdf facade reads every object as pypdf does, on the upstream PDFs.

uv run --all-packages --with "pypdf[full]==6.15.0" python scripts/check_pypdf_reads.py

Each file that real pypdf opens unencrypted is read three ways: as it is, and encrypted by real
pypdf with RC4-128 and with AES-256, then opened with the password. Every object in the xref
and in object streams is read through both libraries and compared: its bytes as each
library's write_to_stream writes them, a stream's decoded data, and the error class when a
read fails. Any difference fails the check.
"""

import collections
import glob
import io
import logging
import sys
import warnings
from collections.abc import Iterator
from typing import Any

import pypdf as real  # ty: ignore[unresolved-import]

from core_pdf_compat import pypdf as facade

PATTERNS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/*.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/*.pdf",
    "tests/upstream/pypdf/resources/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)
ALGORITHMS = ("RC4-128", "AES-256")
PASSWORD = "user"


def variants(data: bytes) -> Iterator[tuple[str, bytes, str | None]]:
    yield "plain", data, None
    for algorithm in ALGORITHMS:
        try:
            writer = real.PdfWriter(clone_from=io.BytesIO(data))
            writer.encrypt(PASSWORD, "owner", algorithm=algorithm)
            output = io.BytesIO()
            writer.write(output)
        except Exception:  # noqa: BLE001, S112
            continue
        yield algorithm, output.getvalue(), PASSWORD


def written(value: Any) -> bytes:
    output = io.BytesIO()
    value.write_to_stream(output)
    return output.getvalue()


def read(library: Any, reader: Any, number: int, generation: int) -> Any:
    try:
        value = reader.get_object(library.generic.IndirectObject(number, generation, reader))
    except Exception as error:  # noqa: BLE001
        return ("error", type(error).__name__)
    if value is None:
        return None
    try:
        body: Any = written(value)
    except Exception as error:  # noqa: BLE001
        body = ("write error", type(error).__name__)
    data: Any = None
    if isinstance(value, library.generic.StreamObject):
        try:
            data = value.get_data()
        except Exception as error:  # noqa: BLE001
            data = ("data error", type(error).__name__)
    return body, data


def objects(library: Any, data: bytes, password: str | None) -> dict[int, Any]:
    reader = library.PdfReader(io.BytesIO(data))
    if password is not None:
        reader.decrypt(password)
    found: dict[int, Any] = {}
    for generation, entries in reader.xref.items():
        for number in entries:
            found[number] = read(library, reader, number, generation)
    for number in reader.xref_objStm:
        if number not in found:
            found[number] = read(library, reader, number, 0)
    return found


def main() -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    counts: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    paths = sorted({path for pattern in PATTERNS for path in glob.glob(pattern, recursive=True)})
    for path in paths:
        with open(path, "rb") as handle:
            data = handle.read()
        try:
            encrypted = real.PdfReader(io.BytesIO(data)).is_encrypted
        except Exception:  # noqa: BLE001
            counts["pypdf cannot open"] += 1
            continue
        if encrypted:
            counts["encrypted source, skipped"] += 1
            continue
        for name, variant, password in variants(data):
            try:
                expected = objects(real, variant, password)
            except Exception:  # noqa: BLE001
                counts[f"{name}: pypdf cannot read"] += 1
                continue
            try:
                actual = objects(facade, variant, password)
            except Exception as error:  # noqa: BLE001
                counts[f"{name}: FACADE FAILS"] += 1
                problems.append(f"{name} {path}: {error!r}"[:200])
                continue
            counts[f"{name}: objects compared"] += len(expected)
            differing = [number for number in expected if expected[number] != actual.get(number)]
            if differing:
                counts[f"{name}: OBJECTS DIFFER"] += len(differing)
                number = differing[0]
                problems.append(
                    f"{name} {path} object {number}: pypdf {str(expected[number])[:120]}, "
                    f"facade {str(actual.get(number))[:120]}"
                )
            else:
                counts[f"{name}: files identical"] += 1
    for key, count in sorted(counts.items()):
        print(f"{count:>8}  {key}")
    for problem in problems[:40]:
        print(problem)
    failed = any(key.split(": ", 1)[-1].isupper() for key in counts)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

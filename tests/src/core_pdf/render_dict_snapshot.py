import dataclasses
import hashlib
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import numpy

from core_pdf import PdfDocument
from core_pdf.impl.capture_program import CapturedProgram
from core_pdf.impl.capture_records import CapturedPath

REPOSITORY = Path(__file__).resolve().parents[3]
FIXTURES = REPOSITORY / "tests" / "fixtures"
SNAPSHOT = Path(__file__).with_name("render_dict_snapshots.json")
PAGE_LIMIT = 2
RENDER_DICT_FIXTURES = (
    "PyMuPDF/tests/resources/test_5054.pdf",
    "PyMuPDF/tests/resources/test_4936.pdf",
    "PyMuPDF/tests/resources/test_3950.pdf",
    "pdfminer.six/samples/contrib/issue_1165_linewidth.pdf",
    "pdfminer.six/samples/nonfree/cmp_itext_logo.pdf",
    "pypdf/resources/Sample_Td-matrix.pdf",
    "pypdf/resources/reportlab-inline-image.pdf",
    "pypdf/sample-files/024-annotations/annotated_pdf.pdf",
    "PyMuPDF/tests/resources/merge-form1.pdf",
    "pikepdf/tests/resources/pal-1bit-trivial.pdf",
    "pikepdf/tests/resources/formxobject.pdf",
)


def canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)) and not isinstance(value, Enum):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Enum):
        return repr(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "bytes:" + hashlib.sha256(bytes(value)).hexdigest()
    if isinstance(value, numpy.ndarray):
        digest = hashlib.sha256(numpy.ascontiguousarray(value).tobytes()).hexdigest()
        return f"ndarray:{value.dtype}:{value.shape}:{digest}"
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, *(canonical(item) for item in value)]
    if isinstance(value, dict):
        return ["dict", *([canonical(key), canonical(item)] for key, item in value.items())]
    if isinstance(value, CapturedPath):
        return ["CapturedPath", canonical(value.subpaths), canonical(value.deferred_columns())]
    if isinstance(value, CapturedProgram):
        return ["CapturedProgram", len(value.commands)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            type(value).__qualname__,
            *(canonical(getattr(value, field.name)) for field in dataclasses.fields(value)),
        ]
    names = getattr(type(value), "__fields__", None)
    if names is None:
        names = [
            name
            for klass in type(value).__mro__
            for name in getattr(klass, "__slots__", ())
            if not name.startswith("_")
        ]
    if not names:
        return type(value).__qualname__
    return [type(value).__qualname__, *(canonical(getattr(value, name, None)) for name in names)]


def render_dict_digests(fixture: str) -> list[str]:
    with PdfDocument(FIXTURES / fixture) as document:
        digests = []
        for index in range(min(document.page_count(), PAGE_LIMIT)):
            payload = canonical(document.pages[index].render().to_dict())
            encoded = json.dumps(payload, separators=(",", ":"))
            digests.append(hashlib.sha256(encoded.encode()).hexdigest())
        return digests


def recorded() -> dict[str, list[str]]:
    return json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}


def update() -> None:
    digests = {fixture: render_dict_digests(fixture) for fixture in RENDER_DICT_FIXTURES}
    SNAPSHOT.write_text(json.dumps(digests, indent=1, sort_keys=True) + "\n")
    print(f"recorded {len(digests)} render dict snapshots", file=sys.stderr)


if __name__ == "__main__":
    update()

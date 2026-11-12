# SPDX-License-Identifier: AGPL-3.0-only

"""Check that StrictPdfLexer parses exactly as the spec's strict lexer does.

StrictPdfLexer reads arrays and dictionaries with core-pdf-cythonized's ObjectScanner in strict
mode, which must bail on everything the strict lexer reads differently. This parses every
top-level object of the upstream test PDFs with both lexers, then objects generated around the
cases where the recovery reader and the strict lexer part ways (invalid and null name escapes,
non-canonical references, string escapes and line ends, malformed numbers, hex strings, stray
keywords, repeated keys, truncation), and fails on the first value, end position or error that
differs.

uv run --all-packages python scripts/check_strict_object_scanner.py [generated objects]
"""

from __future__ import annotations

import glob
import random
import sys
from typing import Any

from core_pdf import PdfDocument
from core_pdf.impl.recovery_lexer import StrictPdfLexer
from core_pdf.impl.spec.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.types import PdfReference

CORPUS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pypdf/resources/**/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
    "tests/upstream/pdfminer.six/samples/**/*.pdf",
)
STRINGS = (
    b"(plain)", b"(a\\nb)", b"(\\r\\n)", b"(\\101\\7\\0012)", b"(\\q)", b"(a\\\r\nb)",
    b"(a\\\rb)", b"(\\\nx)", b"(nested (paren) ok)", b"(\\(x)", b"(unbalanced", b"(\r\n)",
    b"(\r)", b"(\\8)", b"()",
)  # fmt: skip
NUMBERS = (
    b"0", b"-0", b"+5", b"-.5", b".5", b"5.", b"1.2.3", b"--5", b"+-1", b"00012", b"3.14159",
    b"1e5", b"-", b"+", b".", b"12345678901234567890", b"-0.0", b"1_0",
)  # fmt: skip
NAMES = (b"/A", b"/Type", b"/A#20B", b"/A#2", b"/A#GG", b"/A#00", b"/#41", b"/", b"/Contents")
HEX = (b"<41>", b"<4142 43>", b"<414>", b"<>", b"< 4 1 >", b"<4G>", b"<41\n42>")
KEYWORDS = (b"true", b"false", b"null", b"obj", b"R", b"endobj", b"foo", b"stream", b"Null")
SPACES = (b" ", b"\n", b"\r\n", b"\t", b"\x0c", b"\x00", b"%comment\n", b"")
TAILS = (b"\nendobj\n", b" endobj", b"endobj", b"\nstream\r\nxx\nendstream\nendobj", b"")


def canonical(value: Any) -> Any:
    if isinstance(value, PdfStream):
        return ("stream", canonical(value.dictionary), bytes(value.raw_data))
    if isinstance(value, dict):
        return ("dict", tuple((repr(key), canonical(item)) for key, item in value.items()))
    if isinstance(value, list):
        return ("list", tuple(canonical(item) for item in value))
    return (type(value).__name__, repr(value))


def outcome(lexer: PdfLexer, offset: int, reference: PdfReference) -> tuple[Any, ...]:
    lexer.rewind(offset)
    try:
        value = lexer.parse_indirect_object(expected_reference=reference)
    except PdfParseError as error:
        return ("error", str(error))
    except Exception as error:  # noqa: BLE001
        return ("raised", type(error).__name__, str(error))
    return ("ok", canonical(value), lexer.pos)


def generated(rng: random.Random, depth: int = 0) -> bytes:
    roll = rng.random()
    if depth > 3 or roll < 0.55:
        kind = rng.choice((STRINGS, NUMBERS, NAMES, HEX, KEYWORDS))
        value = rng.choice(kind)
        if kind is NUMBERS and rng.random() < 0.2:
            value += b" " + rng.choice(NUMBERS) + b" R"
        return value
    if roll < 0.78:
        items = [generated(rng, depth + 1) for _ in range(rng.randint(0, 5))]
        return b"[" + rng.choice(SPACES).join(items) + rng.choice((b"]", b"]", b"]", b""))
    entries = [
        (rng.choice(NAMES) if rng.random() < 0.9 else rng.choice(NUMBERS))
        + rng.choice(SPACES)
        + generated(rng, depth + 1)
        for _ in range(rng.randint(0, 4))
    ]
    if entries and rng.random() < 0.1:
        entries.append(entries[0])
    return b"<<" + rng.choice(SPACES).join(entries) + rng.choice((b">>", b">>", b">>", b">", b""))


def main(argv: list[str]) -> int:
    count = int(argv[0]) if argv else 30000
    differences: list[str] = []
    objects = 0
    paths = sorted({path for pattern in CORPUS for path in glob.glob(pattern, recursive=True)})
    for path in paths:
        try:
            document = PdfDocument.open(path)
        except Exception:  # noqa: BLE001
            continue
        python = PdfLexer(document.raw_data, reference_resolver=document.resolver.resolve)
        compiled = StrictPdfLexer(document.raw_data, reference_resolver=document.resolver.resolve)
        for key, entry in document.xref.items():
            if not entry.in_use or entry.object_stream is not None or entry.offset < 0:
                continue
            reference = PdfReference(key >> 16, entry.generation)
            objects += 1
            if outcome(python, entry.offset, reference) != outcome(
                compiled, entry.offset, reference
            ):
                differences.append(f"{path} {reference}")
    rng = random.Random(0)
    reference = PdfReference(1, 0)
    for _ in range(count):
        data = b"1 0 obj\n" + generated(rng) + rng.choice(TAILS)
        if outcome(PdfLexer(data), 0, reference) != outcome(StrictPdfLexer(data), 0, reference):
            differences.append(repr(data[:160]))
    summary = f"{objects} objects from {len(paths)} files and {count} generated"
    print(f"{summary}: {len(differences)} differ")
    for difference in differences[:5]:
        print(difference)
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

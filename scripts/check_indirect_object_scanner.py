# SPDX-License-Identifier: AGPL-3.0-only

"""Check that the lexers read indirect objects through ObjectScanner.parse_indirect exactly as
their own code does.

core's PdfLexer and StrictPdfLexer hand a plainly written indirect object to the compiled
scanner whole -- header, value and endobj -- and keep their own code for whatever it declines.
This reads every uncompressed object of the upstream test PDFs (deciphering those of encrypted
files) with each lexer and with a subclass of it, which keeps the lexer's own code, then
objects generated around the headers, values and endings the scanner must decline (leading
zeros, signs, comments and missing white-space in headers, object 0, large generations,
empty objects, glued and missing endobj, streams, a header naming another object), and fails
on the first value, end position or error that differs.

uv run --all-packages python scripts/check_indirect_object_scanner.py [generated objects]
"""

from __future__ import annotations

import glob
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_pdf import PdfDocument
from core_pdf.impl.recovery_lexer import PdfLexer, StrictPdfLexer
from core_pdf.impl.types import PdfReference
from core_pdf_spec.exceptions import PdfParseError
from scripts.check_strict_object_scanner import CORPUS, TAILS, canonical, generated


class OwnCodePdfLexer(PdfLexer):
    """PdfLexer reading indirect objects with its own code, as subclasses do."""

    __slots__ = ()


class OwnCodeStrictPdfLexer(StrictPdfLexer):
    __slots__ = ()


HEADERS = (
    b"1 0 obj", b"1 0 obj", b"01 0 obj", b"1 00 obj", b"1  0\nobj", b"1\r\n0\r\nobj",
    b"1 0 %c\nobj", b"%c\n1 0 obj", b"0 0 obj", b"1 70000 obj", b"1 65535 obj", b"1 0 objx",
    b"1 0obj", b"+1 0 obj", b"12345678901 0 obj", b" \n1 0 obj", b"1\x000 obj", b"2 0 obj",
    b"1 0 obj<<", b"1 0 ob", b"1 0",
)  # fmt: skip
ENDINGS = (*TAILS, b"\nendobjx", b" endobj2 0 obj", b"\nendstream\nendobj", b"\ntrailer")


def outcome(lexer: Any, offset: int, reference: PdfReference | None) -> tuple[Any, ...]:
    lexer.rewind(offset)
    try:
        value = lexer.parse_indirect_object(expected_reference=reference)
    except PdfParseError as error:
        return ("error", str(error))
    except Exception as error:  # noqa: BLE001
        return ("raised", type(error).__name__, str(error))
    return ("ok", canonical(value), lexer.pos)


PAIRS = ((PdfLexer, OwnCodePdfLexer), (StrictPdfLexer, OwnCodeStrictPdfLexer))


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
        resolver = document.resolver
        lexers = [
            tuple(
                cls(document.raw_data, reference_resolver=resolver.resolve, decipher=decipher)
                for cls in pair
            )
            for pair in PAIRS
            for decipher in (None, resolver.decipher)
        ]
        for key, entry in document.xref.items():
            if not entry.in_use or entry.object_stream is not None or entry.offset < 0:
                continue
            reference = PdfReference(key >> 16, entry.generation)
            objects += 1
            differences.extend(
                f"{path} {reference} {type(compiled).__name__}"
                for compiled, own in lexers
                for expected in (reference, None)
                if outcome(compiled, entry.offset, expected) != outcome(own, entry.offset, expected)
            )
    rng = random.Random(0)
    for _ in range(count):
        data = (
            rng.choice(HEADERS)
            + rng.choice((b"\n", b" ", b"", b"%x\n"))
            + generated(rng)
            + rng.choice(ENDINGS)
        )
        expected = rng.choice((PdfReference(1, 0), None))
        for compiled, own in PAIRS:
            if outcome(compiled(data), 0, expected) != outcome(own(data), 0, expected):
                differences.append(f"{compiled.__name__} {data[:160]!r}")
    summary = f"{objects} objects from {len(paths)} files and {count} generated"
    print(f"{summary}: {len(differences)} differ")
    for difference in differences[:5]:
        print(difference)
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

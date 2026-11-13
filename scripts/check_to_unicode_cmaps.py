# SPDX-License-Identifier: AGPL-3.0-only

"""Check that the compiled ToUnicode CMap scan parses exactly what core's Python parser does.

core_pdf.impl.fonts.cmap.parse_to_unicode_cmap parses a ToUnicode CMap with
core-pdf-cythonized's scan_to_unicode_cmap, and with parse_to_unicode_program (the tokenizer
and block parsers) for a CMap the kernel declines. This collects every distinct ToUnicode
stream of the upstream test PDFs, decoded, plus CMaps generated around what the kernel
declines (literal strings, odd hex digit counts, whitespace in hex strings, cidrange blocks,
bfrange blocks that would be rejected or overrun the expansion budget, unterminated tokens,
usecmap), and parses each both ways. Where the kernel does not decline, the parsed CMap must
be the same: its mappings in order, each key and value of the same type and value, its
codespace ranges and usecmap name, or the same error. It fails on the first CMap that
differs, or if the kernel took none.

uv run --all-packages python scripts/check_to_unicode_cmaps.py
"""

from __future__ import annotations

import glob
import hashlib
import logging
import warnings
from collections.abc import Callable, Iterator

from core_pdf import PdfDocument
from core_pdf.impl.fonts.cmap import CMapProgram, parse_to_unicode_program, to_unicode_cmap
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_09_fonts.cmap_tounicode import ParsedToUnicodeCMap
from core_pdf.impl.spec.types import PdfReference
from core_pdf_cythonized import scan_to_unicode_cmap

CORPUS = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/**/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)

HEADER = (
    b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
    b"/CMapName /Check def\n/CMapType 2 def\n"
)
FOOTER = b"endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"


def cmap(body: bytes, codespace: bytes = b"<0000> <FFFF>") -> bytes:
    codespace_block = b"1 begincodespacerange\n" + codespace + b"\nendcodespacerange\n"
    return HEADER + codespace_block + body + FOOTER


GENERATED = (
    # Well formed, which the kernel takes.
    cmap(b"2 beginbfchar\n<0003> <0020>\n<0011> <002E>\nendbfchar\n"),
    cmap(b"1 beginbfrange\n<0010> <0015> <0041>\nendbfrange\n"),
    cmap(b"1 beginbfrange\n<0010> <0012> [<0041> <00660069> <D83DDE00>]\nendbfrange\n"),
    cmap(b"1 beginbfrange\n<0010> <0013> [<0041> %c\n<0042>]\nendbfrange\n"),
    cmap(b"1 beginbfrange\n<00FE> <0101> <D7FF>\nendbfrange\n"),
    cmap(b"1 beginbfchar\n<41> <00410042>\nendbfchar\n", b"<00> <FF>"),
    cmap(b"1 beginbfchar\n<0001> <>\nendbfchar\n"),
    cmap(b"1 beginbfrange\n<0000> <FFFF> <0000>\nendbfrange\n"),
    cmap(b"/Parent usecmap\n1 beginbfchar\n<0001> <0041>\nendbfchar\n"),
    cmap(b"1 beginbfchar\n<0001> <0041>\nendbfchar\n", b"<0000> <FFFF> <00> <FF>"),
    cmap(b"1 beginbfchar\n<0001> <0041>\nendbfchar\n", b"<FFFF> <0000>"),
    cmap(b"1 beginbfchar\n<0001> <0041>\nendbfchar\n", b""),
    cmap(b"1 beginbfchar <0001> <0041> <0002> endbfchar"),
    b"begincmap 1 beginbfchar <01> <0041> endbfchar endcmap 1 beginbfchar <02> <0042> endbfchar",
    b"1 beginbfchar <01> <0041> endbfchar",
    # Literal operands.
    cmap(b"1 beginbfchar\n(\\001) <0041>\nendbfchar\n"),
    cmap(b"1 beginbfchar\n<0001> (A)\nendbfchar\n"),
    cmap(b"1 beginbfrange\n(\\000\\001) (\\000\\003) <0041>\nendbfrange\n"),
    cmap(b"1 beginbfchar\n<0001> <0041>\nendbfchar\n", b"(\\000\\000) (\\377\\377)"),
    # Odd digit counts and whitespace inside hex strings.
    cmap(b"1 beginbfchar\n<001> <0041>\nendbfchar\n"),
    cmap(b"1 beginbfchar\n<0001> <041>\nendbfchar\n"),
    cmap(b"1 beginbfchar\n<00 01> <00 41>\nendbfchar\n"),
    cmap(b"1 beginbfrange\n<0010> <0013> <00\n41>\nendbfrange\n"),
    cmap(b"1 beginbfrange\n<0010> <0012> [<0041> <004 2>]\nendbfrange\n"),
    cmap(b"1 beginbfchar\n<00g1> <0041>\nendbfchar\n"),
    # cidrange blocks.
    cmap(b"1 begincidrange\n<0000> <00FF> 1\nendcidrange\n"),
    cmap(
        b"1 beginbfchar\n<0001> <0041>\nendbfchar\n1 begincidrange\n<0010> <0020> 5\nendcidrange\n"
    ),
    # bfrange blocks that would be rejected, alone and beside a valid one.
    cmap(b"1 beginbfrange\n<0015> <0010> <0041>\nendbfrange\n"),
    cmap(b"1 beginbfrange\n<0010> <0015> <0041>\n<0020> <0010> <0041>\nendbfrange\n"),
    cmap(b"1 beginbfrange\n<00> <0100> <0041>\nendbfrange\n"),
    cmap(b"1 beginbfrange\n<000000> <020000> <0041>\nendbfrange\n"),
    cmap(b"1 beginbfrange\n<0010> <0015> []\nendbfrange\n"),
    cmap(b"1 beginbfrange\n<0010> <0015> [(A)]\nendbfrange\n"),
    cmap(b"1 beginbfrange\n<0010> <0015>\nendbfrange\n"),
    cmap(b"2 beginbfrange\n<0000> <FFFF> <0000>\n<0000> <FFFF> <0000>\nendbfrange\n"),
    cmap(b"3 beginbfrange\n" + b"<0000> <FFFF> <0000>\n" * 3 + b"endbfrange\n"),
    cmap(b"1 beginbfrange\n<0010> <0015> 7\nendbfrange\n"),
    # Unterminated tokens and blocks.
    cmap(b"1 beginbfchar\n<0001> <0041\nendbfchar\n"),
    cmap(b"1 beginbfchar\n<0001> (A\nendbfchar\n"),
    cmap(b"1 beginbfrange\n<0010> <0012> [<0041> <0042>\nendbfrange\n"),
    cmap(b"1 beginbfchar\n<0001> <0041>\n"),
    HEADER + b"1 beginbfchar\n<0001> <0041>\nendbfchar\n",
    b"",
    b"begincmap endcmap",
)


def to_unicode_streams() -> Iterator[tuple[str, bytes]]:
    """(source, decoded bytes) of each distinct ToUnicode stream in the corpus."""
    seen: set[bytes] = set()
    paths = sorted({path for pattern in CORPUS for path in glob.glob(pattern, recursive=True)})
    for path in paths:
        try:
            document = PdfDocument.open(path)
        except Exception:  # noqa: BLE001
            continue
        for key, entry in document.xref.items():
            if not entry.in_use:
                continue
            try:
                value = document.resolver.resolve(PdfReference(key >> 16, entry.generation))
                if not isinstance(value, dict) or "ToUnicode" not in value:
                    continue
                stream = document.resolver.resolve(value["ToUnicode"])
                if not isinstance(stream, PdfStream):
                    continue
                data = bytes(stream.data)
            except Exception:  # noqa: BLE001
                continue
            digest = hashlib.sha256(data).digest()
            if digest not in seen:
                seen.add(digest)
                yield f"{path} {key >> 16} {entry.generation}", data


def exact(parsed: ParsedToUnicodeCMap) -> object:
    """A key equal only for the same mappings in the same order, types included."""
    return (
        tuple((type(key), key, type(value), value) for key, value in parsed.mappings.items()),
        type(parsed.mappings),
        parsed.code_space_ranges,
        type(parsed.usecmap_name),
        parsed.usecmap_name,
    )


def by_python(data: bytes) -> ParsedToUnicodeCMap:
    return parse_to_unicode_program(CMapProgram.parse(data))


def outcome(parse: Callable[..., ParsedToUnicodeCMap], *args: object) -> tuple[object, ...]:
    try:
        return ("ok", exact(parse(*args)))
    except Exception as error:  # noqa: BLE001
        return ("raised", type(error).__name__, str(error))


def main() -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    cases = [*to_unicode_streams()]
    streams = len(cases)
    cases.extend((f"generated {index}", data) for index, data in enumerate(GENERATED))
    taken = declined = 0
    differences: list[str] = []
    for source, data in cases:
        scanned = scan_to_unicode_cmap(data)
        if scanned is None:
            declined += 1
            continue
        taken += 1
        if outcome(to_unicode_cmap, *scanned) != outcome(by_python, data):
            differences.append(source)
    print(
        f"{streams} ToUnicode streams and {len(GENERATED)} generated CMaps,",
        f"{taken} taken by the kernel and {declined} declined: {len(differences)} differ",
    )
    for difference in differences[:10]:
        print(difference)
    return 1 if differences or not taken else 0


if __name__ == "__main__":
    raise SystemExit(main())

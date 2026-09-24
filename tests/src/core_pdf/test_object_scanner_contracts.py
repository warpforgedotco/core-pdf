# SPDX-License-Identifier: AGPL-3.0-only

"""The reader lexer with its compiled object scanner is the reader lexer.

PdfLexer.parse_dictionary and parse_array ask core_pdf_cythonized's
ObjectScanner first and fall back to the Python parse when it declines. The
contract is that nobody can tell: for any input, at any position, the result,
the end position and any exception are those of the Python alone. Reference
below is the reader lexer with the scanner taken out, and every test compares
against it.

The kernel's own golden vectors pin the scanner in isolation; these pin the
composition -- the fallback, the deciphering hand-off, the scanner's
lifetime -- and run the comparison over corpus documents when the fixture
submodules are present.
"""

import re
import struct
from pathlib import Path

import pytest

from core_pdf.impl.document.recovery.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.lexer import PdfLexer as SyntaxLexer
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.types import PdfName, PdfReference, PdfString

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
CORPUS = (
    "pdfminer.six/samples/nonfree/i1040nr.pdf",
    "llama_index/docs/examples/data/10k/lyft_2021.pdf",
    "pypdf/resources/issue-301.pdf",
    "pdf20examples/Simple PDF 2.0 file.pdf",
)
CONTEXTS = (
    None,
    SemanticContext(PdfVersion(1, 1)),
    SemanticContext(PdfVersion(1, 2)),
    SemanticContext(PdfVersion(1, 7)),
    SemanticContext(PdfVersion(2, 0)),
)


class Reference(PdfLexer):
    __slots__ = ()
    parse_dictionary = SyntaxLexer.parse_dictionary
    parse_array = SyntaxLexer.parse_array


def xor_decipher(object_number, generation, value, dictionary):
    return bytes(byte ^ ((object_number + generation) & 0xFF) for byte in value)


def same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is float:
        return struct.pack("<d", left) == struct.pack("<d", right)
    if type(left) is dict:
        return len(left) == len(right) and all(
            same(left_key, right_key) and same(left_value, right_value)
            for (left_key, left_value), (right_key, right_value) in zip(
                left.items(), right.items(), strict=True
            )
        )
    if type(left) is list:
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
    if type(left) is PdfString:
        return left.data == right.data and left.is_literal == right.is_literal
    return left == right


def outcome(lexer, position, *, decipher=None):
    lexer.rewind(position)
    if decipher is not None:
        lexer.decipher = decipher
        lexer.current_obj_num, lexer.current_gen_num = 9, 2
    try:
        if lexer.raw_data[position : position + 2] == b"<<":
            value = lexer.parse_dictionary()
        else:
            value = lexer.parse_array()
    except RecursionError:
        raise
    except Exception as error:
        return ("raised", type(error), str(error))
    return ("value", value, lexer.pos)


def assert_same_outcome(data, position, *, context=None, decipher=None):
    got = outcome(PdfLexer(data, semantic_context=context), position, decipher=decipher)
    want = outcome(Reference(data, semantic_context=context), position, decipher=decipher)
    assert got[0] == want[0], (data[position : position + 80], got, want)
    if got[0] == "raised":
        assert got[1:] == want[1:]
    else:
        assert same(got[1], want[1]), (data[position : position + 80], got, want)
        assert got[2] == want[2]


SYNTHETIC = (
    b"<< /Type /Page /MediaBox [0 0 612 792] /Contents 5 0 R /Rotate 90 >>",
    b"<< /A 1 %comment\r\n /B (x) %c\n\r/C <41> >>",
    b"[1 -2 +3 4. .5 -.5 +5. 0 -0 007 1e5 1_000 inf.]",
    b"[1 2 %c\n 3]",
    b"[1\x002]",
    b"[1 0 R 2 0 R] [1 0 Rx] [1 -1 R] [1 70000 R]",
    b"<< /A trueendobj /B 12endobj /C foo >>",
    b"<< /A#20B 1 /#41 2 /A#4 3 /A#00 5 /# 6 >>",
    b"[(a\\nb\\101\\q) (a\\\r\nb) (a\r\nb) (nested (paren)) <48 65 6> <4G>]",
    b"<< /A << /B 1 >> stream\n>> << /A << /B 1 >> strexm >>",
    b"<< /A 1 /A 2 >> << 1 2 >> << /A 1 > << /A >> << /A ] >>",
    b"<< /Contents <616263> /Type /Sig >> << /Contents <616263> >> << /Contents 5 0 R >>",
    b"[<< /A 1 >> 2] [{1}] [(a]b) 1] [1",
    b"[" * 70 + b"]" * 70,
    b"<< /A " * 70 + b"1" + b" >>" * 70,
)


@pytest.mark.parametrize("data", SYNTHETIC, ids=range(len(SYNTHETIC)))
@pytest.mark.parametrize("context", CONTEXTS, ids=("default", "1.1", "1.2", "1.7", "2.0"))
@pytest.mark.parametrize("decipher", [None, xor_decipher], ids=("plain", "deciphered"))
def test_every_container_parses_as_the_python_alone_would(data, context, decipher):
    for match in re.finditer(rb"<<|\[", data):
        assert_same_outcome(data, match.start(), context=context, decipher=decipher)


def test_signature_contents_stays_undeciphered():
    # The reader defers a hex /Contents until it can see the dictionary is a
    # signature, and then leaves it alone; the scanner must not get there
    # first.
    data = b"<< /Type /Sig /Contents <616263> >>"
    lexer = PdfLexer(data)
    lexer.decipher = xor_decipher
    lexer.current_obj_num, lexer.current_gen_num = 9, 2
    assert lexer.parse_dictionary()[PdfName.of("Contents")] == PdfString(b"abc")


def test_indirect_objects_parse_through_the_scanner():
    data = b"4 0 obj\n<< /Kids [5 0 R 6 0 R] /Count 2 /Type /Pages >>\nendobj\n"
    lexer = PdfLexer(data)
    assert lexer.parse_indirect_object() == {
        PdfName.of("Kids"): [PdfReference(5, 0), PdfReference(6, 0)],
        PdfName.of("Count"): 2,
        PdfName.of("Type"): PdfName.of("Pages"),
    }
    assert lexer.scanner is not None


def test_scanned_names_are_the_interned_instances():
    lexer = PdfLexer(b"<< /Type /Catalog >>")
    ((key, value),) = lexer.parse_dictionary().items()
    assert key is PdfName.of("Type")
    assert value is PdfName.of("Catalog")


def test_close_releases_the_buffer():
    lexer = PdfLexer(bytearray(b"<< /A 1 >>"))
    lexer.parse_dictionary()
    # A live export would make the memoryview refuse to release; close
    # suppresses only ValueError, so a BufferError would escape here.
    lexer.close()
    assert lexer.scanner is None
    assert lexer.data_len == 0


def test_a_rules_change_rebuilds_the_scanner():
    lexer = PdfLexer(b"<< /A#42 1 >>", semantic_context=SemanticContext(PdfVersion(1, 1)))
    assert list(lexer.parse_dictionary()) == [PdfName.of("A#42")]
    lexer.semantic_context = SemanticContext(PdfVersion(2, 0))
    lexer.rewind(0)
    assert list(lexer.parse_dictionary()) == [PdfName.of("AB")]


@pytest.mark.parametrize("path", CORPUS)
def test_corpus_containers_parse_as_the_python_alone_would(path):
    file = FIXTURES / path
    if not file.exists():
        pytest.skip(f"fixture not present, needs a submodule checkout: {path}")
    data = file.read_bytes()
    offsets = [match.start() for match in re.finditer(rb"<<|\[", data)]
    # An even spread. From an offset inside a compressed stream the Python
    # reference can copy and split megabytes looking for a ']', so a full
    # sweep belongs to a profiling run, not the unit suite.
    step = max(1, len(offsets) // 60)
    for position in offsets[::step]:
        assert_same_outcome(data, position)

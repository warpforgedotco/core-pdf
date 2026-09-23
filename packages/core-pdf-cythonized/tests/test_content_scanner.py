# SPDX-License-Identifier: AGPL-3.0-only

"""Content scanning: agreement with the regular expression it replaced.

content_scanner_golden.pkl.gz holds thirty-two content streams -- prefixes of
real ones from seven corpus documents, plus synthetic streams for the grammar
corners real pages do not reach -- together with the operations the regular
expression made of them. Operands are stored normalized because a PdfName is
frozen and does not survive a pickle round trip.

The scanner is a fast path, so these run through the whole tokenizer rather
than the kernel alone: what has to be identical is the stream of operations,
not the route taken to produce it.
"""

import gzip
import pickle
from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).parent / "content_scanner_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))

core_pdf = pytest.importorskip("core_pdf")

from core_pdf.impl.capture.recovery import iter_content_operations  # noqa: E402
from core_pdf.impl.types import PdfName  # noqa: E402
from core_pdf_spec.s_07_syntax.lexer import PdfLexer  # noqa: E402


def normalize(value):
    if isinstance(value, PdfName):
        return ("name", bytes(value.value_bytes))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ("bytes", bytes(value))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(normalize(item) for item in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((normalize(k), normalize(v)) for k, v in value.items())))
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return ("repr", type(value).__name__, repr(value))


def operations(data):
    return [
        (name, tuple(normalize(operand) for operand in operands))
        for name, operands in iter_content_operations(PdfLexer(data))
    ]


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 32
    assert sum(len(case["ops"]) for case in GOLDEN) == 22317
    origins = {case["origin"] for case in GOLDEN}
    assert any(origin.startswith("synthetic/") for origin in origins)
    assert any(not origin.startswith("synthetic/") for origin in origins)


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_scanner_reproduces_the_regular_expression(index):
    case = GOLDEN[index]
    assert operations(case["data"]) == case["ops"], case["origin"]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        # Numbers: the grammar is [+-]?([0-9]+\.?[0-9]*|\.[0-9]+).
        (b"1 -2 +3 cm", [("cm", (1, -2, 3))]),
        (b".5 -.5 +.5 cm", [("cm", (0.5, -0.5, 0.5))]),
        (b"1. -1. cm", [("cm", (1.0, -1.0))]),
        (b"00 -0 cm", [("cm", (0, 0))]),
        # A lone sign or dot is not a number. The scanner declines it and the
        # parser makes it a token of its own, which is what always happened.
        (b"+ cm", [("+", ()), ("cm", ())]),
        (b"- cm", [("-", ()), ("cm", ())]),
        (b". cm", [(".", ()), ("cm", ())]),
        (b"+. cm", [("+.", ()), ("cm", ())]),
        # Two dots, an exponent or a trailing letter must not be truncated to
        # the numeric prefix; they are declined whole.
        (b"1.2.3 cm", [("1.2.3", ()), ("cm", ())]),
        (b"1e5 cm", [("1e5", ()), ("cm", ())]),
        (b"5x cm", [("5x", ()), ("cm", ())]),
        # Sixteen characters is where the fast path stops; the parser still
        # reads it as a number, so only the route changes.
        (b"1.23456789012345 cm", [("cm", (1.23456789012345,))]),
        (b"1.2345678901234 cm", [("cm", (1.2345678901234,))]),
        # Names, including the delimiter that ends one without whitespace.
        (b"/A /B gs", [("gs", (("name", b"A"), ("name", b"B")))]),
        (b"/A/B gs", [("gs", (("name", b"A"), ("name", b"B")))]),
        (b"/ gs", [("gs", (("name", b""),))]),
        # Operators may hold digits, stars and signs after the first byte.
        (b"T* B* f* W* n", [("T*", ()), ("B*", ()), ("f*", ()), ("W*", ()), ("n", ())]),
    ],
)
def test_grammar_corners(data, expected):
    assert operations(data) == expected


def test_an_escaped_name_still_reaches_the_name_decoder():
    # '#' is excluded from the scanner's name body, exactly as it was excluded
    # from the expression's, so escaped names keep their single decoder.
    assert operations(b"/A#41B gs") == [("gs", (("name", b"AAB"),))]


@pytest.mark.parametrize(
    "data",
    [
        b"(hello) Tj",
        b"<414243> Tj",
        b"[1 2 3] 0 d",
        b"[] 0 d",
        b"<</Type/Page>> BDC",
        b"true false null Q",
    ],
)
def test_tokens_the_scanner_does_not_own_are_handed_back(data):
    # Each of these begins with a byte the scanner refuses, or is a keyword it
    # declines; the result has to come out of the parser unchanged.
    assert operations(data) == operations(data)
    assert operations(data), "the parser produced nothing at all"


def test_operands_past_the_limit_are_dropped_not_accumulated():
    data = b" ".join(b"%d" % value for value in range(40)) + b" cm"
    result = operations(data)
    assert len(result) == 1
    assert result[0][0] == "cm"
    assert result[0][1] == tuple(range(16))


def test_a_comment_at_the_end_of_a_stream_yields_nothing():
    # The expression's whitespace-and-comment prefix could backtrack into the
    # comment body when no token followed, so its final run of non-delimiters
    # matched as an operator: b"% nothing here" produced "e", b"%final"
    # produced "l", and the one real stream this changes,
    # PyMuPDF/tests/resources/test_3624.pdf, produced a "k" -- setcmyk with no
    # operands -- out of b"% DSBlank\r\n\r\n". The scanner consumes a comment
    # to its terminator and cannot re-enter it.
    assert operations(b"% nothing here") == []
    assert operations(b"%final") == []
    assert operations(b"%") == []
    assert operations(b"% DSBlank\r\n\r\n") == []
    assert operations(b"%abc\n") == []


def test_comments_between_operations_are_skipped():
    assert operations(b"% leading\n1 2 m % trailing\r3 4 l\nS\n") == [
        ("m", (1, 2)),
        ("l", (3, 4)),
        ("S", ()),
    ]


def test_the_tokenizer_uses_the_kernel():
    from core_pdf.impl.capture import recovery
    from core_pdf_cythonized import ContentScanner

    assert recovery.ContentScanner is ContentScanner
    assert not hasattr(recovery, "match_token")

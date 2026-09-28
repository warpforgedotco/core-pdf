# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from pathlib import Path

import pytest

from core_pdf.impl.capture_recording import (
    GLYPH_PAINT_KEEPING_OPERATORS,
    LINE_MOVING_OPERATORS,
    TEXT_LAYOUT_KEEPING_OPERATORS,
)
from core_pdf.impl.capture_recovery import ContentFallback, content_scanner, iter_content_operations
from core_pdf.impl.types import PdfName
from core_pdf_spec.s_07_syntax.lexer import PdfLexer

GOLDEN_PATH = Path(__file__).parent / "content_scanner_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


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
        (b"1 -2 +3 cm", [("cm", (1, -2, 3))]),
        (b".5 -.5 +.5 cm", [("cm", (0.5, -0.5, 0.5))]),
        (b"1. -1. cm", [("cm", (1.0, -1.0))]),
        (b"00 -0 cm", [("cm", (0, 0))]),
        (b"+ cm", [("+", ()), ("cm", ())]),
        (b"- cm", [("-", ()), ("cm", ())]),
        (b". cm", [(".", ()), ("cm", ())]),
        (b"+. cm", [("+.", ()), ("cm", ())]),
        (b"1.2.3 cm", [("1.2.3", ()), ("cm", ())]),
        (b"1e5 cm", [("1e5", ()), ("cm", ())]),
        (b"5x cm", [("5x", ()), ("cm", ())]),
        (b"1.23456789012345 cm", [("cm", (1.23456789012345,))]),
        (b"1.2345678901234 cm", [("cm", (1.2345678901234,))]),
        (b"/A /B gs", [("gs", (("name", b"A"), ("name", b"B")))]),
        (b"/A/B gs", [("gs", (("name", b"A"), ("name", b"B")))]),
        (b"/ gs", [("gs", (("name", b""),))]),
        (b"T* B* f* W* n", [("T*", ()), ("B*", ()), ("f*", ()), ("W*", ()), ("n", ())]),
    ],
)
def test_grammar_corners(data, expected):
    assert operations(data) == expected


def test_an_escaped_name_still_reaches_the_name_decoder():
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
    assert operations(data) == operations(data)
    assert operations(data), "the parser produced nothing at all"


def test_operands_past_the_limit_are_dropped_not_accumulated():
    data = b" ".join(b"%d" % value for value in range(40)) + b" cm"
    result = operations(data)
    assert len(result) == 1
    assert result[0][0] == "cm"
    assert result[0][1] == tuple(range(16))


def test_a_comment_at_the_end_of_a_stream_yields_nothing():
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
    from core_pdf.impl import capture_recovery
    from core_pdf_cythonized import ContentScanner

    assert capture_recovery.ContentScanner is ContentScanner


class DispatchState:
    shared_glyph_paint = None
    text_layout = None


def dispatched(data, *, stop_at=None):
    lexer = PdfLexer(data)
    scanner = content_scanner(lexer)
    fallback = ContentFallback(lexer, scanner.operands, None, None)
    seen = []
    names = {name for case in GOLDEN for name, _ in case["ops"]} | {"q", "Q", "cm", "BI"}

    def handler_for(name):
        def handler(operands, depth):
            seen.append((name, tuple(normalize(operand) for operand in operands)))
            return (name, lexer.pos) if name == stop_at else None

        return handler

    handlers = {name: handler_for(name) for name in names}
    stops = []
    while True:
        child = scanner.run(
            lexer,
            handlers,
            fallback,
            DispatchState(),
            0,
            GLYPH_PAINT_KEEPING_OPERATORS,
            LINE_MOVING_OPERATORS,
            TEXT_LAYOUT_KEEPING_OPERATORS,
        )
        if child is None:
            return seen, stops
        stops.append(child)


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_compiled_dispatch_sees_the_iterated_operations(index):
    case = GOLDEN[index]
    seen, _ = dispatched(case["data"])
    assert seen == case["ops"], case["origin"]


def test_compiled_dispatch_resumes_after_a_child_frame():
    data = b"q 1 0 0 1 5 5 cm Q q Q"
    seen, stops = dispatched(data, stop_at="Q")
    assert seen == operations(data)
    assert stops == [("Q", data.index(b"Q") + 1), ("Q", len(data))]

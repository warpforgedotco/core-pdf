"""An indirect object header read by one match reads as its three words do."""

import pytest

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf_spec.s_07_syntax.lexer import PdfLexer as SyntaxLexer

HEADERS = [
    b"12 0 obj<</A 1>>",
    b"12 0 obj\n<</A 1>>",
    b"  \n12 0 obj 7",
    b"12\x007\r\nobj(x)",
    b"12 0 obj",
    b"12 0 objx 3",
    b"12 0 % c\nobj 3",
    b"% lead\n12 0 obj 3",
    b"+12 0 obj 3",
    b"12 70000 obj 3",
    b"12<<0 obj",
    b"12 0 ob",
    b"007 0010 obj 1",
]


def read(data: bytes, *, words: bool) -> object:
    lexer = PdfLexer(data)
    try:
        header = SyntaxLexer.read_indirect_header(lexer) if words else lexer.read_indirect_header()
    except (PdfParseError, ValueError) as error:
        return f"raised {type(error).__name__}: {error}"
    finally:
        position = lexer.pos
        lexer.close()
    return header, position


@pytest.mark.parametrize("data", HEADERS)
def test_one_match_reads_as_three_words(data: bytes) -> None:
    assert read(data, words=False) == read(data, words=True)

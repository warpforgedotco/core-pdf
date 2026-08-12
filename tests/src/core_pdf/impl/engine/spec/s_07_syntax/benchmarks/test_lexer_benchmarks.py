# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_content.operations import iter_content_operations
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer

PAGE_DICT_OBJECT = b"""\
<< /Type /Page
   /MediaBox [0 0 612 792]
   /Resources << /Font << /F1 5 0 R /F2 6 0 R >>
                  /XObject << /Im0 12 0 R /Im1 13 0 R >>
                  /ExtGState << /GS0 14 0 R >> >>
   /Contents [7 0 R 8 0 R]
   /Annots [9 0 R 10 0 R 11 0 R]
   /Parent 2 0 R >>
"""

CONTENT_STREAM_BLOCK = b"""\
q
1 0 0 1 100 100 cm
0.2 0.4 0.6 rg
BT
/F1 12 Tf
14 TL
(Hello World) Tj
T*
[(Hello) -250 (World) -300 (Benchmark) 120 (Text)] TJ
ET
100 100 200 200 re
f
Q
"""
CONTENT_STREAM = CONTENT_STREAM_BLOCK * 200


def parse_page_dict() -> object:
    return PdfLexer(PAGE_DICT_OBJECT).parse_object()


def tokenize_content_stream() -> list[object]:
    return list(iter_content_operations(PdfLexer(CONTENT_STREAM)))


def test_parse_object_page_dict_benchmark(benchmark) -> None:
    result = benchmark(parse_page_dict)
    assert result


def test_content_stream_tokenize_benchmark(benchmark) -> None:
    result = benchmark(tokenize_content_stream)
    assert len(result) == 13 * 200

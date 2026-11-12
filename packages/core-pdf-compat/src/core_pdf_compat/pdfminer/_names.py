"""Names as pdfminer.six reads them, where its rule differs from PDF's.

pdfminer.six ends a name at a delimiter or at what Python's ``\\s`` matches, so a NUL byte,
which PDF counts as white-space, stays in the name and a vertical tab ends it. ``#`` starts
an escape of up to two hexadecimal digits, and a ``#`` without one is dropped. Core-pdf
reads names as PDF says; the facade re-reads an object pdfminer.six would name differently.

The rule follows pdfminer.six's ``PSBaseParser._parse_literal`` (pdfminer/psparser.py),
Copyright (c) 2004-2016 Yusuke Shinyama, under the MIT License.
"""

from __future__ import annotations

import re
from typing import Any

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax import lexer as spec_lexer
from core_pdf.impl.spec.s_07_syntax.xref import key_for
from core_pdf.impl.types import PdfReference

# pdfminer.six's END_LITERAL.
END_LITERAL = re.compile(rb"[#/%\[\]()<>{}\s]")
HEX_DIGITS = re.compile(rb"[0-9a-fA-F]{1,2}")
# A name whose end or escapes pdfminer.six reads otherwise: one running into a NUL or a
# vertical tab, or holding a "#" that is not two hexadecimal digits or that writes NUL.
DIFFERENT_NAME = re.compile(
    rb"/[^\t\n\x0c\r /%\[\]()<>{}\x00\x0b]*(?:[\x00\x0b]|#(?![0-9a-fA-F]{2})|#00)"
)
OBJECT_BODY_END = re.compile(rb"\bstream\b|\bendobj\b")


class PdfminerNameLexer(PdfLexer):
    """Core's lexer, reading names as pdfminer.six does.

    Its arrays and dictionaries are read token by token, not by core's compiled scanner,
    so that every name goes through read_name.
    """

    def parse_dictionary(self) -> dict[Any, Any]:
        return spec_lexer.PdfLexer.parse_dictionary(self)

    def parse_array(self) -> list[Any]:
        return spec_lexer.PdfLexer.parse_array(self)

    def read_name(self) -> memoryview:
        data = self.raw_data
        position = self.pos + 1
        name = bytearray()
        while True:
            match = END_LITERAL.search(data, position)
            end = self.data_len if match is None else match.start()
            name += data[position:end]
            if match is None or data[end] != 35:
                break
            digits = HEX_DIGITS.match(data, end + 1)
            if digits is None:
                position = end + 1
            else:
                name.append(int(digits.group(), 16))
                position = digits.end()
        self.pos = end
        return memoryview(bytes(name))


def pdfminer_named_object(document: Any, reference: PdfReference) -> Any | None:
    """The object as pdfminer.six reads it, when one of its names reads otherwise there.

    None when its names read the same, or when the object is not where a re-read finds it.
    """
    entry = document.xref.get(key_for(reference.object_number, reference.generation_number))
    if entry is None or entry.object_stream is not None:
        return None
    data = document.raw_data
    body_end = OBJECT_BODY_END.search(data, entry.offset)
    end = len(data) if body_end is None else body_end.start()
    if DIFFERENT_NAME.search(data, entry.offset, end) is None:
        return None
    lexer = PdfminerNameLexer(
        data, reference_resolver=document.resolver.resolve, decipher=document.decipher
    )
    try:
        lexer.rewind(entry.offset)
        return lexer.parse_indirect_object(expected_reference=reference)
    except PdfParseError, TypeError, ValueError:
        return None
    finally:
        lexer.close()


__all__ = ("PdfminerNameLexer", "pdfminer_named_object")

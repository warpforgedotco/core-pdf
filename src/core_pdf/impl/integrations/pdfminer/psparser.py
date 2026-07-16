# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, TypeAlias

from core_pdf.impl.integrations.pdfminer.psexceptions import PSEOF
from core_pdf.impl.primitives import PdfName

log = logging.getLogger(__name__)
END_KEYWORD = re.compile(rb"[\000\011\012\014\015\040\050\051\074\076\133\135\173\175/\045]")


class PSLiteral:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"/{self.name!r}"

    __str__ = __repr__

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PSLiteral) and self.name == other.name


class PSKeyword:
    def __init__(self, name: bytes) -> None:
        self.name = name


@lru_cache(maxsize=4096)
def LIT(name: str) -> PSLiteral:
    return PSLiteral(name)


@lru_cache(maxsize=4096)
def KWD(name: bytes) -> PSKeyword:
    return PSKeyword(name)


def literal_name(value: Any) -> str:
    if isinstance(value, PSLiteral):
        return value.name
    if isinstance(value, PdfName):
        return value.value
    if isinstance(value, str):
        return value
    raise TypeError(f"literal name required: {value!r}")


PSBaseParserToken: TypeAlias = object


class PSBaseParser:
    """Compatibility shell retained for Unstructured's obsolete parser patch."""

    def __init__(self, fp: Any) -> None:
        self.fp = fp
        self.eof = False
        self._tokens: list[tuple[int, PSBaseParserToken]] = []
        self._curtoken = b""
        self.buf = b""
        self.charpos = 0
        self._parse1 = self._parse_main

    def seek(self, pos: int) -> None:
        self.fp.seek(pos)
        self.charpos = pos

    def _parse_keyword(self, data: bytes, index: int) -> int:
        del data
        return index

    def _parse_main(self, data: bytes, index: int) -> int:
        del data
        return index

    def _add_token(self, token: PSBaseParserToken) -> None:
        self._tokens.append((self.charpos, token))

    def fillbuf(self) -> None:
        raise PSEOF("Unexpected EOF")

    def nexttoken(self) -> tuple[int, PSBaseParserToken]:
        if not self._tokens:
            raise PSEOF("Unexpected EOF")
        return self._tokens.pop(0)


__all__ = (
    "END_KEYWORD",
    "KWD",
    "LIT",
    "PSEOF",
    "PSBaseParser",
    "PSBaseParserToken",
    "PSKeyword",
    "PSLiteral",
    "literal_name",
    "log",
)

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import zlib
from collections.abc import Callable
from typing import Any

from core_pdf.impl.integrations.pdfminer.ascii85 import ascii85decode, asciihexdecode
from core_pdf.impl.integrations.pdfminer.psparser import LIT, PSLiteral, literal_name

LITERALS_FLATE_DECODE = (LIT("FlateDecode"), LIT("Fl"))
LITERALS_ASCII85_DECODE = (LIT("ASCII85Decode"), LIT("A85"))
LITERALS_ASCIIHEX_DECODE = (LIT("ASCIIHexDecode"), LIT("AHx"))


class PDFObjRef:
    def __init__(self, doc: Any, objid: int, genno: int = 0) -> None:
        self.doc = doc
        self.objid = objid
        self.genno = genno

    def resolve(self, default: Any = None) -> Any:
        try:
            return self.doc.getobj(self.objid)
        except Exception:
            return default


class PDFStream:
    def __init__(
        self,
        attrs: dict[str, Any],
        rawdata: bytes,
        decipher: Callable[..., bytes] | None = None,
    ) -> None:
        self.attrs = attrs
        self.rawdata: bytes | None = rawdata
        self.data: bytes | None = None
        self.decipher = decipher
        self.objid: int | None = None
        self.genno: int | None = None

    def get_any(self, keys: tuple[str, ...], default: Any = None) -> Any:
        for key in keys:
            if key in self.attrs:
                return self.attrs[key]
        return default

    def get_rawdata(self) -> bytes | None:
        return self.rawdata

    def get_filters(self) -> list[tuple[PSLiteral, Any]]:
        filters = resolve1(self.get_any(("F", "Filter"), []))
        params = resolve1(self.get_any(("DP", "DecodeParms", "FDecodeParms"), {}))
        if not filters:
            return []
        if not isinstance(filters, list):
            filters = [filters]
        if not isinstance(params, list):
            params = [params] * len(filters)
        return [(LIT(literal_name(name)), param) for name, param in zip(filters, params)]

    def get_data(self) -> bytes:
        if self.data is not None:
            return self.data
        data = self.rawdata or b""
        if self.decipher:
            assert self.objid is not None
            assert self.genno is not None
            data = self.decipher(self.objid, self.genno, data, self.attrs)
        for filter_name, _ in self.get_filters():
            if filter_name in LITERALS_FLATE_DECODE:
                data = zlib.decompress(data)
            elif filter_name in LITERALS_ASCII85_DECODE:
                data = ascii85decode(data)
            elif filter_name in LITERALS_ASCIIHEX_DECODE:
                data = asciihexdecode(data)
        self.data = data
        self.rawdata = None
        return data


def resolve1(value: Any, default: Any = None) -> Any:
    while isinstance(value, PDFObjRef):
        value = value.resolve(default)
    return value


__all__ = (
    "LITERALS_ASCII85_DECODE",
    "LITERALS_ASCIIHEX_DECODE",
    "LITERALS_FLATE_DECODE",
    "PDFObjRef",
    "PDFStream",
    "resolve1",
)

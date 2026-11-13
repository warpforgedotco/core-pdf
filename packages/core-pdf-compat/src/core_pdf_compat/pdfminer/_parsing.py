# Parts ported from pdfminer.six, Copyright (c) 2004-2016 Yusuke Shinyama, under the MIT License.
"""What pdfminer.six 20260107's own parsing raises on a damaged file.

PDFDocument.__init__ finds the last startxref, reads the xref chain from there (each section,
then its /XRefStm, then its /Prev), falls back to scanning the whole file for "N G obj" lines
when the chain has no valid section, and then takes /Info and /Root from the first trailers
that have them. On a damaged file any step can raise: the fallback scan indexes the objects
of an object stream by whatever its header holds (a dictionary there raises TypeError), the
tokenizer asserts on an octal escape above \\377, resolving /Info parses an object that can
be malformed, and with no /Root at all it raises PDFSyntaxError. read_xref_from keeps no
record of the offsets it has read, and int_value turns a /Prev that is no integer into 0, so
a chain that comes back to an offset still being read repeats until Python raises
RecursionError.

core-pdf recovers from all of these, so this module reads the file as pdfminer does, with
its tokenizer, its 4096-byte reads, its object stack, its line reader, its xref loaders,
its fallback scan and its object lookup, all on one shared parser as in pdfminer, and
reports the exception pdfminer raises. It also walks the page tree as PDFPage.create_pages
does, for the page entries pdfplumber reads, and parses a page's content as
PDFContentParser does, for the entries it names when a dictionary there has a key with no
value (PSSyntaxError). Whatever this module does not follow (an
encrypted file, a filter other than Flate, a negative seek, whose error depends on the kind
of file object) ends the reading with no answer, so it never reports an error pdfminer does
not raise.
"""

from __future__ import annotations

import contextlib
import re
import zlib
from collections.abc import Callable, Iterator
from typing import Any

from core_pdf.impl.base.types import PdfByteBuffer

from .casting import safe_int
from .pdfdocument import PDFNoValidXRef
from .pdfexceptions import PDFException, PDFKeyError, PDFNotImplementedError, PDFObjectNotFound
from .pdfparser import PDFSyntaxError
from .psexceptions import PSEOF, PSException, PSSyntaxError, PSTypeError
from .utils import apply_png_predictor, apply_tiff_predictor, choplist, nunpack

EOL = re.compile(rb"[\r\n]")
SPC = re.compile(rb"\s")
NONSPC = re.compile(rb"\S")
HEX = re.compile(rb"[0-9a-fA-F]")
END_LITERAL = re.compile(rb"[#/%\[\]()<>{}\s]")
END_HEX_STRING = re.compile(rb"[^\s0-9a-fA-F]")
HEX_PAIR = re.compile(rb"[0-9a-fA-F]{2}|.")
END_NUMBER = re.compile(rb"[^0-9]")
END_KEYWORD = re.compile(rb"[#/%\[\]()<>{}\s]")
END_STRING = re.compile(rb"[()\134]")
OCT_STRING = re.compile(rb"[0-7]")
ESC_STRING = {b"b": 8, b"t": 9, b"n": 10, b"f": 12, b"r": 13, b"(": 40, b")": 41, b"\\": 92}
LINE = re.compile(rb"[^\r\n]*(?:\r\n?|\n)")
# PDFXRefFallback matches it against a line decoded as Latin-1, so \d and \s are Unicode's.
PDFOBJ_CUE = re.compile(r"^(\d+)\s+(\d+)\s+obj\b")

FLATE = frozenset({"FlateDecode", "Fl"})
UNFOLLOWED_FILTERS = frozenset(
    {
        "LZWDecode",
        "LZW",
        "ASCII85Decode",
        "A85",
        "ASCIIHexDecode",
        "AHx",
        "RunLengthDecode",
        "RL",
        "CCITTFaxDecode",
        "CCF",
    }
)
PASSED_FILTERS = frozenset({"DCTDecode", "DCT", "JBIG2Decode", "JPXDecode"})
INHERITABLE_ATTRS = frozenset({"Resources", "MediaBox", "CropBox", "Rotate"})

MESSAGE = "maximum recursion depth exceeded"


class Unfollowed(Exception):
    """pdfminer would do something here that this module does not reproduce."""


class Loop(Exception):
    """pdfminer recurses here without end, until Python raises RecursionError."""


class PSLiteral:
    """A name, as pdfminer's PSLiteral, with its repr."""

    __slots__ = ("name",)

    def __init__(self, name: str | bytes) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"/{self.name!r}"


class PSKeyword:
    """A keyword, as pdfminer's PSKeyword, with its repr."""

    __slots__ = ("name",)

    def __init__(self, name: bytes) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"/{self.name!r}"


def is_keyword(token: object, name: bytes) -> bool:
    return isinstance(token, PSKeyword) and token.name == name


def is_literal(value: object, names: frozenset[str]) -> bool:
    return isinstance(value, PSLiteral) and value.name in names


class PDFObjRef:
    """pdfminer's PDFObjRef."""

    __slots__ = ("doc", "objid")

    def __init__(self, doc: Document | None, objid: int) -> None:
        self.doc = doc
        self.objid = objid

    def __repr__(self) -> str:
        return f"<PDFObjRef:{self.objid}>"

    def resolve(self, default: object = None) -> Any:
        if self.doc is None:
            raise AssertionError
        try:
            return self.doc.getobj(self.objid)
        except PDFObjectNotFound:
            return default


def resolve1(value: object, default: object = None) -> Any:
    while isinstance(value, PDFObjRef):
        value = value.resolve(default)
    return value


def int_value(value: object) -> int:
    value = resolve1(value)
    return value if isinstance(value, int) else 0


def dict_value(value: object) -> dict[Any, Any]:
    value = resolve1(value)
    return value if isinstance(value, dict) else {}


def list_value(value: object) -> list[Any] | tuple[Any, ...]:
    value = resolve1(value)
    return value if isinstance(value, (list, tuple)) else []


def stream_value(value: object) -> PDFStream:
    value = resolve1(value)
    return value if isinstance(value, PDFStream) else PDFStream({}, b"")


def literal_name(value: object) -> str:
    if isinstance(value, PSLiteral):
        if isinstance(value.name, str):
            return value.name
        try:
            return str(value.name, "utf-8")
        except UnicodeDecodeError:
            return str(value.name)
    return str(value)


class PDFStream:
    """pdfminer's PDFStream, unencrypted."""

    def __init__(self, attrs: dict[str, Any], rawdata: bytes) -> None:
        self.attrs = attrs
        self.rawdata: bytes | None = rawdata
        self.data: bytes | None = None
        self.objid: int | None = None
        self.genno: int | None = None

    def __repr__(self) -> str:
        if self.data is None:
            return f"<PDFStream({self.objid!r}): raw={len(self.rawdata or b'')}, {self.attrs!r}>"
        return f"<PDFStream({self.objid!r}): len={len(self.data)}, {self.attrs!r}>"

    def __contains__(self, name: object) -> bool:
        return name in self.attrs

    def __getitem__(self, name: str) -> Any:
        try:
            return self.attrs[name]
        except KeyError as error:
            raise PDFKeyError(
                f"PDF stream object {self.objid} does not have attribute '{name}'"
            ) from error

    def get(self, name: str, default: object = None) -> Any:
        return self.attrs.get(name, default)

    def get_any(self, names: tuple[str, ...], default: object = None) -> Any:
        for name in names:
            if name in self.attrs:
                return self.attrs[name]
        return default

    def get_data(self) -> bytes:
        if self.data is None:
            self.data = decode(self)
            self.rawdata = None
        return self.data


def decode(stream: PDFStream) -> bytes:
    """PDFStream.decode, for the filters this module follows."""
    data = stream.rawdata or b""
    filters = resolve1(stream.get_any(("F", "Filter"), []))
    params = resolve1(stream.get_any(("DP", "DecodeParms", "FDecodeParms"), {}))
    if not filters:
        return data
    if not isinstance(filters, list):
        filters = [filters]
    if not isinstance(params, list):
        params = [params] * len(filters)
    resolved_filters = [resolve1(f) for f in filters]
    resolved_params = [resolve1(param) for param in params]
    for f, param in zip(resolved_filters, resolved_params, strict=False):
        if is_literal(f, FLATE):
            data = inflate(data)
        elif is_literal(f, UNFOLLOWED_FILTERS):
            raise Unfollowed(f"filter {f!r}")
        elif is_literal(f, PASSED_FILTERS):
            pass
        elif is_literal(f, frozenset({"Crypt"})):
            raise PDFNotImplementedError("/Crypt filter is unsupported")
        else:
            raise PDFNotImplementedError(f"Unsupported filter: {f!r}")
        if param and "Predictor" in param:
            data = predict(param, data)
    return data


def inflate(data: bytes) -> bytes:
    """pdfminer's Flate decoding: a byte at a time once zlib fails, and b"" if that fails."""
    try:
        return zlib.decompress(data)
    except zlib.error:
        pass
    decompressor = zlib.decompressobj()
    result = b""
    for index in range(len(data)):
        try:
            result += decompressor.decompress(data[index : index + 1])
        except zlib.error:
            # pdfminer keeps what it has only when the error is in the last four bytes.
            return b"" if index < len(data) - 3 else result
    return result


def predict(param: Any, data: bytes) -> bytes:
    pred = int_value(param["Predictor"])
    if pred == 1:
        return data
    if pred != 2 and pred < 10:
        raise PDFNotImplementedError(f"Unsupported predictor: {pred!r}")
    colors = int_value(param.get("Colors", 1))
    columns = int_value(param.get("Columns", 1))
    bitspercomponent = int_value(param.get("BitsPerComponent", 8))
    try:
        if pred == 2:
            return apply_tiff_predictor(colors, columns, bitspercomponent, data)
        return apply_png_predictor(pred, colors, columns, bitspercomponent, data)
    except Exception as error:
        raise Unfollowed("a predictor that fails") from error


Token = int | float | bool | bytes | PSLiteral | PSKeyword
Entry = tuple[int, Any]


class Parser:
    """pdfminer's PDFParser over the file's bytes, read 4096 bytes at a time as it reads."""

    BUFSIZ = 4096

    def __init__(self, data: PdfByteBuffer | bytes, doc: Document | None = None) -> None:
        self.data = data
        self.doc = doc
        self.fallback = False
        self.seek(0)

    # PSBaseParser

    def seek(self, pos: int) -> None:
        if pos < 0:
            raise Unfollowed("a negative seek, whose error depends on the file object")
        self.fppos = pos
        self.bufpos = pos
        self.buf = b""
        self.charpos = 0
        self._parse1: Callable[[bytes, int], int] = self._parse_main
        self._curtoken = b""
        self._curtokenpos = 0
        self._tokens: list[tuple[int, Token]] = []
        self.eof = False
        self.reset()

    def read(self, size: int) -> bytes:
        end = len(self.data) if size < 0 else self.fppos + size
        chunk = bytes(self.data[self.fppos : end])
        self.fppos += len(chunk)
        return chunk

    def fillbuf(self) -> None:
        if self.charpos < len(self.buf):
            return
        self.bufpos = self.fppos
        self.buf = self.read(self.BUFSIZ)
        if not self.buf:
            raise PSEOF("Unexpected EOF")
        self.charpos = 0

    def nextline(self) -> tuple[int, bytes]:
        """A line ending in \\r, \\n or \\r\\n; one without an ending is an EOF."""
        linepos = self.bufpos + self.charpos
        data = self.data
        match = EOL.search(data, linepos)
        if match is None:
            raise PSEOF("Unexpected EOF")
        end = match.end()
        if data[match.start() : end] == b"\r":
            if end >= len(data):
                raise PSEOF("Unexpected EOF")
            if data[end : end + 1] == b"\n":
                end += 1
        self.forward(end)
        return linepos, bytes(data[linepos:end])

    def forward(self, pos: int) -> None:
        # The next read starts at pos, as pdfminer's does after reading a line.
        self.fppos = self.bufpos = pos
        self.buf = b""
        self.charpos = 0

    def _parse_main(self, s: bytes, i: int) -> int:  # noqa: C901, PLR0911
        m = NONSPC.search(s, i)
        if not m:
            return len(s)
        j = m.start(0)
        c = s[j : j + 1]
        self._curtokenpos = self.bufpos + j
        if c == b"%":
            self._curtoken = b"%"
            self._parse1 = self._parse_comment
            return j + 1
        if c == b"/":
            self._curtoken = b""
            self._parse1 = self._parse_literal
            return j + 1
        if c in b"-+" or c.isdigit():
            self._curtoken = c
            self._parse1 = self._parse_number
            return j + 1
        if c == b".":
            self._curtoken = c
            self._parse1 = self._parse_float
            return j + 1
        if c.isalpha():
            self._curtoken = c
            self._parse1 = self._parse_keyword
            return j + 1
        if c == b"(":
            self._curtoken = b""
            self.paren = 1
            self._parse1 = self._parse_string
            return j + 1
        if c == b"<":
            self._curtoken = b""
            self._parse1 = self._parse_wopen
            return j + 1
        if c == b">":
            self._curtoken = b""
            self._parse1 = self._parse_wclose
            return j + 1
        if c == b"\x00":
            return j + 1
        self._add_token(PSKeyword(c))
        return j + 1

    def _add_token(self, obj: Token) -> None:
        self._tokens.append((self._curtokenpos, obj))

    def _parse_comment(self, s: bytes, i: int) -> int:
        m = EOL.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        self._parse1 = self._parse_main
        return j

    def _parse_literal(self, s: bytes, i: int) -> int:
        m = END_LITERAL.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        c = s[j : j + 1]
        if c == b"#":
            self.hex = b""
            self._parse1 = self._parse_literal_hex
            return j + 1
        name: str | bytes
        try:
            name = str(self._curtoken, "utf-8")
        except UnicodeDecodeError:
            name = self._curtoken
        self._add_token(PSLiteral(name))
        self._parse1 = self._parse_main
        return j

    def _parse_literal_hex(self, s: bytes, i: int) -> int:
        c = s[i : i + 1]
        if HEX.match(c) and len(self.hex) < 2:
            self.hex += c
            return i + 1
        if self.hex:
            self._curtoken += bytes((int(self.hex, 16),))
        self._parse1 = self._parse_literal
        return i

    def _parse_number(self, s: bytes, i: int) -> int:
        m = END_NUMBER.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        c = s[j : j + 1]
        if c == b".":
            self._curtoken += c
            self._parse1 = self._parse_float
            return j + 1
        with contextlib.suppress(ValueError):
            self._add_token(int(self._curtoken))
        self._parse1 = self._parse_main
        return j

    def _parse_float(self, s: bytes, i: int) -> int:
        m = END_NUMBER.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        with contextlib.suppress(ValueError):
            self._add_token(float(self._curtoken))
        self._parse1 = self._parse_main
        return j

    def _parse_keyword(self, s: bytes, i: int) -> int:
        m = END_KEYWORD.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        token: bool | PSKeyword
        if self._curtoken == b"true":
            token = True
        elif self._curtoken == b"false":
            token = False
        else:
            token = PSKeyword(self._curtoken)
        self._add_token(token)
        self._parse1 = self._parse_main
        return j

    def _parse_string(self, s: bytes, i: int) -> int:
        m = END_STRING.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        c = s[j : j + 1]
        if c == b"\\":
            self.oct = b""
            self._parse1 = self._parse_string_1
            return j + 1
        if c == b"(":
            self.paren += 1
            self._curtoken += c
            return j + 1
        if c == b")":
            self.paren -= 1
            if self.paren:
                self._curtoken += c
                return j + 1
        self._add_token(self._curtoken)
        self._parse1 = self._parse_main
        return j + 1

    def _parse_string_1(self, s: bytes, i: int) -> int:
        c = s[i : i + 1]
        if OCT_STRING.match(c) and len(self.oct) < 3:
            self.oct += c
            return i + 1
        if self.oct:
            chrcode = int(self.oct, 8)
            if chrcode >= 256:  # pdfminer asserts
                raise AssertionError(f"Invalid octal {self.oct!r} ({chrcode})")
            self._curtoken += bytes((chrcode,))
            self._parse1 = self._parse_string
            return i
        if c in ESC_STRING:
            self._curtoken += bytes((ESC_STRING[c],))
        elif c == b"\r" and len(s) > i + 1 and s[i + 1 : i + 2] == b"\n":
            i += 1
        self._parse1 = self._parse_string
        return i + 1

    def _parse_wopen(self, s: bytes, i: int) -> int:
        c = s[i : i + 1]
        if c == b"<":
            self._add_token(PSKeyword(b"<<"))
            self._parse1 = self._parse_main
            i += 1
        else:
            self._parse1 = self._parse_hexstring
        return i

    def _parse_wclose(self, s: bytes, i: int) -> int:
        c = s[i : i + 1]
        if c == b">":
            self._add_token(PSKeyword(b">>"))
            i += 1
        self._parse1 = self._parse_main
        return i

    def _parse_hexstring(self, s: bytes, i: int) -> int:
        m = END_HEX_STRING.search(s, i)
        if not m:
            self._curtoken += s[i:]
            return len(s)
        j = m.start(0)
        self._curtoken += s[i:j]
        token = HEX_PAIR.sub(lambda m: bytes((int(m.group(0), 16),)), SPC.sub(b"", self._curtoken))
        self._add_token(token)
        self._parse1 = self._parse_main
        return j

    def nexttoken(self) -> tuple[int, Token]:
        if self.eof:
            raise PSEOF("Unexpected EOF")
        while not self._tokens:
            try:
                self.fillbuf()
                self.charpos = self._parse1(self.buf, self.charpos)
            except PSEOF:
                self.charpos = self._parse1(b"\n", 0)
                self.eof = True
                if not self._tokens:
                    raise
        return self._tokens.pop(0)

    # PSStackParser

    def reset(self) -> None:
        self.context: list[tuple[int, str | None, list[Entry]]] = []
        self.curtype: str | None = None
        self.curstack: list[Entry] = []
        self.results: list[Entry] = []

    def push(self, *objs: Entry) -> None:
        self.curstack.extend(objs)

    def pop(self, n: int) -> list[Entry]:
        objs = self.curstack[-n:]
        self.curstack[-n:] = []
        return objs

    def popall(self) -> list[Entry]:
        objs = self.curstack
        self.curstack = []
        return objs

    def add_results(self, *objs: Entry) -> None:
        self.results.extend(objs)

    def start_type(self, pos: int, kind: str) -> None:
        self.context.append((pos, self.curtype, self.curstack))
        (self.curtype, self.curstack) = (kind, [])

    def end_type(self, kind: str) -> tuple[int, list[Any]]:
        if self.curtype != kind:
            raise PSTypeError(f"Type mismatch: {self.curtype!r} != {kind!r}")
        objs = [obj for (_, obj) in self.curstack]
        (pos, self.curtype, self.curstack) = self.context.pop()
        return (pos, objs)

    def flush(self) -> None:
        pass

    def nextobject(self) -> Entry:  # noqa: C901
        while not self.results:
            (pos, token) = self.nexttoken()
            if not isinstance(token, PSKeyword):
                self.push((pos, token))
            elif token.name == b"[":
                self.start_type(pos, "a")
            elif token.name == b"]":
                with contextlib.suppress(PSTypeError):
                    self.push(self.end_type("a"))
            elif token.name == b"<<":
                self.start_type(pos, "d")
            elif token.name == b">>":
                try:
                    (pos, objs) = self.end_type("d")
                    if len(objs) % 2 != 0:
                        raise PSSyntaxError(f"Invalid dictionary construct: {objs!r}")
                    d = {literal_name(k): v for (k, v) in choplist(2, objs) if v is not None}
                    self.push((pos, d))
                except PSTypeError:
                    pass
            elif token.name == b"{":
                self.start_type(pos, "p")
            elif token.name == b"}":
                with contextlib.suppress(PSTypeError):
                    self.push(self.end_type("p"))
            else:
                self.do_keyword(pos, token)
            if self.context:
                continue
            self.flush()
        return self.results.pop(0)

    # PDFParser

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        name = token.name
        if name in (b"xref", b"startxref"):
            self.add_results(*self.pop(1))
        elif name == b"endobj":
            self.add_results(*self.pop(4))
        elif name == b"null":
            self.push((pos, None))
        elif name == b"R":
            if len(self.curstack) >= 2:
                (_, object_id), _ = self.pop(2)
                objid = safe_int(object_id)
                if objid is not None:
                    self.push((pos, PDFObjRef(self.doc, objid)))
        elif name == b"stream":
            self.read_stream(pos)
        else:
            self.push((pos, token))

    def read_stream(self, pos: int) -> None:
        popped = self.pop(1)
        try:
            ((_, dic),) = popped
        except ValueError as error:
            raise PDFSyntaxError(f"Invalid stream dictionary: {popped}") from error
        dic = dict_value(dic)
        objlen = 0
        if not self.fallback:
            with contextlib.suppress(KeyError):
                objlen = int_value(dic["Length"])
        self.seek(pos)
        try:
            (_, line) = self.nextline()
        except PSEOF:
            return
        pos += len(line)
        self.fppos = pos
        data = bytearray(self.read(objlen))
        self.seek(pos + objlen)
        while True:
            try:
                (_, line) = self.nextline()
            except PSEOF:
                break
            if b"endstream" in line:
                i = line.index(b"endstream")
                objlen += i
                if self.fallback:
                    data += line[:i]
                break
            objlen += len(line)
            if self.fallback:
                data += line
        self.seek(pos + objlen)
        if self.doc is None:
            raise AssertionError
        self.push((pos, PDFStream(dic, bytes(data))))


class StreamParser(Parser):
    """pdfminer's PDFStreamParser: the objects of an object stream."""

    def flush(self) -> None:
        self.add_results(*self.popall())

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        name = token.name
        if name == b"R":
            (_, object_id), _ = self.pop(2)
            objid = safe_int(object_id)
            if objid is not None:
                self.push((pos, PDFObjRef(self.doc, objid)))
            return
        if name in (b"obj", b"endobj"):
            return
        self.push((pos, token))


class ContentParser(Parser):
    """pdfminer's PDFContentParser over a page's content, to the first error it raises."""

    def flush(self) -> None:
        self.add_results(*self.popall())

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        name = token.name
        if name == b"BI":
            self.start_type(pos, "inline")
        elif name == b"ID":
            try:
                (_, objs) = self.end_type("inline")
                if len(objs) % 2 != 0:
                    raise PSTypeError(f"Invalid dictionary construct: {objs!r}")
                d = {literal_name(k): resolve1(v) for (k, v) in choplist(2, objs)}
                eos = b"EI"
                image_filter = d.get("F")
                if image_filter is not None:
                    if isinstance(image_filter, PSLiteral):
                        image_filter = [image_filter]
                    if is_literal(image_filter[0], frozenset({"ASCII85Decode", "A85"})):
                        eos = b"~>"
                (pos, data) = self.get_inline_data(pos + len(b"ID "), target=eos)
                if eos != b"EI":
                    data += eos
                obj = PDFStream(d, data)
                self.push((pos, obj))
                if eos == b"EI":
                    self.push((pos, PSKeyword(b"EI")))
            except PSTypeError:
                pass
        else:
            self.push((pos, token))

    def get_inline_data(self, pos: int, target: bytes) -> tuple[int, bytes]:
        """PDFContentParser.get_inline_data: up to the target and a whitespace after it."""
        self.seek(pos)
        data = self.data
        end = len(data)
        i = 0
        position = pos
        while i <= len(target):
            if position >= end:
                raise PSEOF("Unexpected EOF, file truncated?")
            if i:
                c = bytes(data[position : position + 1])
                position += 1
                if (len(target) <= i and c.isspace()) or (
                    i < len(target) and c == bytes((target[i],))
                ):
                    i += 1
                else:
                    i = 0
            else:
                j = data.find(target[:1], position)
                if j < 0:
                    position = end
                else:
                    position = j + 1
                    i = 1
        found = bytes(data[pos:position])[: -(len(target) + 1)]
        found = re.sub(rb"(\x0d\x0a|[\x0d\x0a])$", b"", found)
        self.forward(position)
        return (pos, found)


def pdfminer_content_error(data: bytes | memoryview) -> PSException | None:
    """The PSSyntaxError pdfminer raises parsing this content, if it raises one."""
    parser = ContentParser(bytes(data))
    try:
        while True:
            parser.nextobject()
    except PSSyntaxError as error:
        return error
    except Exception:  # noqa: BLE001  # pdfminer stops, or this module does not follow it
        return None


def load_trailer(parser: Parser) -> dict[str, Any]:
    """PDFXRef.load_trailer."""
    try:
        (_, kwd) = parser.nexttoken()
        if not is_keyword(kwd, b"trailer"):
            raise AssertionError(str(kwd))
        (_, dic) = parser.nextobject()
    except PSEOF:
        x = parser.pop(1)
        if not x:
            raise PDFNoValidXRef("Unexpected EOF - file corrupted") from None
        (_, dic) = x[0]
    return dict(dict_value(dic))


class XRef:
    """pdfminer's PDFXRef: a classic xref section."""

    def __init__(self) -> None:
        self.offsets: dict[Any, tuple[int | None, int, int]] = {}
        self.trailer: dict[str, Any] = {}

    def get_trailer(self) -> dict[str, Any]:
        return self.trailer

    def get_objids(self) -> Iterator[Any]:
        return iter(self.offsets.keys())

    def get_pos(self, objid: int) -> tuple[int | None, int, int]:
        return self.offsets[objid]

    def load(self, parser: Parser) -> None:
        while True:
            try:
                (pos, line) = parser.nextline()
                line = line.strip()
                if not line:
                    continue
            except PSEOF as error:
                raise PDFNoValidXRef("Unexpected EOF - file corrupted?") from error
            if line.startswith(b"trailer"):
                parser.seek(pos)
                break
            f = line.split(b" ")
            if len(f) != 2:
                raise PDFNoValidXRef("Trailer not found")
            try:
                (start, nobjs) = map(int, f)
            except ValueError as error:
                raise PDFNoValidXRef("Invalid line") from error
            self.load_entries(parser, start, nobjs)
        self.trailer.update(load_trailer(parser))

    def load_entries(self, parser: Parser, start: int, nobjs: int) -> None:
        """The entry lines of a subsection, read as pdfminer's nextline reads them."""
        data = parser.data
        position = parser.bufpos + parser.charpos
        lines: Iterator[re.Match[bytes]] = LINE.finditer(data, position)
        for objid in range(start, start + nobjs):
            match = next(lines, None)
            if match is None or (match.end() == len(data) and data[match.end() - 1 :] == b"\r"):
                raise PDFNoValidXRef("Unexpected EOF - file corrupted?")
            position = match.end()
            f = match.group().strip().split(b" ")
            if len(f) != 3:
                parser.forward(position)
                raise PDFNoValidXRef("Invalid XRef format")
            (pos_b, genno_b, use_b) = f
            if use_b != b"n":
                continue
            pos_i = safe_int(pos_b)
            genno_i = safe_int(genno_b)
            if pos_i is not None and genno_i is not None:
                self.offsets[objid] = (None, pos_i, genno_i)
        parser.forward(position)


class XRefFallback(XRef):
    """pdfminer's PDFXRefFallback: every "N G obj" line before the first trailer line."""

    def load(self, parser: Parser) -> None:
        parser.seek(0)
        while True:
            try:
                (pos, line_bytes) = parser.nextline()
            except PSEOF:
                break
            if line_bytes.startswith(b"trailer"):
                parser.seek(pos)
                self.trailer.update(load_trailer(parser))
                break
            m = PDFOBJ_CUE.match(line_bytes.decode("latin-1"))
            if not m:
                continue
            (objid_s, genno_s) = m.groups()
            objid = int(objid_s)
            genno = int(genno_s)
            self.offsets[objid] = (None, pos, genno)
            parser.seek(pos)
            (_, obj) = parser.nextobject()
            if isinstance(obj, PDFStream) and is_literal(obj.get("Type"), frozenset({"ObjStm"})):
                stream = stream_value(obj)
                try:
                    n = stream["N"]
                except KeyError:
                    n = 0
                parser1 = StreamParser(stream.get_data())
                objs: list[Any] = []
                try:
                    while True:
                        (_, obj) = parser1.nextobject()
                        objs.append(obj)
                except PSEOF:
                    pass
                n = min(n, len(objs) // 2)
                for index in range(n):
                    objid1 = objs[index * 2]
                    self.offsets[objid1] = (objid, index, 0)


class XRefStream:
    """pdfminer's PDFXRefStream."""

    def __init__(self) -> None:
        self.data: bytes = b""
        self.entlen: Any = None
        self.fl1: Any = None
        self.fl2: Any = None
        self.fl3: Any = None
        self.ranges: list[tuple[Any, ...]] = []
        self.trailer: dict[str, Any] = {}

    def get_trailer(self) -> dict[str, Any]:
        return self.trailer

    def load(self, parser: Parser) -> None:
        parser.nexttoken()
        parser.nexttoken()
        parser.nexttoken()
        (_, stream) = parser.nextobject()
        if not isinstance(stream, PDFStream) or not is_literal(
            stream.get("Type"), frozenset({"XRef"})
        ):
            raise PDFNoValidXRef("Invalid PDF stream spec.")
        size = stream["Size"]
        index_array = stream.get("Index", (0, size))
        if len(index_array) % 2 != 0:
            raise PDFSyntaxError("Invalid index number")
        self.ranges.extend(choplist(2, index_array))
        (self.fl1, self.fl2, self.fl3) = stream["W"]
        if self.fl1 is None or self.fl2 is None or self.fl3 is None:
            raise AssertionError
        self.data = stream.get_data()
        self.entlen = self.fl1 + self.fl2 + self.fl3
        self.trailer = stream.attrs

    def get_objids(self) -> Iterator[Any]:
        for start, nobjs in self.ranges:
            for i in range(nobjs):
                offset = self.entlen * i
                ent = self.data[offset : offset + self.entlen]
                f1 = nunpack(ent[: self.fl1], 1)
                if f1 in {1, 2}:
                    yield start + i

    def get_pos(self, objid: int) -> tuple[int | None, int, int]:
        index = 0
        for start, nobjs in self.ranges:
            if start <= objid and objid < start + nobjs:
                index += objid - start
                break
            index += nobjs
        else:
            raise PDFKeyError(objid)
        offset = self.entlen * index
        ent = self.data[offset : offset + self.entlen]
        f1 = nunpack(ent[: self.fl1], 1)
        f2 = nunpack(ent[self.fl1 : self.fl1 + self.fl2])
        f3 = nunpack(ent[self.fl1 + self.fl2 :])
        if f1 == 1:
            return (None, f2, f3)
        if f1 == 2:
            return (f2, f3, 0)
        raise PDFKeyError(objid)


def find_xref(data: PdfByteBuffer) -> int:
    """PDFDocument.find_xref: the offset on the line after the last startxref line.

    pdfminer reads lines backwards, each from a \\r or \\n to the next, and never reads
    the text before the file's first line ending.
    """
    previous = b""
    end = len(data)
    carriage_return = data.rfind(b"\r", 0, end)
    line_feed = data.rfind(b"\n", 0, end)
    while (start := max(carriage_return, line_feed)) >= 0:
        line = bytes(data[start:end]).strip()
        if line == b"startxref":
            if not previous.isdigit():
                raise PDFNoValidXRef(f"Invalid xref position, no digit: {previous!r}")
            offset = int(previous)
            if offset >= 2**31:
                raise PDFNoValidXRef(f"Invalid xref position, too large: {offset!r}")
            return offset
        if line:
            previous = line
        end = start
        if start == carriage_return:
            carriage_return = data.rfind(b"\r", 0, start)
        else:
            line_feed = data.rfind(b"\n", 0, start)
    raise PDFNoValidXRef("Unexpected EOF")


class Document:
    """pdfminer's PDFDocument, opened with the empty password."""

    def __init__(self, data: PdfByteBuffer) -> None:
        self.xrefs: list[Any] = []
        self.info: list[dict[str, Any]] = []
        self.catalog: dict[str, Any] = {}
        self._cached_objs: dict[Any, tuple[object, int]] = {}
        self._parsed_objs: dict[Any, tuple[list[object], int]] = {}
        # The objects being looked up, and the xref sections being read with the count of
        # lookups made when each was started, to tell pdfminer's endless recursions.
        self.getting: set[Any] = set()
        self.reading: dict[int, int] = {}
        self.lookups = 0
        self.parser = Parser(data, self)
        try:
            pos = find_xref(data)
            self.read_xref_from(pos)
        except PDFNoValidXRef:
            self.parser.fallback = True
            newxref = XRefFallback()
            newxref.load(self.parser)
            self.xrefs.append(newxref)
        for xref in self.xrefs:
            trailer = xref.get_trailer()
            if not trailer:
                continue
            if "Encrypt" in trailer:
                raise Unfollowed("an encrypted file")
            if "Info" in trailer:
                self.info.append(dict_value(trailer["Info"]))
            if "Root" in trailer:
                self.catalog = dict_value(trailer["Root"])
                break
        else:
            raise PDFSyntaxError("No /Root object! - Is this really a PDF?")

    def read_xref_from(self, start: int) -> None:
        """PDFDocument.read_xref_from: the section, then /XRefStm, then /Prev."""
        if start in self.reading:
            if self.reading[start] != self.lookups:
                # An object looked up since may read differently the second time round.
                raise Unfollowed("an xref chain that returns after looking up objects")
            # Every section since was read from its offset alone, so the reading repeats.
            raise Loop
        self.reading[start] = self.lookups
        parser = self.parser
        parser.seek(start)
        try:
            (pos, token) = parser.nexttoken()
        except PSEOF as error:
            raise PDFNoValidXRef("Unexpected EOF") from error
        xref: XRef | XRefStream
        if isinstance(token, int):
            parser.seek(pos)
            xref = XRefStream()
            xref.load(parser)
        else:
            if is_keyword(token, b"xref"):
                parser.nextline()
            xref = XRef()
            xref.load(parser)
        self.xrefs.append(xref)
        trailer = xref.get_trailer()
        if "XRefStm" in trailer:
            self.read_xref_from(int_value(trailer["XRefStm"]))
        if "Prev" in trailer:
            self.read_xref_from(int_value(trailer["Prev"]))
        del self.reading[start]

    def getobj(self, objid: Any) -> object:
        if not self.xrefs:
            raise PDFException("PDFDocument is not initialized")
        if objid in self._cached_objs:
            return self._cached_objs[objid][0]
        if objid in self.getting:
            raise Loop
        self.lookups += 1
        self.getting.add(objid)
        try:
            return self.lookup(objid)
        finally:
            self.getting.discard(objid)

    def lookup(self, objid: Any) -> object:
        obj: object = None
        genno = 0
        for xref in self.xrefs:
            try:
                (strmid, index, genno) = xref.get_pos(objid)
            except KeyError:
                continue
            try:
                if strmid is not None:
                    stream = stream_value(self.getobj(strmid))
                    obj = self._getobj_objstm(stream, index)
                else:
                    obj = self._getobj_parse(index, objid)
                if isinstance(obj, PDFStream):
                    obj.objid = objid
                    obj.genno = genno
                break
            except PSEOF, PDFSyntaxError:
                continue
        else:
            raise PDFObjectNotFound(objid)
        self._cached_objs[objid] = (obj, genno)
        return obj

    def _getobj_objstm(self, stream: PDFStream, index: int) -> object:
        if stream.objid in self._parsed_objs:
            (objs, n) = self._parsed_objs[stream.objid]
        else:
            (objs, n) = self._get_objects(stream)
            self._parsed_objs[stream.objid] = (objs, n)
        i = n * 2 + index
        try:
            obj = objs[i]
        except IndexError as error:
            raise PDFSyntaxError(f"index too big: {index!r}") from error
        return obj

    def _get_objects(self, stream: PDFStream) -> tuple[list[object], int]:
        try:
            n = stream["N"]
        except KeyError:
            n = 0
        parser = StreamParser(stream.get_data(), self)
        objs: list[object] = []
        try:
            while True:
                (_, obj) = parser.nextobject()
                objs.append(obj)
        except PSEOF:
            pass
        return (objs, n)

    def _getobj_parse(self, pos: int, objid: int) -> object:
        parser = self.parser
        parser.seek(pos)
        (_, objid1) = parser.nexttoken()
        (_, _genno) = parser.nexttoken()
        (_, kwd) = parser.nexttoken()
        if objid1 != objid:
            x = []
            while not is_keyword(kwd, b"obj"):
                (_, kwd) = parser.nexttoken()
                x.append(kwd)
            if len(x) >= 2:
                objid1 = x[-2]
        if objid1 != objid:
            raise PDFSyntaxError(f"objid mismatch: {objid1!r}={objid!r}")
        if not is_keyword(kwd, b"obj"):
            raise PDFSyntaxError(f"Invalid object spec: offset={pos!r}")
        (_, obj) = parser.nextobject()
        return obj


def create_pages(document: Document) -> Iterator[dict[str, Any]]:
    """PDFPage.create_pages: the entries of each page, read as PDFPage.__init__ reads them."""
    if "PageLabels" in document.catalog:
        raise Unfollowed("page labels")

    def depth_first_search(
        obj: Any, parent: dict[str, Any], visited: set[Any]
    ) -> Iterator[dict[str, Any]]:
        if isinstance(obj, int):
            object_id = obj
            object_properties = dict_value(document.getobj(object_id)).copy()
        else:
            object_id = obj.objid
            object_properties = dict_value(obj).copy()
        if object_id in visited:
            return
        visited.add(object_id)
        for k, v in parent.items():
            if k in INHERITABLE_ATTRS and k not in object_properties:
                object_properties[k] = v
        object_type = object_properties.get("Type")
        if object_type is None:
            object_type = object_properties.get("type")
        if is_literal(object_type, frozenset({"Pages"})) and "Kids" in object_properties:
            for child in list_value(object_properties["Kids"]):
                yield from depth_first_search(child, object_properties, visited)
        elif is_literal(object_type, frozenset({"Page"})):
            yield object_properties

    pages = False
    if "Pages" in document.catalog:
        for tree in depth_first_search(document.catalog["Pages"], document.catalog, set()):
            yield read_page(tree)
            pages = True
    if not pages:
        for xref in document.xrefs:
            for objid in xref.get_objids():
                try:
                    obj = document.getobj(objid)
                    if isinstance(obj, dict) and is_literal(obj.get("Type"), frozenset({"Page"})):
                        yield read_page(obj)
                except PDFObjectNotFound:
                    pass


def read_page(attrs: object) -> dict[str, Any]:
    """PDFPage.__init__: what it resolves and parses of a page's entries, which can raise."""
    entries = dict_value(attrs)
    resolve1(entries.get("LastModified"))
    resolve1(entries.get("Resources", {}))
    mediabox = parse_box(entries.get("MediaBox"), (0.0, 0.0, 612.0, 792.0))
    parse_box(entries.get("CropBox"), mediabox)
    if entries.get("Contents") is not None:
        resolve1(entries.get("Contents"))
    int_value(entries.get("Rotate", 0))
    return entries


def parse_box(value: object, default: tuple[float, ...]) -> tuple[float, ...]:
    """PDFPage._parse_mediabox and _parse_cropbox: the default when the box is unreadable."""
    if value is None:
        return default
    try:
        (x0, y0, x1, y1) = (resolve1(item) for item in resolve1(value))
        return float(x0), float(y0), float(x1), float(y1)
    except ValueError:
        return default


def open_document(data: PdfByteBuffer) -> Document | BaseException | None:
    """pdfminer's PDFDocument for the file, the exception it raises, or None if unknown."""
    try:
        return Document(data)
    except Loop:
        return RecursionError(MESSAGE)
    except Unfollowed, RecursionError, MemoryError:
        return None
    except Exception as error:  # noqa: BLE001  # what pdfminer raises
        return error


def pdfminer_open_error(data: PdfByteBuffer) -> BaseException | None:
    """The exception pdfminer.six's PDFDocument raises opening this file, if it raises one."""
    opened = open_document(data)
    return opened if isinstance(opened, BaseException) else None


__all__ = (
    "MESSAGE",
    "Document",
    "Loop",
    "Unfollowed",
    "create_pages",
    "open_document",
    "pdfminer_content_error",
    "pdfminer_open_error",
)

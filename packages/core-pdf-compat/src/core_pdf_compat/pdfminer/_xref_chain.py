# Parts ported from pdfminer.six, Copyright (c) 2004-2016 Yusuke Shinyama, under the MIT License.
"""Whether pdfminer.six's PDFDocument follows its xref chain without end.

PDFDocument.read_xref_from reads the section at an offset, then calls itself on the
trailer's /XRefStm and then on its /Prev, with no record of the offsets it has read, and
int_value turns a /Prev that is no integer into 0. Reading a section depends only on the
offset (pdfminer resolves no reference on the paths followed here), so a chain that comes
back to an offset still being read repeats forever, until Python raises RecursionError.

This module reads the chain as pdfminer does: its tokenizer, its 4096-byte reads, its
object stack, its line reader and its xref section loaders, and reports whether the walk
returns to an offset on the current path before anything raises. Whatever pdfminer would
raise on, and whatever this module does not follow (a reference to resolve, a filter it
does not decode), ends the walk with no loop, so a loop is only reported where pdfminer
loops.
"""

from __future__ import annotations

import contextlib
import re
import zlib
from collections.abc import Callable, Iterator
from typing import Any

from core_pdf.impl.types import PdfByteBuffer

from .psexceptions import PSEOF, PSSyntaxError, PSTypeError
from .utils import apply_png_predictor, apply_tiff_predictor

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
ESC_STRING = {
    b"b": 8,
    b"t": 9,
    b"n": 10,
    b"f": 12,
    b"r": 13,
    b"(": 40,
    b")": 41,
    b"\\": 92,
}
LINE = re.compile(rb"[^\r\n]*(?:\r\n?|\n)")

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

MESSAGE = "maximum recursion depth exceeded"


class Unfollowed(Exception):
    """pdfminer would do something here that this module does not reproduce."""


class Literal:
    """A name, as pdfminer's interned PSLiteral: no iteration, no containment."""

    __slots__ = ("name",)

    def __init__(self, name: str | bytes) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"/{self.name!r}"


class Keyword:
    __slots__ = ("name",)

    def __init__(self, name: bytes) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"/{self.name!r}"


class Reference:
    __slots__ = ("objid",)

    def __init__(self, objid: int) -> None:
        self.objid = objid


class Stream:
    """pdfminer's PDFStream, as far as reading an xref stream uses it."""

    def __init__(self, attrs: dict[str, Any], rawdata: bytes) -> None:
        self.attrs = attrs
        self.rawdata = rawdata

    def __contains__(self, name: object) -> bool:
        return name in self.attrs

    def __getitem__(self, name: str) -> Any:
        return self.attrs[name]

    def get(self, name: str, default: object = None) -> Any:
        return self.attrs.get(name, default)

    def get_any(self, names: tuple[str, ...], default: object = None) -> Any:
        for name in names:
            if name in self.attrs:
                return self.attrs[name]
        return default


def resolve1(value: object) -> Any:
    if isinstance(value, Reference):
        raise Unfollowed("pdfminer resolves an object while reading the xref chain")
    return value


def int_value(value: object) -> int:
    value = resolve1(value)
    return value if isinstance(value, int) else 0


def dict_value(value: object) -> dict[Any, Any]:
    value = resolve1(value)
    return value if isinstance(value, dict) else {}


def literal_name(value: object) -> str:
    if isinstance(value, Literal):
        if isinstance(value.name, str):
            return value.name
        try:
            return str(value.name, "utf-8")
        except UnicodeDecodeError:
            return str(value.name)
    # pdfminer keys other values by str(value); none of those is a name read here.
    return f"\0{value!r}"


def is_literal(value: object, names: frozenset[str]) -> bool:
    return isinstance(value, Literal) and value.name in names


Token = int | float | bool | bytes | Literal | Keyword
Entry = tuple[int, Any]


class Parser:
    """pdfminer's PDFParser over the file's bytes, read 4096 bytes at a time as it reads."""

    BUFSIZ = 4096

    def __init__(self, data: PdfByteBuffer) -> None:
        self.data = data
        self.seek(0)

    # PSBaseParser

    def seek(self, pos: int) -> None:
        if pos < 0:
            raise ValueError("negative seek value")
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
        # Every read after a line seeks first, so the buffer can restart at the line's end.
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
        self._add_token(Keyword(c))
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
        self._add_token(Literal(name))
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
        token: bool | Keyword
        if self._curtoken == b"true":
            token = True
        elif self._curtoken == b"false":
            token = False
        else:
            token = Keyword(self._curtoken)
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
            self._add_token(Keyword(b"<<"))
            self._parse1 = self._parse_main
            i += 1
        else:
            self._parse1 = self._parse_hexstring
        return i

    def _parse_wclose(self, s: bytes, i: int) -> int:
        c = s[i : i + 1]
        if c == b">":
            self._add_token(Keyword(b">>"))
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
        token = HEX_PAIR.sub(
            lambda m: bytes((int(m.group(0), 16),)),
            SPC.sub(b"", self._curtoken),
        )
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

    def start_type(self, pos: int, kind: str) -> None:
        self.context.append((pos, self.curtype, self.curstack))
        (self.curtype, self.curstack) = (kind, [])

    def end_type(self, kind: str) -> tuple[int, list[Any]]:
        if self.curtype != kind:
            raise PSTypeError(f"Type mismatch: {self.curtype!r} != {kind!r}")
        objs = [obj for (_, obj) in self.curstack]
        (pos, self.curtype, self.curstack) = self.context.pop()
        return (pos, objs)

    def nextobject(self) -> Entry:  # noqa: C901
        while not self.results:
            (pos, token) = self.nexttoken()
            if not isinstance(token, Keyword):
                self.push((pos, token))
                continue
            name = token.name
            if name == b"[":
                self.start_type(pos, "a")
            elif name in (b"]", b"}"):
                kind = "a" if name == b"]" else "p"
                with contextlib.suppress(PSTypeError):
                    self.push(self.end_type(kind))
            elif name == b"<<":
                self.start_type(pos, "d")
            elif name == b">>":
                try:
                    (pos, objs) = self.end_type("d")
                except PSTypeError:
                    pass
                else:
                    if len(objs) % 2 != 0:
                        raise PSSyntaxError(f"Invalid dictionary construct: {objs!r}")
                    pairs = zip(objs[::2], objs[1::2], strict=True)
                    self.push((pos, {literal_name(k): v for k, v in pairs if v is not None}))
            elif name == b"{":
                self.start_type(pos, "p")
            else:
                self.do_keyword(pos, name, token)
        return self.results.pop(0)

    # PDFParser

    def do_keyword(self, pos: int, name: bytes, token: Keyword) -> None:
        if name in (b"xref", b"startxref"):
            self.results.extend(self.pop(1))
        elif name == b"endobj":
            self.results.extend(self.pop(4))
        elif name == b"null":
            self.push((pos, None))
        elif name == b"R":
            if len(self.curstack) >= 2:
                (_, object_id), _ = self.pop(2)
                with contextlib.suppress(TypeError, ValueError):
                    self.push((pos, Reference(int(object_id))))
        elif name == b"stream":
            self.read_stream(pos)
        else:
            self.push((pos, token))

    def read_stream(self, pos: int) -> None:
        popped = self.pop(1)
        if len(popped) != 1:
            raise PSSyntaxError(f"Invalid stream dictionary: {popped}")
        dic = dict_value(popped[0][1])
        objlen = int_value(dic["Length"]) if "Length" in dic else 0
        self.seek(pos)
        try:
            (_, line) = self.nextline()
        except PSEOF:
            return
        pos += len(line)
        self.fppos = pos
        data = self.read(objlen)
        self.seek(pos + objlen)
        while True:
            try:
                (_, line) = self.nextline()
            except PSEOF:
                break
            if b"endstream" in line:
                objlen += line.index(b"endstream")
                break
            objlen += len(line)
        self.seek(pos + objlen)
        self.push((pos, Stream(dic, data)))


def stream_data(stream: Stream) -> bytes:
    """pdfminer's PDFStream.get_data(), to the point of knowing whether it raises."""
    filters = resolve1(stream.get_any(("F", "Filter"), []))
    params = resolve1(stream.get_any(("DP", "DecodeParms", "FDecodeParms"), {}))
    data = stream.rawdata
    if not filters:
        return data
    if not isinstance(filters, list):
        filters = [filters]
    if not isinstance(params, list):
        params = [params] * len(filters)
    for f, param in zip(
        [resolve1(f) for f in filters], [resolve1(p) for p in params], strict=False
    ):
        if is_literal(f, FLATE):
            data = inflate(data)
        elif is_literal(f, UNFOLLOWED_FILTERS):
            raise Unfollowed(f"filter {f!r}")
        elif not is_literal(f, PASSED_FILTERS):
            raise ValueError(f"Unsupported filter: {f!r}")
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
    colors = int_value(param.get("Colors", 1))
    columns = int_value(param.get("Columns", 1))
    bitspercomponent = int_value(param.get("BitsPerComponent", 8))
    if pred == 2:
        return apply_tiff_predictor(colors, columns, bitspercomponent, data)
    if pred >= 10:
        return apply_png_predictor(pred, colors, columns, bitspercomponent, data)
    raise ValueError(f"Unsupported predictor: {pred!r}")


def load_xref_stream(parser: Parser) -> dict[str, Any]:
    """PDFXRefStream.load: the stream's dictionary is the trailer."""
    parser.nexttoken()
    parser.nexttoken()
    parser.nexttoken()
    (_, stream) = parser.nextobject()
    if not isinstance(stream, Stream) or not is_literal(stream.get("Type"), frozenset({"XRef"})):
        raise ValueError("Invalid PDF stream spec.")
    size = stream["Size"]
    if len(stream.get("Index", (0, size))) % 2 != 0:
        raise ValueError("Invalid index number")
    fields = stream["W"]
    (fl1, fl2, fl3) = fields
    if fl1 is None or fl2 is None or fl3 is None:
        raise AssertionError(fields)
    stream_data(stream)
    _ = fl1 + fl2 + fl3  # pdfminer adds the widths, which can raise
    return stream.attrs


def load_xref_table(parser: Parser) -> dict[str, Any]:
    """PDFXRef.load: subsections of entry lines up to the trailer line."""
    while True:
        (pos, line) = parser.nextline()
        line = line.strip()
        if not line:
            continue
        if line.startswith(b"trailer"):
            parser.seek(pos)
            break
        fields = line.split(b" ")
        if len(fields) != 2:
            raise ValueError(f"Trailer not found: line={line!r}")
        (start, nobjs) = map(int, fields)
        skip_entry_lines(parser, len(range(start, start + nobjs)))
    return load_trailer(parser)


def skip_entry_lines(parser: Parser, count: int) -> None:
    """Read count entry lines; pdfminer raises unless each splits into three fields."""
    data = parser.data
    position = parser.bufpos + parser.charpos
    lines: Iterator[re.Match[bytes]] = LINE.finditer(data, position)
    for _ in range(count):
        match = next(lines, None)
        if match is None or (match.end() == len(data) and data[match.end() - 1 :] == b"\r"):
            raise PSEOF("Unexpected EOF - file corrupted?")
        if len(match.group().strip().split(b" ")) != 3:
            raise ValueError(f"Invalid XRef format: line={match.group()!r}")
        position = match.end()
    parser.forward(position)


def load_trailer(parser: Parser) -> dict[str, Any]:
    try:
        (_, keyword) = parser.nexttoken()
        if not (isinstance(keyword, Keyword) and keyword.name == b"trailer"):
            raise AssertionError(str(keyword))
        (_, dic) = parser.nextobject()
    except PSEOF:
        popped = parser.pop(1)
        if not popped:
            raise
        (_, dic) = popped[0]
    return dict_value(dic)


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
                raise ValueError(f"Invalid xref position, no digit: {previous!r}")
            offset = int(previous)
            if offset >= 2**31:
                raise ValueError(f"Invalid xref position, too large: {offset!r}")
            return offset
        if line:
            previous = line
        end = start
        if start == carriage_return:
            carriage_return = data.rfind(b"\r", 0, start)
        else:
            line_feed = data.rfind(b"\n", 0, start)
    raise ValueError("Unexpected EOF")


def read_section(parser: Parser, start: int) -> dict[str, Any]:
    """The trailer of the xref section pdfminer reads at start: a stream after a number."""
    parser.seek(start)
    (pos, token) = parser.nexttoken()
    if isinstance(token, int):
        parser.seek(pos)
        return load_xref_stream(parser)
    if isinstance(token, Keyword) and token.name == b"xref":
        parser.nextline()
    return load_xref_table(parser)


def pdfminer_xref_chain_loops(data: PdfByteBuffer) -> bool:
    """Whether pdfminer.six's PDFDocument recurses without end reading this file's xrefs."""
    try:
        start = find_xref(data)
    except Exception:  # noqa: BLE001  # pdfminer falls back to scanning the file
        return False
    parser = Parser(data)
    path: set[int] = set()
    done: set[int] = set()

    def loops_from(start: int) -> bool:
        """PDFDocument.read_xref_from: the section, then /XRefStm, then /Prev."""
        if start in path:
            return True
        if start in done:
            # Read before without raising or looping; reading it again does the same.
            return False
        path.add(start)
        trailer = read_section(parser, start)
        if "XRefStm" in trailer and loops_from(int_value(trailer["XRefStm"])):
            return True
        if "Prev" in trailer and loops_from(int_value(trailer["Prev"])):
            return True
        path.discard(start)
        done.add(start)
        return False

    try:
        return loops_from(start)
    except Exception:  # noqa: BLE001  # pdfminer raises, or falls back, before any loop
        return False


__all__ = ("MESSAGE", "pdfminer_xref_chain_loops")

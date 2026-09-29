# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
from cython.cimports.core_pdf_cythonized._cmap import (
    ARRAY,
    BFCHAR,
    BFRANGE,
    CODESPACE,
    DELIMITER,
    HEX,
    LITERAL,
    MAX_DEPTH,
    MAX_RANGE_SPAN,
    NO_BLOCK,
    PROCEDURE,
    WORD,
    PyUnicode_DecodeLatin1,
    PyUnicode_DecodeUTF16,
    PyUnicode_FromOrdinal,
)
from cython.cimports.cpython.bytes import PyBytes_AS_STRING, PyBytes_FromStringAndSize
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc, PyMem_Realloc
from cython.cimports.libc.string import memchr, memcmp

WHITESPACE = cython.declare(cython.uchar[256])
SEPARATOR = cython.declare(cython.uchar[256])
HEX_VALUE = cython.declare(cython.uchar[256])
_index = cython.declare(cython.int)
for _index in range(256):
    WHITESPACE[_index] = 0
    SEPARATOR[_index] = 0
    HEX_VALUE[_index] = 255
for _index in b"\x00\t\n\x0c\r ":
    WHITESPACE[_index] = 1
    SEPARATOR[_index] = 1
for _index in b"()<>[]/%":
    SEPARATOR[_index] = 1
for _index in range(10):
    HEX_VALUE[48 + _index] = _index
for _index in range(6):
    HEX_VALUE[65 + _index] = 10 + _index
    HEX_VALUE[97 + _index] = 10 + _index


Tokens = cython.struct(
    start=cython.p_Py_ssize_t,
    end=cython.p_Py_ssize_t,
    kind=cython.p_uchar,
    count=cython.Py_ssize_t,
    capacity=cython.Py_ssize_t,
)


@cython.cfunc
@cython.exceptval(check=False)
def push(
    tokens: cython.pointer[Tokens],
    start: cython.Py_ssize_t,
    end: cython.Py_ssize_t,
    kind: cython.uchar,
) -> cython.bint:
    capacity: cython.Py_ssize_t
    grown: cython.p_void
    if tokens.count == tokens.capacity:
        capacity = tokens.capacity * 2 if tokens.capacity else 256
        grown = PyMem_Realloc(tokens.start, capacity * cython.sizeof(cython.Py_ssize_t))
        if grown == cython.NULL:
            return False
        tokens.start = cython.cast(cython.p_Py_ssize_t, grown)
        grown = PyMem_Realloc(tokens.end, capacity * cython.sizeof(cython.Py_ssize_t))
        if grown == cython.NULL:
            return False
        tokens.end = cython.cast(cython.p_Py_ssize_t, grown)
        grown = PyMem_Realloc(tokens.kind, capacity)
        if grown == cython.NULL:
            return False
        tokens.kind = cython.cast(cython.p_uchar, grown)
        tokens.capacity = capacity
    tokens.start[tokens.count] = start
    tokens.end[tokens.count] = end
    tokens.kind[tokens.count] = kind
    tokens.count += 1
    return True


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def literal_end(
    data: cython.p_const_uchar,
    n: cython.Py_ssize_t,
    pos: cython.Py_ssize_t,
    terminated: cython.pointer[cython.bint],
) -> cython.Py_ssize_t:
    end: cython.Py_ssize_t = pos + 1
    depth: cython.int = 1
    current: cython.uchar
    while end < n and depth:
        current = data[end]
        if current == 92:
            end += 1
            if end < n:
                if data[end] == 13 and end + 1 < n and data[end + 1] == 10:
                    end += 2
                else:
                    end += 1
            continue
        if current == 40:
            depth += 1
        elif current == 41:
            depth -= 1
        end += 1
    terminated[0] = depth == 0
    return end if end < n else n


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def find_byte(
    data: cython.p_const_uchar,
    n: cython.Py_ssize_t,
    pos: cython.Py_ssize_t,
    byte: cython.uchar,
) -> cython.Py_ssize_t:
    found: cython.pointer[cython.const[cython.void]]
    if pos >= n:
        return -1
    found = memchr(data + pos, byte, n - pos)
    if found == cython.NULL:
        return -1
    return cython.cast(cython.p_const_uchar, found) - data


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def composite_end(
    data: cython.p_const_uchar,
    n: cython.Py_ssize_t,
    pos: cython.Py_ssize_t,
    depth: cython.int,
    status: cython.p_int,
) -> cython.Py_ssize_t:
    closing: cython.uchar = 93 if data[pos] == 91 else 125
    end: cython.Py_ssize_t = pos + 1
    close: cython.Py_ssize_t
    current: cython.uchar
    ignored = cython.declare(cython.bint)
    if depth > MAX_DEPTH:
        status[0] = -1
        return n
    while end < n:
        current = data[end]
        if current == 37:
            while end < n and data[end] != 10 and data[end] != 13:
                end += 1
        elif current == 40:
            end = literal_end(data, n, end, cython.address(ignored))
        elif current == 60:
            if end + 1 < n and data[end + 1] == 60:
                end += 2
                continue
            close = find_byte(data, n, end + 1, 62)
            if close < 0:
                status[0] = 0
                return n
            end = close + 1
        elif current == 91 or current == 123:
            end = composite_end(data, n, end, depth + 1, status)
            if status[0] != 1:
                return n
        elif current == closing:
            status[0] = 1
            return end + 1
        else:
            end += 1
    status[0] = 0
    return n


@cython.cfunc
@cython.exceptval(check=False)
def tokenize(
    data: cython.p_const_uchar,
    n: cython.Py_ssize_t,
    tokens: cython.pointer[Tokens],
) -> cython.bint:
    pos: cython.Py_ssize_t = 0
    end: cython.Py_ssize_t
    byte: cython.uchar
    terminated = cython.declare(cython.bint)
    status: cython.int
    while pos < n:
        byte = data[pos]
        if WHITESPACE[byte]:
            pos += 1
            continue
        if byte == 37:
            while pos < n and data[pos] != 10 and data[pos] != 13:
                pos += 1
            continue
        if byte == 40:
            end = literal_end(data, n, pos, cython.address(terminated))
            if not terminated or not push(tokens, pos, end, LITERAL):
                return False
        elif byte == 60:
            if pos + 1 < n and data[pos + 1] == 60:
                end = pos + 2
                if not push(tokens, pos, end, DELIMITER):
                    return False
            else:
                end = find_byte(data, n, pos + 1, 62)
                if end < 0:
                    return False
                end += 1
                if not push(tokens, pos, end, HEX):
                    return False
        elif byte == 62:
            end = pos + (2 if pos + 1 < n and data[pos + 1] == 62 else 1)
            if not push(tokens, pos, end, DELIMITER):
                return False
        elif byte == 91 or byte == 123:
            status = 0
            end = composite_end(data, n, pos, 0, cython.address(status))
            if status != 1 or not push(tokens, pos, end, ARRAY if byte == 91 else PROCEDURE):
                return False
        elif byte == 93 or byte == 41 or byte == 125:
            end = pos + 1
            if not push(tokens, pos, end, DELIMITER):
                return False
        else:
            end = pos + 1
            while end < n and not SEPARATOR[data[end]]:
                end += 1
            if not push(tokens, pos, end, WORD):
                return False
        pos = end
    return True


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def is_word(
    data: cython.p_const_uchar,
    tokens: cython.pointer[Tokens],
    index: cython.Py_ssize_t,
    keyword: cython.p_const_char,
    size: cython.Py_ssize_t,
) -> cython.bint:
    return (
        tokens.kind[index] == WORD
        and tokens.end[index] - tokens.start[index] == size
        and memcmp(data + tokens.start[index], keyword, size) == 0
    )


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def clean_hex_length(
    data: cython.p_const_uchar,
    start: cython.Py_ssize_t,
    end: cython.Py_ssize_t,
) -> cython.Py_ssize_t:
    # The digits between < and >, when every one is a hex digit and they pair up;
    # -1 otherwise, which declines the whole CMap to the Python parser.
    index: cython.Py_ssize_t
    length: cython.Py_ssize_t = end - start - 2
    if length & 1:
        return -1
    for index in range(start + 1, end - 1):
        if HEX_VALUE[data[index]] == 255:
            return -1
    return length // 2


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def unhex(
    data: cython.p_const_uchar,
    start: cython.Py_ssize_t,
    size: cython.Py_ssize_t,
    out: cython.p_uchar,
) -> cython.void:
    index: cython.Py_ssize_t
    for index in range(size):
        out[index] = (HEX_VALUE[data[start + 1 + 2 * index]] << 4) | HEX_VALUE[
            data[start + 2 + 2 * index]
        ]


@cython.cfunc
def utf16be_text(buffer: cython.p_uchar, size: cython.Py_ssize_t) -> object:
    padded: cython.p_uchar
    index: cython.Py_ssize_t
    if size == 0:
        return ""
    if size >= 2 and buffer[0] == 0xFE and buffer[1] == 0xFF:
        buffer += 2
        size -= 2
    if size == 1:
        return PyUnicode_FromOrdinal(buffer[0])
    if size & 1:
        padded = cython.cast(cython.p_uchar, PyMem_Malloc(size + 1))
        if padded == cython.NULL:
            raise MemoryError()
        try:
            padded[0] = 0
            for index in range(size):
                padded[index + 1] = buffer[index]
            return utf16be_decode(padded, size + 1)
        finally:
            PyMem_Free(padded)
    return utf16be_decode(buffer, size)


@cython.cfunc
def utf16be_decode(buffer: cython.p_uchar, size: cython.Py_ssize_t) -> object:
    byteorder: cython.int = 1
    try:
        return PyUnicode_DecodeUTF16(
            cython.cast(cython.p_const_char, buffer),
            size,
            "strict",
            cython.address(byteorder),
        )
    except UnicodeDecodeError:
        byteorder = 1
        return PyUnicode_DecodeUTF16(
            cython.cast(cython.p_const_char, buffer),
            size,
            "replace",
            cython.address(byteorder),
        )


@cython.cfunc
def hex_text(
    data: cython.p_const_uchar,
    start: cython.Py_ssize_t,
    size: cython.Py_ssize_t,
    scratch: cython.p_uchar,
) -> object:
    unhex(data, start, size, scratch)
    return utf16be_text(scratch, size)


@cython.cfunc
@cython.inline
def scalar_or_replacement(codepoint: cython.longlong) -> object:
    if 0 <= codepoint < 0x110000 and not 0xD800 <= codepoint <= 0xDFFF:
        return PyUnicode_FromOrdinal(cython.cast(cython.int, codepoint))
    return "�"


@cython.cfunc
def range_key(value: cython.ulonglong, width: cython.Py_ssize_t) -> object:
    out = cython.declare(cython.uchar[4])
    index: cython.Py_ssize_t
    for index in range(width):
        out[width - 1 - index] = (value >> (8 * index)) & 0xFF
    return PyBytes_FromStringAndSize(cython.cast(cython.p_const_char, out), width)


@cython.cclass
class Scan:
    data: cython.p_const_uchar
    tokens: cython.pointer[Tokens]
    scratch: cython.p_uchar
    mappings: dict
    invalid: cython.Py_ssize_t
    valid: cython.Py_ssize_t

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def bfchar(self, first: cython.Py_ssize_t, last: cython.Py_ssize_t) -> cython.bint:
        tokens: cython.pointer[Tokens] = self.tokens
        data: cython.p_const_uchar = self.data
        index: cython.Py_ssize_t
        pending: cython.Py_ssize_t = -1
        source_size: cython.Py_ssize_t
        target_size: cython.Py_ssize_t
        source: object
        for index in range(first, last):
            if tokens.kind[index] == LITERAL:
                return False
            if tokens.kind[index] != HEX:
                continue
            if pending < 0:
                pending = index
                continue
            source_size = clean_hex_length(data, tokens.start[pending], tokens.end[pending])
            target_size = clean_hex_length(data, tokens.start[index], tokens.end[index])
            if source_size < 0 or target_size < 0:
                return False
            if source_size:
                unhex(data, tokens.start[pending], source_size, self.scratch)
                source = PyBytes_FromStringAndSize(
                    cython.cast(cython.p_const_char, self.scratch),
                    source_size,
                )
                self.mappings[source] = hex_text(
                    data, tokens.start[index], target_size, self.scratch
                )
            pending = -1
        return True

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def bfrange(self, first: cython.Py_ssize_t, last: cython.Py_ssize_t) -> cython.bint:
        tokens: cython.pointer[Tokens] = self.tokens
        data: cython.p_const_uchar = self.data
        operands = cython.declare(cython.Py_ssize_t[3])
        count: cython.Py_ssize_t = 0
        total: cython.Py_ssize_t = 0
        index: cython.Py_ssize_t
        start_size: cython.Py_ssize_t
        end_size: cython.Py_ssize_t
        target_size: cython.Py_ssize_t
        low: cython.ulonglong
        high: cython.ulonglong
        value: cython.ulonglong
        byte: cython.Py_ssize_t
        base: object
        prefix: object
        final: cython.longlong
        mappings: dict = self.mappings
        for index in range(first, last):
            if tokens.kind[index] == LITERAL:
                return False
            if tokens.kind[index] == HEX or tokens.kind[index] == ARRAY:
                total += 1
        if total % 3:
            self.invalid += 1
        for index in range(first, last):
            if tokens.kind[index] != HEX and tokens.kind[index] != ARRAY:
                continue
            operands[count] = index
            count += 1
            if count < 3:
                continue
            count = 0
            total -= 3
            if tokens.kind[operands[0]] != HEX or tokens.kind[operands[1]] != HEX:
                self.invalid += 1
                continue
            start_size = clean_hex_length(data, tokens.start[operands[0]], tokens.end[operands[0]])
            end_size = clean_hex_length(data, tokens.start[operands[1]], tokens.end[operands[1]])
            if start_size < 0 or end_size < 0:
                return False
            if not 1 <= start_size <= 4 or start_size != end_size:
                self.invalid += 1
                continue
            unhex(data, tokens.start[operands[0]], start_size, self.scratch)
            low = 0
            for byte in range(start_size):
                low = (low << 8) | self.scratch[byte]
            unhex(data, tokens.start[operands[1]], end_size, self.scratch)
            high = 0
            for byte in range(end_size):
                high = (high << 8) | self.scratch[byte]
            if low > high:
                self.invalid += 1
                continue
            if tokens.kind[operands[2]] == ARRAY:
                if not self.array_targets(operands[2], low, high, start_size):
                    return False
                continue
            target_size = clean_hex_length(data, tokens.start[operands[2]], tokens.end[operands[2]])
            if target_size < 0:
                return False
            base = hex_text(data, tokens.start[operands[2]], target_size, self.scratch)
            if high - low + 1 > MAX_RANGE_SPAN:
                self.invalid += 1
                continue
            if not base:
                for value in range(low, high + 1):
                    mappings[range_key(value, start_size)] = ""
            else:
                prefix = base[: len(base) - 1]
                final = ord(base[len(base) - 1])
                if prefix:
                    for value in range(low, high + 1):
                        mappings[range_key(value, start_size)] = prefix + scalar_or_replacement(
                            final + cython.cast(cython.longlong, value - low)
                        )
                else:
                    for value in range(low, high + 1):
                        mappings[range_key(value, start_size)] = scalar_or_replacement(
                            final + cython.cast(cython.longlong, value - low)
                        )
            self.valid += 1
        return True

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def array_targets(
        self,
        token: cython.Py_ssize_t,
        low: cython.ulonglong,
        high: cython.ulonglong,
        width: cython.Py_ssize_t,
    ) -> cython.bint:
        # The bracketed destinations of one bfrange, re-read the way cmap_tokens
        # reads them: only hex strings, whitespace and comments are accepted.
        data: cython.p_const_uchar = self.data
        pos: cython.Py_ssize_t = self.tokens.start[token] + 1
        stop: cython.Py_ssize_t = self.tokens.end[token] - 1
        close: cython.Py_ssize_t
        size: cython.Py_ssize_t
        offset: cython.ulonglong = 0
        added: cython.bint = False
        pending: list = []
        while pos < stop:
            if WHITESPACE[data[pos]]:
                pos += 1
                continue
            if data[pos] == 37:
                while pos < stop and data[pos] != 10 and data[pos] != 13:
                    pos += 1
                continue
            if data[pos] != 60 or (pos + 1 < stop and data[pos + 1] == 60):
                return False
            close = find_byte(data, stop, pos + 1, 62)
            if close < 0:
                return False
            size = clean_hex_length(data, pos, close + 1)
            if size < 0:
                return False
            pending.append((pos, size))
            pos = close + 1
        if not pending:
            self.invalid += 1
            return True
        for pos, size in pending:
            if offset > high - low:
                break
            self.mappings[range_key(low + offset, width)] = hex_text(data, pos, size, self.scratch)
            offset += 1
            added = True
        if added:
            self.valid += 1
        else:
            self.invalid += 1
        return True


def scan_to_unicode_cmap(data: typing.Optional[bytes]):
    """Parse a ToUnicode CMap the way core's Python parser does, or return None.

    Returns (mappings, codespace_blocks, usecmap_name): the mappings dict in the
    Python parser's insertion order, each codespacerange block's hex and literal
    operands for the Python codespace validation, and the first usecmap name.
    None declines: the caller parses the same bytes in Python.
    """
    raw: cython.p_const_uchar = cython.cast(cython.p_const_uchar, PyBytes_AS_STRING(data))
    n: cython.Py_ssize_t = len(data)
    tokens = cython.declare(Tokens)
    scan: Scan
    scope_start: cython.Py_ssize_t = 0
    scope_end: cython.Py_ssize_t
    index: cython.Py_ssize_t
    block_first: cython.Py_ssize_t = 0
    previous_word: cython.Py_ssize_t = -1
    longest: cython.Py_ssize_t = 0
    block: cython.int = NO_BLOCK
    usecmap_name: object = None
    codespace_blocks: list = []
    operands: list
    tokens.start = cython.NULL
    tokens.end = cython.NULL
    tokens.kind = cython.NULL
    tokens.count = 0
    tokens.capacity = 0
    scan = Scan.__new__(Scan)
    scan.scratch = cython.NULL
    try:
        if not tokenize(raw, n, cython.address(tokens)):
            return None
        for index in range(tokens.count):
            if tokens.end[index] - tokens.start[index] > longest:
                longest = tokens.end[index] - tokens.start[index]
        scan.scratch = cython.cast(cython.p_uchar, PyMem_Malloc(longest // 2 + 4))
        if scan.scratch == cython.NULL:
            raise MemoryError()

        for index in range(tokens.count):
            if is_word(raw, cython.address(tokens), index, b"begincmap", 9):
                scope_start = index + 1
                break
        scope_end = tokens.count
        for index in range(scope_start, tokens.count):
            if is_word(raw, cython.address(tokens), index, b"endcmap", 7):
                scope_end = index
                break

        for index in range(scope_start, scope_end):
            if tokens.kind[index] != WORD:
                continue
            if previous_word >= 0 and is_word(raw, cython.address(tokens), index, b"usecmap", 7):
                if raw[tokens.start[previous_word]] == 47:
                    usecmap_name = PyUnicode_DecodeLatin1(
                        cython.cast(cython.p_const_char, raw) + tokens.start[previous_word] + 1,
                        tokens.end[previous_word] - tokens.start[previous_word] - 1,
                        cython.NULL,
                    )
                break
            previous_word = index

        scan.data = raw
        scan.tokens = cython.address(tokens)
        scan.mappings = {}
        scan.invalid = 0
        scan.valid = 0
        for index in range(scope_start, scope_end):
            if block == NO_BLOCK:
                if tokens.kind[index] != WORD:
                    continue
                if is_word(raw, cython.address(tokens), index, b"beginbfchar", 11):
                    block = BFCHAR
                elif is_word(raw, cython.address(tokens), index, b"beginbfrange", 12):
                    block = BFRANGE
                elif is_word(raw, cython.address(tokens), index, b"begincidrange", 13):
                    return None
                else:
                    continue
                block_first = index + 1
                continue
            if block == BFCHAR and is_word(raw, cython.address(tokens), index, b"endbfchar", 9):
                if not scan.bfchar(block_first, index):
                    return None
                block = NO_BLOCK
            elif block == BFRANGE and is_word(
                raw, cython.address(tokens), index, b"endbfrange", 10
            ):
                if not scan.bfrange(block_first, index):
                    return None
                block = NO_BLOCK
        if scan.invalid and not scan.valid:
            return None

        block = NO_BLOCK
        for index in range(scope_start, scope_end):
            if block == NO_BLOCK:
                if is_word(raw, cython.address(tokens), index, b"begincodespacerange", 19):
                    block = CODESPACE
                    operands = []
                continue
            if is_word(raw, cython.address(tokens), index, b"endcodespacerange", 17):
                codespace_blocks.append(operands)
                block = NO_BLOCK
            elif tokens.kind[index] == HEX or tokens.kind[index] == LITERAL:
                operands.append(
                    PyBytes_FromStringAndSize(
                        cython.cast(cython.p_const_char, raw) + tokens.start[index],
                        tokens.end[index] - tokens.start[index],
                    )
                )
        return scan.mappings, codespace_blocks, usecmap_name
    finally:
        PyMem_Free(tokens.start)
        PyMem_Free(tokens.end)
        PyMem_Free(tokens.kind)
        PyMem_Free(scan.scratch)

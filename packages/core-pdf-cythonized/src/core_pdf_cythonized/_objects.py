# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
from cython.cimports.core_pdf_cythonized._objects import (
    FAST_INT_DIGITS,
    MAX_DEPTH,
    PyLong_FromString,
    PyOS_string_to_double,
)
from cython.cimports.cpython.bytes import PyBytes_AS_STRING, PyBytes_FromStringAndSize
from cython.cimports.libc.stdint import int64_t
from cython.cimports.libc.string import memchr

BAIL = cython.declare(object, object())

HEX_VALUE = cython.declare(cython.uchar[256])
_index = cython.declare(cython.int)
for _index in range(256):
    HEX_VALUE[_index] = 255
for _index in range(10):
    HEX_VALUE[48 + _index] = _index
for _index in range(6):
    HEX_VALUE[65 + _index] = 10 + _index
    HEX_VALUE[97 + _index] = 10 + _index


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def is_digit(byte: cython.uchar) -> cython.bint:
    return 48 <= byte <= 57


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def is_python_space(byte: cython.uchar) -> cython.bint:
    return byte == 32 or 9 <= byte <= 13


@cython.cclass
class ObjectScanner:
    view: cython.const[cython.uchar][::1]
    data: cython.p_const_uchar
    length: cython.Py_ssize_t
    whitespace: cython.uchar[256]
    separator: cython.uchar[256]
    name_escapes: cython.bint
    split_whitespace_compatible: cython.bint
    names: dict
    name_of: object
    string_type: object
    reference_type: object
    decipher: object
    object_number: object
    generation: object
    name_is_contents: cython.bint

    def __init__(
        self,
        data,
        whitespace_table: bytes,
        separator_table: bytes,
        name_escapes: cython.bint,
        split_whitespace_compatible: cython.bint,
        names: dict,
        name_of,
        string_type,
        reference_type,
    ):
        if len(whitespace_table) != 256 or len(separator_table) != 256:
            raise ValueError("lexical tables must have 256 entries")
        self.view = data
        self.length = self.view.shape[0]
        self.data = cython.address(self.view[0]) if self.length else cython.NULL
        index: cython.int
        for index in range(256):
            self.whitespace[index] = whitespace_table[index]
            self.separator[index] = separator_table[index]
        self.name_escapes = name_escapes
        self.split_whitespace_compatible = split_whitespace_compatible
        self.names = names
        self.name_of = name_of
        self.string_type = string_type
        self.reference_type = reference_type

    def release(self):
        self.view = None
        self.data = cython.NULL
        self.length = 0

    def parse_dictionary(
        self, pos: cython.Py_ssize_t, decipher=None, object_number=0, generation=0
    ):
        return self.parse(pos, True, decipher, object_number, generation)

    def parse_array(self, pos: cython.Py_ssize_t, decipher=None, object_number=0, generation=0):
        return self.parse(pos, False, decipher, object_number, generation)

    @cython.cfunc
    def parse(
        self,
        pos: cython.Py_ssize_t,
        dictionary: cython.bint,
        decipher,
        object_number,
        generation,
    ) -> object:
        if pos < 0 or self.data == cython.NULL:
            return None
        self.decipher = decipher
        self.object_number = object_number
        self.generation = generation
        cursor: cython.Py_ssize_t = pos
        try:
            value = (
                self.dictionary(cython.address(cursor), 0)
                if dictionary
                else self.array(cython.address(cursor), 0)
            )
        finally:
            self.decipher = None
        if value is BAIL:
            return None
        return value, cursor

    @cython.cfunc
    @cython.exceptval(check=False)
    def skip_ignored(self, pos: cython.Py_ssize_t) -> cython.Py_ssize_t:
        data: cython.p_const_uchar = self.data
        length: cython.Py_ssize_t = self.length
        while pos < length:
            if self.whitespace[data[pos]]:
                pos += 1
            elif data[pos] == 37:
                pos += 1
                while pos < length and data[pos] != 13 and data[pos] != 10:
                    pos += 1
                if pos < length:
                    if pos + 1 < length and (
                        (data[pos] == 13 and data[pos + 1] == 10)
                        or (data[pos] == 10 and data[pos + 1] == 13)
                    ):
                        pos += 2
                    else:
                        pos += 1
            else:
                break
        return pos

    @cython.cfunc
    @cython.exceptval(check=False)
    def word_end(self, pos: cython.Py_ssize_t) -> cython.Py_ssize_t:
        if self.separator[self.data[pos]]:
            return pos + 1
        pos += 1
        while pos < self.length and not self.separator[self.data[pos]]:
            pos += 1
        return pos

    @cython.cfunc
    @cython.exceptval(check=False)
    def number_kind(self, start: cython.Py_ssize_t, end: cython.Py_ssize_t) -> cython.int:
        data: cython.p_const_uchar = self.data
        pos: cython.Py_ssize_t = start
        digits: cython.Py_ssize_t = 0
        dots: cython.Py_ssize_t = 0
        if pos < end and (data[pos] == 43 or data[pos] == 45):
            pos += 1
        if pos >= end:
            return 0
        while pos < end:
            if is_digit(data[pos]):
                digits += 1
            elif data[pos] == 46:
                dots += 1
            else:
                return 0
            pos += 1
        if digits == 0 or dots > 1:
            return 0
        return 2 if dots else 1

    @cython.cfunc
    @cython.exceptval(check=False)
    def is_integer_token(self, start: cython.Py_ssize_t, end: cython.Py_ssize_t) -> cython.bint:
        pos: cython.Py_ssize_t = start
        if pos < end and (self.data[pos] == 43 or self.data[pos] == 45):
            pos += 1
        if pos >= end:
            return False
        while pos < end:
            if not is_digit(self.data[pos]):
                return False
            pos += 1
        return True

    @cython.cfunc
    def integer(self, start: cython.Py_ssize_t, end: cython.Py_ssize_t) -> object:
        data: cython.p_const_uchar = self.data
        pos: cython.Py_ssize_t = start
        negative: cython.bint = False
        value: int64_t = 0
        if data[pos] == 43 or data[pos] == 45:
            negative = data[pos] == 45
            pos += 1
        if end - pos <= FAST_INT_DIGITS:
            while pos < end:
                value = value * 10 + (data[pos] - 48)
                pos += 1
            return -value if negative else value
        token = PyBytes_FromStringAndSize(
            cython.cast(cython.p_const_char, data) + start, end - start
        )
        try:
            return PyLong_FromString(PyBytes_AS_STRING(token), cython.NULL, 10)
        except ValueError:
            return BAIL

    @cython.cfunc
    def real(self, start: cython.Py_ssize_t, end: cython.Py_ssize_t) -> object:
        token = PyBytes_FromStringAndSize(
            cython.cast(cython.p_const_char, self.data) + start, end - start
        )
        return PyOS_string_to_double(PyBytes_AS_STRING(token), cython.NULL, cython.NULL)

    @cython.cfunc
    def keyword(self, start: cython.Py_ssize_t, end: cython.Py_ssize_t) -> object:
        data: cython.p_const_uchar = self.data + start
        size: cython.Py_ssize_t = end - start
        if size == 4 and data[0] == 116 and data[1] == 114 and data[2] == 117 and data[3] == 101:
            return True
        if (
            size == 5
            and data[0] == 102
            and data[1] == 97
            and data[2] == 108
            and data[3] == 115
            and data[4] == 101
        ):
            return False
        if size == 4 and data[0] == 110 and data[1] == 117 and data[2] == 108 and data[3] == 108:
            return None
        return BAIL

    @cython.cfunc
    def name(self, pos: cython.p_Py_ssize_t) -> object:
        data: cython.p_const_uchar = self.data
        start: cython.Py_ssize_t = pos[0] + 1
        end: cython.Py_ssize_t = start
        escaped: cython.bint = False
        while end < self.length and not self.separator[data[end]]:
            if data[end] == 35:
                escaped = True
            end += 1
        pos[0] = end
        raw: bytes
        decoded: bytearray
        index: cython.Py_ssize_t
        high: cython.uchar
        low: cython.uchar
        if escaped and self.name_escapes:
            decoded = bytearray()
            index = start
            while index < end:
                if data[index] != 35:
                    decoded.append(data[index])
                    index += 1
                    continue
                high = HEX_VALUE[data[index + 1]] if index + 1 < end else 255
                low = HEX_VALUE[data[index + 2]] if index + 2 < end else 255
                if high == 255 or low == 255:
                    decoded.append(35)
                    index += 1
                else:
                    decoded.append((high << 4) | low)
                    index += 3
            raw = bytes(decoded)
        else:
            raw = PyBytes_FromStringAndSize(
                cython.cast(cython.p_const_char, data) + start, end - start
            )
        self.name_is_contents = raw == b"Contents"
        interned = self.names.get(raw)
        if interned is not None:
            return interned
        return self.name_of(raw)

    @cython.cfunc
    def literal_string(self, pos: cython.p_Py_ssize_t) -> object:
        data: cython.p_const_uchar = self.data
        length: cython.Py_ssize_t = self.length
        start: cython.Py_ssize_t = pos[0] + 1
        cursor: cython.Py_ssize_t = start
        while cursor < length:
            if data[cursor] == 41:
                pos[0] = cursor + 1
                return self.string(
                    PyBytes_FromStringAndSize(
                        cython.cast(cython.p_const_char, data) + start,
                        cursor - start,
                    ),
                    True,
                )
            if data[cursor] == 40 or data[cursor] == 92 or data[cursor] == 13 or data[cursor] == 10:
                break
            cursor += 1
        end: cython.Py_ssize_t = self.literal_string_end(start)
        if end < 0:
            return BAIL
        out = PyBytes_FromStringAndSize(cython.NULL, end - start)
        buffer: cython.p_uchar = cython.cast(cython.p_uchar, PyBytes_AS_STRING(out))
        size: cython.Py_ssize_t = 0
        depth: cython.Py_ssize_t = 1
        byte: cython.uchar
        escape: cython.uchar
        octal: cython.int
        count: cython.int
        cursor = start
        while cursor < length:
            byte = data[cursor]
            cursor += 1
            if byte == 40:
                depth += 1
                buffer[size] = byte
                size += 1
            elif byte == 41:
                depth -= 1
                if depth == 0:
                    break
                buffer[size] = byte
                size += 1
            elif byte == 92:
                if cursor >= length:
                    continue
                escape = data[cursor]
                cursor += 1
                if 48 <= escape <= 55:
                    octal = escape - 48
                    count = 1
                    while count < 3 and cursor < length and 48 <= data[cursor] <= 55:
                        octal = (octal << 3) | (data[cursor] - 48)
                        cursor += 1
                        count += 1
                    buffer[size] = octal & 0xFF
                    size += 1
                elif escape == 10 or escape == 13:
                    if cursor < length and self.line_end_pair(escape, data[cursor]):
                        cursor += 1
                else:
                    buffer[size] = self.escaped_byte(escape)
                    size += 1
            elif byte == 13 or byte == 10:
                buffer[size] = 10
                size += 1
                if cursor < length and self.line_end_pair(byte, data[cursor]):
                    cursor += 1
            else:
                buffer[size] = byte
                size += 1
        pos[0] = cursor
        return self.string(out[:size], True)

    @cython.cfunc
    def string(self, value: typing.Optional[bytes], is_literal: cython.bint) -> object:
        if self.decipher is not None:
            try:
                deciphered = self.decipher(self.object_number, self.generation, value, None)
            except Exception:
                return BAIL
            if type(deciphered) is memoryview:
                deciphered = deciphered.tobytes()
            return self.string_type(deciphered, is_literal=is_literal)
        return self.string_type(value, is_literal=is_literal)

    @cython.cfunc
    @cython.exceptval(check=False)
    def literal_string_end(self, start: cython.Py_ssize_t) -> cython.Py_ssize_t:
        data: cython.p_const_uchar = self.data
        length: cython.Py_ssize_t = self.length
        cursor: cython.Py_ssize_t = start
        depth: cython.Py_ssize_t = 1
        while cursor < length:
            if data[cursor] == 92:
                cursor += 2
                continue
            if data[cursor] == 40:
                depth += 1
            elif data[cursor] == 41:
                depth -= 1
                if depth == 0:
                    return cursor + 1
            cursor += 1
        return -1

    @cython.cfunc
    @cython.inline
    @cython.exceptval(check=False)
    def line_end_pair(self, first: cython.uchar, second: cython.uchar) -> cython.bint:
        return (first == 13 and second == 10) or (first == 10 and second == 13)

    @cython.cfunc
    @cython.inline
    @cython.exceptval(check=False)
    def escaped_byte(self, escape: cython.uchar) -> cython.uchar:
        if escape == 110:
            return 10
        if escape == 114:
            return 13
        if escape == 116:
            return 9
        if escape == 98:
            return 8
        if escape == 102:
            return 12
        return escape

    @cython.cfunc
    def hex_string(self, pos: cython.p_Py_ssize_t) -> object:
        data: cython.p_const_uchar = self.data
        start: cython.Py_ssize_t = pos[0] + 1
        if start > self.length:
            return BAIL
        found: cython.p_const_uchar = cython.cast(
            cython.p_const_uchar,
            memchr(data + start, 62, self.length - start),
        )
        if found == cython.NULL:
            return BAIL
        marker: cython.Py_ssize_t = found - data
        out = PyBytes_FromStringAndSize(cython.NULL, (marker - start + 1) // 2)
        buffer: cython.p_uchar = cython.cast(cython.p_uchar, PyBytes_AS_STRING(out))
        size: cython.Py_ssize_t = 0
        cursor: cython.Py_ssize_t
        nibble: cython.uchar
        high: cython.bint = True
        for cursor in range(start, marker):
            nibble = HEX_VALUE[data[cursor]]
            if nibble == 255:
                if self.whitespace[data[cursor]]:
                    continue
                return BAIL
            if high:
                buffer[size] = nibble << 4
            else:
                buffer[size] |= nibble
                size += 1
            high = not high
        if not high:
            size += 1
        pos[0] = marker + 1
        return self.string(out[:size], False)

    @cython.cfunc
    def number_or_reference(
        self, start: cython.Py_ssize_t, end: cython.Py_ssize_t, pos: cython.p_Py_ssize_t
    ) -> object:
        if self.number_kind(start, end) == 2:
            pos[0] = end
            return self.real(start, end)
        next_start: cython.Py_ssize_t = self.skip_ignored(end)
        next_end: cython.Py_ssize_t
        marker: cython.Py_ssize_t
        marker_end: cython.Py_ssize_t
        byte: cython.uchar
        if next_start < self.length:
            byte = self.data[next_start]
            if byte == 43 or byte == 45 or is_digit(byte):
                next_end = self.word_end(next_start)
                if self.is_integer_token(next_start, next_end):
                    marker = self.skip_ignored(next_end)
                    if marker < self.length:
                        marker_end = self.word_end(marker)
                        if marker_end - marker == 1 and self.data[marker] == 82:
                            return self.reference(start, end, next_start, next_end, pos, marker_end)
        pos[0] = end
        return self.integer(start, end)

    @cython.cfunc
    def reference(
        self,
        start: cython.Py_ssize_t,
        end: cython.Py_ssize_t,
        next_start: cython.Py_ssize_t,
        next_end: cython.Py_ssize_t,
        pos: cython.p_Py_ssize_t,
        marker_end: cython.Py_ssize_t,
    ) -> object:
        object_number = self.integer(start, end)
        generation = self.integer(next_start, next_end)
        if object_number is BAIL or generation is BAIL:
            return BAIL
        if object_number < 0 or not 0 <= generation <= 65535:
            return BAIL
        pos[0] = marker_end
        return self.reference_type(object_number, generation)

    @cython.cfunc
    def value(
        self, pos: cython.p_Py_ssize_t, depth: cython.int, strip_endobj: cython.bint
    ) -> object:
        start: cython.Py_ssize_t = self.skip_ignored(pos[0])
        if start >= self.length:
            return BAIL
        byte: cython.uchar = self.data[start]
        pos[0] = start
        if byte == 40:
            return self.literal_string(pos)
        if byte == 60:
            if start + 1 < self.length and self.data[start + 1] == 60:
                return self.nested_dictionary(pos, depth)
            return self.hex_string(pos)
        if byte == 91:
            return self.array(pos, depth)
        if byte == 47:
            return self.name(pos)
        end: cython.Py_ssize_t = self.word_end(start)
        if (
            strip_endobj
            and end - start > 6
            and self.data[end - 6] == 101
            and self.data[end - 5] == 110
            and self.data[end - 4] == 100
            and self.data[end - 3] == 111
            and self.data[end - 2] == 98
            and self.data[end - 1] == 106
        ):
            end -= 6
        if self.number_kind(start, end):
            return self.number_or_reference(start, end, pos)
        pos[0] = end
        return self.keyword(start, end)

    @cython.cfunc
    def dictionary(self, pos: cython.p_Py_ssize_t, depth: cython.int) -> object:
        if depth > MAX_DEPTH or pos[0] + 2 > self.length:
            return BAIL
        values: dict = {}
        cursor: cython.Py_ssize_t = pos[0] + 2
        while True:
            cursor = self.skip_ignored(cursor)
            if cursor >= self.length:
                return BAIL
            if self.data[cursor] == 62:
                if cursor + 1 < self.length and self.data[cursor + 1] == 62:
                    cursor += 2
                    break
                return BAIL
            if self.data[cursor] != 47:
                return BAIL
            key = self.name(cython.address(cursor))
            if self.name_is_contents and self.decipher is not None and self.hex_follows(cursor):
                return BAIL
            value = self.value(cython.address(cursor), depth + 1, True)
            if value is BAIL:
                return BAIL
            values[key] = value
        pos[0] = cursor
        return values

    @cython.cfunc
    @cython.exceptval(check=False)
    def hex_follows(self, pos: cython.Py_ssize_t) -> cython.bint:
        pos = self.skip_ignored(pos)
        return (
            pos < self.length
            and self.data[pos] == 60
            and (pos + 1 >= self.length or self.data[pos + 1] != 60)
        )

    @cython.cfunc
    def nested_dictionary(self, pos: cython.p_Py_ssize_t, depth: cython.int) -> object:
        values = self.dictionary(pos, depth + 1)
        if values is BAIL:
            return BAIL
        cursor: cython.Py_ssize_t = self.skip_ignored(pos[0])
        index: cython.Py_ssize_t
        mismatches: cython.int = 0
        keyword: cython.p_const_char = b"stream"
        if cursor + 6 <= self.length:
            for index in range(6):
                if self.data[cursor + index] != cython.cast(cython.uchar, keyword[index]):
                    mismatches += 1
            if mismatches <= 1:
                return BAIL
        pos[0] = cursor
        return values

    @cython.cfunc
    def array(self, pos: cython.p_Py_ssize_t, depth: cython.int) -> object:
        if depth > MAX_DEPTH:
            return BAIL
        values = self.numeric_array(pos)
        if values is not None:
            return values
        items: list = []
        cursor: cython.Py_ssize_t = pos[0] + 1
        byte: cython.uchar
        while True:
            cursor = self.skip_ignored(cursor)
            if cursor >= self.length:
                return BAIL
            byte = self.data[cursor]
            if byte == 93:
                pos[0] = cursor + 1
                return items
            item = self.value(cython.address(cursor), depth + 1, False)
            if item is BAIL:
                return BAIL
            items.append(item)

    @cython.cfunc
    def numeric_array(self, pos: cython.p_Py_ssize_t) -> object:
        values = self.split_numeric_array(pos)
        if values is not None:
            return values
        return self.scanned_numeric_array(pos)

    @cython.cfunc
    def split_numeric_array(self, pos: cython.p_Py_ssize_t) -> object:
        if not self.split_whitespace_compatible:
            return None
        data: cython.p_const_uchar = self.data
        start: cython.Py_ssize_t = pos[0] + 1
        if start > self.length:
            return None
        found: cython.p_const_uchar = cython.cast(
            cython.p_const_uchar,
            memchr(data + start, 93, self.length - start),
        )
        if found == cython.NULL:
            return None
        close: cython.Py_ssize_t = found - data
        cursor: cython.Py_ssize_t
        for cursor in range(start, close):
            if data[cursor] == 37 or data[cursor] == 91 or data[cursor] == 11:
                return None
        values: list = []
        token_start: cython.Py_ssize_t
        cursor = start
        while True:
            while cursor < close and is_python_space(data[cursor]):
                cursor += 1
            if cursor >= close:
                break
            token_start = cursor
            while cursor < close and not is_python_space(data[cursor]):
                cursor += 1
            value = self.numeric_word(token_start, cursor)
            if value is None or value is BAIL:
                return value
            values.append(value)
        pos[0] = close + 1
        return values

    @cython.cfunc
    def scanned_numeric_array(self, pos: cython.p_Py_ssize_t) -> object:
        values: list = []
        cursor: cython.Py_ssize_t = pos[0] + 1
        end: cython.Py_ssize_t
        while True:
            cursor = self.skip_ignored(cursor)
            if cursor >= self.length:
                return None
            if self.data[cursor] == 93:
                pos[0] = cursor + 1
                return values
            end = self.word_end(cursor)
            value = self.numeric_word(cursor, end)
            if value is None or value is BAIL:
                return value
            values.append(value)
            cursor = end

    @cython.cfunc
    def numeric_word(self, start: cython.Py_ssize_t, end: cython.Py_ssize_t) -> object:
        kind: cython.int = self.number_kind(start, end)
        if kind == 2:
            return self.real(start, end)
        if kind == 1:
            return self.integer(start, end)
        return BAIL if self.python_might_read(start, end) else None

    @cython.cfunc
    @cython.exceptval(check=False)
    def python_might_read(self, start: cython.Py_ssize_t, end: cython.Py_ssize_t) -> cython.bint:
        dotted: cython.bint = memchr(self.data + start, 46, end - start) != cython.NULL
        byte: cython.uchar
        cursor: cython.Py_ssize_t
        for cursor in range(start, end):
            byte = self.data[cursor]
            if is_digit(byte) or byte == 43 or byte == 45 or byte == 95:
                continue
            if dotted and (byte == 46 or (byte | 32) in b"einfaty"):
                continue
            return False
        return True

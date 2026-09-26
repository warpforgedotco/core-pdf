# SPDX-License-Identifier: AGPL-3.0-only

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from cpython.object cimport PyObject
from libc.stdint cimport int64_t
from libc.string cimport memchr


cdef extern from "Python.h":
    object PyLong_FromString(const char *text, char **end, int base)
    double PyOS_string_to_double(
        const char *text, char **end, PyObject *overflow_exception
    ) except? -1.0


cdef enum:
    MAX_DEPTH = 64
    FAST_INT_DIGITS = 18

cdef object BAIL = object()

cdef unsigned char HEX_VALUE[256]
cdef int _index
for _index in range(256):
    HEX_VALUE[_index] = 255
for _index in range(10):
    HEX_VALUE[48 + _index] = _index
for _index in range(6):
    HEX_VALUE[65 + _index] = 10 + _index
    HEX_VALUE[97 + _index] = 10 + _index


cdef inline bint is_digit(unsigned char byte) noexcept nogil:
    return 48 <= byte <= 57


cdef inline bint is_python_space(unsigned char byte) noexcept nogil:
    return byte == 32 or 9 <= byte <= 13


cdef class ObjectScanner:

    cdef const unsigned char[::1] view
    cdef const unsigned char *data
    cdef Py_ssize_t length
    cdef unsigned char whitespace[256]
    cdef unsigned char separator[256]
    cdef bint name_escapes
    cdef bint split_whitespace_compatible
    cdef dict names
    cdef object name_of
    cdef object string_type
    cdef object reference_type
    cdef object decipher
    cdef object object_number
    cdef object generation
    cdef bint name_is_contents

    def __init__(
        self,
        data,
        bytes whitespace_table not None,
        bytes separator_table not None,
        bint name_escapes,
        bint split_whitespace_compatible,
        dict names not None,
        name_of,
        string_type,
        reference_type,
    ):
        if len(whitespace_table) != 256 or len(separator_table) != 256:
            raise ValueError("lexical tables must have 256 entries")
        self.view = data
        self.length = self.view.shape[0]
        self.data = &self.view[0] if self.length else NULL
        cdef int index
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
        self.data = NULL
        self.length = 0

    def parse_dictionary(self, Py_ssize_t pos, decipher=None, object_number=0, generation=0):
        return self.parse(pos, True, decipher, object_number, generation)

    def parse_array(self, Py_ssize_t pos, decipher=None, object_number=0, generation=0):
        return self.parse(pos, False, decipher, object_number, generation)

    cdef object parse(self, Py_ssize_t pos, bint dictionary, decipher, object_number, generation):
        if pos < 0 or self.data == NULL:
            return None
        self.decipher = decipher
        self.object_number = object_number
        self.generation = generation
        cdef Py_ssize_t cursor = pos
        try:
            value = self.dictionary(&cursor, 0) if dictionary else self.array(&cursor, 0)
        finally:
            self.decipher = None
        if value is BAIL:
            return None
        return value, cursor


    cdef Py_ssize_t skip_ignored(self, Py_ssize_t pos) noexcept:
        cdef const unsigned char *data = self.data
        cdef Py_ssize_t length = self.length
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

    cdef Py_ssize_t word_end(self, Py_ssize_t pos) noexcept:
        if self.separator[self.data[pos]]:
            return pos + 1
        pos += 1
        while pos < self.length and not self.separator[self.data[pos]]:
            pos += 1
        return pos

    cdef int number_kind(self, Py_ssize_t start, Py_ssize_t end) noexcept:
        cdef const unsigned char *data = self.data
        cdef Py_ssize_t pos = start
        cdef Py_ssize_t digits = 0
        cdef Py_ssize_t dots = 0
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

    cdef bint is_integer_token(self, Py_ssize_t start, Py_ssize_t end) noexcept:
        cdef Py_ssize_t pos = start
        if pos < end and (self.data[pos] == 43 or self.data[pos] == 45):
            pos += 1
        if pos >= end:
            return False
        while pos < end:
            if not is_digit(self.data[pos]):
                return False
            pos += 1
        return True

    cdef object integer(self, Py_ssize_t start, Py_ssize_t end):
        cdef const unsigned char *data = self.data
        cdef Py_ssize_t pos = start
        cdef bint negative = False
        cdef int64_t value = 0
        if data[pos] == 43 or data[pos] == 45:
            negative = data[pos] == 45
            pos += 1
        if end - pos <= FAST_INT_DIGITS:
            while pos < end:
                value = value * 10 + (data[pos] - 48)
                pos += 1
            return -value if negative else value
        token = PyBytes_FromStringAndSize(<const char *> data + start, end - start)
        try:
            return PyLong_FromString(PyBytes_AS_STRING(token), NULL, 10)
        except ValueError:
            return BAIL

    cdef object real(self, Py_ssize_t start, Py_ssize_t end):
        token = PyBytes_FromStringAndSize(<const char *> self.data + start, end - start)
        return PyOS_string_to_double(PyBytes_AS_STRING(token), NULL, NULL)

    cdef object keyword(self, Py_ssize_t start, Py_ssize_t end):
        cdef const unsigned char *data = self.data + start
        cdef Py_ssize_t size = end - start
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


    cdef object name(self, Py_ssize_t *pos):
        cdef const unsigned char *data = self.data
        cdef Py_ssize_t start = pos[0] + 1
        cdef Py_ssize_t end = start
        cdef bint escaped = False
        while end < self.length and not self.separator[data[end]]:
            if data[end] == 35:
                escaped = True
            end += 1
        pos[0] = end
        cdef bytes raw
        cdef bytearray decoded
        cdef Py_ssize_t index
        cdef unsigned char high, low
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
            raw = PyBytes_FromStringAndSize(<const char *> data + start, end - start)
        self.name_is_contents = raw == b"Contents"
        interned = self.names.get(raw)
        if interned is not None:
            return interned
        return self.name_of(raw)

    cdef object literal_string(self, Py_ssize_t *pos):
        cdef const unsigned char *data = self.data
        cdef Py_ssize_t length = self.length
        cdef Py_ssize_t start = pos[0] + 1
        cdef Py_ssize_t cursor = start
        while cursor < length:
            if data[cursor] == 41:
                pos[0] = cursor + 1
                return self.string(
                    PyBytes_FromStringAndSize(<const char *> data + start, cursor - start), True
                )
            if data[cursor] == 40 or data[cursor] == 92 or data[cursor] == 13 or data[cursor] == 10:
                break
            cursor += 1
        cdef Py_ssize_t end = self.literal_string_end(start)
        if end < 0:
            return BAIL
        out = PyBytes_FromStringAndSize(NULL, end - start)
        cdef unsigned char *buffer = <unsigned char *> PyBytes_AS_STRING(out)
        cdef Py_ssize_t size = 0
        cdef Py_ssize_t depth = 1
        cdef unsigned char byte, escape
        cdef int octal, count
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

    cdef object string(self, bytes value, bint is_literal):
        if self.decipher is not None:
            try:
                deciphered = self.decipher(self.object_number, self.generation, value, None)
            except Exception:
                return BAIL
            if type(deciphered) is memoryview:
                deciphered = deciphered.tobytes()
            return self.string_type(deciphered, is_literal=is_literal)
        return self.string_type(value, is_literal=is_literal)

    cdef Py_ssize_t literal_string_end(self, Py_ssize_t start) noexcept:
        cdef const unsigned char *data = self.data
        cdef Py_ssize_t length = self.length
        cdef Py_ssize_t cursor = start
        cdef Py_ssize_t depth = 1
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

    cdef inline bint line_end_pair(self, unsigned char first, unsigned char second) noexcept:
        return (first == 13 and second == 10) or (first == 10 and second == 13)

    cdef inline unsigned char escaped_byte(self, unsigned char escape) noexcept:
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

    cdef object hex_string(self, Py_ssize_t *pos):
        cdef const unsigned char *data = self.data
        cdef Py_ssize_t start = pos[0] + 1
        if start > self.length:
            return BAIL
        cdef const unsigned char *found = <const unsigned char *> memchr(
            data + start, 62, self.length - start
        )
        if found == NULL:
            return BAIL
        cdef Py_ssize_t marker = found - data
        out = PyBytes_FromStringAndSize(NULL, (marker - start + 1) // 2)
        cdef unsigned char *buffer = <unsigned char *> PyBytes_AS_STRING(out)
        cdef Py_ssize_t size = 0
        cdef Py_ssize_t cursor
        cdef unsigned char nibble
        cdef bint high = True
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

    cdef object number_or_reference(self, Py_ssize_t start, Py_ssize_t end, Py_ssize_t *pos):
        if self.number_kind(start, end) == 2:
            pos[0] = end
            return self.real(start, end)
        cdef Py_ssize_t next_start = self.skip_ignored(end)
        cdef Py_ssize_t next_end, marker, marker_end
        cdef unsigned char byte
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

    cdef object reference(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t next_start,
        Py_ssize_t next_end,
        Py_ssize_t *pos,
        Py_ssize_t marker_end,
    ):
        object_number = self.integer(start, end)
        generation = self.integer(next_start, next_end)
        if object_number is BAIL or generation is BAIL:
            return BAIL
        if object_number < 0 or not 0 <= generation <= 65535:
            return BAIL
        pos[0] = marker_end
        return self.reference_type(object_number, generation)

    cdef object value(self, Py_ssize_t *pos, int depth, bint strip_endobj):
        cdef Py_ssize_t start = self.skip_ignored(pos[0])
        if start >= self.length:
            return BAIL
        cdef unsigned char byte = self.data[start]
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
        cdef Py_ssize_t end = self.word_end(start)
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

    cdef object dictionary(self, Py_ssize_t *pos, int depth):
        if depth > MAX_DEPTH or pos[0] + 2 > self.length:
            return BAIL
        cdef dict values = {}
        cdef Py_ssize_t cursor = pos[0] + 2
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
            key = self.name(&cursor)
            if self.name_is_contents and self.decipher is not None and self.hex_follows(cursor):
                return BAIL
            value = self.value(&cursor, depth + 1, True)
            if value is BAIL:
                return BAIL
            values[key] = value
        pos[0] = cursor
        return values

    cdef bint hex_follows(self, Py_ssize_t pos) noexcept:
        pos = self.skip_ignored(pos)
        return (
            pos < self.length
            and self.data[pos] == 60
            and (pos + 1 >= self.length or self.data[pos + 1] != 60)
        )

    cdef object nested_dictionary(self, Py_ssize_t *pos, int depth):
        values = self.dictionary(pos, depth + 1)
        if values is BAIL:
            return BAIL
        cdef Py_ssize_t cursor = self.skip_ignored(pos[0])
        cdef Py_ssize_t index
        cdef int mismatches = 0
        cdef const char *keyword = b"stream"
        if cursor + 6 <= self.length:
            for index in range(6):
                if self.data[cursor + index] != <unsigned char> keyword[index]:
                    mismatches += 1
            if mismatches <= 1:
                return BAIL
        pos[0] = cursor
        return values

    cdef object array(self, Py_ssize_t *pos, int depth):
        if depth > MAX_DEPTH:
            return BAIL
        values = self.numeric_array(pos)
        if values is not None:
            return values
        cdef list items = []
        cdef Py_ssize_t cursor = pos[0] + 1
        cdef unsigned char byte
        while True:
            cursor = self.skip_ignored(cursor)
            if cursor >= self.length:
                return BAIL
            byte = self.data[cursor]
            if byte == 93:
                pos[0] = cursor + 1
                return items
            item = self.value(&cursor, depth + 1, False)
            if item is BAIL:
                return BAIL
            items.append(item)

    cdef object numeric_array(self, Py_ssize_t *pos):
        values = self.split_numeric_array(pos)
        if values is not None:
            return values
        return self.scanned_numeric_array(pos)

    cdef object split_numeric_array(self, Py_ssize_t *pos):
        if not self.split_whitespace_compatible:
            return None
        cdef const unsigned char *data = self.data
        cdef Py_ssize_t start = pos[0] + 1
        if start > self.length:
            return None
        cdef const unsigned char *found = <const unsigned char *> memchr(
            data + start, 93, self.length - start
        )
        if found == NULL:
            return None
        cdef Py_ssize_t close = found - data
        cdef Py_ssize_t cursor
        for cursor in range(start, close):
            if data[cursor] == 37 or data[cursor] == 91 or data[cursor] == 11:
                return None
        cdef list values = []
        cdef Py_ssize_t token_start
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

    cdef object scanned_numeric_array(self, Py_ssize_t *pos):
        cdef list values = []
        cdef Py_ssize_t cursor = pos[0] + 1
        cdef Py_ssize_t end
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

    cdef object numeric_word(self, Py_ssize_t start, Py_ssize_t end):
        cdef int kind = self.number_kind(start, end)
        if kind == 2:
            return self.real(start, end)
        if kind == 1:
            return self.integer(start, end)
        return BAIL if self.python_might_read(start, end) else None

    cdef bint python_might_read(self, Py_ssize_t start, Py_ssize_t end) noexcept:
        cdef bint dotted = memchr(self.data + start, 46, end - start) != NULL
        cdef unsigned char byte
        cdef Py_ssize_t cursor
        for cursor in range(start, end):
            byte = self.data[cursor]
            if is_digit(byte) or byte == 43 or byte == 45 or byte == 95:
                continue
            if dotted and (byte == 46 or (byte | 32) in b"einfaty"):
                continue
            return False
        return True

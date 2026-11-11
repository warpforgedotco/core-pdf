# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports.cpython.bytes import PyBytes_AS_STRING, PyBytes_FromStringAndSize
from cython.cimports.cpython.long import PyLong_AsLongLongAndOverflow
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Realloc
from cython.cimports.libc.stdio import snprintf
from cython.cimports.libc.string import memcpy

# Containers nested deeper than this go to the spec's serializer, whose Python recursion
# raises RecursionError for a cyclic or too deep value, where C recursion would overflow.
MAX_DEPTH = cython.declare(cython.int, 64)

HEX_UPPER = cython.declare(cython.p_const_char, b"0123456789ABCDEF")
HEX_LOWER = cython.declare(cython.p_const_char, b"0123456789abcdef")

# 7.2.3: white-space and delimiter characters, written as #xx inside a name with every byte
# outside the printable range, as core_pdf_spec.s_07_syntax.serialize.NAME_ESCAPED says.
NAME_ESCAPED = cython.declare(cython.uchar[256])
# A literal string's escapes in the compact layout: the byte after the backslash, or 0.
LITERAL_ESCAPE = cython.declare(cython.uchar[256])
_byte = cython.declare(cython.int)
for _byte in range(256):
    NAME_ESCAPED[_byte] = 1 if _byte < 0x21 or _byte > 0x7E else 0
    LITERAL_ESCAPE[_byte] = 0
for _byte in b"\x00\t\n\x0c\r ()<>[]{}/%#":
    NAME_ESCAPED[_byte] = 1
for _byte, _escape in (
    (0x0A, b"n"),
    (0x0D, b"r"),
    (0x09, b"t"),
    (0x08, b"b"),
    (0x0C, b"f"),
    (0x28, b"("),
    (0x29, b")"),
    (0x5C, b"\\"),
):
    LITERAL_ESCAPE[_byte] = _escape[0]


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def is_spaced_plain(byte: cython.uchar) -> cython.bint:
    # bytes((byte,)).isalnum() or a space: what the spaced layout leaves unescaped.
    return (
        byte == 0x20 or (0x30 <= byte <= 0x39) or (0x41 <= byte <= 0x5A) or (0x61 <= byte <= 0x7A)
    )


@cython.cclass
class ObjectWriter:
    """core_pdf_spec.s_07_syntax.serialize.serialize_object, in C, for one layout.

    It writes None, booleans, ints, names (and the plain strings the lexer gives some names
    as), PdfStrings, references, streams, and exact dicts, lists and tuples of them byte for
    byte as the spec's serializer does, and hands everything else -- reals, whose text is
    the spec's, bytes, other mappings and sequences -- to fallback, the spec's serializer for
    the same layout, which returns its bytes.

    A writer holds the buffer of one serialize call at a time; make one per call.
    """

    spaced: cython.bint
    name_type: object
    string_type: object
    reference_type: object
    stream_type: object
    fallback: object
    buffer: cython.p_char
    size: cython.Py_ssize_t
    capacity: cython.Py_ssize_t
    depth: cython.int

    def __cinit__(self):
        self.buffer = cython.NULL
        self.size = 0
        self.capacity = 0
        self.depth = 0

    def __init__(
        self,
        spaced: cython.bint,
        name_type,
        string_type,
        reference_type,
        stream_type,
        fallback,
    ):
        self.spaced = spaced
        self.name_type = name_type
        self.string_type = string_type
        self.reference_type = reference_type
        self.stream_type = stream_type
        self.fallback = fallback

    def __dealloc__(self):
        PyMem_Free(self.buffer)

    def serialize(self, value) -> bytes:
        """The bytes of value, as serialize_object(value, layout) gives them."""
        self.size = 0
        self.depth = 0
        self.write_value(value)
        return PyBytes_FromStringAndSize(self.buffer, self.size)

    @cython.cfunc
    def reserve(self, extra: cython.Py_ssize_t) -> cython.int:
        needed: cython.Py_ssize_t = self.size + extra
        capacity: cython.Py_ssize_t
        grown: cython.p_char
        if needed <= self.capacity:
            return 0
        capacity = self.capacity * 2 if self.capacity else 256
        while capacity < needed:
            capacity *= 2
        grown = cython.cast(cython.p_char, PyMem_Realloc(self.buffer, capacity))
        if grown == cython.NULL:
            raise MemoryError()
        self.buffer = grown
        self.capacity = capacity
        return 0

    @cython.cfunc
    def write(self, data: cython.p_const_char, length: cython.Py_ssize_t) -> cython.int:
        self.reserve(length)
        memcpy(self.buffer + self.size, data, length)
        self.size += length
        return 0

    @cython.cfunc
    def write_byte(self, byte: cython.char) -> cython.int:
        self.reserve(1)
        self.buffer[self.size] = byte
        self.size += 1
        return 0

    @cython.cfunc
    def write_bytes(self, data: bytes) -> cython.int:
        return self.write(PyBytes_AS_STRING(data), len(data))

    @cython.cfunc
    def write_integer(self, value) -> cython.int:
        text = cython.declare(cython.char[32])
        length: cython.int
        number: cython.longlong
        overflow: cython.int = 0
        if type(value) is int:
            number = PyLong_AsLongLongAndOverflow(value, cython.address(overflow))
            if not overflow:
                length = snprintf(text, 32, b"%lld", number)
                return self.write(text, length)
        # Large ints, and int subclasses, as the spec's b"%d" % value writes them.
        return self.write_bytes(b"%d" % value)

    @cython.cfunc
    def write_name_bytes(self, raw: bytes) -> cython.int:
        data: cython.p_const_uchar = cython.cast(cython.p_const_uchar, PyBytes_AS_STRING(raw))
        length: cython.Py_ssize_t = len(raw)
        index: cython.Py_ssize_t
        byte: cython.uchar
        self.reserve(1 + 3 * length)
        self.buffer[self.size] = 0x2F
        self.size += 1
        for index in range(length):
            byte = data[index]
            if NAME_ESCAPED[byte]:
                self.buffer[self.size] = 0x23
                self.buffer[self.size + 1] = HEX_UPPER[byte >> 4]
                self.buffer[self.size + 2] = HEX_UPPER[byte & 15]
                self.size += 3
            else:
                self.buffer[self.size] = byte
                self.size += 1
        return 0

    @cython.cfunc
    def write_name(self, value) -> cython.int:
        # serialize_name: a PdfName's bytes, a string's Latin-1, other values' bytes().
        if isinstance(value, self.name_type):
            return self.write_name_bytes(value.value_bytes)
        if isinstance(value, str):
            return self.write_name_bytes(value.encode("latin-1"))
        return self.write_name_bytes(bytes(value))

    @cython.cfunc
    def write_string(self, value) -> cython.int:
        raw: bytes = bytes(value.data)
        data: cython.p_const_uchar = cython.cast(cython.p_const_uchar, PyBytes_AS_STRING(raw))
        length: cython.Py_ssize_t = len(raw)
        index: cython.Py_ssize_t
        byte: cython.uchar
        digits: cython.p_const_char
        escape: cython.uchar
        text = cython.declare(cython.char[8])
        if value.is_literal is False:
            digits = HEX_LOWER if self.spaced else HEX_UPPER
            self.reserve(2 + 2 * length)
            self.buffer[self.size] = 0x3C
            self.size += 1
            for index in range(length):
                byte = data[index]
                self.buffer[self.size] = digits[byte >> 4]
                self.buffer[self.size + 1] = digits[byte & 15]
                self.size += 2
            self.buffer[self.size] = 0x3E
            self.size += 1
            return 0
        self.reserve(2 + 4 * length)
        self.buffer[self.size] = 0x28
        self.size += 1
        for index in range(length):
            byte = data[index]
            if self.spaced:
                if is_spaced_plain(byte):
                    self.buffer[self.size] = byte
                    self.size += 1
                else:
                    snprintf(text, 8, b"\\%03o", byte)
                    memcpy(self.buffer + self.size, text, 4)
                    self.size += 4
                continue
            escape = LITERAL_ESCAPE[byte]
            if escape:
                self.buffer[self.size] = 0x5C
                self.buffer[self.size + 1] = escape
                self.size += 2
            else:
                self.buffer[self.size] = byte
                self.size += 1
        self.buffer[self.size] = 0x29
        self.size += 1
        return 0

    @cython.cfunc
    def write_dictionary(self, dictionary: dict) -> cython.int:
        key: object
        item: object
        first: cython.bint = True
        if self.spaced:
            self.write(b"<<\n", 3)
            for key, item in dictionary.items():
                self.write_name(key)
                self.write_byte(0x20)
                self.write_value(item)
                self.write_byte(0x0A)
            self.write(b">>", 2)
            return 0
        self.write(b"<< ", 3)
        for key, item in dictionary.items():
            if not first:
                self.write_byte(0x20)
            first = False
            self.write_name(key)
            self.write_byte(0x20)
            self.write_value(item)
        self.write(b" >>", 3)
        return 0

    @cython.cfunc
    def write_array(self, items) -> cython.int:
        item: object
        first: cython.bint = True
        self.write_byte(0x5B)
        for item in items:
            if self.spaced:
                self.write_byte(0x20)
            elif not first:
                self.write_byte(0x20)
            first = False
            self.write_value(item)
        if self.spaced:
            self.write(b" ]", 2)
        else:
            self.write_byte(0x5D)
        return 0

    @cython.cfunc
    def write_stream(self, stream) -> cython.int:
        # serialize_stream: the dictionary with its Length rewritten, then the data.
        dictionary: dict = dict(stream.dictionary)
        data: bytes = bytes(stream.raw_data)
        dictionary["Length"] = len(data)
        self.write_value(dictionary)
        self.write(b"\nstream\n", 8)
        self.write_bytes(data)
        self.write(b"\nendstream", 10)
        return 0

    @cython.cfunc
    def write_value(self, value) -> cython.int:
        # The order of serialize_object's tests, which subclasses of several types rely on.
        cls = type(value)
        if value is None:
            return self.write(b"null", 4)
        if value is True:
            return self.write(b"true", 4)
        if value is False:
            return self.write(b"false", 5)
        if isinstance(value, int):
            return self.write_integer(value)
        if isinstance(value, float):
            return self.write_bytes(self.fallback(value))
        if isinstance(value, (self.name_type, str)):
            return self.write_name(value)
        if isinstance(value, self.string_type):
            return self.write_string(value)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return self.write_bytes(self.fallback(value))
        if isinstance(value, self.reference_type):
            self.write_integer(value.object_number)
            self.write_byte(0x20)
            self.write_integer(value.generation_number)
            return self.write(b" R", 2)
        if self.depth >= MAX_DEPTH or not (
            cls is dict or cls is list or cls is tuple or isinstance(value, self.stream_type)
        ):
            return self.write_bytes(self.fallback(value))
        self.depth += 1
        try:
            if cls is dict:
                return self.write_dictionary(value)
            if cls is list or cls is tuple:
                return self.write_array(value)
            return self.write_stream(value)
        finally:
            self.depth -= 1


def collect_references(value, reference_type, stream_type) -> list:
    """Every reference in value, depth first, in the order
    core_pdf.impl.document_editing.references_in has always given them.

    It pops the last pending value: a reference is found, a stream's dictionary values,
    a dict's values, and a list's or tuple's items are pushed, in their order; anything
    else is skipped. Subclasses go through their own values() and iteration.
    """
    found: list = []
    pending: list = [value]
    item: object
    cls: type
    while pending:
        item = pending.pop()
        cls = type(item)
        if cls is list or cls is tuple:
            pending.extend(item)
        elif cls is dict:
            pending.extend(cython.cast(dict, item).values())
        elif isinstance(item, reference_type):
            found.append(item)
        elif isinstance(item, stream_type):
            pending.extend(item.dictionary.values())
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return found

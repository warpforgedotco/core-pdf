# SPDX-License-Identifier: AGPL-3.0-only

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc
from libc.string cimport memchr, memcmp


cdef extern from "Python.h":
    object PyUnicode_DecodeUTF16(
        const char *text, Py_ssize_t size, const char *errors, int *byteorder
    )
    object PyUnicode_FromOrdinal(int ordinal)
    object PyUnicode_DecodeLatin1(const char *text, Py_ssize_t size, const char *errors)


cdef enum:
    WORD = 0
    HEX = 1
    LITERAL = 2
    ARRAY = 3
    PROCEDURE = 4
    DELIMITER = 5
    MAX_DEPTH = 64
    MAX_RANGE_SPAN = 65536

cdef enum:
    NO_BLOCK = 0
    BFCHAR = 1
    BFRANGE = 2
    CIDRANGE = 3
    CODESPACE = 4

cdef unsigned char WHITESPACE[256]
cdef unsigned char SEPARATOR[256]
cdef unsigned char HEX_VALUE[256]
cdef int _index
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


cdef struct Tokens:
    Py_ssize_t *start
    Py_ssize_t *end
    unsigned char *kind
    Py_ssize_t count
    Py_ssize_t capacity


cdef bint push(Tokens *tokens, Py_ssize_t start, Py_ssize_t end, unsigned char kind) noexcept:
    cdef Py_ssize_t capacity
    cdef void *grown
    if tokens.count == tokens.capacity:
        capacity = tokens.capacity * 2 if tokens.capacity else 256
        grown = PyMem_Realloc(tokens.start, capacity * sizeof(Py_ssize_t))
        if grown == NULL:
            return False
        tokens.start = <Py_ssize_t *>grown
        grown = PyMem_Realloc(tokens.end, capacity * sizeof(Py_ssize_t))
        if grown == NULL:
            return False
        tokens.end = <Py_ssize_t *>grown
        grown = PyMem_Realloc(tokens.kind, capacity)
        if grown == NULL:
            return False
        tokens.kind = <unsigned char *>grown
        tokens.capacity = capacity
    tokens.start[tokens.count] = start
    tokens.end[tokens.count] = end
    tokens.kind[tokens.count] = kind
    tokens.count += 1
    return True


cdef Py_ssize_t literal_end(
    const unsigned char *data, Py_ssize_t n, Py_ssize_t pos, bint *terminated
) noexcept nogil:
    cdef Py_ssize_t end = pos + 1
    cdef int depth = 1
    cdef unsigned char current
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


cdef Py_ssize_t find_byte(
    const unsigned char *data, Py_ssize_t n, Py_ssize_t pos, unsigned char byte
) noexcept nogil:
    cdef const void *found
    if pos >= n:
        return -1
    found = memchr(data + pos, byte, n - pos)
    if found == NULL:
        return -1
    return <const unsigned char *>found - data


cdef Py_ssize_t composite_end(
    const unsigned char *data, Py_ssize_t n, Py_ssize_t pos, int depth, int *status
) noexcept nogil:
    cdef unsigned char closing = 93 if data[pos] == 91 else 125
    cdef Py_ssize_t end = pos + 1
    cdef Py_ssize_t close
    cdef unsigned char current
    cdef bint ignored
    if depth > MAX_DEPTH:
        status[0] = -1
        return n
    while end < n:
        current = data[end]
        if current == 37:
            while end < n and data[end] != 10 and data[end] != 13:
                end += 1
        elif current == 40:
            end = literal_end(data, n, end, &ignored)
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


cdef bint tokenize(const unsigned char *data, Py_ssize_t n, Tokens *tokens) noexcept:
    cdef Py_ssize_t pos = 0
    cdef Py_ssize_t end
    cdef unsigned char byte
    cdef bint terminated
    cdef int status
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
            end = literal_end(data, n, pos, &terminated)
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
            end = composite_end(data, n, pos, 0, &status)
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


cdef inline bint is_word(
    const unsigned char *data, Tokens *tokens, Py_ssize_t index, const char *keyword, Py_ssize_t size
) noexcept nogil:
    return (
        tokens.kind[index] == WORD
        and tokens.end[index] - tokens.start[index] == size
        and memcmp(data + tokens.start[index], keyword, size) == 0
    )


cdef inline Py_ssize_t clean_hex_length(
    const unsigned char *data, Py_ssize_t start, Py_ssize_t end
) noexcept nogil:
    # The digits between < and >, when every one is a hex digit and they pair up;
    # -1 otherwise, which declines the whole CMap to the Python parser.
    cdef Py_ssize_t index
    cdef Py_ssize_t length = end - start - 2
    if length & 1:
        return -1
    for index in range(start + 1, end - 1):
        if HEX_VALUE[data[index]] == 255:
            return -1
    return length // 2


cdef inline void unhex(
    const unsigned char *data, Py_ssize_t start, Py_ssize_t size, unsigned char *out
) noexcept nogil:
    cdef Py_ssize_t index
    for index in range(size):
        out[index] = (HEX_VALUE[data[start + 1 + 2 * index]] << 4) | HEX_VALUE[
            data[start + 2 + 2 * index]
        ]


cdef object utf16be_text(unsigned char *buffer, Py_ssize_t size):
    cdef unsigned char *padded
    cdef Py_ssize_t index
    if size == 0:
        return ""
    if size >= 2 and buffer[0] == 0xFE and buffer[1] == 0xFF:
        buffer += 2
        size -= 2
    if size == 1:
        return PyUnicode_FromOrdinal(buffer[0])
    if size & 1:
        padded = <unsigned char *>PyMem_Malloc(size + 1)
        if padded == NULL:
            raise MemoryError()
        try:
            padded[0] = 0
            for index in range(size):
                padded[index + 1] = buffer[index]
            return utf16be_decode(padded, size + 1)
        finally:
            PyMem_Free(padded)
    return utf16be_decode(buffer, size)


cdef object utf16be_decode(unsigned char *buffer, Py_ssize_t size):
    cdef int byteorder = 1
    try:
        return PyUnicode_DecodeUTF16(<const char *>buffer, size, "strict", &byteorder)
    except UnicodeDecodeError:
        byteorder = 1
        return PyUnicode_DecodeUTF16(<const char *>buffer, size, "replace", &byteorder)


cdef object hex_text(const unsigned char *data, Py_ssize_t start, Py_ssize_t size, unsigned char *scratch):
    unhex(data, start, size, scratch)
    return utf16be_text(scratch, size)


cdef inline object scalar_or_replacement(long long codepoint):
    if 0 <= codepoint < 0x110000 and not 0xD800 <= codepoint <= 0xDFFF:
        return PyUnicode_FromOrdinal(<int>codepoint)
    return "�"


cdef object range_key(unsigned long long value, Py_ssize_t width):
    cdef unsigned char out[4]
    cdef Py_ssize_t index
    for index in range(width):
        out[width - 1 - index] = (value >> (8 * index)) & 0xFF
    return PyBytes_FromStringAndSize(<const char *>out, width)


cdef class Scan:
    cdef const unsigned char *data
    cdef Tokens *tokens
    cdef unsigned char *scratch
    cdef dict mappings
    cdef Py_ssize_t invalid
    cdef Py_ssize_t valid

    cdef bint bfchar(self, Py_ssize_t first, Py_ssize_t last) except -1:
        cdef Tokens *tokens = self.tokens
        cdef const unsigned char *data = self.data
        cdef Py_ssize_t index
        cdef Py_ssize_t pending = -1
        cdef Py_ssize_t source_size
        cdef Py_ssize_t target_size
        cdef object source
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
                source = PyBytes_FromStringAndSize(<const char *>self.scratch, source_size)
                self.mappings[source] = hex_text(data, tokens.start[index], target_size, self.scratch)
            pending = -1
        return True

    cdef bint bfrange(self, Py_ssize_t first, Py_ssize_t last) except -1:
        cdef Tokens *tokens = self.tokens
        cdef const unsigned char *data = self.data
        cdef Py_ssize_t operands[3]
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t total = 0
        cdef Py_ssize_t index
        cdef Py_ssize_t start_size
        cdef Py_ssize_t end_size
        cdef Py_ssize_t target_size
        cdef unsigned long long low
        cdef unsigned long long high
        cdef unsigned long long value
        cdef Py_ssize_t byte
        cdef object base
        cdef object prefix
        cdef long long final
        cdef dict mappings = self.mappings
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
                            final + <long long>(value - low)
                        )
                else:
                    for value in range(low, high + 1):
                        mappings[range_key(value, start_size)] = scalar_or_replacement(
                            final + <long long>(value - low)
                        )
            self.valid += 1
        return True

    cdef bint array_targets(
        self, Py_ssize_t token, unsigned long long low, unsigned long long high, Py_ssize_t width
    ) except -1:
        # The bracketed destinations of one bfrange, re-read the way cmap_tokens
        # reads them: only hex strings, whitespace and comments are accepted.
        cdef const unsigned char *data = self.data
        cdef Py_ssize_t pos = self.tokens.start[token] + 1
        cdef Py_ssize_t stop = self.tokens.end[token] - 1
        cdef Py_ssize_t close
        cdef Py_ssize_t size
        cdef unsigned long long offset = 0
        cdef bint added = False
        cdef bint any_target = False
        cdef list pending = []
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


def scan_to_unicode_cmap(bytes data):
    """Parse a ToUnicode CMap the way core's Python parser does, or return None.

    Returns (mappings, codespace_blocks, usecmap_name): the mappings dict in the
    Python parser's insertion order, each codespacerange block's hex and literal
    operands for the Python codespace validation, and the first usecmap name.
    None declines: the caller parses the same bytes in Python.
    """
    cdef const unsigned char *raw = <const unsigned char *>PyBytes_AS_STRING(data)
    cdef Py_ssize_t n = len(data)
    cdef Tokens tokens
    cdef Scan scan
    cdef Py_ssize_t scope_start = 0
    cdef Py_ssize_t scope_end
    cdef Py_ssize_t index
    cdef Py_ssize_t block_first = 0
    cdef Py_ssize_t previous_word = -1
    cdef Py_ssize_t longest = 0
    cdef int block = NO_BLOCK
    cdef bint usecmap_checked = False
    cdef object usecmap_name = None
    cdef list codespace_blocks = []
    cdef list operands
    tokens.start = NULL
    tokens.end = NULL
    tokens.kind = NULL
    tokens.count = 0
    tokens.capacity = 0
    scan = Scan.__new__(Scan)
    scan.scratch = NULL
    try:
        if not tokenize(raw, n, &tokens):
            return None
        for index in range(tokens.count):
            if tokens.end[index] - tokens.start[index] > longest:
                longest = tokens.end[index] - tokens.start[index]
        scan.scratch = <unsigned char *>PyMem_Malloc(longest // 2 + 4)
        if scan.scratch == NULL:
            raise MemoryError()

        for index in range(tokens.count):
            if is_word(raw, &tokens, index, b"begincmap", 9):
                scope_start = index + 1
                break
        scope_end = tokens.count
        for index in range(scope_start, tokens.count):
            if is_word(raw, &tokens, index, b"endcmap", 7):
                scope_end = index
                break

        for index in range(scope_start, scope_end):
            if tokens.kind[index] != WORD:
                continue
            if previous_word >= 0 and is_word(raw, &tokens, index, b"usecmap", 7):
                if raw[tokens.start[previous_word]] == 47:
                    usecmap_name = PyUnicode_DecodeLatin1(
                        <const char *>raw + tokens.start[previous_word] + 1,
                        tokens.end[previous_word] - tokens.start[previous_word] - 1,
                        NULL,
                    )
                break
            previous_word = index

        scan.data = raw
        scan.tokens = &tokens
        scan.mappings = {}
        scan.invalid = 0
        scan.valid = 0
        for index in range(scope_start, scope_end):
            if block == NO_BLOCK:
                if tokens.kind[index] != WORD:
                    continue
                if is_word(raw, &tokens, index, b"beginbfchar", 11):
                    block = BFCHAR
                elif is_word(raw, &tokens, index, b"beginbfrange", 12):
                    block = BFRANGE
                elif is_word(raw, &tokens, index, b"begincidrange", 13):
                    return None
                else:
                    continue
                block_first = index + 1
                continue
            if block == BFCHAR and is_word(raw, &tokens, index, b"endbfchar", 9):
                if not scan.bfchar(block_first, index):
                    return None
                block = NO_BLOCK
            elif block == BFRANGE and is_word(raw, &tokens, index, b"endbfrange", 10):
                if not scan.bfrange(block_first, index):
                    return None
                block = NO_BLOCK
        if scan.invalid and not scan.valid:
            return None

        block = NO_BLOCK
        for index in range(scope_start, scope_end):
            if block == NO_BLOCK:
                if is_word(raw, &tokens, index, b"begincodespacerange", 19):
                    block = CODESPACE
                    operands = []
                continue
            if is_word(raw, &tokens, index, b"endcodespacerange", 17):
                codespace_blocks.append(operands)
                block = NO_BLOCK
            elif tokens.kind[index] == HEX or tokens.kind[index] == LITERAL:
                operands.append(
                    PyBytes_FromStringAndSize(
                        <const char *>raw + tokens.start[index],
                        tokens.end[index] - tokens.start[index],
                    )
                )
        return scan.mappings, codespace_blocks, usecmap_name
    finally:
        PyMem_Free(tokens.start)
        PyMem_Free(tokens.end)
        PyMem_Free(tokens.kind)
        PyMem_Free(scan.scratch)

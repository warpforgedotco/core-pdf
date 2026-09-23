# SPDX-License-Identifier: AGPL-3.0-only
"""Content stream token scanning (core_pdf.impl.capture.recovery).

The tokenizer's fast path was a single regular expression matched once per
token. The expression is C, but every match allocates a match object and every
operand allocates a bytes slice out of it, and a vector-heavy page runs that
about a million times: one page in the corpus spends a quarter of its render
here. Scanning the bytes directly drops both allocations, and the operand is
built once, straight from the buffer.

The grammar below is the regular expression, transcribed:

    (?:[\\x00\\t\\n\\f\\r ]+|%[^\\r\\n]*)*
    (?: (?P<num>[+-]?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+))
      | (?P<name>/[^\\x00\\t\\n\\f\\r ()<>\\[\\]/%#]*)
      | (?P<op>[^\\x00\\t\\n\\f\\r ()<>\\[\\]/%+\\-.0-9][^\\x00\\t\\n\\f\\r ()<>\\[\\]/%]*)
    )(?=[\\x00\\t\\n\\f\\r ()<>\\[\\]/%]|$)

Numbers are converted with CPython's own parsers rather than a C one.
PyOS_string_to_double is what float() calls and PyLong_FromString is what int()
calls, so the values are not merely close to what the old path produced, they
come off the same code. That is the only way a scanner like this can promise
bit-exactness for every input rather than for the inputs someone thought to
test.

The scanner handles the three token shapes above and nothing else. Strings,
arrays, dictionaries, inline images and anything malformed are handed back to
the caller as a byte offset, so every recovery path stays in Python where it
was. On the densest corpus page that hand-back happens for one operation in
seven -- the dash arrays -- and the scan is still more than twice as fast.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.list cimport PyList_Append, PyList_GET_SIZE
from cpython.long cimport PyLong_FromString

cdef extern from "Python.h":
    double PyOS_string_to_double(
        const char* s, char** endptr, object overflow_exception
    ) except? -1.0

# Returned in place of a byte offset. A caller sees a negative code as a
# status and any other value as the position it must parse from.
SCAN_OPERATOR = -1
SCAN_NAME = -2

cdef Py_ssize_t C_SCAN_OPERATOR = SCAN_OPERATOR
cdef Py_ssize_t C_SCAN_NAME = SCAN_NAME

# Returned alongside a byte offset. An operator token and a name token are both
# non-empty, so an empty word is unambiguous, and keeping the type of the pair
# fixed saves the caller a narrowing check on its hottest loop.
cdef bytes EMPTY = b""

# An operand list never grows past this; the tokenizer has always dropped the
# rest rather than let a malformed stream accumulate without bound.
cdef Py_ssize_t OPERAND_LIMIT = 16
# A numeric token this long or longer goes to the slow path, as it always has.
cdef Py_ssize_t NUMBER_LIMIT = 16

cdef bint IS_SPACE[256]
cdef bint IS_DELIM[256]
cdef bint IS_NUMERIC_START[256]
cdef bint IS_DIGIT[256]

cdef int _i
for _i in range(256):
    IS_SPACE[_i] = 0
    IS_DELIM[_i] = 0
    IS_NUMERIC_START[_i] = 0
    IS_DIGIT[_i] = 0
# NUL, tab, newline, form feed, carriage return, space.
for _i in (0x00, 0x09, 0x0A, 0x0C, 0x0D, 0x20):
    IS_SPACE[_i] = 1
    IS_DELIM[_i] = 1
# ( ) < > [ ] / %
for _i in (0x28, 0x29, 0x3C, 0x3E, 0x5B, 0x5D, 0x2F, 0x25):
    IS_DELIM[_i] = 1
for _i in (0x2B, 0x2D, 0x2E):  # + - .
    IS_NUMERIC_START[_i] = 1
for _i in range(0x30, 0x3A):   # 0-9
    IS_NUMERIC_START[_i] = 1
    IS_DIGIT[_i] = 1


cdef class ContentScanner:
    """Scans content stream tokens, deferring anything it does not own."""

    cdef const unsigned char[::1] view
    cdef const unsigned char* buf
    cdef Py_ssize_t size
    cdef Py_ssize_t cursor
    cdef object keywords

    def __cinit__(self, data, keywords):
        self.view = data
        self.size = self.view.shape[0]
        self.buf = &self.view[0] if self.size else NULL
        self.cursor = 0
        self.keywords = keywords

    @property
    def pos(self):
        return self.cursor

    @pos.setter
    def pos(self, Py_ssize_t value):
        self.cursor = value

    cdef inline Py_ssize_t skip_ignored(self, Py_ssize_t p) noexcept nogil:
        """Consume whitespace runs and comments, as the expression's prefix did."""
        cdef const unsigned char* b = self.buf
        cdef Py_ssize_t n = self.size
        while p < n:
            if IS_SPACE[b[p]]:
                p += 1
            elif b[p] == 0x25:  # %
                # A comment ends before its terminator; the loop then eats it
                # as whitespace, exactly as the expression's alternation did.
                while p < n and b[p] != 0x0D and b[p] != 0x0A:
                    p += 1
            else:
                break
        return p

    def next_operation(self, list operands, dict names):
        """Scan up to and including the next operator.

        Appends operands to ``operands`` as it goes and returns a pair. The
        second element is ``SCAN_OPERATOR`` when the first is an operator's
        bytes, ``SCAN_NAME`` when it is a name token whose interned form is not
        yet in ``names`` and the caller must add it, and otherwise a byte
        offset the caller must parse from, with the first element ``None``.
        """
        cdef const unsigned char* b = self.buf
        cdef Py_ssize_t n = self.size
        cdef Py_ssize_t p, start, digits_before, digits_after
        cdef unsigned char c
        cdef bint has_dot
        cdef char* end
        cdef bytes word
        cdef object value

        while True:
            with nogil:
                p = self.skip_ignored(self.cursor)
            self.cursor = p
            if p >= n:
                # Nothing but whitespace left. The caller's parser turns this
                # into the end of the stream; reproducing that here would be a
                # second place for it to be decided.
                return EMPTY, p
            c = b[p]

            if IS_NUMERIC_START[c]:
                start = p
                has_dot = 0
                digits_before = 0
                digits_after = 0
                if c == 0x2B or c == 0x2D:
                    p += 1
                while p < n and IS_DIGIT[b[p]]:
                    p += 1
                    digits_before += 1
                if p < n and b[p] == 0x2E:
                    has_dot = 1
                    p += 1
                    while p < n and IS_DIGIT[b[p]]:
                        p += 1
                        digits_after += 1
                # [0-9]+\.?[0-9]*  or  \.[0-9]+ -- a lone sign, a lone dot and
                # a sign followed by a dot are all rejected, as they were.
                if digits_before == 0 and not (has_dot and digits_after > 0):
                    return EMPTY, start
                # A second dot, a letter, an exponent: the expression required
                # a delimiter here and so does this.
                if p < n and not IS_DELIM[b[p]]:
                    return EMPTY, start
                if p - start >= NUMBER_LIMIT:
                    return EMPTY, start
                self.cursor = p
                if PyList_GET_SIZE(operands) < OPERAND_LIMIT:
                    if has_dot:
                        # endptr is required: the buffer is not NUL terminated
                        # at the end of the token, and without it the parser
                        # would reject everything after it.
                        value = PyOS_string_to_double(<const char*> (b + start), &end, None)
                    else:
                        word = PyBytes_FromStringAndSize(
                            <const char*> (b + start), p - start
                        )
                        value = PyLong_FromString(word, NULL, 10)
                    PyList_Append(operands, value)
                continue

            if c == 0x2F:  # /
                start = p
                p += 1
                # '#' is excluded from the name body, so an escaped name falls
                # back and keeps its one decoder.
                while p < n and not (IS_DELIM[b[p]] or b[p] == 0x23):
                    p += 1
                if p < n and not IS_DELIM[b[p]]:
                    return EMPTY, start
                word = PyBytes_FromStringAndSize(<const char*> (b + start), p - start)
                self.cursor = p
                value = names.get(word)
                if value is None:
                    return word, C_SCAN_NAME
                if PyList_GET_SIZE(operands) < OPERAND_LIMIT:
                    PyList_Append(operands, value)
                continue

            if IS_DELIM[c]:
                # A string, array, dictionary or stray delimiter.
                return EMPTY, p

            # An operator: the first byte is neither numeric nor a delimiter,
            # which is exactly the expression's first character class.
            start = p
            p += 1
            while p < n and not IS_DELIM[b[p]]:
                p += 1
            word = PyBytes_FromStringAndSize(<const char*> (b + start), p - start)
            if word in self.keywords:
                # BI, true, false and null are not operators; the expression
                # declined them too and left them to the parser.
                return EMPTY, start
            self.cursor = p
            return word, C_SCAN_OPERATOR

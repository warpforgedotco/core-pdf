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

An operation comes back whole: the operator's name interned as the str the
handler tables are keyed by, and its operands as the tuple the handler is
passed. The caller's loop then runs once per operation and does nothing but
hand it on; building the pair in Python, with its two dictionary lookups, was
most of what that loop cost.

Given a capture state (set_path_state), the path-construction operators --
m, l, c, v, y, re and h -- do not come back at all: the scanner applies
them to the state's current path itself, as ContentInterpreter's and
RecoveringTextState's handlers would, when each has exactly its operands,
all numbers. Numbers are held as doubles until their operator is known and
become Python objects only if it is not one of those, so a path's
coordinates are parsed and stored without one. An integer converts as
float(int(token)) does -- through the integer, so "-0" is 0.0 -- and a
real as PyOS_string_to_double reads it, as the Python float was. The
current point, the subpath start, and the path are the state's again, and
its shared glyph paint and text layout dropped as dispatch_frame drops them
for these operators, before control goes back to Python.
"""

from cpython cimport array
from cpython.bytearray cimport PyByteArray_AS_STRING, PyByteArray_GET_SIZE, PyByteArray_Resize
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.float cimport PyFloat_AsDouble
from cpython.list cimport PyList_Append, PyList_AsTuple, PyList_GET_SIZE
from cpython.long cimport PyLong_FromString
from cpython.number cimport PyNumber_Float
from cpython.object cimport PyObject

cdef extern from "Python.h":
    double PyOS_string_to_double(
        const char* s, char** endptr, object overflow_exception
    ) except? -1.0
    # Declared here rather than cimported: cpython.list types the item list as
    # an object, which cannot be NULL, and NULL is what deletes the slice.
    int PyList_SetSlice(object list, Py_ssize_t low, Py_ssize_t high, PyObject* items) except -1

# An operand list never grows past this; the tokenizer has always dropped the
# rest rather than let a malformed stream accumulate without bound.
cdef enum:
    OPERAND_CAPACITY = 16
cdef Py_ssize_t OPERAND_LIMIT = OPERAND_CAPACITY
# A numeric token this long or longer goes to the slow path, as it always has.
cdef Py_ssize_t NUMBER_LIMIT = 16

# The path operators, as the codes PdfPath stores; re is stored as r.
cdef enum:
    NOT_PATH = 0
    PATH_M = 0x6D
    PATH_L = 0x6C
    PATH_C = 0x63
    PATH_V = 0x76
    PATH_Y = 0x79
    PATH_H = 0x68
    PATH_RE = 0x72

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


cdef inline int path_operator(const unsigned char* word, Py_ssize_t length) noexcept nogil:
    if length == 1:
        if word[0] == PATH_M or word[0] == PATH_L or word[0] == PATH_C or word[0] == PATH_V \
                or word[0] == PATH_Y or word[0] == PATH_H:
            return word[0]
        return NOT_PATH
    if length == 2 and word[0] == 0x72 and word[1] == 0x65:  # re
        return PATH_RE
    return NOT_PATH


cdef inline int path_operand_count(int op) noexcept nogil:
    if op == PATH_M or op == PATH_L:
        return 2
    if op == PATH_C:
        return 6
    if op == PATH_V or op == PATH_Y or op == PATH_RE:
        return 4
    return 0


cdef class ContentScanner:
    """Scans content stream operations, deferring anything it does not own."""

    cdef const unsigned char[::1] view
    cdef const unsigned char* buf
    cdef Py_ssize_t size
    cdef Py_ssize_t cursor
    cdef object keywords
    cdef object object_keywords
    cdef object make_name
    cdef dict names
    cdef dict operators
    cdef readonly list operands

    # Numbers scanned but not yet made into Python objects: they follow
    # whatever `operands` holds.
    cdef Py_ssize_t pending
    cdef double pending_value[OPERAND_CAPACITY]
    cdef Py_ssize_t pending_start[OPERAND_CAPACITY]
    cdef Py_ssize_t pending_end[OPERAND_CAPACITY]
    cdef bint pending_integer[OPERAND_CAPACITY]

    # The capture state path operators apply to, or None; and while they
    # are being applied, its path and points as they now stand.
    cdef object path_state
    cdef bint paths_loaded
    cdef bint paths_declined
    cdef bint paths_changed
    cdef object path_ops
    cdef array.array path_coords
    cdef bint current_set
    cdef double current_x, current_y
    cdef bint start_set
    cdef double start_x, start_y
    cdef bint curve_loaded
    cdef double curve_matrix[4]
    cdef double flatness

    def __cinit__(self, data, keywords, object_keywords, make_name):
        self.view = data
        self.size = self.view.shape[0]
        self.buf = &self.view[0] if self.size else NULL
        self.cursor = 0
        self.keywords = keywords
        self.object_keywords = object_keywords
        self.make_name = make_name
        self.names = {}
        self.operators = {}
        self.operands = []
        self.pending = 0
        self.path_state = None
        self.paths_loaded = False
        self.paths_changed = False

    @property
    def pos(self):
        return self.cursor

    @pos.setter
    def pos(self, Py_ssize_t value):
        self.cursor = value

    def set_path_state(self, state):
        """Apply path operators to `state` rather than return them; None stops it.

        The caller vouches that `state`'s handlers for them are the
        tolerant capture's own.
        """
        self.flush_paths()
        self.path_state = state

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

    cdef int materialize(self) except -1:
        """Make the pending numbers Python objects, after what `operands` holds."""
        cdef Py_ssize_t i
        cdef bytes word
        cdef object value
        for i in range(self.pending):
            if self.pending_integer[i]:
                word = PyBytes_FromStringAndSize(
                    <const char*> (self.buf + self.pending_start[i]),
                    self.pending_end[i] - self.pending_start[i],
                )
                value = PyLong_FromString(word, NULL, 10)
            else:
                value = self.pending_value[i]
            PyList_Append(self.operands, value)
        self.pending = 0
        return 0

    cdef bint load_point(self, object point, double* x, double* y) except -1:
        """Read a (x, y) point of two floats; False if it is anything else."""
        if type(point) is not tuple or len(point) != 2:
            return False
        first = point[0]
        second = point[1]
        if type(first) is not float or type(second) is not float:
            return False
        x[0] = PyFloat_AsDouble(first)
        y[0] = PyFloat_AsDouble(second)
        return True

    cdef int load_paths(self) except -1:
        """Take the state's path and points, or decline to apply operators to it."""
        state = self.path_state
        self.paths_loaded = True
        self.paths_declined = False
        self.paths_changed = False
        self.curve_loaded = False
        path = state.current_path
        ops = path.ops
        coords = path.coords
        if type(ops) is not bytearray or not isinstance(coords, array.array) or coords.typecode != "d":
            self.paths_declined = True
            return 0
        self.path_ops = ops
        self.path_coords = coords
        current = state.current_point
        start = state.subpath_start
        self.current_set = current is not None
        self.start_set = start is not None
        if self.current_set and not self.load_point(current, &self.current_x, &self.current_y):
            self.paths_declined = True
        if self.start_set and not self.load_point(start, &self.start_x, &self.start_y):
            self.paths_declined = True
        return 0

    cdef bint load_curve(self) except -1:
        """The CTM's linear part and the flatness a curve records; False if unreadable."""
        graphics = self.path_state.graphics
        ctm = graphics.ctm
        cdef Py_ssize_t i
        for i in range(4):
            value = ctm[i]
            if type(value) is not float and type(value) is not int:
                return False
        flatness = graphics.flatness
        if type(flatness) is not float and type(flatness) is not int:
            return False
        for i in range(4):
            self.curve_matrix[i] = PyFloat_AsDouble(PyNumber_Float(ctm[i]))
        self.flatness = PyFloat_AsDouble(PyNumber_Float(flatness))
        self.curve_loaded = True
        return True

    cdef int flush_paths(self) except -1:
        """Hand the path and points back to the state, if operators changed them."""
        if not self.paths_loaded:
            return 0
        self.paths_loaded = False
        self.path_ops = None
        self.path_coords = None
        if not self.paths_changed:
            return 0
        self.paths_changed = False
        state = self.path_state
        state.current_point = (self.current_x, self.current_y) if self.current_set else None
        state.subpath_start = (self.start_x, self.start_y) if self.start_set else None
        # What dispatch_frame does before any operator outside the text ones.
        state.shared_glyph_paint = None
        state.text_layout = None
        return 0

    cdef int append_op(self, unsigned char op) except -1:
        cdef Py_ssize_t n = PyByteArray_GET_SIZE(self.path_ops)
        PyByteArray_Resize(self.path_ops, n + 1)
        PyByteArray_AS_STRING(self.path_ops)[n] = <char> op
        return 0

    cdef int append_coords(self, const double* values, Py_ssize_t count) except -1:
        cdef array.array coords = self.path_coords
        cdef Py_ssize_t n = len(coords), i
        array.resize_smart(coords, n + count)
        for i in range(count):
            coords.data.as_doubles[n + i] = values[i]
        return 0

    cdef bint apply_path(self, int op) except -1:
        """Apply `op` to the state with the pending numbers; False to decline it."""
        cdef double* v = self.pending_value
        cdef double curve[13]
        cdef Py_ssize_t i
        if not self.paths_loaded:
            self.load_paths()
        if self.paths_declined:
            return False
        if op == PATH_M:
            self.append_op(PATH_M)
            self.append_coords(v, 2)
            self.current_x = self.start_x = v[0]
            self.current_y = self.start_y = v[1]
            self.current_set = self.start_set = True
        elif op == PATH_L:
            # RecoveringTextState.op_l: no current point, no line.
            if self.current_set:
                self.append_op(PATH_L)
                self.append_coords(v, 2)
                self.current_x = v[0]
                self.current_y = v[1]
        elif op == PATH_RE:
            self.append_op(PATH_RE)
            self.append_coords(v, 4)
            self.current_x = self.start_x = v[0]
            self.current_y = self.start_y = v[1]
            self.current_set = self.start_set = True
        elif op == PATH_H:
            if self.current_set and self.start_set:
                self.append_op(PATH_H)
                self.current_x = self.start_x
                self.current_y = self.start_y
        else:
            # c, v and y, through RecoveringTextState.append_cubic_curve: a
            # curve with no current point only sets it, and v and y need one.
            if not self.current_set:
                if op != PATH_C:
                    return True
                self.current_x = v[4]
                self.current_y = v[5]
                self.current_set = True
                self.paths_changed = True
                return True
            if not self.curve_loaded and not self.load_curve():
                return False
            curve[0] = self.current_x
            curve[1] = self.current_y
            if op == PATH_C:
                for i in range(6):
                    curve[2 + i] = v[i]
            elif op == PATH_V:
                curve[2] = self.current_x
                curve[3] = self.current_y
                for i in range(4):
                    curve[4 + i] = v[i]
            else:
                curve[2] = v[0]
                curve[3] = v[1]
                curve[4] = v[2]
                curve[5] = v[3]
                curve[6] = v[2]
                curve[7] = v[3]
            for i in range(4):
                curve[8 + i] = self.curve_matrix[i]
            curve[12] = self.flatness
            self.append_op(PATH_C)
            self.append_coords(curve, 13)
            self.current_x = curve[6]
            self.current_y = curve[7]
        self.paths_changed = True
        return True

    def next_operation(self):
        """Scan up to and including the next operation.

        Appends operands to ``operands`` as it goes. Returns the operation as
        a ``(name, operands)`` pair, with the operand list emptied into the
        tuple, or else a byte offset the caller must parse from; what the
        caller parses there goes on the same list. Operations named in
        ``object_keywords`` are dropped with their operands, as the caller
        dropped them, and with a path state, path operators it can apply are
        applied and not returned.
        """
        try:
            return self.scan_operation()
        finally:
            if self.pending:
                self.materialize()
            self.flush_paths()

    cdef object scan_operation(self):
        cdef const unsigned char* b = self.buf
        cdef Py_ssize_t n = self.size
        cdef Py_ssize_t p, start, digits_before, digits_after
        cdef unsigned char c
        cdef bint has_dot
        cdef char* end
        cdef bytes word
        cdef object value
        cdef object name
        cdef tuple arguments
        cdef list operands = self.operands
        cdef long long integer
        cdef Py_ssize_t q, slot
        cdef int op
        cdef bint negative

        while True:
            with nogil:
                p = self.skip_ignored(self.cursor)
            self.cursor = p
            if p >= n:
                # Nothing but whitespace left. The caller's parser turns this
                # into the end of the stream; reproducing that here would be a
                # second place for it to be decided.
                return p
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
                    return start
                # A second dot, a letter, an exponent: the expression required
                # a delimiter here and so does this.
                if p < n and not IS_DELIM[b[p]]:
                    return start
                if p - start >= NUMBER_LIMIT:
                    return start
                self.cursor = p
                if PyList_GET_SIZE(operands) + self.pending < OPERAND_LIMIT:
                    slot = self.pending
                    self.pending_start[slot] = start
                    self.pending_end[slot] = p
                    self.pending_integer[slot] = not has_dot
                    if has_dot:
                        # endptr is required: the buffer is not NUL terminated
                        # at the end of the token, and without it the parser
                        # would reject everything after it.
                        self.pending_value[slot] = PyOS_string_to_double(
                            <const char*> (b + start), &end, None
                        )
                    else:
                        # float(int(token)): at most fifteen digits, exact in a
                        # long long and in a double, and "-0" is zero.
                        negative = b[start] == 0x2D
                        integer = 0
                        for q in range(start + (1 if b[start] == 0x2B or negative else 0), p):
                            integer = integer * 10 + (b[q] - 0x30)
                        self.pending_value[slot] = <double> (-integer if negative else integer)
                    self.pending += 1
                continue

            if c == 0x2F:  # /
                start = p
                p += 1
                # '#' is excluded from the name body, so an escaped name falls
                # back and keeps its one decoder.
                while p < n and not (IS_DELIM[b[p]] or b[p] == 0x23):
                    p += 1
                if p < n and not IS_DELIM[b[p]]:
                    return start
                word = PyBytes_FromStringAndSize(<const char*> (b + start), p - start)
                self.cursor = p
                value = self.names.get(word)
                if value is None:
                    value = self.names[word] = self.make_name(word[1:])
                if PyList_GET_SIZE(operands) + self.pending < OPERAND_LIMIT:
                    if self.pending:
                        self.materialize()
                    PyList_Append(operands, value)
                continue

            if IS_DELIM[c]:
                # A string, array, dictionary or stray delimiter.
                return p

            # An operator: the first byte is neither numeric nor a delimiter,
            # which is exactly the expression's first character class.
            start = p
            p += 1
            while p < n and not IS_DELIM[b[p]]:
                p += 1
            if self.path_state is not None and PyList_GET_SIZE(operands) == 0:
                op = path_operator(b + start, p - start)
                if (
                    op != NOT_PATH
                    and self.pending == path_operand_count(op)
                    and self.apply_path(op)
                ):
                    self.cursor = p
                    self.pending = 0
                    continue
            word = PyBytes_FromStringAndSize(<const char*> (b + start), p - start)
            if word in self.keywords:
                # BI, true, false and null are not operators; the expression
                # declined them too and left them to the parser.
                return start
            self.cursor = p
            if self.pending:
                self.materialize()
            name = self.operators.get(word)
            if name is None:
                name = self.operators[word] = word.decode("latin-1")
            arguments = PyList_AsTuple(operands)
            PyList_SetSlice(operands, 0, PyList_GET_SIZE(operands), NULL)
            if name in self.object_keywords:
                continue
            return name, arguments

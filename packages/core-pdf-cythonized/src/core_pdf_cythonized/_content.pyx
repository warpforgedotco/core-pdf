# SPDX-License-Identifier: AGPL-3.0-only

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
    int PyList_SetSlice(object list, Py_ssize_t low, Py_ssize_t high, PyObject* items) except -1

cdef enum:
    OPERAND_CAPACITY = 16
cdef Py_ssize_t OPERAND_LIMIT = OPERAND_CAPACITY
cdef Py_ssize_t NUMBER_LIMIT = 16

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
for _i in (0x00, 0x09, 0x0A, 0x0C, 0x0D, 0x20):
    IS_SPACE[_i] = 1
    IS_DELIM[_i] = 1
for _i in (0x28, 0x29, 0x3C, 0x3E, 0x5B, 0x5D, 0x2F, 0x25):
    IS_DELIM[_i] = 1
for _i in (0x2B, 0x2D, 0x2E):
    IS_NUMERIC_START[_i] = 1
for _i in range(0x30, 0x3A):
    IS_NUMERIC_START[_i] = 1
    IS_DIGIT[_i] = 1


cdef inline int path_operator(const unsigned char* word, Py_ssize_t length) noexcept nogil:
    if length == 1:
        if word[0] == PATH_M or word[0] == PATH_L or word[0] == PATH_C or word[0] == PATH_V \
                or word[0] == PATH_Y or word[0] == PATH_H:
            return word[0]
        return NOT_PATH
    if length == 2 and word[0] == 0x72 and word[1] == 0x65:
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

    cdef Py_ssize_t pending
    cdef double pending_value[OPERAND_CAPACITY]
    cdef Py_ssize_t pending_start[OPERAND_CAPACITY]
    cdef Py_ssize_t pending_end[OPERAND_CAPACITY]
    cdef bint pending_integer[OPERAND_CAPACITY]

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
        self.flush_paths()
        self.path_state = state

    cdef inline Py_ssize_t skip_ignored(self, Py_ssize_t p) noexcept nogil:
        cdef const unsigned char* b = self.buf
        cdef Py_ssize_t n = self.size
        while p < n:
            if IS_SPACE[b[p]]:
                p += 1
            elif b[p] == 0x25:
                while p < n and b[p] != 0x0D and b[p] != 0x0A:
                    p += 1
            else:
                break
        return p

    cdef int materialize(self) except -1:
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

    cdef object scan_numeric_array(self, Py_ssize_t p):
        cdef const unsigned char* b = self.buf
        cdef Py_ssize_t n = self.size
        cdef Py_ssize_t start, digits_before, digits_after
        cdef bint has_dot
        cdef char* end
        cdef bytes word
        cdef list values = []
        p += 1
        while True:
            while p < n and IS_SPACE[b[p]]:
                p += 1
            if p >= n:
                return None
            if b[p] == 0x5D:
                self.cursor = p + 1
                return values
            if not IS_NUMERIC_START[b[p]]:
                return None
            start = p
            has_dot = 0
            digits_before = 0
            digits_after = 0
            if b[p] == 0x2B or b[p] == 0x2D:
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
            if digits_before == 0 and not (has_dot and digits_after > 0):
                return None
            if p >= n or not (IS_SPACE[b[p]] or b[p] == 0x5D):
                return None
            if p - start >= NUMBER_LIMIT:
                return None
            if has_dot:
                PyList_Append(values, PyOS_string_to_double(<const char*> (b + start), &end, None))
            else:
                word = PyBytes_FromStringAndSize(<const char*> (b + start), p - start)
                PyList_Append(values, PyLong_FromString(word, NULL, 10))

    def run(
        self,
        lexer,
        dict handlers,
        fallback,
        state,
        depth,
        frozenset keep_paint,
        frozenset line_moving,
        frozenset keep_layout,
    ):
        """Scan and dispatch operations until a handler returns a child frame.

        fallback parses a token the scanner hands back: it returns the next
        operation, False to resume scanning, or None at the end of the stream.
        lexer.pos is kept at the end of the current operation.
        """
        cdef object result, name, operands, handler, child, layout
        while True:
            self.cursor = lexer.pos
            try:
                result = self.scan_operation()
            finally:
                if self.pending:
                    self.materialize()
                self.flush_paths()
            if type(result) is tuple:
                lexer.pos = self.cursor
            else:
                result = fallback(result)
                if result is None:
                    return None
                if result is False:
                    continue
            name, operands = result
            handler = handlers.get(name)
            if handler is None:
                continue
            if name not in keep_paint:
                state.shared_glyph_paint = None
                state.text_layout = None
            elif name in line_moving:
                layout = state.text_layout
                if layout is not None:
                    layout.style = None
            elif name not in keep_layout:
                state.text_layout = None
            child = handler(operands, depth)
            if child is not None:
                return child

    def next_operation(self):
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
                if digits_before == 0 and not (has_dot and digits_after > 0):
                    return start
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
                        self.pending_value[slot] = PyOS_string_to_double(
                            <const char*> (b + start), &end, None
                        )
                    else:
                        negative = b[start] == 0x2D
                        integer = 0
                        for q in range(start + (1 if b[start] == 0x2B or negative else 0), p):
                            integer = integer * 10 + (b[q] - 0x30)
                        self.pending_value[slot] = <double> (-integer if negative else integer)
                    self.pending += 1
                continue

            if c == 0x2F:
                start = p
                p += 1
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

            if c == 0x5B:
                value = self.scan_numeric_array(p)
                if value is None:
                    return p
                if PyList_GET_SIZE(operands) + self.pending < OPERAND_LIMIT:
                    if self.pending:
                        self.materialize()
                    PyList_Append(operands, value)
                continue

            if IS_DELIM[c]:
                return p

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

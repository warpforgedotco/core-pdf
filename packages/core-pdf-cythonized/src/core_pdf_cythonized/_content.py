# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
from cython.cimports.core_pdf_cythonized._content import (
    NOT_PATH,
    OPERAND_CAPACITY,
    PATH_C,
    PATH_H,
    PATH_L,
    PATH_M,
    PATH_RE,
    PATH_V,
    PATH_Y,
    PyList_SetSlice,
    PyOS_string_to_double,
)
from cython.cimports.cpython import array
from cython.cimports.cpython.bytearray import (
    PyByteArray_AS_STRING,
    PyByteArray_GET_SIZE,
    PyByteArray_Resize,
)
from cython.cimports.cpython.bytes import PyBytes_AS_STRING, PyBytes_FromStringAndSize
from cython.cimports.cpython.float import PyFloat_AsDouble
from cython.cimports.cpython.list import PyList_Append, PyList_AsTuple, PyList_GET_SIZE
from cython.cimports.cpython.long import PyLong_FromString
from cython.cimports.cpython.number import PyNumber_Float

OPERAND_LIMIT = cython.declare(cython.Py_ssize_t, OPERAND_CAPACITY)
NUMBER_LIMIT = cython.declare(cython.Py_ssize_t, 16)

IS_SPACE = cython.declare(cython.bint[256])
IS_DELIM = cython.declare(cython.bint[256])
IS_NUMERIC_START = cython.declare(cython.bint[256])
IS_DIGIT = cython.declare(cython.bint[256])

HEX_NIBBLE = cython.declare(cython.uchar[256])

_i = cython.declare(cython.int)
for _i in range(256):
    HEX_NIBBLE[_i] = 255
for _i in range(10):
    HEX_NIBBLE[0x30 + _i] = _i
for _i in range(6):
    HEX_NIBBLE[0x41 + _i] = 10 + _i
    HEX_NIBBLE[0x61 + _i] = 10 + _i
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


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def path_operator(word: cython.p_const_uchar, length: cython.Py_ssize_t) -> cython.int:
    if length == 1:
        if (
            word[0] == PATH_M
            or word[0] == PATH_L
            or word[0] == PATH_C
            or word[0] == PATH_V
            or word[0] == PATH_Y
            or word[0] == PATH_H
        ):
            return word[0]
        return NOT_PATH
    if length == 2 and word[0] == 0x72 and word[1] == 0x65:
        return PATH_RE
    return NOT_PATH


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def path_operand_count(op: cython.int) -> cython.int:
    if op == PATH_M or op == PATH_L:
        return 2
    if op == PATH_C:
        return 6
    if op == PATH_V or op == PATH_Y or op == PATH_RE:
        return 4
    return 0


@cython.cclass
class ContentScanner:
    view: cython.const[cython.uchar][::1]
    buf: cython.p_const_uchar
    size: cython.Py_ssize_t
    cursor: cython.Py_ssize_t
    keywords: object
    object_keywords: object
    make_name: object
    names: dict
    operators: dict
    operands = cython.declare(list, visibility="readonly")
    string_type: object

    pending: cython.Py_ssize_t
    pending_value: cython.double[OPERAND_CAPACITY]
    pending_start: cython.Py_ssize_t[OPERAND_CAPACITY]
    pending_end: cython.Py_ssize_t[OPERAND_CAPACITY]
    pending_integer: cython.bint[OPERAND_CAPACITY]

    path_state: object
    paths_loaded: cython.bint
    paths_declined: cython.bint
    paths_changed: cython.bint
    path_ops: object
    path_coords: array.array
    current_set: cython.bint
    current_x: cython.double
    current_y: cython.double
    start_set: cython.bint
    start_x: cython.double
    start_y: cython.double
    curve_loaded: cython.bint
    curve_matrix: cython.double[4]
    flatness: cython.double

    def __cinit__(self, data, keywords, object_keywords, make_name):
        self.view = data
        self.size = self.view.shape[0]
        self.buf = cython.address(self.view[0]) if self.size else cython.NULL
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
        self.string_type = None

    @property
    def pos(self):
        return self.cursor

    @pos.setter
    def pos(self, value: cython.Py_ssize_t):
        self.cursor = value

    def set_path_state(self, state):
        self.flush_paths()
        self.path_state = state

    @cython.cfunc
    @cython.inline
    @cython.nogil
    @cython.exceptval(check=False)
    def skip_ignored(self, p: cython.Py_ssize_t) -> cython.Py_ssize_t:
        b: cython.p_const_uchar = self.buf
        n: cython.Py_ssize_t = self.size
        while p < n:
            if IS_SPACE[b[p]]:
                p += 1
            elif b[p] == 0x25:
                while p < n and b[p] != 0x0D and b[p] != 0x0A:
                    p += 1
            else:
                break
        return p

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def materialize(self) -> cython.int:
        i: cython.Py_ssize_t
        word: bytes
        value: object
        for i in range(self.pending):
            if self.pending_integer[i]:
                word = PyBytes_FromStringAndSize(
                    cython.cast(
                        cython.p_const_char,
                        (self.buf + self.pending_start[i]),
                    ),
                    self.pending_end[i] - self.pending_start[i],
                )
                value = PyLong_FromString(word, cython.NULL, 10)
            else:
                value = self.pending_value[i]
            PyList_Append(self.operands, value)
        self.pending = 0
        return 0

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def load_point(self, point: object, x: cython.p_double, y: cython.p_double) -> cython.bint:
        if type(point) is not tuple or len(point) != 2:
            return False
        first = point[0]
        second = point[1]
        if type(first) is not float or type(second) is not float:
            return False
        x[0] = PyFloat_AsDouble(first)
        y[0] = PyFloat_AsDouble(second)
        return True

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def load_paths(self) -> cython.int:
        state = self.path_state
        self.paths_loaded = True
        self.paths_declined = False
        self.paths_changed = False
        self.curve_loaded = False
        path = state.current_path
        ops = path.ops
        coords = path.coords
        if (
            type(ops) is not bytearray
            or not isinstance(coords, array.array)
            or coords.typecode != "d"
        ):
            self.paths_declined = True
            return 0
        self.path_ops = ops
        self.path_coords = coords
        current = state.current_point
        start = state.subpath_start
        self.current_set = current is not None
        self.start_set = start is not None
        if self.current_set and not self.load_point(
            current, cython.address(self.current_x), cython.address(self.current_y)
        ):
            self.paths_declined = True
        if self.start_set and not self.load_point(
            start, cython.address(self.start_x), cython.address(self.start_y)
        ):
            self.paths_declined = True
        return 0

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def load_curve(self) -> cython.bint:
        graphics = self.path_state.graphics
        ctm = graphics.ctm
        i: cython.Py_ssize_t
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

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def flush_paths(self) -> cython.int:
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

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def append_op(self, op: cython.uchar) -> cython.int:
        n: cython.Py_ssize_t = PyByteArray_GET_SIZE(self.path_ops)
        PyByteArray_Resize(self.path_ops, n + 1)
        PyByteArray_AS_STRING(self.path_ops)[n] = cython.cast(cython.char, op)
        return 0

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def append_coords(self, values: cython.p_const_double, count: cython.Py_ssize_t) -> cython.int:
        coords: array.array = self.path_coords
        n: cython.Py_ssize_t = len(coords)
        i: cython.Py_ssize_t
        array.resize_smart(coords, n + count)
        for i in range(count):
            coords.data.as_doubles[n + i] = values[i]
        return 0

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def apply_path(self, op: cython.int) -> cython.bint:
        v: cython.p_double = self.pending_value
        curve = cython.declare(cython.double[13])
        i: cython.Py_ssize_t
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

    def enable_strings(self, string_type):
        """Parse plain literal and hex strings as string_type(data, is_literal=...)."""
        self.string_type = string_type

    @cython.cfunc
    def scan_string(self, p: cython.Py_ssize_t, end_out: cython.p_Py_ssize_t) -> object:
        # Only strings every lexer reads the same way: a literal string with no
        # escape, nesting or line end, or an even run of hex digits. None otherwise.
        b: cython.p_const_uchar = self.buf
        n: cython.Py_ssize_t = self.size
        q: cython.Py_ssize_t = p + 1
        c: cython.uchar
        data: bytes
        out: cython.p_char
        i: cython.Py_ssize_t
        if self.string_type is None:
            return None
        if b[p] == 0x28:
            while q < n:
                c = b[q]
                if c == 0x29:
                    end_out[0] = q + 1
                    data = PyBytes_FromStringAndSize(
                        cython.cast(cython.p_const_char, (b + p + 1)),
                        q - p - 1,
                    )
                    return self.string_type(data, is_literal=True)
                if c == 0x28 or c == 0x5C or c == 0x0D or c == 0x0A:
                    return None
                q += 1
            return None
        if b[p] != 0x3C or (q < n and b[q] == 0x3C):
            return None
        while q < n and HEX_NIBBLE[b[q]] != 255:
            q += 1
        if q >= n or b[q] != 0x3E or (q - p - 1) & 1:
            return None
        data = PyBytes_FromStringAndSize(cython.NULL, (q - p - 1) // 2)
        out = PyBytes_AS_STRING(data)
        for i in range((q - p - 1) // 2):
            out[i] = cython.cast(
                cython.char,
                ((HEX_NIBBLE[b[p + 1 + 2 * i]] << 4) | HEX_NIBBLE[b[p + 2 + 2 * i]]),
            )
        end_out[0] = q + 1
        return self.string_type(data, is_literal=False)

    @cython.cfunc
    def scan_numeric_array(self, p: cython.Py_ssize_t) -> object:
        b: cython.p_const_uchar = self.buf
        n: cython.Py_ssize_t = self.size
        start: cython.Py_ssize_t
        digits_before: cython.Py_ssize_t
        digits_after: cython.Py_ssize_t
        has_dot: cython.bint
        end = cython.declare(cython.p_char)
        word: bytes
        value: object
        string_end: cython.Py_ssize_t = 0
        values: list = []
        p += 1
        while True:
            while p < n and IS_SPACE[b[p]]:
                p += 1
            if p >= n:
                return None
            if b[p] == 0x5D:
                self.cursor = p + 1
                return values
            if b[p] == 0x28 or b[p] == 0x3C:
                value = self.scan_string(p, cython.address(string_end))
                if value is None:
                    return None
                PyList_Append(values, value)
                p = string_end
                continue
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
            if p >= n or not (IS_SPACE[b[p]] or b[p] == 0x5D or b[p] == 0x28 or b[p] == 0x3C):
                return None
            if p - start >= NUMBER_LIMIT:
                return None
            if has_dot:
                PyList_Append(
                    values,
                    PyOS_string_to_double(
                        cython.cast(cython.p_const_char, (b + start)),
                        cython.address(end),
                        None,
                    ),
                )
            else:
                word = PyBytes_FromStringAndSize(
                    cython.cast(cython.p_const_char, (b + start)),
                    p - start,
                )
                PyList_Append(values, PyLong_FromString(word, cython.NULL, 10))

    def run(
        self,
        lexer,
        handlers: typing.Optional[dict],
        fallback,
        state,
        depth,
        keep_paint: typing.Optional[frozenset],
        line_moving: typing.Optional[frozenset],
        keep_layout: typing.Optional[frozenset],
    ):
        """Scan and dispatch operations until a handler returns a child frame.

        fallback parses a token the scanner hands back: it returns the next
        operation, False to resume scanning, or None at the end of the stream.
        lexer.pos is kept at the end of the current operation.
        """
        result: object
        name: object
        operands: object
        handler: object
        child: object
        layout: object
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

    @cython.cfunc
    def scan_operation(self) -> object:
        b: cython.p_const_uchar = self.buf
        n: cython.Py_ssize_t = self.size
        p: cython.Py_ssize_t
        start: cython.Py_ssize_t
        digits_before: cython.Py_ssize_t
        digits_after: cython.Py_ssize_t
        c: cython.uchar
        has_dot: cython.bint
        end = cython.declare(cython.p_char)
        word: bytes
        value: object
        name: object
        arguments: tuple
        operands: list = self.operands
        integer: cython.longlong
        q = cython.declare(cython.Py_ssize_t)
        slot: cython.Py_ssize_t
        op: cython.int
        negative: cython.bint

        while True:
            with cython.nogil:
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
                            cython.cast(cython.p_const_char, (b + start)),
                            cython.address(end),
                            None,
                        )
                    else:
                        negative = b[start] == 0x2D
                        integer = 0
                        for q in range(start + (1 if b[start] == 0x2B or negative else 0), p):
                            integer = integer * 10 + (b[q] - 0x30)
                        self.pending_value[slot] = cython.cast(
                            cython.double, (-integer if negative else integer)
                        )
                    self.pending += 1
                continue

            if c == 0x2F:
                start = p
                p += 1
                while p < n and not (IS_DELIM[b[p]] or b[p] == 0x23):
                    p += 1
                if p < n and not IS_DELIM[b[p]]:
                    return start
                word = PyBytes_FromStringAndSize(
                    cython.cast(cython.p_const_char, (b + start)),
                    p - start,
                )
                self.cursor = p
                value = self.names.get(word)
                if value is None:
                    value = self.names[word] = self.make_name(word[1:])
                if PyList_GET_SIZE(operands) + self.pending < OPERAND_LIMIT:
                    if self.pending:
                        self.materialize()
                    PyList_Append(operands, value)
                continue

            if c == 0x28 or c == 0x3C:
                value = self.scan_string(p, cython.address(q))
                if value is None:
                    return p
                self.cursor = q
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
            word = PyBytes_FromStringAndSize(
                cython.cast(cython.p_const_char, (b + start)),
                p - start,
            )
            if word in self.keywords:
                return start
            self.cursor = p
            if self.pending:
                self.materialize()
            name = self.operators.get(word)
            if name is None:
                name = self.operators[word] = word.decode("latin-1")
            arguments = PyList_AsTuple(operands)
            PyList_SetSlice(operands, 0, PyList_GET_SIZE(operands), cython.NULL)
            if name in self.object_keywords:
                continue
            return name, arguments

# SPDX-License-Identifier: AGPL-3.0-only

from cpython.object cimport PyObject


cdef extern from "Python.h":
    double PyOS_string_to_double(
        const char* s, char** endptr, object overflow_exception
    ) except? -1.0
    int PyList_SetSlice(object list, Py_ssize_t low, Py_ssize_t high, PyObject* items) except -1

cdef enum:
    OPERAND_CAPACITY = 16

cdef enum:
    NOT_PATH = 0
    PATH_M = 0x6D
    PATH_L = 0x6C
    PATH_C = 0x63
    PATH_V = 0x76
    PATH_Y = 0x79
    PATH_H = 0x68
    PATH_RE = 0x72

# Operators applied to a capture state without calling their handlers.
cdef enum:
    NOT_NATIVE = 0
    NATIVE_STROKE_GRAY = 1
    NATIVE_FILL_GRAY = 2
    NATIVE_STROKE_RGB = 3
    NATIVE_FILL_RGB = 4
    NATIVE_STROKE_CMYK = 5
    NATIVE_FILL_CMYK = 6
    NATIVE_LINE_WIDTH = 7
    NATIVE_DASH = 8
    NATIVE_LINE_CAP = 9
    NATIVE_LINE_JOIN = 10
    NATIVE_MITER_LIMIT = 11
    NATIVE_MOVE_TEXT = 12
    NATIVE_MOVE_TEXT_LEADING = 13
    NATIVE_TEXT_MATRIX = 14
    NATIVE_NEXT_LINE = 15
    NATIVE_LEADING = 16
    NATIVE_SAVE = 17
    NATIVE_RESTORE = 18
    NATIVE_CONCAT = 19
    NATIVE_COUNT = 20

# What the dispatch loop does to the glyph paint and text layout before a handler.
cdef enum:
    CLEAR_PAINT = 0
    RESET_LINE_STYLE = 1
    CLEAR_LAYOUT = 2
    KEEP_LAYOUT = 3

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

# SPDX-License-Identifier: AGPL-3.0-only

from cpython.object cimport PyObject


cdef extern from "Python.h":
    object PyLong_FromString(const char *text, char **end, int base)
    double PyOS_string_to_double(
        const char *text, char **end, PyObject *overflow_exception
    ) except? -1.0


cdef enum:
    MAX_DEPTH = 64
    FAST_INT_DIGITS = 18

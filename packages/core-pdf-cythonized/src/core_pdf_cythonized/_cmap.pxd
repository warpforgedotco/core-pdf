# SPDX-License-Identifier: AGPL-3.0-only

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

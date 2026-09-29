# SPDX-License-Identifier: AGPL-3.0-only

cdef extern from "Python.h":
    ctypedef struct PyMemberDef:
        const char* name
        int type
        Py_ssize_t offset
    ctypedef struct PyMemberDescrObject:
        PyMemberDef* d_member
    int Py_T_OBJECT_EX


# DecodedGlyph fields, in this order.
cdef enum:
    G_CODE_BYTES = 0
    G_WIDTH_CODE = 1
    G_UNICODE = 2
    G_BITMAP_CODE = 3
    G_SPLIT_UNICODE = 4
    G_UNICODE_SOURCE = 5
    G_ALTERNATES = 6
    G_CHAR_CODE = 7
    G_CID = 8
    G_GID = 9

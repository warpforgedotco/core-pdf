# SPDX-License-Identifier: AGPL-3.0-only

cdef extern from "math.h" nogil:
    double sqrt(double)
    double fabs(double)

cdef enum:
    T1_STACK = 256
    T1_MAX_DEPTH = 64

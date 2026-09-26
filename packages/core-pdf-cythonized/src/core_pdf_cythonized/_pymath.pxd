# SPDX-License-Identifier: AGPL-3.0-only

from libc.math cimport isinf, isnan


cdef inline double py_max(double a, double b) noexcept nogil:
    return b if b > a else a


cdef inline double py_min(double a, double b) noexcept nogil:
    return b if b < a else a


cdef inline int raise_not_integral(bint nan) except -1:
    if nan:
        raise ValueError("cannot convert float NaN to integer")
    raise OverflowError("cannot convert float infinity to integer")


cdef inline int check_integral(double value) except -1:
    if isnan(value):
        return raise_not_integral(True)
    if isinf(value):
        return raise_not_integral(False)
    return 0

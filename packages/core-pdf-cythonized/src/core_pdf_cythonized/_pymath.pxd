# SPDX-License-Identifier: AGPL-3.0-only
# Python's own semantics for a few float operations the kernels replace:
# max and min with their tie-breaking, and the errors math.floor and
# math.ceil raise on a float with no integer value. Shared so every kernel
# reproduces them the same way.

from libc.math cimport isinf, isnan


cdef inline double py_max(double a, double b) noexcept nogil:
    # Python's max(a, b): a unless b is strictly greater.
    return b if b > a else a


cdef inline double py_min(double a, double b) noexcept nogil:
    # Python's min(a, b): a unless b is strictly less.
    return b if b < a else a


cdef inline int raise_not_integral(bint nan) except -1:
    # What math.floor and math.ceil raise converting a NaN or an infinity.
    if nan:
        raise ValueError("cannot convert float NaN to integer")
    raise OverflowError("cannot convert float infinity to integer")


cdef inline int check_integral(double value) except -1:
    if isnan(value):
        return raise_not_integral(True)
    if isinf(value):
        return raise_not_integral(False)
    return 0

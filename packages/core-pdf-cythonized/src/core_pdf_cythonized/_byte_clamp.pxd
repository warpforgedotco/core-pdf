# SPDX-License-Identifier: AGPL-3.0-only

from libc.math cimport rint


cdef inline unsigned char double_to_byte(double value) noexcept nogil:
    if value < 0.0:
        return 0
    if value > 255.0:
        return 255
    return <unsigned char> <int> value


cdef inline unsigned char float_to_byte(float value) noexcept nogil:
    cdef float ZERO = 0.0
    cdef float SCALE = 255.0
    if value < ZERO:
        return 0
    if value > SCALE:
        return 255
    return <unsigned char> <int> value


cdef inline int round_to_byte(double value) noexcept nogil:
    cdef double rounded = rint(value)
    if rounded < 0.0:
        return 0
    if rounded > 255.0:
        return 255
    return <int> rounded


cdef inline double unit_to_byte(double value) noexcept nogil:
    value = rint(value * 255.0)
    if value < 0.0:
        return 0.0
    if value > 255.0:
        return 255.0
    return value


cdef inline int unit_to_byte_checked(double value) except -1:
    cdef double scaled = rint(value * 255.0)
    if scaled < 0.0 or scaled > 255.0:
        raise ValueError("source_alpha must lie in [0, 1]")
    return <int> scaled

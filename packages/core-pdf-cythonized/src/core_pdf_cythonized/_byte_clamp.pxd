# SPDX-License-Identifier: AGPL-3.0-only
# Clamping a colour or alpha value into a byte, once per meaning, so the
# kernels that write pixels share one copy of each rather than several copies
# that differ only in name. Each keeps the exact comparison and conversion its
# callers were pinned with: which of them rounds, in which precision, and
# whether the input is a unit fraction or already on the 0..255 scale.

from libc.math cimport rint


cdef inline unsigned char double_to_byte(double value) noexcept nogil:
    # numpy.clip(v, 0, 255).astype(uint8) on an already rounded double.
    if value < 0.0:
        return 0
    if value > 255.0:
        return 255
    return <unsigned char> <int> value


cdef inline unsigned char float_to_byte(float value) noexcept nogil:
    # numpy.clip(v, 0, 255).astype(uint8) on an already rounded float32.
    cdef float ZERO = 0.0
    cdef float SCALE = 255.0
    if value < ZERO:
        return 0
    if value > SCALE:
        return 255
    return <unsigned char> <int> value


cdef inline int round_to_byte(double value) noexcept nogil:
    # max(0, min(255, round(value))) on the 0..255 scale, as Python rounds.
    cdef double rounded = rint(value)
    if rounded < 0.0:
        return 0
    if rounded > 255.0:
        return 255
    return <int> rounded


cdef inline double unit_to_byte(double value) noexcept nogil:
    # numpy.clip(numpy.rint(v * 255), 0, 255) for a unit fraction, still a
    # double for the caller's uint8 cast.
    value = rint(value * 255.0)
    if value < 0.0:
        return 0.0
    if value > 255.0:
        return 255.0
    return value

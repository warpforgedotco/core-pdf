# SPDX-License-Identifier: AGPL-3.0-only

import numpy

__all__ = ("object_headers_match",)

cdef bint IS_SPACE[256]
cdef bint IS_DELIM[256]
cdef int _i
for _i in range(256):
    IS_SPACE[_i] = 0
    IS_DELIM[_i] = 0
for _i in (0x00, 0x09, 0x0A, 0x0C, 0x0D, 0x20):
    IS_SPACE[_i] = 1
    IS_DELIM[_i] = 1
for _i in (0x28, 0x29, 0x3C, 0x3E, 0x5B, 0x5D, 0x7B, 0x7D, 0x2F, 0x25):
    IS_DELIM[_i] = 1

cdef long long LARGEST = 9223372036854775807


cdef inline Py_ssize_t read_number(
    const unsigned char* b, Py_ssize_t p, Py_ssize_t n, long long* value
) noexcept nogil:
    cdef Py_ssize_t start = p
    cdef long long total = 0
    cdef int digit
    cdef bint overflowed = False
    while p < n and 0x30 <= b[p] <= 0x39:
        digit = b[p] - 0x30
        if not overflowed:
            if total > (LARGEST - digit) // 10:
                overflowed = True
            else:
                total = total * 10 + digit
        p += 1
    if p == start:
        return -1
    value[0] = -1 if overflowed else total
    return p


cdef enum:
    DIFFERENT = 0
    SAME = 1
    UNDECIDED = 2

cdef long long HUGE = -2


cdef inline unsigned char header_state(
    const unsigned char* b, Py_ssize_t n, long long offset, long long number, long long generation
) noexcept nogil:
    cdef Py_ssize_t p, q
    cdef long long found_number, found_generation
    if offset < 0 or offset >= n:
        return DIFFERENT
    p = read_number(b, offset, n, &found_number)
    if p < 0 or p >= n or not IS_SPACE[b[p]]:
        return DIFFERENT
    while p < n and IS_SPACE[b[p]]:
        p += 1
    p = read_number(b, p, n, &found_generation)
    if p < 0 or p >= n or not IS_SPACE[b[p]]:
        return DIFFERENT
    while p < n and IS_SPACE[b[p]]:
        p += 1
    if p + 3 > n or b[p] != 0x6F or b[p + 1] != 0x62 or b[p + 2] != 0x6A:
        return DIFFERENT
    q = p + 3
    if q < n and not IS_DELIM[b[q]]:
        return DIFFERENT
    if found_generation != generation:
        return DIFFERENT
    if number == HUGE:
        return UNDECIDED if found_number == -1 else DIFFERENT
    return SAME if found_number == number else DIFFERENT


def object_headers_match(
    const unsigned char[::1] data,
    const long long[::1] numbers,
    const long long[::1] generations,
    const long long[::1] offsets,
):
    cdef Py_ssize_t count = numbers.shape[0], i
    if generations.shape[0] != count or offsets.shape[0] != count:
        raise ValueError("numbers, generations and offsets differ in length")
    result = numpy.zeros(count, dtype=numpy.uint8)
    cdef unsigned char[::1] out = result
    cdef Py_ssize_t n = data.shape[0]
    if n == 0:
        return result
    cdef const unsigned char* b = &data[0]
    with nogil:
        for i in range(count):
            out[i] = header_state(b, n, offsets[i], numbers[i], generations[i])
    return result

# SPDX-License-Identifier: AGPL-3.0-only
"""Checking xref offsets against object headers (core_pdf.impl.document_document).

Opening a document checks that every in-use xref entry's offset points at
its own "N G obj" header, and repairs the ones that do not. That was one
regular-expression match per entry -- a match object, two int() calls and a
comparison, 471,099 of them over the corpus -- for a question with a yes on
nearly every entry. This answers it for all of them in one pass.

The expression it replaces was

    (\\d+)[\\0\\t\\n\\f\\r ]+(\\d+)[\\0\\t\\n\\f\\r ]+obj(?=[\\0\\t\\n\\f\\r ()<>\\[\\]{}/%]|\\Z)

anchored at the offset. Every part of it is a maximal run -- a shorter digit
run leaves a digit where whitespace must follow, a shorter whitespace run
leaves whitespace where a digit or "obj" must -- so it matches exactly when
the greedy scan below does, and its groups are the runs the scan finds. The
groups were compared as int(): leading zeros do not count, and a run too
large for 63 bits cannot equal a generation, which is at most 65535, and can
equal an object number only if that number is as large: the caller passes
such numbers as -2, and settles the few headers this cannot by comparing
digits.
"""

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
# ( ) < > [ ] { } / %
for _i in (0x28, 0x29, 0x3C, 0x3E, 0x5B, 0x5D, 0x7B, 0x7D, 0x2F, 0x25):
    IS_DELIM[_i] = 1

cdef long long LARGEST = 9223372036854775807


cdef inline Py_ssize_t read_number(
    const unsigned char* b, Py_ssize_t p, Py_ssize_t n, long long* value
) noexcept nogil:
    """Read a maximal digit run at p into value; the end of the run, or -1 if none.

    value is what int() makes of the run, or -1 when that exceeds 63 bits,
    which only an object number passed as too large can equal.
    """
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


# What header_state answers.
cdef enum:
    DIFFERENT = 0
    SAME = 1
    UNDECIDED = 2

# An object number too large for 63 bits, as the caller passes it.
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
    if p + 3 > n or b[p] != 0x6F or b[p + 1] != 0x62 or b[p + 2] != 0x6A:  # obj
        return DIFFERENT
    q = p + 3
    if q < n and not IS_DELIM[b[q]]:
        return DIFFERENT
    if found_generation != generation:
        return DIFFERENT
    if number == HUGE:
        # Only a run too long to hold can equal it, and then only by digits.
        return UNDECIDED if found_number == -1 else DIFFERENT
    return SAME if found_number == number else DIFFERENT


def object_headers_match(
    const unsigned char[::1] data,
    const long long[::1] numbers,
    const long long[::1] generations,
    const long long[::1] offsets,
):
    """For each entry, whether its object header starts at its offset.

    ``numbers`` holds each entry's object number, or -2 where it needs more
    than 63 bits; ``generations`` its generation. Returns a uint8 array: 1
    where the header is there, 0 where it is not, and 2 where the header is
    there with an object number too long to compare here, which the caller
    compares by its digits.
    """
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

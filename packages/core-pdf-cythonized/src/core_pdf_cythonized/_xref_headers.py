# SPDX-License-Identifier: AGPL-3.0-only

import cython
import numpy
from cython.cimports.core_pdf_cythonized._xref_headers import DIFFERENT, SAME, UNDECIDED

__all__ = ("object_headers_match",)

IS_SPACE = cython.declare(cython.bint[256])
IS_DELIM = cython.declare(cython.bint[256])
_i = cython.declare(cython.int)
for _i in range(256):
    IS_SPACE[_i] = 0
    IS_DELIM[_i] = 0
for _i in (0x00, 0x09, 0x0A, 0x0C, 0x0D, 0x20):
    IS_SPACE[_i] = 1
    IS_DELIM[_i] = 1
for _i in (0x28, 0x29, 0x3C, 0x3E, 0x5B, 0x5D, 0x7B, 0x7D, 0x2F, 0x25):
    IS_DELIM[_i] = 1

LARGEST = cython.declare(cython.longlong, 9223372036854775807)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def read_number(
    b: cython.p_const_uchar,
    p: cython.Py_ssize_t,
    n: cython.Py_ssize_t,
    value: cython.p_longlong,
) -> cython.Py_ssize_t:
    start: cython.Py_ssize_t = p
    total: cython.longlong = 0
    digit: cython.int
    overflowed: cython.bint = False
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


HUGE = cython.declare(cython.longlong, -2)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def header_state(
    b: cython.p_const_uchar,
    n: cython.Py_ssize_t,
    offset: cython.longlong,
    number: cython.longlong,
    generation: cython.longlong,
) -> cython.uchar:
    p: cython.Py_ssize_t
    q: cython.Py_ssize_t
    found_number = cython.declare(cython.longlong)
    found_generation = cython.declare(cython.longlong)
    if offset < 0 or offset >= n:
        return DIFFERENT
    p = read_number(b, offset, n, cython.address(found_number))
    if p < 0 or p >= n or not IS_SPACE[b[p]]:
        return DIFFERENT
    while p < n and IS_SPACE[b[p]]:
        p += 1
    p = read_number(b, p, n, cython.address(found_generation))
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
    data: cython.const[cython.uchar][::1],
    numbers: cython.const[cython.longlong][::1],
    generations: cython.const[cython.longlong][::1],
    offsets: cython.const[cython.longlong][::1],
):
    count: cython.Py_ssize_t = numbers.shape[0]
    i: cython.Py_ssize_t
    if generations.shape[0] != count or offsets.shape[0] != count:
        raise ValueError("numbers, generations and offsets differ in length")
    result = numpy.zeros(count, dtype=numpy.uint8)
    out: cython.uchar[::1] = result
    n: cython.Py_ssize_t = data.shape[0]
    if n == 0:
        return result
    b: cython.p_const_uchar = cython.address(data[0])
    with cython.nogil:
        for i in range(count):
            out[i] = header_state(b, n, offsets[i], numbers[i], generations[i])
    return result

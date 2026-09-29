# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc
from cython.cimports.libc.string import memchr, memcmp


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def anchor_of(needle: cython.p_const_uchar, length: cython.Py_ssize_t) -> cython.Py_ssize_t:
    # "/" opens every PDF name, so a needle is found by a byte after it.
    index: cython.Py_ssize_t
    for index in range(length):
        if needle[index] != 0x2F:
            return index
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def find_needle(
    data: cython.p_const_uchar,
    start: cython.Py_ssize_t,
    end: cython.Py_ssize_t,
    needle: cython.p_const_uchar,
    length: cython.Py_ssize_t,
    anchor: cython.Py_ssize_t,
) -> cython.Py_ssize_t:
    """The first position p >= start with the needle wholly in [p, end), or -1."""
    hit: cython.p_const_uchar
    cursor: cython.Py_ssize_t = start + anchor
    last: cython.Py_ssize_t = end - length + anchor
    key: cython.uchar = needle[anchor]
    while cursor <= last:
        hit = cython.cast(
            cython.p_const_uchar,
            memchr(data + cursor, key, cython.cast(cython.size_t, last - cursor + 1)),
        )
        if hit == cython.NULL:
            return -1
        cursor = hit - data
        if memcmp(data + cursor - anchor, needle, cython.cast(cython.size_t, length)) == 0:
            return cursor - anchor
        cursor += 1
    return -1


def markers_present(data: cython.const[cython.uchar][::1], needles: tuple):
    """For each needle, whether it occurs anywhere in data."""
    n: cython.Py_ssize_t = data.shape[0]
    length: cython.Py_ssize_t
    buffer: cython.p_const_uchar = cython.address(data[0]) if n else cython.NULL
    needle_bytes: cython.p_const_uchar
    present: list = []
    needle: bytes
    for needle in needles:
        length = len(needle)
        if length == 0:
            raise ValueError("needles must not be empty")
        needle_bytes = cython.cast(cython.p_const_uchar, needle)
        present.append(
            buffer is not cython.NULL
            and find_needle(buffer, 0, n, needle_bytes, length, anchor_of(needle_bytes, length))
            >= 0
        )
    return tuple(present)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def region_of(
    starts: cython.p_const_Py_ssize_t,
    count: cython.Py_ssize_t,
    position: cython.Py_ssize_t,
) -> cython.Py_ssize_t:
    # The last region starting at or before position.
    low: cython.Py_ssize_t = 0
    high: cython.Py_ssize_t = count
    middle: cython.Py_ssize_t
    while low < high:
        middle = (low + high) >> 1
        if starts[middle] <= position:
            low = middle + 1
        else:
            high = middle
    return low - 1


def regions_with_markers(
    data: cython.const[cython.uchar][::1],
    needles: tuple,
    starts,
    data_len: cython.Py_ssize_t,
):
    """For each needle, the indexes of the regions [starts[i], next start or data_len)
    that hold a whole occurrence of it. starts must be ascending."""
    count: cython.Py_ssize_t = len(starts)
    index: cython.Py_ssize_t
    position: cython.Py_ssize_t
    region: cython.Py_ssize_t
    end: cython.Py_ssize_t
    length: cython.Py_ssize_t
    anchor: cython.Py_ssize_t
    found: list = []
    regions: set
    needle: bytes
    needle_bytes: cython.p_const_uchar
    n: cython.Py_ssize_t = data.shape[0] if data.shape[0] < data_len else data_len
    buffer: cython.p_const_uchar = cython.address(data[0]) if data.shape[0] else cython.NULL
    offsets: cython.p_Py_ssize_t = cython.cast(
        cython.p_Py_ssize_t, PyMem_Malloc((count or 1) * cython.sizeof(cython.Py_ssize_t))
    )
    if offsets == cython.NULL:
        raise MemoryError
    try:
        for index in range(count):
            offsets[index] = starts[index]
            if offsets[index] < 0:
                raise ValueError("starts must not be negative")
        for needle in needles:
            length = len(needle)
            if length == 0:
                raise ValueError("needles must not be empty")
            needle_bytes = cython.cast(cython.p_const_uchar, needle)
            anchor = anchor_of(needle_bytes, length)
            regions = set()
            found.append(regions)
            if count == 0 or buffer is cython.NULL:
                continue
            position = offsets[0]
            while True:
                position = find_needle(buffer, position, n, needle_bytes, length, anchor)
                if position < 0:
                    break
                region = region_of(offsets, count, position)
                end = offsets[region + 1] if region + 1 < count else data_len
                if position + length <= end:
                    regions.add(region)
                    position = end
                else:
                    position += 1
    finally:
        PyMem_Free(offsets)
    return found

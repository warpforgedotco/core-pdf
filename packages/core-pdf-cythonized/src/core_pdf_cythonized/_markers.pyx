# SPDX-License-Identifier: AGPL-3.0-only

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.string cimport memchr, memcmp


cdef inline Py_ssize_t anchor_of(const unsigned char* needle, Py_ssize_t length) noexcept nogil:
    # "/" opens every PDF name, so a needle is found by a byte after it.
    cdef Py_ssize_t index
    for index in range(length):
        if needle[index] != 0x2F:
            return index
    return 0


cdef Py_ssize_t find_needle(
    const unsigned char* data,
    Py_ssize_t start,
    Py_ssize_t end,
    const unsigned char* needle,
    Py_ssize_t length,
    Py_ssize_t anchor,
) noexcept nogil:
    """The first position p >= start with the needle wholly in [p, end), or -1."""
    cdef const unsigned char* hit
    cdef Py_ssize_t cursor = start + anchor
    cdef Py_ssize_t last = end - length + anchor
    cdef unsigned char key = needle[anchor]
    while cursor <= last:
        hit = <const unsigned char*> memchr(data + cursor, key, <size_t> (last - cursor + 1))
        if hit == NULL:
            return -1
        cursor = hit - data
        if memcmp(data + cursor - anchor, needle, <size_t> length) == 0:
            return cursor - anchor
        cursor += 1
    return -1


def markers_present(const unsigned char[::1] data, tuple needles):
    """For each needle, whether it occurs anywhere in data."""
    cdef Py_ssize_t n = data.shape[0], length
    cdef const unsigned char* buffer = &data[0] if n else NULL
    cdef const unsigned char* needle_bytes
    cdef list present = []
    cdef bytes needle
    for needle in needles:
        length = len(needle)
        if length == 0:
            raise ValueError("needles must not be empty")
        needle_bytes = <const unsigned char*> needle
        present.append(
            buffer is not NULL
            and find_needle(buffer, 0, n, needle_bytes, length, anchor_of(needle_bytes, length)) >= 0
        )
    return tuple(present)


cdef inline Py_ssize_t region_of(const Py_ssize_t* starts, Py_ssize_t count, Py_ssize_t position) noexcept nogil:
    # The last region starting at or before position.
    cdef Py_ssize_t low = 0, high = count, middle
    while low < high:
        middle = (low + high) >> 1
        if starts[middle] <= position:
            low = middle + 1
        else:
            high = middle
    return low - 1


def regions_with_markers(
    const unsigned char[::1] data, tuple needles, starts, Py_ssize_t data_len
):
    """For each needle, the indexes of the regions [starts[i], next start or data_len)
    that hold a whole occurrence of it. starts must be ascending."""
    cdef Py_ssize_t count = len(starts), index, position, region, end, length, anchor
    cdef list found = []
    cdef set regions
    cdef bytes needle
    cdef const unsigned char* needle_bytes
    cdef Py_ssize_t n = data.shape[0] if data.shape[0] < data_len else data_len
    cdef const unsigned char* buffer = &data[0] if data.shape[0] else NULL
    cdef Py_ssize_t* offsets = <Py_ssize_t*> PyMem_Malloc((count or 1) * sizeof(Py_ssize_t))
    if offsets == NULL:
        raise MemoryError
    try:
        for index in range(count):
            offsets[index] = starts[index]
        for needle in needles:
            length = len(needle)
            if length == 0:
                raise ValueError("needles must not be empty")
            needle_bytes = <const unsigned char*> needle
            anchor = anchor_of(needle_bytes, length)
            regions = set()
            found.append(regions)
            if count == 0 or buffer is NULL:
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

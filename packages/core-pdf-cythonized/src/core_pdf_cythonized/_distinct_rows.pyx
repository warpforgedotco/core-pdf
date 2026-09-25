# SPDX-License-Identifier: AGPL-3.0-only
"""Distinct colour rows, and the scatter back (core_pdf.impl.graphics.icc_profiles).

An ICC image conversion hands lcms every pixel, but a photograph uses few
of the colours it could: 908,510 distinct among 36.6 million pixels on
SCORE-Bench 153rd-Omaha-Pow-Wow, 66,646 among 10.6 million on another page.
lcms converts each pixel independently, so converting each distinct colour
once and scattering the results is the same image -- and lcms was 2.6 s of
that page. numpy.unique would sort the rows to find them, which alone took
over a second; this finds them with a hash table in one pass.

distinct_uint16_rows returns the distinct rows in first-seen order and each
row's index into them, or None once there are more than `limit` distinct
rows, when converting everything directly is cheaper than the scatter.
gather_uint8_rows is the scatter, taking the uint32 indices directly rather
than having numpy widen them to intp first.

Nothing here is arithmetic on colour values: rows are compared bit for bit,
and lcms still does every conversion.
"""

from libc.stdlib cimport free, malloc, realloc
from libc.stdint cimport int32_t, uint16_t, uint32_t, uint64_t
from libc.string cimport memcpy

import numpy

__all__ = ("distinct_uint16_rows", "gather_uint8_rows")

cdef uint64_t MULTIPLIER = 0x9E3779B97F4A7C15ULL


cdef inline uint64_t row_key(const uint16_t[:, ::1] samples, Py_ssize_t row, Py_ssize_t channels) noexcept nogil:
    cdef uint64_t key = 0
    cdef Py_ssize_t k
    for k in range(channels):
        key = (key << 16) | samples[row, k]
    return key


cdef inline Py_ssize_t slot_of(uint64_t key, int shift) noexcept nogil:
    return <Py_ssize_t> ((key * MULTIPLIER) >> shift)


def distinct_uint16_rows(const uint16_t[:, ::1] samples, Py_ssize_t limit):
    """(distinct rows, uint32 index of each row into them), or None past `limit` distinct."""
    cdef Py_ssize_t rows = samples.shape[0]
    cdef Py_ssize_t channels = samples.shape[1]
    if channels < 1 or channels > 4:
        raise ValueError("rows must have one to four channels")
    if rows >= 0xFFFFFFFF:
        raise ValueError("too many rows for uint32 indices")
    inverse = numpy.empty(rows, dtype=numpy.uint32)
    cdef uint32_t[::1] index_out = inverse
    cdef int shift = 64 - 16
    cdef Py_ssize_t capacity = 1 << 16
    cdef Py_ssize_t count = 0
    cdef uint64_t *keys = <uint64_t *> malloc(capacity * sizeof(uint64_t))
    cdef int32_t *slots = <int32_t *> malloc(capacity * sizeof(int32_t))
    cdef uint64_t *distinct = <uint64_t *> malloc((capacity // 2) * sizeof(uint64_t))
    cdef uint64_t *grown_keys
    cdef int32_t *grown_slots
    cdef uint64_t *grown_distinct
    cdef Py_ssize_t i, j, slot, old_capacity
    cdef uint64_t key
    cdef bint over_limit = False
    cdef bint out_of_memory = False
    if keys == NULL or slots == NULL or distinct == NULL:
        free(keys)
        free(slots)
        free(distinct)
        raise MemoryError()
    try:
        with nogil:
            for j in range(capacity):
                slots[j] = -1
            for i in range(rows):
                key = row_key(samples, i, channels)
                slot = slot_of(key, shift)
                while slots[slot] != -1 and keys[slot] != key:
                    slot = (slot + 1) & (capacity - 1)
                if slots[slot] != -1:
                    index_out[i] = <uint32_t> slots[slot]
                    continue
                if count >= limit:
                    over_limit = True
                    break
                keys[slot] = key
                slots[slot] = <int32_t> count
                distinct[count] = key
                index_out[i] = <uint32_t> count
                count += 1
                if count * 2 >= capacity:
                    # Grow and rehash, keeping the load at or under a half.
                    old_capacity = capacity
                    capacity *= 2
                    shift -= 1
                    grown_distinct = <uint64_t *> realloc(
                        distinct, (capacity // 2) * sizeof(uint64_t)
                    )
                    grown_keys = <uint64_t *> malloc(capacity * sizeof(uint64_t))
                    grown_slots = <int32_t *> malloc(capacity * sizeof(int32_t))
                    if grown_distinct == NULL or grown_keys == NULL or grown_slots == NULL:
                        if grown_distinct != NULL:
                            distinct = grown_distinct
                        free(grown_keys)
                        free(grown_slots)
                        out_of_memory = True
                        break
                    distinct = grown_distinct
                    free(keys)
                    free(slots)
                    keys = grown_keys
                    slots = grown_slots
                    for j in range(capacity):
                        slots[j] = -1
                    for j in range(count):
                        slot = slot_of(distinct[j], shift)
                        while slots[slot] != -1:
                            slot = (slot + 1) & (capacity - 1)
                        keys[slot] = distinct[j]
                        slots[slot] = <int32_t> j
        if out_of_memory:
            raise MemoryError()
        if over_limit:
            return None
        rows_out = numpy.empty((count, channels), dtype=numpy.uint16)
        unpack_keys(rows_out, distinct, count, channels)
        return rows_out, inverse
    finally:
        free(keys)
        free(slots)
        free(distinct)


cdef void unpack_keys(uint16_t[:, ::1] out, const uint64_t *keys, Py_ssize_t count, Py_ssize_t channels) noexcept:
    cdef Py_ssize_t i, k
    cdef uint64_t key
    for i in range(count):
        key = keys[i]
        for k in range(channels - 1, -1, -1):
            out[i, k] = <uint16_t> (key & 0xFFFF)
            key >>= 16


def gather_uint8_rows(const unsigned char[:, ::1] table, const uint32_t[::1] indices):
    """table[indices], row by row, into a new (len(indices), columns) uint8 array."""
    cdef Py_ssize_t count = indices.shape[0]
    cdef Py_ssize_t columns = table.shape[1]
    cdef Py_ssize_t available = table.shape[0]
    output = numpy.empty((count, columns), dtype=numpy.uint8)
    cdef unsigned char[:, ::1] out = output
    cdef Py_ssize_t i
    cdef uint32_t row
    cdef bint out_of_range = False
    if count == 0 or columns == 0:
        return output
    with nogil:
        for i in range(count):
            row = indices[i]
            if row >= available:
                out_of_range = True
                break
            memcpy(&out[i, 0], &table[row, 0], columns)
    if out_of_range:
        raise IndexError("row index out of range")
    return output

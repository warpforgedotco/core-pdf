# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
import numpy
from cython.cimports.libc.stdint import int32_t, uint16_t, uint32_t, uint64_t
from cython.cimports.libc.stdlib import free, malloc, realloc
from cython.cimports.libc.string import memcpy

__all__ = ("code_presence", "distinct_uint16_rows", "gather_uint8_rows")

MULTIPLIER = cython.declare(uint64_t, 0x9E3779B97F4A7C15)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def row_key(
    samples: cython.const[uint16_t][:, ::1], row: cython.Py_ssize_t, channels: cython.Py_ssize_t
) -> uint64_t:
    key: uint64_t = 0
    k: cython.Py_ssize_t
    for k in range(channels):
        key = (key << 16) | samples[row, k]
    return key


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def slot_of(key: uint64_t, shift: cython.int) -> cython.Py_ssize_t:
    return cython.cast(cython.Py_ssize_t, (key * MULTIPLIER) >> shift)


def distinct_uint16_rows(samples: cython.const[uint16_t][:, ::1], limit: cython.Py_ssize_t):
    rows: cython.Py_ssize_t = samples.shape[0]
    channels: cython.Py_ssize_t = samples.shape[1]
    if channels < 1 or channels > 4:
        raise ValueError("rows must have one to four channels")
    if rows >= 0xFFFFFFFF:
        raise ValueError("too many rows for uint32 indices")
    inverse = numpy.empty(rows, dtype=numpy.uint32)
    index_out: uint32_t[::1] = inverse
    shift: cython.int = 64 - 16
    capacity: cython.Py_ssize_t = 1 << 16
    count: cython.Py_ssize_t = 0
    keys: cython.pointer[uint64_t] = cython.cast(
        cython.pointer[uint64_t], malloc(capacity * cython.sizeof(uint64_t))
    )
    slots: cython.pointer[int32_t] = cython.cast(
        cython.pointer[int32_t], malloc(capacity * cython.sizeof(int32_t))
    )
    distinct: cython.pointer[uint64_t] = cython.cast(
        cython.pointer[uint64_t], malloc((capacity // 2) * cython.sizeof(uint64_t))
    )
    grown_keys: cython.pointer[uint64_t]
    grown_slots: cython.pointer[int32_t]
    grown_distinct: cython.pointer[uint64_t]
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    slot: cython.Py_ssize_t
    key: uint64_t
    over_limit: cython.bint = False
    out_of_memory: cython.bint = False
    if keys == cython.NULL or slots == cython.NULL or distinct == cython.NULL:
        free(keys)
        free(slots)
        free(distinct)
        raise MemoryError()
    try:
        with cython.nogil:
            for j in range(capacity):
                slots[j] = -1
            for i in range(rows):
                key = row_key(samples, i, channels)
                slot = slot_of(key, shift)
                while slots[slot] != -1 and keys[slot] != key:
                    slot = (slot + 1) & (capacity - 1)
                if slots[slot] != -1:
                    index_out[i] = cython.cast(uint32_t, slots[slot])
                    continue
                if count >= limit:
                    over_limit = True
                    break
                keys[slot] = key
                slots[slot] = cython.cast(int32_t, count)
                distinct[count] = key
                index_out[i] = cython.cast(uint32_t, count)
                count += 1
                if count * 2 >= capacity:
                    capacity *= 2
                    shift -= 1
                    grown_distinct = cython.cast(
                        cython.pointer[uint64_t],
                        realloc(distinct, (capacity // 2) * cython.sizeof(uint64_t)),
                    )
                    grown_keys = cython.cast(
                        cython.pointer[uint64_t], malloc(capacity * cython.sizeof(uint64_t))
                    )
                    grown_slots = cython.cast(
                        cython.pointer[int32_t], malloc(capacity * cython.sizeof(int32_t))
                    )
                    if (
                        grown_distinct == cython.NULL
                        or grown_keys == cython.NULL
                        or grown_slots == cython.NULL
                    ):
                        if grown_distinct != cython.NULL:
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
                        slots[slot] = cython.cast(int32_t, j)
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


@cython.cfunc
@cython.exceptval(check=False)
def unpack_keys(
    out: typing.Optional[uint16_t[:, ::1]],
    keys: cython.pointer[cython.const[uint64_t]],
    count: cython.Py_ssize_t,
    channels: cython.Py_ssize_t,
) -> cython.void:
    i: cython.Py_ssize_t
    k: cython.Py_ssize_t
    key: uint64_t
    for i in range(count):
        key = keys[i]
        for k in range(channels - 1, -1, -1):
            out[i, k] = cython.cast(uint16_t, key & 0xFFFF)
            key >>= 16


row_index = cython.fused_type(uint16_t, uint32_t)


def gather_uint8_rows(
    table: cython.const[cython.uchar][:, ::1], indices: cython.const[row_index][::1]
):
    count: cython.Py_ssize_t = indices.shape[0]
    columns: cython.Py_ssize_t = table.shape[1]
    available: cython.Py_ssize_t = table.shape[0]
    output = numpy.empty((count, columns), dtype=numpy.uint8)
    out: cython.uchar[:, ::1] = output
    i: cython.Py_ssize_t
    row: cython.Py_ssize_t
    out_of_range: cython.bint = False
    if count == 0 or columns == 0:
        return output
    source: cython.p_const_uchar = cython.address(table[0, 0])
    target: cython.p_uchar = cython.address(out[0, 0])
    with cython.nogil:
        for i in range(count):
            row = indices[i]
            if row >= available:
                out_of_range = True
                break
            if columns == 3:
                target[3 * i] = source[3 * row]
                target[3 * i + 1] = source[3 * row + 1]
                target[3 * i + 2] = source[3 * row + 2]
            elif columns == 1:
                target[i] = source[row]
            else:
                memcpy(target + columns * i, source + columns * row, columns)
    if out_of_range:
        raise IndexError("row index out of range")
    return output


def code_presence(codes: cython.const[uint16_t][::1]):
    present = numpy.zeros(65536, dtype=numpy.bool_)
    marks: cython.uchar[::1] = present.view(numpy.uint8)
    i: cython.Py_ssize_t
    count: cython.Py_ssize_t = codes.shape[0]
    largest: cython.int = -1
    code: uint16_t
    with cython.nogil:
        for i in range(count):
            code = codes[i]
            marks[code] = 1
            if code > largest:
                largest = code
    return present, largest

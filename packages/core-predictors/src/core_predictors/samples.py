# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from operator import index
from typing import Any

import numpy


def unpack_subbyte_rows(
    packed_rows: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    samples_per_row: int,
    bits_per_component: int,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    binary = numpy.unpackbits(packed_rows, axis=1, bitorder="big")[
        :, : samples_per_row * bits_per_component
    ]
    groups = binary.reshape(len(packed_rows), samples_per_row, bits_per_component)
    shifts = numpy.arange(bits_per_component - 1, -1, -1, dtype=numpy.uint8)
    return numpy.sum(groups << shifts, axis=2, dtype=numpy.uint8)


def uint8_view(
    buffer: bytes | bytearray | memoryview | numpy.ndarray[Any, Any],
    *,
    count: int = -1,
    offset: int = 0,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    count = index(count)
    offset = index(offset)
    if isinstance(buffer, numpy.ndarray):
        array = numpy.asarray(buffer)
        if array.dtype.type is numpy.uint8:
            if array.flags.c_contiguous:
                view = array.reshape(-1)
            else:
                view = numpy.ascontiguousarray(array).reshape(-1)
        else:
            view = numpy.asarray(array, dtype=numpy.uint8).reshape(-1)
        if offset < 0 or offset > view.size:
            raise ValueError("offset must be non-negative and no greater than buffer length")
        if count > view.size - offset:
            raise ValueError("buffer is smaller than requested size")
        if offset:
            view = view[offset:]
        if count >= 0:
            view = view[:count]
        return view
    return numpy.frombuffer(buffer, dtype=numpy.uint8, count=count, offset=offset)


__all__ = ("uint8_view", "unpack_subbyte_rows")

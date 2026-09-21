# SPDX-License-Identifier: AGPL-3.0-only
"""Sub-byte sample unpacking shared by TIFF differencing and image decoding."""

from __future__ import annotations

from typing import Any

import numpy


def unpack_subbyte_rows(
    packed_rows: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    samples_per_row: int,
    bits_per_component: int,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    """Unpack prepared byte rows into MSB-first samples, discarding row padding.

    Callers supply a two-dimensional uint8 array containing enough bytes for
    each row, normally with sample depths of 1, 2, or 4. Input validation and
    incomplete-row policy belong to the caller.
    """
    binary = numpy.unpackbits(packed_rows, axis=1, bitorder="big")[
        :, : samples_per_row * bits_per_component
    ]
    groups = binary.reshape(len(packed_rows), samples_per_row, bits_per_component)
    shifts = numpy.arange(bits_per_component - 1, -1, -1, dtype=numpy.uint8)
    return numpy.sum(groups << shifts, axis=2, dtype=numpy.uint8)


__all__ = ("unpack_subbyte_rows",)

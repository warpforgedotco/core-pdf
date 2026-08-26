# SPDX-License-Identifier: AGPL-3.0-only
"""Borrowed byte views used by the JBIG2 bitmap kernels."""

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl.runtime.array_views import uint8_view


def uint8_matrix_view(
    buffer: bytes | bytearray | memoryview | numpy.ndarray[Any, Any],
    rows: int,
    columns: int,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    """Return a validated mutable/read-only matrix view over packed rows."""
    return uint8_view(buffer, count=rows * columns).reshape(rows, columns)


__all__ = ("uint8_matrix_view", "uint8_view")

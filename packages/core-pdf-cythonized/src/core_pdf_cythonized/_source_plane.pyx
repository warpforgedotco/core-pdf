# SPDX-License-Identifier: AGPL-3.0-only
"""Source alpha and shape accumulation (core_pdf.impl.render.target.record_plane).

A transparency group tracks, per pixel, how much of it its sources covered:
plane = plane + (1 - plane) * coverage / 255 * scale. RasterTarget calls this
once per painted element with a window of about thirty pixels -- 78,000 times
over three pages each of five corpus documents -- and numpy spent almost all
of it building four temporaries around the arithmetic.

This handles the call every corpus page makes: a two-dimensional window, a
uint8 coverage plane of the same shape, no visibility mask. The caller keeps
numpy for anything else (a scalar coverage, a row, a mask), which takes a
different promotion path and is not what this reproduces.

Bit-exactness with numpy follows its promotion step by step:

* `1.0 - previous` is float32: a Python float does not widen a float32 array.
* `coverage / 255.0` is float64, since a uint8 array divided by a Python
  float is; multiplying by the float shape scale stays float64.
* The float32 difference times that float64 source, plus the float32
  previous value, is float64; the store narrows to float32 by rounding.

setup.py builds with -ffp-contract=off, so none of this is fused into FMAs.
"""

from core_pdf_cythonized._alpha_blend cimport accumulate_plane

__all__ = ("accumulate_source_plane",)


def accumulate_source_plane(float[:, :] plane, const unsigned char[:, :] coverage, double scale):
    """plane = plane + (1 - plane) * (coverage / 255.0 * scale), in place, as numpy does it."""
    cdef Py_ssize_t rows = plane.shape[0]
    cdef Py_ssize_t cols = plane.shape[1]
    if coverage.shape[0] != rows or coverage.shape[1] != cols:
        raise ValueError("coverage and plane windows differ in shape")
    cdef Py_ssize_t i, j
    with nogil:
        for i in range(rows):
            for j in range(cols):
                plane[i, j] = accumulate_plane(plane[i, j], coverage[i, j], scale)

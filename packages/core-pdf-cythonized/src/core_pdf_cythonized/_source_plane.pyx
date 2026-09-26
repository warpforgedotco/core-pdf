# SPDX-License-Identifier: AGPL-3.0-only

from core_pdf_cythonized._alpha_blend cimport accumulate_plane

__all__ = ("accumulate_source_plane",)


def accumulate_source_plane(float[:, :] plane, const unsigned char[:, :] coverage, double scale):
    cdef Py_ssize_t rows = plane.shape[0]
    cdef Py_ssize_t cols = plane.shape[1]
    if coverage.shape[0] != rows or coverage.shape[1] != cols:
        raise ValueError("coverage and plane windows differ in shape")
    cdef Py_ssize_t i, j
    with nogil:
        for i in range(rows):
            for j in range(cols):
                plane[i, j] = accumulate_plane(plane[i, j], coverage[i, j], scale)

# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports.core_pdf_cythonized._alpha_blend import accumulate_plane

__all__ = ("accumulate_source_plane",)


def accumulate_source_plane(
    plane: cython.float[:, :],
    coverage: cython.const[cython.uchar][:, :],
    scale: cython.double,
) -> None:
    rows: cython.Py_ssize_t = plane.shape[0]
    cols: cython.Py_ssize_t = plane.shape[1]
    if coverage.shape[0] != rows or coverage.shape[1] != cols:
        raise ValueError("coverage and plane windows differ in shape")
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    with cython.nogil:
        for i in range(rows):
            for j in range(cols):
                plane[i, j] = accumulate_plane(plane[i, j], coverage[i, j], scale)

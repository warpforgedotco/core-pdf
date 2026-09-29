# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports.core_pdf_cythonized._alpha_blend import blend_one, opaque_channel


def blend_normal_alpha_array_numpy(target, rgba, alpha) -> None:
    if target.size == 0:
        return

    red: cython.float = cython.cast(cython.float, cython.cast(cython.int, rgba[0]))
    green: cython.float = cython.cast(cython.float, cython.cast(cython.int, rgba[1]))
    blue: cython.float = cython.cast(cython.float, cython.cast(cython.int, rgba[2]))
    cap: cython.int = cython.cast(cython.int, rgba[3])

    plane: cython.uchar[:, :, :]
    plane_alpha: cython.uchar[:, :]
    row: cython.uchar[:, :]
    row_alpha: cython.uchar[:]
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    rows: cython.Py_ssize_t
    cols: cython.Py_ssize_t
    raw: cython.int
    opaque_from: cython.int = 255 if cap >= 255 else 256
    opaque_red: cython.uchar = opaque_channel(red)
    opaque_green: cython.uchar = opaque_channel(green)
    opaque_blue: cython.uchar = opaque_channel(blue)

    if target.ndim == 3:
        plane = target
        plane_alpha = alpha
        rows = plane.shape[0]
        cols = plane.shape[1]
        with cython.nogil:
            for i in range(rows):
                for j in range(cols):
                    raw = plane_alpha[i, j]
                    if raw == 0:
                        continue
                    if raw >= opaque_from:
                        plane[i, j, 0] = opaque_red
                        plane[i, j, 1] = opaque_green
                        plane[i, j, 2] = opaque_blue
                        plane[i, j, 3] = 255
                        continue
                    blend_one(
                        cython.address(plane[i, j, 0]),
                        cython.address(plane[i, j, 1]),
                        cython.address(plane[i, j, 2]),
                        cython.address(plane[i, j, 3]),
                        raw,
                        cap,
                        red,
                        green,
                        blue,
                    )
    elif target.ndim == 2:
        row = target
        row_alpha = alpha
        cols = row.shape[0]
        with cython.nogil:
            for j in range(cols):
                raw = row_alpha[j]
                if raw == 0:
                    continue
                if raw >= opaque_from:
                    row[j, 0] = opaque_red
                    row[j, 1] = opaque_green
                    row[j, 2] = opaque_blue
                    row[j, 3] = 255
                    continue
                blend_one(
                    cython.address(row[j, 0]),
                    cython.address(row[j, 1]),
                    cython.address(row[j, 2]),
                    cython.address(row[j, 3]),
                    raw,
                    cap,
                    red,
                    green,
                    blue,
                )
    else:
        raise ValueError(f"unsupported target rank {target.ndim}")

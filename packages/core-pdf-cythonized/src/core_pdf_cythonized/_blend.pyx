# SPDX-License-Identifier: AGPL-3.0-only

from core_pdf_cythonized._alpha_blend cimport blend_one, opaque_channel


def blend_normal_alpha_array_numpy(target, rgba, alpha):
    if target.size == 0:
        return

    cdef float red = <float> <int> rgba[0]
    cdef float green = <float> <int> rgba[1]
    cdef float blue = <float> <int> rgba[2]
    cdef int cap = <int> rgba[3]

    cdef unsigned char[:, :, :] plane
    cdef unsigned char[:, :] plane_alpha
    cdef unsigned char[:, :] row
    cdef unsigned char[:] row_alpha
    cdef Py_ssize_t i, j, rows, cols
    cdef int raw
    cdef int opaque_from = 255 if cap >= 255 else 256
    cdef unsigned char opaque_red = opaque_channel(red)
    cdef unsigned char opaque_green = opaque_channel(green)
    cdef unsigned char opaque_blue = opaque_channel(blue)

    if target.ndim == 3:
        plane = target
        plane_alpha = alpha
        rows = plane.shape[0]
        cols = plane.shape[1]
        with nogil:
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
                    blend_one(&plane[i, j, 0], &plane[i, j, 1], &plane[i, j, 2],
                              &plane[i, j, 3], raw, cap, red, green, blue)
    elif target.ndim == 2:
        row = target
        row_alpha = alpha
        cols = row.shape[0]
        with nogil:
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
                blend_one(&row[j, 0], &row[j, 1], &row[j, 2], &row[j, 3],
                          raw, cap, red, green, blue)
    else:
        raise ValueError(f"unsupported target rank {target.ndim}")

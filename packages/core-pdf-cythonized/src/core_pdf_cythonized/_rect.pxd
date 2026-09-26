# SPDX-License-Identifier: AGPL-3.0-only
# The rectangle fill's pixel core, for the stroke kernel's square joins and caps.

cdef int fill_rect_pixels(
    unsigned char* base,
    Py_ssize_t row_stride,
    Py_ssize_t ix0,
    Py_ssize_t ix1,
    Py_ssize_t iy0,
    Py_ssize_t iy1,
    double left,
    double right,
    double top,
    double bottom,
    int red_byte,
    int green_byte,
    int blue_byte,
    int cap,
) noexcept nogil

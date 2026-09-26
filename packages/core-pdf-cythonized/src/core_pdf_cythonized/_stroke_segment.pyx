# SPDX-License-Identifier: AGPL-3.0-only

from core_pdf_cythonized._pixel_blend cimport blend_normal_pixel, coverage_alpha

__all__ = ("stroke_segment_samples",)

cdef double[4] OFFSETS = [0.125, 0.375, 0.625, 0.875]


cdef bint segment_samples(
    unsigned char* pixels,
    Py_ssize_t row_stride,
    Py_ssize_t ix0,
    Py_ssize_t iy0,
    Py_ssize_t ix1,
    Py_ssize_t iy1,
    double crop_x0,
    double crop_y1,
    double scale,
    double x0,
    double y0,
    double dx,
    double dy,
    double seg_len2,
    double half2,
    double projection_extension,
    int red,
    int green,
    int blue,
    int alpha,
    const unsigned char* allowed,
    unsigned char* counts,
    Py_ssize_t* covered_box,
) noexcept nogil:
    cdef double cross_limit = half2 * seg_len2
    cdef Py_ssize_t px, py, sx, sy
    cdef double page_x[4]
    cdef double page_y[4]
    cdef double offset_x, offset_y, projection, cross
    cdef int covered, sa
    cdef Py_ssize_t low_x = ix1, low_y = iy1, high_x = ix0, high_y = iy0
    cdef Py_ssize_t box_width = ix1 - ix0
    for py in range(iy0, iy1):
        for sy in range(4):
            page_y[sy] = crop_y1 - (<double> py + OFFSETS[sy]) / scale
        for px in range(ix0, ix1):
            if allowed != NULL and not allowed[(py - iy0) * box_width + (px - ix0)]:
                continue
            for sx in range(4):
                page_x[sx] = crop_x0 + (<double> px + OFFSETS[sx]) / scale
            covered = 0
            for sy in range(4):
                offset_y = page_y[sy] - y0
                for sx in range(4):
                    offset_x = page_x[sx] - x0
                    projection = offset_x * dx + offset_y * dy
                    if projection < -projection_extension or projection > seg_len2 + projection_extension:
                        continue
                    cross = offset_x * dy - offset_y * dx
                    if cross * cross <= cross_limit:
                        covered += 1
            if not covered:
                continue
            if px < low_x:
                low_x = px
            if px + 1 > high_x:
                high_x = px + 1
            if py < low_y:
                low_y = py
            if py + 1 > high_y:
                high_y = py + 1
            if counts != NULL:
                counts[(py - iy0) * box_width + (px - ix0)] = <unsigned char> covered
                continue
            sa = coverage_alpha(alpha, covered)
            if sa <= 0:
                continue
            blend_normal_pixel(
                pixels + (py - iy0) * row_stride + (px - ix0) * 4, red, green, blue, sa
            )
    covered_box[0] = low_x
    covered_box[1] = low_y
    covered_box[2] = high_x
    covered_box[3] = high_y
    return high_x > low_x


def stroke_segment_samples(
    unsigned char[:, :, ::1] pixels,
    Py_ssize_t origin_x,
    Py_ssize_t origin_y,
    Py_ssize_t ix0,
    Py_ssize_t iy0,
    Py_ssize_t ix1,
    Py_ssize_t iy1,
    double crop_x0,
    double crop_y1,
    double scale,
    double x0,
    double y0,
    double dx,
    double dy,
    double seg_len2,
    double half2,
    double projection_extension,
    int red,
    int green,
    int blue,
    int alpha,
    const unsigned char[::1] allowed=None,
    unsigned char[:, ::1] counts=None,
):
    if ix0 < origin_x or iy0 < origin_y or ix1 > origin_x + pixels.shape[1] or iy1 > origin_y + pixels.shape[0]:
        raise ValueError("segment box runs past the pixels given")
    cdef Py_ssize_t box_width = ix1 - ix0
    if allowed is not None and allowed.shape[0] != box_width * (iy1 - iy0):
        raise ValueError("allowed must hold one byte per pixel of the box")
    if counts is not None and (counts.shape[0] != iy1 - iy0 or counts.shape[1] != box_width):
        raise ValueError("counts must hold one byte per pixel of the box")
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    cdef Py_ssize_t box[4]
    cdef bint any_covered
    cdef const unsigned char* allowed_bytes = &allowed[0] if allowed is not None and allowed.shape[0] else NULL
    cdef unsigned char* count_bytes = &counts[0, 0] if counts is not None else NULL
    cdef unsigned char* origin = &pixels[iy0 - origin_y, ix0 - origin_x, 0]
    with nogil:
        any_covered = segment_samples(
            origin, pixels.strides[0], ix0, iy0, ix1, iy1, crop_x0, crop_y1, scale,
            x0, y0, dx, dy, seg_len2, half2, projection_extension,
            red, green, blue, alpha, allowed_bytes, count_bytes, box,
        )
    if not any_covered:
        return None
    return box[0], box[1], box[2], box[3]

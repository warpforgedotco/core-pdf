# SPDX-License-Identifier: AGPL-3.0-only

from libc.math cimport ceil, isnan
from libc.stdlib cimport free, malloc

from core_pdf_cythonized._pymath cimport py_max, py_min


def horizontal_glyph_geometry(
    list offsets,
    list advances,
    list glyph_boxes,
    tuple basis,
    double font_ascent,
    double font_descent,
    double rise,
    double font_scale,
    double advance_scale,
    double font_size,
    clip_primary,
    clip_page,
    bint visible,
    list want_bitmap,
    bint want_transform=True,
):
    cdef Py_ssize_t n = len(offsets)
    cdef Py_ssize_t i, j, k

    if len(advances) != n or len(want_bitmap) != n:
        raise ValueError("offsets, advances and want_bitmap must be the same length")
    if len(glyph_boxes) != 4 * n:
        raise ValueError("glyph_boxes must hold four floats per glyph")
    if len(basis) != 6:
        raise ValueError("basis must hold six floats")

    cdef double base_x = basis[0]
    cdef double base_y = basis[1]
    cdef double a = basis[2]
    cdef double b = basis[3]
    cdef double c = basis[4]
    cdef double d = basis[5]
    cdef bint axis_aligned = (b == 0.0) and (c == 0.0)

    cdef double transform_a = advance_scale * a
    cdef double transform_b = advance_scale * b
    cdef double transform_c = font_scale * c
    cdef double transform_d = font_scale * d
    cdef double rise_offset_x = rise * c
    cdef double rise_offset_y = rise * d

    cdef double ar = font_ascent + rise
    cdef double dr = font_descent + rise

    cdef double axis_y0 = base_y + dr * d
    cdef double axis_y1 = base_y + ar * d
    cdef double swap
    if axis_y0 > axis_y1:
        swap = axis_y0
        axis_y0 = axis_y1
        axis_y1 = swap
    cdef double axis_baseline_y = base_y + rise * d

    cdef bint has_primary = clip_primary is not None
    cdef bint has_page = clip_page is not None
    cdef double px0 = 0.0, py0 = 0.0, px1 = 0.0, py1 = 0.0
    cdef double qx0 = 0.0, qy0 = 0.0, qx1 = 0.0, qy1 = 0.0
    if has_primary:
        px0 = clip_primary[0]
        py0 = clip_primary[1]
        px1 = clip_primary[2]
        py1 = clip_primary[3]
    if has_page:
        qx0 = clip_page[0]
        qy0 = clip_page[1]
        qx1 = clip_page[2]
        qy1 = clip_page[3]

    cdef double *off = <double *> malloc(n * sizeof(double))
    cdef double *adv = <double *> malloc(n * sizeof(double))
    cdef double *box = <double *> malloc(4 * n * sizeof(double))
    cdef char *bmp = <char *> malloc(n * sizeof(char))
    if off is NULL or adv is NULL or box is NULL or bmp is NULL:
        free(off); free(adv); free(box); free(bmp)
        raise MemoryError

    cdef list out_advance = [None] * n
    cdef list out_baseline = [None] * n
    cdef list out_transform = [None] * n
    cdef list out_ink = [None] * n
    cdef list out_visible = [0] * n
    cdef list out_bitmap = [0] * (2 * n)

    cdef double offset, advance
    cdef double ax0, ax1, abx0, abx1, aby0, aby1
    cdef double blx0, bly0, blx1, bly1
    cdef double tx0, tx1, cx0, cx1, cx2, cx3, cy0, cy1, cy2, cy3
    cdef double gx0, gy0, gx1, gy1
    cdef double ix0, ix1, iy0, iy1, rx0, rx1, ry0, ry1
    cdef double ex0, ex1, ex2, ex3, ey0, ey1, ey2, ey3
    cdef double inx0, iny0, inx1, iny1
    cdef double fb_w, fb_h, r_w, r_h, width, height, size, scaled, ratio
    cdef bint have_box
    cdef int vis, bw, bh
    cdef double ua0 = 0.0, ua1 = 0.0, ua2 = 0.0, ua3 = 0.0
    cdef double ui0 = 0.0, ui1 = 0.0, ui2 = 0.0, ui3 = 0.0

    try:
        for i in range(n):
            off[i] = offsets[i]
            adv[i] = advances[i]
            bmp[i] = 1 if want_bitmap[i] else 0
        for i in range(4 * n):
            box[i] = glyph_boxes[i]

        for i in range(n):
            offset = off[i]
            advance = adv[i]
            j = 4 * i
            k = 6 * i

            if axis_aligned:
                ax0 = base_x + offset * a
                ax1 = base_x + (offset + advance) * a
                if ax1 < ax0:
                    abx0 = ax1
                    abx1 = ax0
                else:
                    abx0 = ax0
                    abx1 = ax1
                aby0 = axis_y0
                aby1 = axis_y1
                blx0 = ax0
                bly0 = axis_baseline_y
                blx1 = ax1
                bly1 = axis_baseline_y
            else:
                tx0 = offset
                tx1 = offset + advance
                cx0 = tx0 * a + dr * c + base_x
                cx1 = tx1 * a + dr * c + base_x
                cx2 = tx0 * a + ar * c + base_x
                cx3 = tx1 * a + ar * c + base_x
                cy0 = tx0 * b + dr * d + base_y
                cy1 = tx1 * b + dr * d + base_y
                cy2 = tx0 * b + ar * d + base_y
                cy3 = tx1 * b + ar * d + base_y
                abx0 = py_min(py_min(py_min(cx0, cx1), cx2), cx3)
                abx1 = py_max(py_max(py_max(cx0, cx1), cx2), cx3)
                aby0 = py_min(py_min(py_min(cy0, cy1), cy2), cy3)
                aby1 = py_max(py_max(py_max(cy0, cy1), cy2), cy3)
                blx0 = base_x + tx0 * a + rise * c
                bly0 = base_y + tx0 * b + rise * d
                blx1 = base_x + tx1 * a + rise * c
                bly1 = base_y + tx1 * b + rise * d

            out_advance[i] = (abx0, aby0, abx1, aby1)
            out_baseline[i] = (blx0, bly0, blx1, bly1)
            if want_transform:
                out_transform[i] = (
                    transform_a,
                    transform_b,
                    transform_c,
                    transform_d,
                    base_x + offset * a + rise_offset_x,
                    base_y + offset * b + rise_offset_y,
                )

            vis = 0
            if visible:
                vis = 1
                if has_primary and (
                    abx1 <= px0 or abx0 >= px1 or aby1 <= py0 or aby0 >= py1
                ):
                    vis = 0
                elif has_page and (
                    abx1 <= qx0 or abx0 >= qx1 or aby1 <= qy0 or aby0 >= qy1
                ):
                    vis = 0
            out_visible[i] = vis

            gx0 = box[j]
            gy0 = box[j + 1]
            gx1 = box[j + 2]
            gy1 = box[j + 3]
            have_box = not isnan(gx0)

            if (
                have_box
                and axis_aligned
                and gx0 == 0.0
                and gy0 * font_scale == font_descent
                and gx1 * advance_scale == advance
                and gy1 * font_scale == font_ascent
            ):
                inx0 = abx0
                iny0 = aby0
                inx1 = abx1
                iny1 = aby1
            elif (not have_box) or gx1 <= gx0 or gy1 <= gy0:
                inx0 = abx0
                iny0 = aby0
                inx1 = abx1
                iny1 = aby1
            else:
                ix0 = offset + gx0 * advance_scale
                ix1 = offset + gx1 * advance_scale
                iy0 = rise + gy0 * font_scale
                iy1 = rise + gy1 * font_scale
                if axis_aligned:
                    rx0 = ix0 * a + base_x
                    rx1 = ix1 * a + base_x
                    ry0 = iy0 * d + base_y
                    ry1 = iy1 * d + base_y
                    if rx1 < rx0:
                        swap = rx0
                        rx0 = rx1
                        rx1 = swap
                    if ry1 < ry0:
                        swap = ry0
                        ry0 = ry1
                        ry1 = swap
                else:
                    ex0 = ix0 * a + iy0 * c + base_x
                    ex1 = ix1 * a + iy0 * c + base_x
                    ex2 = ix0 * a + iy1 * c + base_x
                    ex3 = ix1 * a + iy1 * c + base_x
                    ey0 = ix0 * b + iy0 * d + base_y
                    ey1 = ix1 * b + iy0 * d + base_y
                    ey2 = ix0 * b + iy1 * d + base_y
                    ey3 = ix1 * b + iy1 * d + base_y
                    rx0 = py_min(py_min(py_min(ex0, ex1), ex2), ex3)
                    rx1 = py_max(py_max(py_max(ex0, ex1), ex2), ex3)
                    ry0 = py_min(py_min(py_min(ey0, ey1), ey2), ey3)
                    ry1 = py_max(py_max(py_max(ey0, ey1), ey2), ey3)
                fb_w = abx1 - abx0
                fb_h = aby1 - aby0
                r_w = rx1 - rx0
                r_h = ry1 - ry0
                if (
                    r_w <= 0.01
                    or r_h <= 0.01
                    or (fb_w > 0.0 and r_w > fb_w * 4.0)
                    or (fb_h > 0.0 and r_h > fb_h * 1.5)
                ):
                    inx0 = abx0
                    iny0 = aby0
                    inx1 = abx1
                    iny1 = aby1
                else:
                    inx0 = rx0
                    iny0 = ry0
                    inx1 = rx1
                    iny1 = ry1

            out_ink[i] = (inx0, iny0, inx1, iny1)
            if i == 0:
                ua0 = abx0
                ua1 = aby0
                ua2 = abx1
                ua3 = aby1
                ui0 = inx0
                ui1 = iny0
                ui2 = inx1
                ui3 = iny1
            else:
                ua0 = ua0 if ua0 < abx0 else abx0
                ua1 = ua1 if ua1 < aby0 else aby0
                ua2 = ua2 if ua2 > abx1 else abx1
                ua3 = ua3 if ua3 > aby1 else aby1
                ui0 = ui0 if ui0 < inx0 else inx0
                ui1 = ui1 if ui1 < iny0 else iny0
                ui2 = ui2 if ui2 > inx1 else inx1
                ui3 = ui3 if ui3 > iny1 else iny1

            if bmp[i]:
                bw = 24
                bh = 32
                if have_box:
                    width = gx1 - gx0
                    height = gy1 - gy0
                    if width > 0.0 and height > 0.0:
                        size = font_size if font_size > 1.0 else 1.0
                        scaled = ceil(size * 2.5)
                        if scaled < 16.0:
                            bh = 16
                        elif scaled > 64.0:
                            bh = 64
                        else:
                            bh = <int> scaled
                        ratio = ceil(bh * width / height)
                        if ratio < 1.0:
                            bw = 1
                        elif ratio > 96.0:
                            bw = 96
                        else:
                            bw = <int> ratio
                out_bitmap[2 * i] = bw
                out_bitmap[2 * i + 1] = bh
    finally:
        free(off)
        free(adv)
        free(box)
        free(bmp)

    advance_union = (ua0, ua1, ua2, ua3) if n else None
    ink_union = (ui0, ui1, ui2, ui3) if n else None
    return (
        out_advance,
        out_baseline,
        out_transform,
        out_ink,
        out_visible,
        out_bitmap,
        advance_union,
        ink_union,
    )

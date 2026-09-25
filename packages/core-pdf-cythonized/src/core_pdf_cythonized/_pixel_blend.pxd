# SPDX-License-Identifier: AGPL-3.0-only
# RasterTarget.blend_px's per-pixel arithmetic, for kernels that replace one
# of its Python loops. Shared so there is one copy of it, as _bezier.pxd is
# for curve sampling.

from libc.math cimport rint


cdef inline int clamp_byte(double value) noexcept nogil:
    cdef double rounded = rint(value)
    if rounded < 0.0:
        return 0
    if rounded > 255.0:
        return 255
    return <int> rounded


cdef inline int coverage_alpha(int alpha, int covered) noexcept nogil:
    # blend_coverage_pixel: max(0, min(255, round(alpha * covered / 16))).
    return clamp_byte(<double> (alpha * covered) / 16.0)


cdef inline void blend_normal_pixel(
    unsigned char *pixel, int red, int green, int blue, int sa
) noexcept nogil:
    # blend_px in normal mode, for sa > 0, in double as Python computes it.
    cdef int dr, dg, db, da
    cdef double src_a, dst_a, out_a
    if sa >= 255:
        pixel[0] = <unsigned char> red
        pixel[1] = <unsigned char> green
        pixel[2] = <unsigned char> blue
        pixel[3] = 255
        return
    dr = pixel[0]
    dg = pixel[1]
    db = pixel[2]
    da = pixel[3]
    src_a = sa / 255.0
    dst_a = da / 255.0
    out_a = src_a + dst_a * (1.0 - src_a)
    pixel[0] = <unsigned char> clamp_byte(((red / 255.0 * 255.0) * src_a + (dr * dst_a) * (1.0 - src_a)) / out_a)
    pixel[1] = <unsigned char> clamp_byte(((green / 255.0 * 255.0) * src_a + (dg * dst_a) * (1.0 - src_a)) / out_a)
    pixel[2] = <unsigned char> clamp_byte(((blue / 255.0 * 255.0) * src_a + (db * dst_a) * (1.0 - src_a)) / out_a)
    pixel[3] = <unsigned char> clamp_byte(out_a * 255.0)


cdef inline float plane_accumulate(float previous, double source) noexcept nogil:
    # record_plane for one pixel: numpy float32 scalars against a Python
    # float, which narrows to float32 first -- every step is float32 here,
    # unlike the array path, where the source stays float64.
    cdef float narrowed = <float> source
    cdef float ONE = 1.0
    cdef float remaining = ONE - previous
    cdef float product = remaining * narrowed
    return previous + product

# SPDX-License-Identifier: AGPL-3.0-only
# RasterTarget.blend_px's per-pixel arithmetic, for kernels that replace one
# of its Python loops. Shared so there is one copy of it, as _bezier.pxd is
# for curve sampling.

from core_pdf_cythonized._byte_clamp cimport round_to_byte


# The blend modes blend_px treats apart; any other composites as normal.
cdef enum:
    MODE_NORMAL = 0
    MODE_MULTIPLY = 1
    MODE_SCREEN = 2
    MODE_COLOR_DODGE = 3
    MODE_COLOR_BURN = 4


cdef inline int coverage_alpha(int alpha, int covered) noexcept nogil:
    # blend_coverage_pixel: max(0, min(255, round(alpha * covered / 16))).
    return round_to_byte(<double> (alpha * covered) / 16.0)


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
    pixel[0] = <unsigned char> round_to_byte(((red / 255.0 * 255.0) * src_a + (dr * dst_a) * (1.0 - src_a)) / out_a)
    pixel[1] = <unsigned char> round_to_byte(((green / 255.0 * 255.0) * src_a + (dg * dst_a) * (1.0 - src_a)) / out_a)
    pixel[2] = <unsigned char> round_to_byte(((blue / 255.0 * 255.0) * src_a + (db * dst_a) * (1.0 - src_a)) / out_a)
    pixel[3] = <unsigned char> round_to_byte(out_a * 255.0)


cdef inline float plane_accumulate(float previous, double source) noexcept nogil:
    # record_plane for one pixel: numpy float32 scalars against a Python
    # float, which narrows to float32 first -- every step is float32 here,
    # unlike the array path, where the source stays float64.
    cdef float narrowed = <float> source
    cdef float ONE = 1.0
    cdef float remaining = ONE - previous
    cdef float product = remaining * narrowed
    return previous + product


cdef inline double color_dodge(double backdrop, double source, bint revised) noexcept nogil:
    # blend_component's ColorDodge.
    if revised and backdrop == 0.0:
        return 0.0
    if backdrop >= 1.0 - source:
        return 1.0
    return backdrop / (1.0 - source)


cdef inline double color_burn(double backdrop, double source, bint revised) noexcept nogil:
    # blend_component's ColorBurn.
    if revised and backdrop == 1.0:
        return 1.0
    if 1.0 - backdrop >= source:
        return 0.0
    return 1.0 - (1.0 - backdrop) / source


cdef inline void blend_mode_pixel(
    unsigned char* pixel, int red, int green, int blue, int sa, int mode, bint revised
) noexcept nogil:
    # blend_px for multiply, screen, color dodge and color burn, in double as
    # Python computes it: the source adjusted against the backdrop, then the
    # general compositing, with no opaque shortcut.
    cdef double src_a = sa / 255.0
    cdef double dst_a = pixel[3] / 255.0
    cdef double source[3]
    cdef double backdrop
    cdef int k
    cdef int destination[3]
    source[0] = red / 255.0
    source[1] = green / 255.0
    source[2] = blue / 255.0
    for k in range(3):
        destination[k] = pixel[k]
        backdrop = destination[k] / 255.0
        if mode == MODE_MULTIPLY:
            source[k] = source[k] * (1.0 - dst_a) + dst_a * (source[k] * backdrop)
        elif mode == MODE_SCREEN:
            source[k] = source[k] * (1.0 - dst_a) + dst_a * (
                1.0 - (1.0 - source[k]) * (1.0 - backdrop)
            )
        elif mode == MODE_COLOR_DODGE:
            source[k] = source[k] * (1.0 - dst_a) + dst_a * color_dodge(backdrop, source[k], revised)
        else:
            source[k] = source[k] * (1.0 - dst_a) + dst_a * color_burn(backdrop, source[k], revised)
    cdef double out_a = src_a + dst_a * (1.0 - src_a)
    for k in range(3):
        pixel[k] = <unsigned char> round_to_byte(
            ((source[k] * 255.0) * src_a + destination[k] * dst_a * (1.0 - src_a)) / out_a
        )
    pixel[3] = <unsigned char> round_to_byte(out_a * 255.0)

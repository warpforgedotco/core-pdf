# SPDX-License-Identifier: AGPL-3.0-only

from libc.math cimport fabs, sqrt

from core_pdf_cythonized._byte_clamp cimport round_to_byte


# The blend codes: a mode's index in core_pdf.impl.spec.s_11_transparency.blend.BLEND_MODES,
# and core_pdf.impl.render.blend.BlendOp's value. Codes from MODE_HUE to MODE_LUMINOSITY are
# non-separable. MODE_BACKDROP and MODE_SOURCE are no PDF blend modes: B(cb, cs) = cb and
# B(cb, cs) = cs, the rules ISO 32000-2 11.3.5.3 gives the K component of Hue, Saturation
# and Color, and of Luminosity, in a CMYK blending space. Unlike MODE_NORMAL, MODE_SOURCE
# blends: the source colour becomes (1 - ab) cs + ab cs.
cdef enum:
    MODE_NORMAL = 0
    MODE_MULTIPLY = 1
    MODE_SCREEN = 2
    MODE_COLOR_DODGE = 3
    MODE_COLOR_BURN = 4
    MODE_OVERLAY = 5
    MODE_DARKEN = 6
    MODE_LIGHTEN = 7
    MODE_HARD_LIGHT = 8
    MODE_SOFT_LIGHT = 9
    MODE_DIFFERENCE = 10
    MODE_EXCLUSION = 11
    MODE_HUE = 12
    MODE_SATURATION = 13
    MODE_COLOR = 14
    MODE_LUMINOSITY = 15
    MODE_BACKDROP = 16
    MODE_SOURCE = 17
    MODE_COUNT = 18


cdef inline int coverage_alpha(int alpha, int covered) noexcept nogil:
    return round_to_byte(<double> (alpha * covered) / 16.0)


cdef inline void blend_normal_pixel(
    unsigned char *pixel, int red, int green, int blue, int sa
) noexcept nogil:
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
    cdef float narrowed = <float> source
    cdef float ONE = 1.0
    cdef float remaining = ONE - previous
    cdef float product = remaining * narrowed
    return previous + product


cdef inline double color_dodge(double backdrop, double source, bint revised) noexcept nogil:
    if revised and backdrop == 0.0:
        return 0.0
    if backdrop >= 1.0 - source:
        return 1.0
    return backdrop / (1.0 - source)


cdef inline double color_burn(double backdrop, double source, bint revised) noexcept nogil:
    if revised and backdrop == 1.0:
        return 1.0
    if 1.0 - backdrop >= source:
        return 0.0
    return 1.0 - (1.0 - backdrop) / source


# The blend functions of core_pdf.impl.spec.s_11_transparency.blend, operation for operation,
# so that with contraction off they give its float64 bits.


cdef inline double hard_light(double cb, double cs) noexcept nogil:
    if cs <= 0.5:
        return cb * (2.0 * cs)
    return 1.0 - (1.0 - cb) * (1.0 - (2.0 * cs - 1.0))


cdef inline double soft_light(double cb, double cs) noexcept nogil:
    cdef double d
    if cs <= 0.5:
        return cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
    if cb <= 0.25:
        d = ((16.0 * cb - 12.0) * cb + 4.0) * cb
    else:
        d = sqrt(cb)
    return cb + (2.0 * cs - 1.0) * (d - cb)


cdef inline double separable_blend(
    double cb, double cs, int mode, bint revised
) noexcept nogil:
    if mode == MODE_MULTIPLY:
        return cb * cs
    if mode == MODE_SCREEN:
        return 1.0 - (1.0 - cb) * (1.0 - cs)
    if mode == MODE_COLOR_DODGE:
        return color_dodge(cb, cs, revised)
    if mode == MODE_COLOR_BURN:
        return color_burn(cb, cs, revised)
    if mode == MODE_OVERLAY:
        return hard_light(cs, cb)
    if mode == MODE_DARKEN:
        return cs if cs < cb else cb
    if mode == MODE_LIGHTEN:
        return cs if cs > cb else cb
    if mode == MODE_HARD_LIGHT:
        return hard_light(cb, cs)
    if mode == MODE_SOFT_LIGHT:
        return soft_light(cb, cs)
    if mode == MODE_DIFFERENCE:
        return fabs(cb - cs)
    if mode == MODE_EXCLUSION:
        return cb + cs - 2.0 * cb * cs
    return cs


cdef inline double lum(double r, double g, double b) noexcept nogil:
    return 0.3 * r + 0.59 * g + 0.11 * b


cdef inline double min3(double r, double g, double b) noexcept nogil:
    cdef double low = g if g < r else r
    return b if b < low else low


cdef inline double max3(double r, double g, double b) noexcept nogil:
    cdef double high = g if g > r else r
    return b if b > high else high


cdef inline void clip_color(double *c) noexcept nogil:
    cdef double l = lum(c[0], c[1], c[2])
    cdef double n = min3(c[0], c[1], c[2])
    cdef double x = max3(c[0], c[1], c[2])
    cdef double span
    cdef int k
    if n < 0.0:
        span = l - n
        if span != 0.0:
            for k in range(3):
                c[k] = l + ((c[k] - l) * l) / span
    if x > 1.0:
        span = x - l
        if span != 0.0:
            for k in range(3):
                c[k] = l + ((c[k] - l) * (1.0 - l)) / span


cdef inline void set_lum(double *c, double l) noexcept nogil:
    cdef double d = l - lum(c[0], c[1], c[2])
    c[0] = c[0] + d
    c[1] = c[1] + d
    c[2] = c[2] + d
    clip_color(c)


cdef inline double sat(const double *c) noexcept nogil:
    return max3(c[0], c[1], c[2]) - min3(c[0], c[1], c[2])


cdef inline void set_sat(double *c, double s) noexcept nogil:
    cdef double high = max3(c[0], c[1], c[2])
    cdef double low = min3(c[0], c[1], c[2])
    cdef double span
    cdef int k
    if not high > low:
        c[0] = 0.0
        c[1] = 0.0
        c[2] = 0.0
        return
    span = high - low
    for k in range(3):
        if c[k] == high:
            c[k] = s
        elif c[k] == low:
            c[k] = 0.0
        else:
            c[k] = ((c[k] - low) * s) / span


cdef inline void blend_rgb(
    const double *cb, const double *cs, double *out, int mode, bint revised
) noexcept nogil:
    """B(Cb, Cs) into out, for any code below MODE_COUNT; out may not alias cb or cs."""
    cdef int k
    if mode < MODE_HUE:
        for k in range(3):
            out[k] = separable_blend(cb[k], cs[k], mode, revised)
        return
    if mode == MODE_BACKDROP:
        out[0] = cb[0]
        out[1] = cb[1]
        out[2] = cb[2]
        return
    if mode == MODE_SOURCE:
        out[0] = cs[0]
        out[1] = cs[1]
        out[2] = cs[2]
        return
    if mode == MODE_HUE or mode == MODE_SATURATION:
        if mode == MODE_HUE:
            out[0] = cs[0]
            out[1] = cs[1]
            out[2] = cs[2]
            set_sat(out, sat(cb))
        else:
            out[0] = cb[0]
            out[1] = cb[1]
            out[2] = cb[2]
            set_sat(out, sat(cs))
        set_lum(out, lum(cb[0], cb[1], cb[2]))
    elif mode == MODE_COLOR:
        out[0] = cs[0]
        out[1] = cs[1]
        out[2] = cs[2]
        set_lum(out, lum(cb[0], cb[1], cb[2]))
    else:
        out[0] = cb[0]
        out[1] = cb[1]
        out[2] = cb[2]
        set_lum(out, lum(cs[0], cs[1], cs[2]))


cdef inline void blend_mode_pixel(
    unsigned char* pixel, int red, int green, int blue, int sa, int mode, bint revised
) noexcept nogil:
    """Composite one source colour over a pixel in any mode but normal (0)."""
    cdef double src_a = sa / 255.0
    cdef double dst_a = pixel[3] / 255.0
    cdef double source[3]
    cdef double backdrop[3]
    cdef double blended[3]
    cdef int k
    cdef int destination[3]
    source[0] = red / 255.0
    source[1] = green / 255.0
    source[2] = blue / 255.0
    for k in range(3):
        destination[k] = pixel[k]
        backdrop[k] = destination[k] / 255.0
    blend_rgb(backdrop, source, blended, mode, revised)
    for k in range(3):
        source[k] = source[k] * (1.0 - dst_a) + dst_a * blended[k]
    cdef double out_a = src_a + dst_a * (1.0 - src_a)
    for k in range(3):
        pixel[k] = <unsigned char> round_to_byte(
            ((source[k] * 255.0) * src_a + destination[k] * dst_a * (1.0 - src_a)) / out_a
        )
    pixel[3] = <unsigned char> round_to_byte(out_a * 255.0)

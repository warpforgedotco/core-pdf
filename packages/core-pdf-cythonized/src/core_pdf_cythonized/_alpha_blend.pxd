# SPDX-License-Identifier: AGPL-3.0-only
# Normal-mode compositing of a quantized coverage into RGBA (see _blend.pyx
# for why it is float32 throughout) and a group plane's per-element update
# (see _source_plane.pyx for numpy's promotion it follows). Shared so the
# fused glyph fill in _coverage.pyx is these same expressions, not a copy.

from libc.math cimport rintf

from core_pdf_cythonized._byte_clamp cimport float_to_byte


cdef inline unsigned char opaque_channel(float value) noexcept nogil:
    # blend_one's result for a channel when sa == 1: rintf((v * 1 + d * 0) / 1).
    return float_to_byte(rintf(value))


cdef inline void blend_one(
    unsigned char* c0,
    unsigned char* c1,
    unsigned char* c2,
    unsigned char* c3,
    int raw,
    int cap,
    float red,
    float green,
    float blue,
) noexcept nogil:
    cdef float ZERO = 0.0
    cdef float ONE = 1.0
    cdef float SCALE = 255.0
    cdef float sa = <float> (raw if raw < cap else cap) / SCALE
    cdef float d0 = <float> c0[0]
    cdef float d1 = <float> c1[0]
    cdef float d2 = <float> c2[0]
    cdef float da = <float> c3[0] / SCALE
    cdef float oa = sa + da * (ONE - sa)
    cdef float safe = oa if oa > ZERO else ONE
    cdef float weight = da * (ONE - sa)

    c0[0] = float_to_byte(rintf((red * sa + d0 * weight) / safe))
    c1[0] = float_to_byte(rintf((green * sa + d1 * weight) / safe))
    c2[0] = float_to_byte(rintf((blue * sa + d2 * weight) / safe))
    c3[0] = float_to_byte(rintf(oa * SCALE))


cdef inline float accumulate_plane(float previous, unsigned char coverage, double scale) noexcept nogil:
    # plane = plane + (1 - plane) * (coverage / 255.0 * scale), as numpy does it.
    cdef float ONE = 1.0
    cdef double SCALE = 255.0
    cdef float remaining = ONE - previous
    cdef double source = (<double> coverage / SCALE) * scale
    return <float> (<double> previous + <double> remaining * source)

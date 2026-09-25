# SPDX-License-Identifier: AGPL-3.0-only
"""Normal-mode alpha compositing (core_pdf.impl.render.blend).

The numpy original ran about fifteen array operations over a buffer averaging
thirty-eight elements, 26,315 times on a corpus page. All of the cost was
per-call overhead rather than arithmetic.

Two things keep this bit-exact with the original:

* The arithmetic is float32, not double. numpy promoted the uint8 buffers to
  float32 and did every step there, so this uses C float and never widens.
  The constants below are typed rather than written inline for that reason: a
  bare 1.0 is a double in C, which would promote each expression to double,
  compute there and narrow on assignment -- different results. Cython has no
  float-literal suffix, so named float constants are how this is expressed.
  Building with -ffp-contract=off (see setup.py) stops the compiler folding
  `sa + da * (1 - sa)` into an FMA, which would round differently too.
* Channel three is written last. The original assigns the three colour
  channels into its float scratch before overwriting alpha, so every channel
  is computed from the original destination alpha.

Pixels where alpha is zero are left untouched, matching the `where=alpha > 0`
mask on the original's final copyto.

A pixel painted at full coverage with an opaque colour takes a shortcut: there
sa is exactly 1, the destination's weight is exactly 0 and the output alpha
exactly 1, so blend_one can only produce the clamped source colour and 255.
Those bytes are worked out once, by the same rintf and clamp, and stored. The
interior of a filled rectangle is all such pixels, and the four divisions per
pixel were most of the cost once callers passed whole rectangles (70,000
pixels on average on test_3450) rather than the short spans this was built
for.

Both ranks are handled separately rather than reshaped to one flat loop.
Every caller passes a slice of a larger raster -- a row, or a rectangle --
so the buffers are not contiguous, and reshaping one would silently copy and
throw the in-place write away.
"""

from libc.math cimport rintf


cdef inline unsigned char opaque_channel(float value) noexcept nogil:
    # blend_one's result for a channel when sa == 1: rintf((v * 1 + d * 0) / 1).
    cdef float ZERO = 0.0
    cdef float SCALE = 255.0
    value = rintf(value)
    return <unsigned char> (ZERO if value < ZERO else (SCALE if value > SCALE else value))


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
    cdef float value

    value = rintf((red * sa + d0 * weight) / safe)
    c0[0] = <unsigned char> (ZERO if value < ZERO else (SCALE if value > SCALE else value))
    value = rintf((green * sa + d1 * weight) / safe)
    c1[0] = <unsigned char> (ZERO if value < ZERO else (SCALE if value > SCALE else value))
    value = rintf((blue * sa + d2 * weight) / safe)
    c2[0] = <unsigned char> (ZERO if value < ZERO else (SCALE if value > SCALE else value))
    value = rintf(oa * SCALE)
    c3[0] = <unsigned char> (ZERO if value < ZERO else (SCALE if value > SCALE else value))


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
    # Coverage at or above this is opaque; 256 when the colour itself is not.
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

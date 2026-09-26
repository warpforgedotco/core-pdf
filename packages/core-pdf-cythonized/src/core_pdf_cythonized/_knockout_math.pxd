# SPDX-License-Identifier: AGPL-3.0-only
# The knockout composite's per-component arithmetic (ISO 32000-2 11.4.x, see
# _knockout.pyx), shared so the fused glyph knockout in _coverage.pyx is these
# same expressions, not a copy.

from libc.math cimport rint


cdef inline double clamp_byte(double value) noexcept nogil:
    value = rint(value * 255.0)
    if value < 0.0:
        return 0.0
    if value > 255.0:
        return 255.0
    return value


cdef inline double knockout_component(
    double element_k,
    double element_complete,
    double remaining,
    double color_k,
    double complete,
    double backdrop_k,
    double initial,
    double result_alpha,
) noexcept nogil:
    # numpy computed this as
    #   element * element_complete + remaining * (color * complete - backdrop * initial)
    # then divided by result_alpha where that was non-zero, leaving zero
    # elsewhere. The order is reproduced exactly.
    cdef double premultiplied = (
        element_k * element_complete
        + remaining * (color_k * complete - backdrop_k * initial)
    )
    if result_alpha == 0.0:
        return 0.0
    return premultiplied / result_alpha

# SPDX-License-Identifier: AGPL-3.0-only

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
    cdef double premultiplied = (
        element_k * element_complete
        + remaining * (color_k * complete - backdrop_k * initial)
    )
    if result_alpha == 0.0:
        return 0.0
    return premultiplied / result_alpha

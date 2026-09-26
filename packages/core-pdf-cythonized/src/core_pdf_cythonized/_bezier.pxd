# SPDX-License-Identifier: AGPL-3.0-only

cdef enum:
    CUBIC_SAMPLE_CAPACITY = 4101

cdef int sample_times_c(double x0, double y0, double x1, double y1,
                        double x2, double y2, double x3, double y3,
                        double* out) noexcept nogil

cdef void extrema(double p0, double p1, double p2, double p3,
                  double* out, int* n) noexcept nogil

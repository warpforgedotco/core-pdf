# SPDX-License-Identifier: AGPL-3.0-only
# C-level entry point so other kernels can flatten a cubic without going
# through Python. _type2 uses it: sharing the declaration is what keeps one
# implementation of the sampling, rather than a second copy that could drift.

# How many times sample_times_c can write. rec() appends one time per leaf;
# depth is capped, so leaves <= 2**12, plus 1.0 and up to four extrema.
cdef enum:
    CUBIC_SAMPLE_CAPACITY = 4101

cdef int sample_times_c(double x0, double y0, double x1, double y1,
                        double x2, double y2, double x3, double y3,
                        double* out) noexcept nogil

# Cubic extrema in one axis, shared for the same reason. Matches
# charstrings.cubic_extrema_times: at most two roots strictly inside (0, 1),
# de-duplicated.
cdef void extrema(double p0, double p1, double p2, double p3,
                  double* out, int* n) noexcept nogil

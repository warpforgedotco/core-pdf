# SPDX-License-Identifier: AGPL-3.0-only

cdef enum:
    FAILED_NAN = 1
    FAILED_INFINITY = 2
    FAILED_MEMORY = 3

cdef enum:
    CLIP_NONE = 0
    CLIP_RECT = 1
    CLIP_ROWS = 2

# The line cap and line join styles (ISO 32000-2 8.4.3.3 and 8.4.3.4).
cdef enum:
    CAP_BUTT = 0
    CAP_ROUND = 1
    CAP_SQUARE = 2

cdef enum:
    JOIN_MITER = 0
    JOIN_ROUND = 1
    JOIN_BEVEL = 2

# The shapes stroke_coverage covers the samples of.
cdef enum:
    COVER_POLYGON = 0
    COVER_DISC = 1

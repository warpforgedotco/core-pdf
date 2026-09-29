# SPDX-License-Identifier: AGPL-3.0-only

cdef enum:
    FAILED_NAN = 1
    FAILED_INFINITY = 2
    FAILED_MEMORY = 3

cdef enum:
    CLIP_NONE = 0
    CLIP_RECT = 1
    CLIP_ROWS = 2

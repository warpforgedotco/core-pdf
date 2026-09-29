# SPDX-License-Identifier: AGPL-3.0-only

cdef enum:
    ON_CURVE = 0x01
    X_SHORT = 0x02
    Y_SHORT = 0x04
    REPEAT = 0x08
    X_SAME = 0x10
    Y_SAME = 0x20
    CUBIC = 0x80

cdef enum:
    WORDS = 0x0001
    XY_VALUES = 0x0002
    HAVE_SCALE = 0x0008
    MORE = 0x0020
    XY_SCALE = 0x0040
    TWO_BY_TWO = 0x0080
    INSTRUCTIONS = 0x0100

cdef enum:
    DONE = 0
    DECLINED = 1
    NO_MEMORY = 2

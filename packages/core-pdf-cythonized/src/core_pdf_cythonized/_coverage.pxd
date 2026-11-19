# SPDX-License-Identifier: AGPL-3.0-only

cdef enum:
    STACK_PIECES = 512
    # Glyph fills keep their device edges and accumulation cells on the stack
    # up to these sizes; typical glyphs have about a hundred edges in a 5x7 window.
    STACK_EDGES = 256
    STACK_CELLS = 1024
    # Four doubles per device edge. Pure-mode array sizes must be a single constant.
    STACK_EDGE_VALUES = STACK_EDGES * 4
    # The row engine keeps a glyph's edges, one row's pieces and its runs on the stack, in
    # SCRATCH_DOUBLES doubles, up to these counts.
    SCRATCH_EDGES = 128
    SCRATCH_ENTRIES = 256
    SCRATCH_RUNS = 128
    SCRATCH_DOUBLES = 4096
    # A window of at most this many cells, (width + 2) x height, is summed densely.
    DENSE_CELLS = 262144
    # Rows the row engine counts edges into on the stack.
    TALLY_STACK = 512

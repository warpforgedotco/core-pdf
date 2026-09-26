# SPDX-License-Identifier: AGPL-3.0-only
"""JBIG2 arithmetic generic region decoding (ITU-T T.88 6.2, Annex E).

Moved here from core_jbig2.codec. Like the knockout kernel it owns a
standard-defined algorithm rather than mirroring a helper: core_jbig2 keeps
segment parsing and page composition, and its base decoder now reports
arithmetic generic regions as unsupported, the way it already reported MMR
and text regions. Core's recovery decoder calls this instead. core_jbig2 --
and spec above it -- stay pure Python.

It came here because it is the whole cost of a JBIG2 page: a per-pixel MQ
decode (T.88 E.3) over a 65,536-entry context table, pure integer arithmetic
on byte buffers. no_bad_redactions.4.1.pdf spent 1.60s of a 1.69s render in
it.

Only generic template 0 with its default adaptive pixels and no typical
prediction is implemented; callers check the region header first. The
arithmetic is integer throughout, so there is no float contract to keep, and
the loop follows the Python original step for step, including reading 0xFF
past the end of the data (T.88 E.3.4).
"""

from libc.string cimport memset
from cpython.mem cimport PyMem_Free, PyMem_Malloc

__all__ = ("decode_arithmetic_generic_template0",)


# T.88 Table E.1: Qe, NMPS, NLPS and SWITCH for each of the 47 states.
cdef unsigned int MQ_QE[47]
cdef unsigned char MQ_NMPS[47]
cdef unsigned char MQ_NLPS[47]
cdef unsigned char MQ_SWITCH[47]

MQ_QE[:] = [
    0x5601, 0x3401, 0x1801, 0x0AC1, 0x0521, 0x0221, 0x5601, 0x5401,
    0x4801, 0x3801, 0x3001, 0x2401, 0x1C01, 0x1601, 0x5601, 0x5401,
    0x5101, 0x4801, 0x3801, 0x3401, 0x3001, 0x2801, 0x2401, 0x2201,
    0x1C01, 0x1801, 0x1601, 0x1401, 0x1201, 0x1101, 0x0AC1, 0x09C1,
    0x08A1, 0x0521, 0x0441, 0x02A1, 0x0221, 0x0141, 0x0111, 0x0085,
    0x0049, 0x0025, 0x0015, 0x0009, 0x0005, 0x0001, 0x5601,
]
MQ_NMPS[:] = [
    1, 2, 3, 4, 5, 38, 7, 8, 9, 10, 11, 12, 13, 29, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 45, 46,
]
MQ_NLPS[:] = [
    1, 6, 9, 12, 29, 33, 6, 14, 14, 14, 17, 18, 20, 21, 14, 14,
    15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46,
]
MQ_SWITCH[:] = [
    1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]

# Context bits kept when the template slides one pixel right (T.88 6.2.5.3).
cdef unsigned int OLD_PIXEL_MASK = 0x7BF7


cdef struct MQState:
    const unsigned char *data
    Py_ssize_t data_end
    Py_ssize_t bp
    unsigned int a
    unsigned int chigh
    unsigned int clow
    int ct


cdef inline unsigned int byte_at(const MQState *s, Py_ssize_t pos) noexcept nogil:
    return s.data[pos] if pos < s.data_end else 0xFF


cdef inline void byte_in(MQState *s) noexcept nogil:
    # T.88 E.3.4 BYTEIN.
    cdef unsigned int current = byte_at(s, s.bp)
    cdef unsigned int following = byte_at(s, s.bp + 1)
    if current == 0xFF:
        if following > 0x8F:
            s.clow += 0xFF00
            s.ct = 8
        else:
            s.bp += 1
            s.clow += byte_at(s, s.bp) << 9
            s.ct = 7
    else:
        s.bp += 1
        s.clow += byte_at(s, s.bp) << 8
        s.ct = 8
    if s.clow > 0xFFFF:
        s.chigh += s.clow >> 16
        s.clow &= 0xFFFF


cdef inline void init_decoder(MQState *s) noexcept nogil:
    # T.88 E.3.5 INITDEC.
    s.bp = 0
    s.chigh = s.data[0] if s.data_end > 0 else 0xFF
    s.clow = 0
    s.ct = 0
    byte_in(s)
    s.chigh = ((s.chigh << 7) & 0xFFFF) | ((s.clow >> 9) & 0x7F)
    s.clow = (s.clow << 7) & 0xFFFF
    s.ct -= 7
    s.a = 0x8000


cdef inline unsigned int decode_bit(MQState *s, unsigned char *contexts, unsigned int context) noexcept nogil:
    # T.88 E.3.2 DECODE, with the MPS/LPS exchanges of E.3.3 folded in.
    cdef unsigned int packed = contexts[context]
    cdef unsigned int idx = packed >> 1
    cdef unsigned int mps = packed & 1
    cdef unsigned int qe = MQ_QE[idx]
    cdef unsigned int next_a = s.a - qe
    cdef unsigned int pixel
    if s.chigh < qe:
        if next_a < qe:
            next_a = qe
            pixel = mps
            idx = MQ_NMPS[idx]
        else:
            next_a = qe
            pixel = 1 ^ mps
            if MQ_SWITCH[idx]:
                mps = pixel
            idx = MQ_NLPS[idx]
    else:
        s.chigh -= qe
        if next_a & 0x8000:
            s.a = next_a
            return mps
        if next_a < qe:
            pixel = 1 ^ mps
            if MQ_SWITCH[idx]:
                mps = pixel
            idx = MQ_NLPS[idx]
        else:
            pixel = mps
            idx = MQ_NMPS[idx]
    # RENORMD (E.3.3).
    while not (next_a & 0x8000):
        if s.ct == 0:
            byte_in(s)
        next_a <<= 1
        s.chigh = ((s.chigh << 1) & 0xFFFF) | ((s.clow >> 15) & 1)
        s.clow = (s.clow << 1) & 0xFFFF
        s.ct -= 1
    s.a = next_a
    contexts[context] = <unsigned char>((idx << 1) | mps)
    return pixel


cdef void decode_template0(
    MQState *s,
    unsigned char *contexts,
    unsigned char *bitmap,
    Py_ssize_t width,
    Py_ssize_t height,
    Py_ssize_t row_byte_length,
    unsigned char *buffers,
) noexcept nogil:
    # Three rows of width + 4 pixels, one byte each: the row being decoded
    # and the two above it. Before the second and third rows exist, the
    # template reads the current row in their place, which is all zeros
    # ahead of the pixel being decoded -- as T.88 6.2.5.2 requires.
    cdef Py_ssize_t stride = width + 4
    cdef unsigned char *row
    cdef unsigned char *row1
    cdef unsigned char *row2
    cdef unsigned char *previous_row = buffers
    cdef unsigned char *previous_previous_row = buffers + stride
    cdef unsigned char *spare = buffers + 2 * stride
    cdef unsigned char *out_row
    cdef Py_ssize_t row_index, col
    cdef unsigned int context, pixel
    for row_index in range(height):
        row = spare
        memset(row, 0, stride)
        row1 = row if row_index < 1 else previous_row
        row2 = row if row_index < 2 else previous_previous_row
        out_row = bitmap + row_index * row_byte_length
        context = (
            (row2[0] << 13)
            | (row2[1] << 12)
            | (row2[2] << 11)
            | (row1[0] << 7)
            | (row1[1] << 6)
            | (row1[2] << 5)
            | (row1[3] << 4)
        )
        for col in range(width):
            pixel = decode_bit(s, contexts, context)
            row[col] = <unsigned char>pixel
            if pixel:
                out_row[col >> 3] |= 0x80 >> (col & 7)
            context = (
                ((context & OLD_PIXEL_MASK) << 1)
                | (row2[col + 3] << 11)
                | (row1[col + 4] << 4)
                | pixel
            )
        spare = previous_previous_row
        previous_previous_row = previous_row
        previous_row = row


def decode_arithmetic_generic_template0(
    const unsigned char[:] data, Py_ssize_t width, Py_ssize_t height
) -> bytearray:
    """Decode a template-0 generic region to a packed, MSB-first bitmap.

    Rows are (width + 7) // 8 bytes. A 1 bit is a black pixel in T.88
    polarity; inverting it for PDF is the caller's job.
    """
    if width < 0 or height < 0:
        raise ValueError("negative JBIG2 generic region size")
    cdef Py_ssize_t row_byte_length = (width + 7) // 8
    bitmap = bytearray(row_byte_length * height)
    if not bitmap:
        return bitmap
    cdef unsigned char[::1] out = bitmap
    cdef MQState state
    state.data = &data[0] if data.shape[0] > 0 else NULL
    state.data_end = data.shape[0]
    cdef unsigned char *contexts = <unsigned char *>PyMem_Malloc(65536)
    cdef unsigned char *buffers = <unsigned char *>PyMem_Malloc(3 * (width + 4))
    if contexts == NULL or buffers == NULL:
        PyMem_Free(contexts)
        PyMem_Free(buffers)
        raise MemoryError()
    memset(contexts, 0, 65536)
    memset(buffers, 0, 3 * (width + 4))
    try:
        with nogil:
            init_decoder(&state)
            decode_template0(
                &state, contexts, &out[0], width, height, row_byte_length, buffers
            )
    finally:
        PyMem_Free(contexts)
        PyMem_Free(buffers)
    return bitmap

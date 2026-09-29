# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc
from cython.cimports.libc.string import memset

__all__ = ("decode_arithmetic_generic_template0",)


MQ_QE = cython.declare(cython.uint[47])
MQ_NMPS = cython.declare(cython.uchar[47])
MQ_NLPS = cython.declare(cython.uchar[47])
MQ_SWITCH = cython.declare(cython.uchar[47])

# fmt: off
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
# fmt: on

OLD_PIXEL_MASK = cython.declare(cython.uint, 0x7BF7)


MQState = cython.struct(
    data=cython.p_const_uchar,
    data_end=cython.Py_ssize_t,
    bp=cython.Py_ssize_t,
    a=cython.uint,
    chigh=cython.uint,
    clow=cython.uint,
    ct=cython.int,
)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def byte_at(s: cython.pointer[cython.const[MQState]], pos: cython.Py_ssize_t) -> cython.uint:
    return s.data[pos] if pos < s.data_end else 0xFF


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def byte_in(s: cython.pointer[MQState]) -> cython.void:
    current: cython.uint = byte_at(s, s.bp)
    following: cython.uint = byte_at(s, s.bp + 1)
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


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def init_decoder(s: cython.pointer[MQState]) -> cython.void:
    s.bp = 0
    s.chigh = s.data[0] if s.data_end > 0 else 0xFF
    s.clow = 0
    s.ct = 0
    byte_in(s)
    s.chigh = ((s.chigh << 7) & 0xFFFF) | ((s.clow >> 9) & 0x7F)
    s.clow = (s.clow << 7) & 0xFFFF
    s.ct -= 7
    s.a = 0x8000


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def decode_bit(
    s: cython.pointer[MQState], contexts: cython.p_uchar, context: cython.uint
) -> cython.uint:
    packed: cython.uint = contexts[context]
    idx: cython.uint = packed >> 1
    mps: cython.uint = packed & 1
    qe: cython.uint = MQ_QE[idx]
    next_a: cython.uint = s.a - qe
    pixel: cython.uint
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
    while not (next_a & 0x8000):
        if s.ct == 0:
            byte_in(s)
        next_a <<= 1
        s.chigh = ((s.chigh << 1) & 0xFFFF) | ((s.clow >> 15) & 1)
        s.clow = (s.clow << 1) & 0xFFFF
        s.ct -= 1
    s.a = next_a
    contexts[context] = cython.cast(cython.uchar, (idx << 1) | mps)
    return pixel


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def decode_template0(
    s: cython.pointer[MQState],
    contexts: cython.p_uchar,
    bitmap: cython.p_uchar,
    width: cython.Py_ssize_t,
    height: cython.Py_ssize_t,
    row_byte_length: cython.Py_ssize_t,
    buffers: cython.p_uchar,
) -> cython.void:
    stride: cython.Py_ssize_t = width + 4
    row: cython.p_uchar
    row1: cython.p_uchar
    row2: cython.p_uchar
    previous_row: cython.p_uchar = buffers
    previous_previous_row: cython.p_uchar = buffers + stride
    spare: cython.p_uchar = buffers + 2 * stride
    out_row: cython.p_uchar
    row_index: cython.Py_ssize_t
    col: cython.Py_ssize_t
    context: cython.uint
    pixel: cython.uint
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
            row[col] = cython.cast(cython.uchar, pixel)
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
    data: cython.const[cython.uchar][:], width: cython.Py_ssize_t, height: cython.Py_ssize_t
) -> bytearray:
    if width < 0 or height < 0:
        raise ValueError("negative JBIG2 generic region size")
    row_byte_length: cython.Py_ssize_t = (width + 7) // 8
    bitmap = bytearray(row_byte_length * height)
    if not bitmap:
        return bitmap
    out: cython.uchar[::1] = bitmap
    state = cython.declare(MQState)
    state.data = cython.address(data[0]) if data.shape[0] > 0 else cython.NULL
    state.data_end = data.shape[0]
    contexts: cython.p_uchar = cython.cast(cython.p_uchar, PyMem_Malloc(65536))
    buffers: cython.p_uchar = cython.cast(cython.p_uchar, PyMem_Malloc(3 * (width + 4)))
    if contexts == cython.NULL or buffers == cython.NULL:
        PyMem_Free(contexts)
        PyMem_Free(buffers)
        raise MemoryError()
    memset(contexts, 0, 65536)
    memset(buffers, 0, 3 * (width + 4))
    try:
        with cython.nogil:
            init_decoder(cython.address(state))
            decode_template0(
                cython.address(state),
                contexts,
                cython.address(out[0]),
                width,
                height,
                row_byte_length,
                buffers,
            )
    finally:
        PyMem_Free(contexts)
        PyMem_Free(buffers)
    return bitmap

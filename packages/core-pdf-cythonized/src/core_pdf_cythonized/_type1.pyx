# SPDX-License-Identifier: AGPL-3.0-only

__all__ = ("decrypt_type1",)


def decrypt_type1(const unsigned char[::1] data, int key):
    cdef Py_ssize_t n = data.shape[0], i
    result = bytearray(n)
    cdef unsigned char[::1] out = result
    cdef unsigned int state = key & 0xFFFF
    cdef unsigned int cipher
    with nogil:
        for i in range(n):
            cipher = data[i]
            out[i] = <unsigned char> (cipher ^ (state >> 8))
            state = ((cipher + state) * 52845 + 22719) & 0xFFFF
    return bytes(result)

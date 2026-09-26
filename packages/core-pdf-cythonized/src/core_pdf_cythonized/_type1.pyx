# SPDX-License-Identifier: AGPL-3.0-only
"""Type 1 eexec and charstring decryption (core_pdf.impl.fonts_font_program).

A Type 1 program's private section is eexec-encrypted, and every
subroutine and charstring in it is encrypted again (Adobe Type 1 Font
Format, 7.1): r = (c + r) * 52845 + 22719, each plain byte c ^ (r >> 8).
core_adobe_fonts.type1.program.decrypt_type1 does that a byte at a time in
Python, about 100 ns a byte, and a Type 1 font is decrypted whole when it
is loaded -- 1.36 s of a 400-document extraction sample, over 41,241 calls.

This mirrors it, as ObjectScanner mirrors a lexer that stays: the CMap and
font package is pure Python by design and keeps its decrypt_type1 for its
own readers; core decrypts with this. The arithmetic is 16-bit and
integral, so the result is the same bytes.
"""

__all__ = ("decrypt_type1",)


def decrypt_type1(const unsigned char[::1] data, int key):
    """The plaintext of `data` under the Type 1 cipher starting at `key`."""
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

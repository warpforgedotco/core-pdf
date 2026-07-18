# SPDX-License-Identifier: AGPL-3.0-only
class CryptRC4:
    def __init__(self, key: bytes) -> None:
        self.sbox = self.init_sbox(key)

    def init_sbox(self, key: bytes) -> list[int]:
        sbox = list(range(256))
        j = 0
        key_len = len(key)

        key_mod_fn = key.__getitem__
        for i in range(256):
            j = (j + sbox[i] + key_mod_fn(i % key_len)) & 0xFF
            sbox[i], sbox[j] = sbox[j], sbox[i]
        return sbox

    def encrypt(self, data: bytes) -> bytes:
        sbox = self.sbox.copy()
        i = 0
        j = 0
        out = bytearray()
        append = out.append

        for byte in data:
            i = (i + 1) & 0xFF
            j = (j + sbox[i]) & 0xFF
            sbox[i], sbox[j] = sbox[j], sbox[i]
            append(byte ^ sbox[(sbox[i] + sbox[j]) & 0xFF])

        return bytes(out)

    def decrypt(self, data: bytes) -> bytes:
        return self.encrypt(data)

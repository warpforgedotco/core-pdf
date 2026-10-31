from collections.abc import Sequence


class Arcfour:
    def __init__(self, key: Sequence[int]) -> None:
        self.key = bytes(key)

    def process(self, data: bytes) -> bytes:
        # RC4 is keyed from any length here; the cryptography backend refuses short keys.
        state = list(range(256))
        j = 0
        for i in range(256):
            j = (j + state[i] + self.key[i % len(self.key)]) & 255
            state[i], state[j] = state[j], state[i]
        i = j = 0
        output = bytearray()
        for byte in data:
            i = (i + 1) & 255
            j = (j + state[i]) & 255
            state[i], state[j] = state[j], state[i]
            output.append(byte ^ state[(state[i] + state[j]) & 255])
        return bytes(output)

    encrypt = decrypt = process


__all__ = ("Arcfour",)

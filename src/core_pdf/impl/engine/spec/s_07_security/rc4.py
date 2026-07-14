# SPDX-License-Identifier: AGPL-3.0-only
class CryptRC4:
    """RC4 encryption/decryption stream cipher.

    RC4 is a symmetric stream cipher that is used in older PDF encryption
    standards (RC4-40 and RC4-128).
    """

    def __init__(self, key: bytes) -> None:
        """Initialize RC4 with a key.

        Args:
            key: The encryption key (5-128 bytes for PDF RC4).
        """
        self.sbox = self.init_sbox(key)

    def init_sbox(self, key: bytes) -> list[int]:
        """Initialize the S-box using the key-scheduling algorithm (KSA)."""
        sbox = list(range(256))
        j = 0
        key_len = len(key)
        # Local binding for speed in hot loop
        key_mod_fn = key.__getitem__
        for i in range(256):
            j = (j + sbox[i] + key_mod_fn(i % key_len)) & 0xFF
            sbox[i], sbox[j] = sbox[j], sbox[i]
        return sbox

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt or decrypt data.

        RC4 is symmetric: encrypt() == decrypt().

        Args:
            data: The plaintext or ciphertext bytes.

        Returns:
            The encrypted or decrypted bytes.
        """
        # Copy S-box per-call (RC4 is stateful)
        sbox = self.sbox.copy()
        i = 0
        j = 0
        out = bytearray(len(data))

        # Local bindings for hot loop speed
        sbox_fn = sbox.__getitem__
        data_fn = data.__getitem__

        for k in range(len(data)):
            i = (i + 1) & 0xFF
            j = (j + sbox_fn(i)) & 0xFF
            si, sj = sbox[i], sbox[j]
            sbox[i], sbox[j] = sj, si
            t = (sj + si) & 0xFF
            out[k] = data_fn(k) ^ sbox_fn(t)

        return bytes(out)

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data.

        RC4 is symmetric, so this is identical to encrypt.
        """
        return self.encrypt(data)

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from hashlib import sha256, sha384, sha512
from typing import Callable

from core_pdf.impl.spec.s_07_security.aes import AES
from core_pdf.impl.spec.s_07_security.saslprep import saslprep
from core_pdf.impl.spec.s_07_security.standard_v4 import (
    PdfStandardSecurityHandlerV4,
)
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import coerce_to_bytes
from core_pdf.impl.spec.s_07_syntax_primitives.pdfdict import lookup_dict_key


class PdfStandardSecurityHandlerV5(PdfStandardSecurityHandlerV4):
    supported_revisions = (5, 6)
    internal_aes_cipher: AES

    def init_params(self) -> None:
        super().init_params()
        self.length = 256
        self.oe = coerce_to_bytes(lookup_dict_key(self.param, "OE"))
        self.ue = coerce_to_bytes(lookup_dict_key(self.param, "UE"))
        self.o_hash = self.o[:32]
        self.o_validation_salt = self.o[32:40]
        self.o_key_salt = self.o[40:]
        self.u_hash = self.u[:32]
        self.u_validation_salt = self.u[32:40]
        self.u_key_salt = self.u[40:]

    def get_cfm(self, name: str) -> Callable[[int, int, bytes], bytes] | None:
        if name == "AESV3":
            return self.decrypt_aes256
        return None

    def authenticate(self, password: str) -> bytes | None:
        password_b = self.normalize_password(password)

        hash_val = self.password_hash(password_b, self.o_validation_salt, self.u)
        if hash_val == self.o_hash:
            hash_val = self.password_hash(password_b, self.o_key_salt, self.u)
            cipher = AES(hash_val)
            return cipher.decrypt_cbc(b"\0" * 16, self.oe, padding=False)

        hash_val = self.password_hash(password_b, self.u_validation_salt)
        if hash_val == self.u_hash:
            hash_val = self.password_hash(password_b, self.u_key_salt)
            cipher = AES(hash_val)
            return cipher.decrypt_cbc(b"\0" * 16, self.ue, padding=False)
        return None

    def normalize_password(self, password: str) -> bytes:
        if self.r == 6:
            if not password:
                return b""
            password = saslprep(password)
        return password.encode("utf-8")[:127]

    def password_hash(
        self,
        password: bytes,
        salt: bytes,
        vector: bytes | None = None,
    ) -> bytes:
        if self.r == 5:
            return self.r5_password(password, salt, vector)
        return self.r6_password(password, salt[0:8], vector)

    def r5_password(
        self,
        password: bytes,
        salt: bytes,
        vector: bytes | None = None,
    ) -> bytes:
        h = sha256(password)
        h.update(salt)
        if vector is not None:
            h.update(vector)
        return h.digest()

    def r6_password(
        self,
        password: bytes,
        salt: bytes,
        vector: bytes | None = None,
    ) -> bytes:
        initial_hash = sha256(password)
        initial_hash.update(salt)
        if vector is not None:
            initial_hash.update(vector)
        k = initial_hash.digest()
        hashes = (sha256, sha384, sha512)
        round_no = last_byte_val = 0
        while round_no < 64 or last_byte_val > round_no - 32:
            k1 = (password + k + (vector or b"")) * 64
            cipher = AES(k[:16])
            e = cipher.encrypt_cbc(k[16:32], k1, padding=False)
            next_hash = hashes[self.bytes_mod_3(e[:16])]
            k = next_hash(e).digest()
            last_byte_val = e[len(e) - 1]
            round_no += 1
        return k[:32]

    @staticmethod
    def bytes_mod_3(input_bytes: bytes) -> int:
        return sum(b % 3 for b in input_bytes) % 3

    def init_key(self) -> None:
        # AESV3 uses one file key for every object, unlike the per-object keys V4
        # derives, so the key schedule is expanded once here rather than per
        # encrypted string and stream.
        super().init_key()
        assert self.key is not None
        self.internal_aes_cipher = AES(self.key)

    def decrypt_aes256(self, objid: int, genno: int, data: bytes) -> bytes:
        initialization_vector = data[:16]
        ciphertext = data[16:]
        return self.internal_aes_cipher.decrypt_cbc(initialization_vector, ciphertext, padding=True)

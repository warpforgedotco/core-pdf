# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import struct
from hashlib import md5
from typing import Sequence

from core_pdf.impl.engine.spec.s_07_objects.coercion import coerce_to_bytes
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import (
    lookup_dict_key,
    lookup_dict_key_default,
)
from core_pdf.impl.engine.spec.s_07_security.crypto_constants import PDF_PADDING
from core_pdf.impl.engine.spec.s_07_security.errors import (
    PDFEncryptionError,
    PDFPasswordIncorrect,
)
from core_pdf.impl.engine.spec.s_07_security.rc4 import CryptRC4
from core_pdf.impl.engine.spec.s_07_security.values import get_int, get_uint
from core_pdf.impl.objects import MISSING
from core_pdf.impl.types import PdfDict


class PdfStandardSecurityHandler:
    supported_revisions: tuple[int, ...] = (2, 3)

    def __init__(
        self,
        docid: Sequence[object],
        param: PdfDict,
        password: str = "",
    ) -> None:
        self.docid = docid
        self.param = param
        self.password = password
        self.encrypt_metadata = True
        self.init()

    def init(self) -> None:
        try:
            self.init_params()
        except (TypeError, ValueError, PDFEncryptionError) as exc:
            raise PDFEncryptionError(f"invalid encryption dictionary: {exc}") from exc
        if self.r not in self.supported_revisions:
            raise PDFEncryptionError(f"Unsupported revision: param={self.param!r}")
        self.init_key()

    def init_params(self) -> None:
        raw_v = lookup_dict_key(self.param, "V")
        if raw_v is None:
            raise PDFEncryptionError("missing encryption algorithm")
        self.v = get_int(raw_v)

        raw_r = lookup_dict_key(self.param, "R")
        if raw_r is None:
            raise PDFEncryptionError("missing revision")
        self.r = get_int(raw_r)

        raw_p = lookup_dict_key_default(self.param, "P", MISSING)
        if raw_p is MISSING:
            raise PDFEncryptionError("missing encryption permissions")
        if raw_p is None:
            raise PDFEncryptionError("invalid encryption permissions")
        self.p = get_uint(raw_p, 32)
        self.o = coerce_to_bytes(lookup_dict_key(self.param, "O"))
        self.u = coerce_to_bytes(lookup_dict_key(self.param, "U"))
        raw_length = lookup_dict_key_default(self.param, "Length", MISSING)
        if raw_length is not MISSING:
            if raw_length is None:
                raise PDFEncryptionError("invalid encryption length")
            self.length = get_int(raw_length)
        else:
            self.length = 40

    def init_key(self) -> None:
        self.key = self.authenticate(self.password)
        if self.key is None:
            raise PDFPasswordIncorrect("Incorrect password")

    def is_printable(self) -> bool:
        return bool(self.p & 4)

    def is_modifiable(self) -> bool:
        return bool(self.p & 8)

    def is_extractable(self) -> bool:
        return bool(self.p & 16)

    def compute_u(self, key: bytes) -> bytes:
        if self.r == 2:
            return CryptRC4(key).encrypt(PDF_PADDING)
        else:
            h = md5(PDF_PADDING)
            docid_list = self.docid
            first_id = (
                coerce_to_bytes(docid_list[0])
                if docid_list and len(docid_list) > 0
                else b""
            )
            h.update(first_id)
            result = CryptRC4(key).encrypt(h.digest())
            for i in range(1, 20):
                k = bytes(c ^ i for c in key)
                result = CryptRC4(k).encrypt(result)
            result += result
            return result

    def compute_encryption_key(self, password: bytes) -> bytes:
        password = (password + PDF_PADDING)[:32]
        h = md5(password)
        h.update(self.o)
        h.update(struct.pack("<L", self.p))
        docid_list = self.docid
        first_id = (
            coerce_to_bytes(docid_list[0])
            if docid_list and len(docid_list) > 0
            else b""
        )
        h.update(first_id)
        if self.r >= 4 and not self.encrypt_metadata:
            h.update(b"\xff\xff\xff\xff")
        result = h.digest()
        n = 5
        if self.r >= 3:
            n = self.length // 8
            for ignored in range(50):
                result = md5(result[:n]).digest()
        return result[:n]

    def authenticate(self, password: str) -> bytes | None:
        password_bytes = password.encode("latin1")
        key = self.authenticate_user_password(password_bytes)
        if key is None:
            key = self.authenticate_owner_password(password_bytes)
        return key

    def authenticate_user_password(self, password: bytes) -> bytes | None:
        key = self.compute_encryption_key(password)
        if self.verify_encryption_key(key):
            return key
        return None

    def verify_encryption_key(self, key: bytes) -> bool:
        u = self.compute_u(key)
        if self.r == 2:
            return u == self.u
        return u[:16] == self.u[:16]

    def authenticate_owner_password(self, password: bytes) -> bytes | None:
        password = (password + PDF_PADDING)[:32]
        h = md5(password)
        if self.r >= 3:
            for ignored in range(50):
                h = md5(h.digest())
        n = 5
        if self.r >= 3:
            n = self.length // 8
        key = h.digest()[:n]
        if self.r == 2:
            user_password = CryptRC4(key).decrypt(self.o)
        else:
            user_password = self.o
            for i in range(19, -1, -1):
                k = bytes(c ^ i for c in key)
                user_password = CryptRC4(k).decrypt(user_password)
        return self.authenticate_user_password(user_password)

    def decrypt(
        self,
        objid: int,
        genno: int,
        data: bytes,
        attrs: PdfDict | None = None,
    ) -> bytes:
        return self.decrypt_rc4(objid, genno, data)

    def decrypt_rc4(self, objid: int, genno: int, data: bytes) -> bytes:
        assert self.key is not None
        key = self.key + struct.pack("<L", objid)[:3] + struct.pack("<L", genno)[:2]
        h = md5(key)
        key = h.digest()[: min(len(key), 16)]
        return CryptRC4(key).decrypt(data)

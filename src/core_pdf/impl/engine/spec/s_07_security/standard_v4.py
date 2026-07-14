# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import struct
from hashlib import md5
from typing import Callable

from core_pdf.impl.engine.spec.s_07_objects.pdfdict import (
    lookup_dict_key,
    lookup_dict_key_default,
)
from core_pdf.impl.engine.spec.s_07_security.aes import AES
from core_pdf.impl.engine.spec.s_07_security.errors import PDFEncryptionError
from core_pdf.impl.engine.spec.s_07_security.standard import PdfStandardSecurityHandler
from core_pdf.impl.engine.spec.s_07_security.values import get_name
from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.objects import MISSING
from core_pdf.impl.types import PdfDict


class PdfStandardSecurityHandlerV4(PdfStandardSecurityHandler):
    supported_revisions: tuple[int, ...] = (4,)
    cf: PdfDict
    cfm: dict[str, Callable[[int, int, bytes], bytes]]

    def init_params(self) -> None:
        super().init_params()
        self.length = 128
        cf = lookup_dict_key(self.param, "CF")
        if cf is None:
            self.cf = {}
        elif not isinstance(cf, dict):
            raise PDFEncryptionError("Invalid crypt filter dictionary: CF")
        else:
            self.cf = cf
        stmf_raw = lookup_dict_key_default(self.param, "StmF", MISSING)
        self.stmf = get_name("Identity" if stmf_raw is MISSING else stmf_raw) or "Identity"
        strf_raw = lookup_dict_key_default(self.param, "StrF", MISSING)
        self.strf = get_name("Identity" if strf_raw is MISSING else strf_raw) or "Identity"
        encrypt_metadata = lookup_dict_key_default(self.param, "EncryptMetadata", MISSING)
        if encrypt_metadata is MISSING:
            encrypt_metadata = True
        if type(encrypt_metadata) is not bool:
            raise PDFEncryptionError("Invalid encryption metadata flag")
        self.encrypt_metadata = encrypt_metadata
        self.cfm = {}
        for k, v in self.cf.items():
            if not isinstance(v, dict):
                raise PDFEncryptionError(f"Invalid crypt filter dictionary: {k!r}")
            f = self.get_cfm(get_name(lookup_dict_key(v, "CFM") or ""))
            if f is None:
                raise PDFEncryptionError(
                    f"Unknown crypt filter method CFM: {lookup_dict_key(v, 'CFM')}"
                )
            self.cfm[get_name(k)] = f
        if self.strf != "Identity" and self.strf not in self.cfm:
            raise PDFEncryptionError(f"Undefined crypt filter: {self.strf}")

    def get_cfm(self, name: str) -> Callable[[int, int, bytes], bytes] | None:
        if name == "V2":
            return self.decrypt_rc4
        elif name == "AESV2":
            return self.decrypt_aes128
        return None

    def decrypt(
        self,
        objid: int,
        genno: int,
        data: bytes,
        attrs: PdfDict | None = None,
        name: str | None = None,
    ) -> bytes:
        if not self.encrypt_metadata and attrs is not None:
            t = lookup_dict_key(attrs, "Type")
            if t is not None and get_name(t) == "Metadata":
                return data
        if name is None:
            name = self.stmf if attrs is not None else self.strf
        if name == "Identity":
            return data
        fn = self.cfm.get(name)
        if fn is None:
            raise PdfUnsupportedError(f"Undefined crypt filter: {name}")
        return fn(objid, genno, data)

    def decrypt_aes128(self, objid: int, genno: int, data: bytes) -> bytes:
        assert self.key is not None
        key = self.key + struct.pack("<L", objid)[:3] + struct.pack("<L", genno)[:2] + b"sAlT"
        h = md5(key)
        key = h.digest()[: min(len(key), 16)]
        initialization_vector = data[:16]
        ciphertext = data[16:]
        cipher = AES(key)
        return cipher.decrypt_cbc(initialization_vector, ciphertext, padding=True)

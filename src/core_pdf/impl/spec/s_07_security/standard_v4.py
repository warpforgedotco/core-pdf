# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Callable, cast

from core_pdf.impl.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf.impl.primitives import MISSING
from core_pdf.impl.spec.s_07_filters.decode_spec import normalize_stream_decode_spec
from core_pdf.impl.spec.s_07_security.ciphers import internal_aes_cbc_decrypt
from core_pdf.impl.spec.s_07_security.errors import PDFEncryptionError
from core_pdf.impl.spec.s_07_security.standard import PdfStandardSecurityHandler
from core_pdf.impl.spec.s_07_security.values import get_name
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    is_pdf_null,
    normalize_pdf_name,
)
from core_pdf.impl.spec.s_07_syntax_primitives.pdfdict import (
    lookup_dict_key,
    lookup_dict_key_default,
)


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
            self.cf = cast(PdfDict, cf)
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
            name = self.stream_crypt_filter_name(attrs) if attrs is not None else self.strf
        if name == "Identity":
            return data
        fn = self.cfm.get(name)
        if fn is None:
            raise PdfUnsupportedError(f"Undefined crypt filter: {name}")
        return fn(objid, genno, data)

    def stream_crypt_filter_name(self, attrs: PdfDict) -> str:
        spec = normalize_stream_decode_spec(attrs)
        crypt_indexes = [
            index for index, filter_name in enumerate(spec.filters) if filter_name == "Crypt"
        ]
        if not crypt_indexes:
            return self.stmf
        if len(crypt_indexes) != 1 or crypt_indexes[0] != 0:
            raise PdfParseError("Crypt must be the first and only Crypt stream filter")

        params = spec.params[crypt_indexes[0]]
        if is_pdf_null(params):
            return "Identity"
        if not isinstance(params, dict):
            raise PdfParseError("invalid Crypt filter params")
        raw_name = lookup_dict_key(params, "Name")
        if is_pdf_null(raw_name):
            return "Identity"
        filter_name = normalize_pdf_name(raw_name)
        if filter_name is None:
            raise PdfParseError("invalid Crypt filter name")
        return filter_name

    def decrypt_aes128(self, objid: int, genno: int, data: bytes) -> bytes:
        key = self.object_key(objid, genno, b"sAlT")
        initialization_vector = data[:16]
        ciphertext = data[16:]
        return internal_aes_cbc_decrypt(
            key,
            initialization_vector,
            ciphertext,
            use_padding=True,
        )

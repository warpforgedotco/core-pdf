# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_security.crypto_handlers import (
    SECURITY_HANDLER_REGISTRY,
)
from core_pdf.impl.engine.spec.s_07_security.errors import (
    PDFEncryptionError,
    PDFPasswordIncorrect,
)
from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.objects import PdfReference
from core_pdf.impl.types import Decipher, PdfDict


class DocumentSecurityResolver(Protocol):
    def resolve(self, ref: object) -> object: ...

    def resolve_dict(self, ref: object) -> object: ...

    def resolve_int(self, value: object) -> int | None: ...

    def resolve_name(self, value: object) -> str | None: ...


class DocumentSecurityMixin:
    decipher: Decipher | None
    resolver: DocumentSecurityResolver
    trailer_dict: PdfDict

    def init_security(self, password: str) -> None:
        encrypt_ref = lookup_dict_key(self.trailer_dict, "Encrypt")
        if encrypt_ref is None:
            return

        encrypt_dict = self.resolver.resolve_dict(encrypt_ref)
        if not isinstance(encrypt_dict, dict):
            raise PdfUnsupportedError("Invalid Encrypt dictionary")
        encrypt_dict = cast(PdfDict, encrypt_dict)

        filter_name = self.resolver.resolve_name(lookup_dict_key(encrypt_dict, "Filter"))
        if filter_name is None:
            raise PdfUnsupportedError("Invalid encryption dictionary")
        if filter_name in {"Adobe.PubSec", "PubSec"}:
            raise PdfUnsupportedError("Public-key encryption is not supported")
        if filter_name != "Standard":
            raise PdfUnsupportedError(f"Unsupported encryption filter: {filter_name}")

        raw_v = lookup_dict_key(encrypt_dict, "V")
        if raw_v is None:
            raise PdfUnsupportedError("Invalid encryption dictionary")
        v_opt = self.resolver.resolve_int(raw_v)
        if type(v_opt) is not int:
            raise PdfUnsupportedError("Invalid encryption dictionary")
        v = v_opt
        handler_cls = SECURITY_HANDLER_REGISTRY.get(v)
        if handler_cls is None:
            raise PdfUnsupportedError(f"Unsupported standard encryption algorithm V={v}")

        docid = lookup_dict_key(self.trailer_dict, "ID")
        if docid is None:
            docid = [b""]
        if isinstance(docid, PdfReference):
            docid = self.resolver.resolve(docid)
        if not isinstance(docid, (list, tuple)) or len(docid) == 0:
            raise PdfUnsupportedError("Invalid trailer ID array")
        docid_list: Sequence[object] = docid

        try:
            handler = handler_cls(docid_list, encrypt_dict, password)
        except PDFPasswordIncorrect as exc:
            raise PdfUnsupportedError("Incorrect password") from exc
        except PDFEncryptionError as exc:
            raise PdfUnsupportedError("Invalid encryption dictionary") from exc
        self.decipher = handler.decrypt


__all__ = ("DocumentSecurityMixin",)

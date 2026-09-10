# SPDX-License-Identifier: AGPL-3.0-only
"""Document security initialization and authenticated extension validation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_07_security.pdf_mac import (
    validate_pdf_mac_extension,
    validate_pdf_mac_if_present,
)
from core_pdf_spec.s_07_security.standard import (
    StandardSecurityHandler,
    create_standard_security_handler,
)
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.types import Decipher, PdfDict
from core_pdf_spec.types import PdfByteBuffer, PdfReference


def initialize_document_security(
    data: PdfByteBuffer,
    trailer: PdfDict,
    resolver: ObjectResolver,
    password: str,
    *,
    handler_factory: Callable[
        [Sequence[object], PdfDict, str], StandardSecurityHandler
    ] = create_standard_security_handler,
) -> Decipher | None:
    encrypt_ref = trailer.get("Encrypt")
    if encrypt_ref is None:
        # ISO/TS 32004:2024, Table 5 defines AuthCode only for encrypted
        # documents whose Encrypt dictionary has V >= 5.
        if "AuthCode" in trailer:
            raise PdfUnsupportedError("AuthCode requires an encrypted document")
        return None

    encrypt_dict = resolver.resolve_dict(encrypt_ref)
    if not isinstance(encrypt_dict, dict):
        raise PdfUnsupportedError("Invalid Encrypt dictionary")

    docid: object = trailer.get("ID")
    if isinstance(docid, PdfReference):
        docid = resolver.resolve(docid)
    if not isinstance(docid, (list, tuple)) or len(docid) == 0:
        raise PdfUnsupportedError("Invalid trailer ID array")
    docid_list: Sequence[object] = docid

    security_handler = handler_factory(docid_list, encrypt_dict, password)
    # ISO/TS 32004:2024 integrity validation authenticates the complete
    # serialized file. Perform it before installing the object decipher so
    # no decrypted string, stream, catalog, or page can be exposed first.
    has_pdf_mac = validate_pdf_mac_if_present(
        data,
        trailer,
        security_handler,
    )
    decipher = security_handler.decrypt
    if has_pdf_mac:
        # ISO/TS 32004:2024, clause 4 and Table 1 require this exact
        # declaration. Its text-string fields are encrypted, so resolve it
        # only after authenticating the complete file and installing the
        # decipher, but still before returning the document to the caller.
        resolver.decipher = decipher
        catalog = resolver.resolve(trailer.get("Root"))
        if not isinstance(catalog, dict):
            raise PdfUnsupportedError("invalid trailer Root dictionary")
        extensions = resolver.resolve(catalog.get("Extensions"))
        iso_declarations: object = None
        if isinstance(extensions, dict):
            iso_declarations = resolver.resolve(extensions.get("ISO_"))
        if isinstance(iso_declarations, list):
            iso_declarations = [resolver.resolve(value) for value in iso_declarations]
        validate_pdf_mac_extension(iso_declarations)
    return decipher


__all__ = ("initialize_document_security",)

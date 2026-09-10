# SPDX-License-Identifier: AGPL-3.0-only
"""Reader recovery around strict document encryption and authentication."""

from __future__ import annotations

from collections.abc import Sequence

from core_pdf.impl.spec.s_07_security.document import (
    initialize_document_security as initialize_spec_document_security,
)
from core_pdf.impl.spec.s_07_security.standard import (
    create_standard_security_handler,
    create_standard_user_security_handler,
    internal_StandardSecurityHandler,
)
from core_pdf.impl.spec.s_07_syntax.resolver import ObjectResolver
from core_pdf.impl.spec.s_07_syntax.types import Decipher, PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import coerce_to_bytes, parse_int
from core_pdf.impl.types import PdfByteBuffer


def initialize_document_security(
    data: PdfByteBuffer, trailer: PdfDict, resolver: ObjectResolver, password: str
) -> Decipher | None:
    if trailer.get("Encrypt") is not None and trailer.get("ID") is None:
        # Preserve the reader's legacy handling of missing encrypted-file IDs.
        trailer = dict(trailer)
        trailer["ID"] = [b""]
    return initialize_spec_document_security(
        data, trailer, resolver, password, handler_factory=internal_create_security_handler
    )


def internal_create_security_handler(
    document_id: Sequence[object], params: PdfDict, password: str
) -> internal_StandardSecurityHandler:
    if internal_has_malformed_modern_owner(params):
        # PDFMiner-compatible recovery: R5/R6 user authentication does not use
        # the owner credential. All remaining dictionary inputs, U/UE and Perms
        # are still validated, and the caller still requires PDF MAC validation.
        return create_standard_user_security_handler(document_id, params, password)
    return create_standard_security_handler(document_id, params, password)


def internal_has_malformed_modern_owner(params: PdfDict) -> bool:
    if parse_int(params.get("V")) != 5 or parse_int(params.get("R")) not in (5, 6):
        return False
    try:
        owner_entry = coerce_to_bytes(params.get("O"))
        owner_encrypted_key = coerce_to_bytes(params.get("OE"))
    except TypeError:
        # Missing entries and invalid object types are outside this recovery.
        return False
    return len(owner_entry) != 48 or len(owner_encrypted_key) != 32

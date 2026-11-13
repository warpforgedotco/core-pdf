# SPDX-License-Identifier: AGPL-3.0-only

"""Encryption: negotiating the security handler, and recovering a damaged one.

The PdfDocument methods here are functions taking the document as self;
PdfDocument lists them as its methods.
"""

from __future__ import annotations

import mmap
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
)

from core_pdf.impl.base.exceptions import (
    PdfUnsupportedError,
)
from core_pdf.impl.base.types import (
    PdfName,
    PdfReference,
    PdfString,
)
from core_pdf.impl.base.values import recover_pdf_name
from core_pdf.impl.document.standards import (
    bootstrap_security_context,
    discover_document_standards,
    discover_header_standards,
    preserve_historical_version,
)
from core_pdf.impl.recovery.lexer import PdfLexer
from core_pdf.impl.recovery.resolver import ObjectResolver
from core_pdf.impl.spec.s_07_security.document import initialize_document_security
from core_pdf.impl.spec.s_07_security.standard import (
    StandardSecurityHandler,
    create_standard_security_handler,
)
from core_pdf.impl.spec.s_07_syntax.types import (
    PdfDict,
)
from core_pdf.impl.spec.s_07_syntax.xref import (
    key_for,
)
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_int
lazy from core_pdf_cythonized import (
    security_marker_present,
)

if TYPE_CHECKING:
    pass


from core_pdf.impl.document.navigation import (
    legacy_name_context,
)

if TYPE_CHECKING:
    from core_pdf.impl.document.pdf_document import PdfDocument


SECURITY_TRAILER_KEYS = frozenset({"Encrypt", "AuthCode"})


def has_security_metadata_marker(data: bytes | mmap.mmap) -> bool:
    # Encrypt or AuthCode anywhere, or the names /Encrypt and /AuthCode with #-escaped
    # bytes, in one compiled pass instead of two full-file finds and a regex search.
    return security_marker_present(data)


def check_security_aliases(trailer: PdfDict, resolver: ObjectResolver) -> None:
    pending: list[tuple[dict, bool]] = [(trailer, True)]
    seen: set[int] = set()
    security_resolver: ObjectResolver | None = None
    try:
        while pending:
            dictionary, top = pending.pop()
            if id(dictionary) in seen:
                continue
            seen.add(id(dictionary))
            names: set[bytes] = set()
            for key, value in dictionary.items():
                raw_name = key.value if isinstance(key, PdfName) else key
                if "#" in raw_name:
                    lexer = PdfLexer(b"/" + raw_name.encode("latin-1"))
                    try:
                        name = bytes(lexer.read_name())
                    finally:
                        lexer.close()
                else:
                    name = raw_name.encode("latin-1")
                if top and name not in {b"Encrypt", b"AuthCode", b"ID"}:
                    continue
                if name in names:
                    raise PdfUnsupportedError("Ambiguous security dictionary name aliases")
                names.add(name)
                if top and name in {b"Encrypt", b"AuthCode"}:
                    if security_resolver is None:
                        security_resolver = ObjectResolver(
                            resolver.data, resolver.xref, semantic_context=resolver.semantic_context
                        )
                    value = security_resolver.resolve(value)
                if isinstance(value, dict):
                    pending.append((value, False))
    finally:
        if security_resolver is not None:
            security_resolver.close()


def normalize_values(
    values: PdfDict, integer_fields: tuple[str, ...], name_fields: tuple[str, ...]
) -> PdfDict:
    normalized = values
    for name in integer_fields:
        value = values.get(name)
        number = parse_int(value)
        if number is not None and type(value) is not int:
            if normalized is values:
                normalized = dict(values)
            normalized[name] = number
    for name in name_fields:
        value = values.get(name)
        decoded = recover_pdf_name(value)
        if decoded is not None and not isinstance(value, PdfName) and decoded != value:
            if normalized is values:
                normalized = dict(values)
            normalized[name] = PdfName.of(decoded)
    return normalized


AES256_REVISION = 5
AES256_ENTRY_LENGTH = 48


def create_recovered_security_handler(
    document_id: Sequence[object], params: PdfDict, password: str | bytes = ""
) -> StandardSecurityHandler:
    normalized = normalize_values(
        params, ("V", "R", "P", "Length"), ("Filter", "StmF", "StrF", "EFF")
    )
    filters = params.get("CF")
    if isinstance(filters, dict):
        normalized_filters = filters
        for key, value in filters.items():
            name = recover_pdf_name(key)
            normalized_value = value
            if name == "StdCF" and isinstance(value, dict):
                normalized_value = normalize_values(
                    value, ("Length",), ("Type", "CFM", "AuthEvent")
                )
            normalized_key = (
                PdfName.of(name)
                if name is not None and not isinstance(key, PdfName) and name != key
                else key
            )
            if normalized_key != key or normalized_value is not value:
                if normalized_filters is filters:
                    normalized_filters = dict(filters)
                if normalized_key != key:
                    del normalized_filters[key]
                normalized_filters[normalized_key] = normalized_value
        if normalized_filters is not filters:
            if normalized is params:
                normalized = dict(params)
            normalized["CF"] = normalized_filters
    revision = parse_int(normalized.get("R"))
    if revision is not None and revision >= AES256_REVISION:
        # From revision 5, /O and /U hold a 32-byte hash and two 8-byte salts, and readers
        # use only those first 48 bytes, so trailing bytes some producers write are ignored.
        for name in ("O", "U"):
            value = normalized.get(name)
            data = value.data if isinstance(value, PdfString) else None
            if data is not None and len(data) > AES256_ENTRY_LENGTH:
                if normalized is params:
                    normalized = dict(params)
                normalized[name] = PdfString(bytes(data[:AES256_ENTRY_LENGTH]), is_literal=False)
    return create_standard_security_handler(document_id, normalized, password)


def negotiate_security(self: PdfDocument[Any], password: str | bytes) -> None:
    self._standards = discover_header_standards(self.raw_data)
    header = self._standards
    context = header.context if legacy_name_context(header.context) else None
    self.resolver = ObjectResolver(self.raw_data, self.xref, semantic_context=context)
    self.rescan_xref()
    if legacy_name_context(context):
        selected = bootstrap_security_context(header, self.raw_data, self.xref, self.trailer_dict)
        if not legacy_name_context(selected):
            check_security_aliases(self.trailer_dict, self.resolver)
            self.resolver.semantic_context = selected
            self.rescan_xref()

    for attempt in range(2):
        self.init_security(password)
        self.resolver.decipher = self.decipher
        self._standards = discover_document_standards(header, self.resolver, self.trailer_dict)
        self._standards = preserve_historical_version(
            self._standards,
            self.raw_data,
            self.trailer_dict,
            self.decipher,
            recovered=self.xref_was_recovered,
            trailer_context=self.resolver.semantic_context,
        )
        selected = self._standards.context
        if legacy_name_context(selected) == legacy_name_context(self.resolver.semantic_context):
            encrypt_ref = self.trailer_dict.get("Encrypt")
            encrypt_object = (
                self.resolver.resolve(encrypt_ref)
                if self.decipher is not None and isinstance(encrypt_ref, PdfReference)
                else None
            )
            self.resolver.semantic_context = selected
            if encrypt_object is not None and isinstance(encrypt_ref, PdfReference):
                self.resolver.objects[
                    key_for(encrypt_ref.object_number, encrypt_ref.generation_number)
                ] = encrypt_object
            break
        if attempt:
            raise PdfUnsupportedError("Unstable security dictionary name semantics")
        if not legacy_name_context(selected):
            check_security_aliases(self.trailer_dict, self.resolver)
        self.resolver.close()
        self.decipher = None
        self.resolver = ObjectResolver(self.raw_data, self.xref, semantic_context=selected)
        self.rescan_xref()


def rescan_xref(self: PdfDocument[Any]) -> None:
    self.scan_xref()
    self.resolver.xref = self.xref


def init_security(self: PdfDocument[Any], password: str | bytes) -> None:
    trailer = self.trailer_dict
    if trailer.get("Encrypt") is not None and trailer.get("ID") is None:
        trailer = dict(trailer)
        trailer["ID"] = [b""]
    self.decipher = initialize_document_security(
        self.raw_data,
        trailer,
        self.resolver,
        password,
        handler_factory=create_recovered_security_handler,
    )

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence

from core_pdf.impl.document_contracts import DocumentState
from core_pdf.impl.document_standards import (
    bootstrap_security_context,
    discover_document_standards,
    discover_header_standards,
    preserve_historical_version,
)
from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf.impl.recovery_resolver import ObjectResolver
from core_pdf.impl.types import PdfName, PdfReference
from core_pdf_spec.s_07_security.document import initialize_document_security
from core_pdf_spec.s_07_security.standard import (
    StandardSecurityHandler,
    create_standard_security_handler,
)
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_07_syntax_primitives.coercion import parse_int
from core_pdf_spec.standards import PdfVersion, SemanticContext


def legacy_name_context(context: SemanticContext | None) -> bool:
    return context is not None and context.version in {PdfVersion(1, 0), PdfVersion(1, 1)}


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
                lexer = PdfLexer(b"/" + raw_name.encode("latin-1"))
                try:
                    name = bytes(lexer.read_name())
                finally:
                    lexer.close()
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


def create_recovered_security_handler(
    document_id: Sequence[object], params: PdfDict, password: str = ""
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
    return create_standard_security_handler(document_id, normalized, password)


class SecuritySetupMixin(DocumentState):
    __slots__ = ()

    def negotiate_security(self, password: str) -> None:
        self._standards = discover_header_standards(self.raw_data)
        header = self._standards
        context = header.context if legacy_name_context(header.context) else None
        self.resolver = ObjectResolver(self.raw_data, self.xref, semantic_context=context)
        self.scan_xref()
        self.resolver.xref = self.xref
        if legacy_name_context(context):
            selected = bootstrap_security_context(
                header, self.raw_data, self.xref, self.trailer_dict
            )
            if not legacy_name_context(selected):
                check_security_aliases(self.trailer_dict, self.resolver)
                self.resolver.semantic_context = selected
                self.scan_xref()
                self.resolver.xref = self.xref

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
            self.scan_xref()
            self.resolver.xref = self.xref

    def init_security(self, password: str) -> None:
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


__all__ = (
    "SecuritySetupMixin",
    "check_security_aliases",
    "create_recovered_security_handler",
    "legacy_name_context",
    "normalize_values",
)

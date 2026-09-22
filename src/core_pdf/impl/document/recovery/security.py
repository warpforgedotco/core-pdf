# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf_spec.s_07_security.standard import (
    StandardSecurityHandler,
    create_standard_security_handler,
)
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax_primitives.coercion import parse_int
from core_pdf_spec.types import PdfName


def internal_normalize_values(
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
    normalized = internal_normalize_values(
        params, ("V", "R", "P", "Length"), ("Filter", "StmF", "StrF", "EFF")
    )
    filters = params.get("CF")
    if isinstance(filters, dict):
        normalized_filters = filters
        for key, value in filters.items():
            name = recover_pdf_name(key)
            normalized_value = value
            if name == "StdCF" and isinstance(value, dict):
                normalized_value = internal_normalize_values(
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
            normalized["CF"] = cast(PdfDict, normalized_filters)
    return create_standard_security_handler(document_id, normalized, password)

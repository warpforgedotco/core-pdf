# SPDX-License-Identifier: AGPL-3.0-only
"""Reader handling of malformed resource dictionaries."""

from __future__ import annotations

from core_pdf.impl.spec.s_07_syntax.resources import resolve_resource_dict as resolve_spec_resources
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver


def resolve_resource_dict(value: object, resolver: PdfValueResolver) -> PdfDict | None:
    return resolve_spec_resources(value, resolver, on_invalid=lambda value: None)

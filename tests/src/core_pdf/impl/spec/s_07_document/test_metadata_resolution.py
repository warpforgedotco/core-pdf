# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import cast

from core_pdf.impl.primitives import PdfName
from core_pdf.impl.spec.s_07_document.metadata import resolve_info_metadata
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver
from tests.helpers.resolvers import IdentityResolver


def test_trapped_info_value_accepts_pdf_name() -> None:
    info = {"Trapped": PdfName.of("False")}

    result = resolve_info_metadata(
        cast(PdfValueResolver, IdentityResolver()), cast(PdfDict, {"Info": info})
    )

    assert result["Trapped"] == PdfName.of("False")

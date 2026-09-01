# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from operator import eq
from typing import Any, cast

from core_pdf.impl.primitives import PdfName


def test_pdf_name_bytes_equality_supports_mapping_lookup() -> None:
    name = PdfName.of(b"N\xe1me")
    mapping = {name: "value"}
    cross_type_mapping = cast(dict[Any, str], mapping)

    assert name == b"N\xe1me"
    assert eq(b"N\xe1me", name)
    assert hash(name) == hash(b"N\xe1me")
    assert cross_type_mapping[b"N\xe1me"] == "value"

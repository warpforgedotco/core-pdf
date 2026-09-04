# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import cast

import pytest

from core_pdf.impl.primitives import PdfName
from core_pdf.impl.spec.s_07_document.metadata import (
    parse_xmp_metadata,
    resolve_info_metadata,
)
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver
from tests.helpers.resolvers import IdentityResolver


def test_trapped_info_value_accepts_pdf_name() -> None:
    info = {"Trapped": PdfName.of("False")}

    result = resolve_info_metadata(
        cast(PdfValueResolver, IdentityResolver()), cast(PdfDict, {"Info": info})
    )

    assert result["Trapped"] == PdfName.of("False")


def xmp_stream(payload: bytes) -> PdfStream:
    return PdfStream({}, b"", decoded_data=payload)


def test_xmp_metadata_parses_a_well_formed_packet() -> None:
    packet = xmp_stream(
        b'<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf>hello</rdf></x:xmpmeta>'
    )

    result = parse_xmp_metadata(packet)

    assert result is not None
    assert result["tag"] == "xmpmeta"
    assert [child["tag"] for child in result["children"]] == ["rdf"]


def test_xmp_metadata_rejects_entity_expansion() -> None:
    # A "billion laughs" packet: ElementTree expands these declarations without
    # limit, so parsing must refuse the entities rather than materialise them.
    bomb = xmp_stream(
        b'<?xml version="1.0"?>'
        b"<!DOCTYPE lolz ["
        b'<!ENTITY lol "lol">'
        b'<!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
        b'<!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">'
        b'<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
        b"]>"
        b"<lolz>&lol3;</lolz>"
    )

    with pytest.raises(ValueError, match="invalid XMP metadata"):
        parse_xmp_metadata(bomb)

    assert parse_xmp_metadata(bomb, recover=True) == {"parse_error": "invalid XMP metadata"}


def test_xmp_metadata_rejects_external_entity_references() -> None:
    external = xmp_stream(
        b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><r>&xxe;</r>'
    )

    assert parse_xmp_metadata(external, recover=True) == {"parse_error": "invalid XMP metadata"}

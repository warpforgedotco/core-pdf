# SPDX-License-Identifier: AGPL-3.0-only
from core_pdf.impl.engine.spec.s_07_syntax.text_string import decode_pdf_text_string


def test_decode_pdfdoc_encoding_accent_and_quote_bytes() -> None:
    assert decode_pdf_text_string(bytes(range(0x18, 0x20))) == "˘ˇˆ˙˝˛˚˜"
    assert decode_pdf_text_string(bytes(range(0x8D, 0x91))) == "“”‘’"


def test_decode_pdf_text_string_accepts_utf8_bom() -> None:
    assert decode_pdf_text_string(b"\xef\xbb\xbfPrice \xe2\x82\xac") == "Price €"

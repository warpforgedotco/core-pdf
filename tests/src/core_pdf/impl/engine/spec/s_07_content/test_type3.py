from types import SimpleNamespace

from core_pdf.impl.engine.spec.s_07_content.capture import type3_glyph_names
from core_pdf.impl.objects import PdfName


def test_type3_names_include_base_encoding_and_differences() -> None:
    decoder = SimpleNamespace(base_encoding="StandardEncoding")
    names = type3_glyph_names(
        {"Encoding": {"BaseEncoding": "StandardEncoding", "Differences": [65, PdfName(b"custom")]}},
        decoder,
    )

    assert names[32] == "space"
    assert names[65] == "custom"
    assert 1 not in names

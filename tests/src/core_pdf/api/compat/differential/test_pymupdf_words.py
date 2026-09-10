import re

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_geometry import internal_geometry_expected
from .test_pymupdf_text import internal_unicode_source, real_pymupdf

pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize(
    "codepoints",
    [
        [0x200D],
        [0x200D, 65],
        [65, 0x200D, 66],
        [65, 0x202A, 66],
        [65, 0x202E, 66],
        [65, 0x5D0, 66],
        [65, 0x661, 66],
    ],
)
@pytest.mark.parametrize("delimiters", [None, "A"])
def test_word_boundaries_for_joiners_and_directional_characters(
    codepoints: list[int], delimiters: str | None
) -> None:
    with real_pymupdf.open(stream=internal_unicode_source(codepoints)) as document:
        for xref in document[0].get_contents():
            content = document.xref_stream(xref)
            content = re.sub(
                rb"<([0-9a-fA-F]+)>",
                lambda match: (
                    b"<" + bytes.fromhex(match[1].decode()).replace(b" ", b"").hex().encode() + b">"
                ),
                content,
            )
            document.update_stream(xref, content)
        source = document.tobytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text("words", delimiters=delimiters) == internal_geometry_expected(
            expected[0].get_text("words", delimiters=delimiters)
        )

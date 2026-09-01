# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
import zlib

from core_pdf.impl.spec.s_09_fonts.font_program_truetype import tt_font_for_data
from tests.helpers.paths import score_bench_pdf


def internal_first_embedded_truetype(pdf: bytes) -> bytes:
    match = re.search(rb"/FontFile2\s+(\d+)\s+0\s+R", pdf)
    assert match is not None
    obj = int(match.group(1))
    stream = re.search(rb"[^0-9]%d 0 obj(.*?)stream\r?\n" % obj, pdf, re.S)
    assert stream is not None
    body = pdf[stream.end() : pdf.find(b"endstream", stream.end())]
    return zlib.decompress(body) if b"FlateDecode" in stream.group(1) else body


def test_macintosh_only_cmap_resolves_glyphs() -> None:
    """Regression: subset fonts with no Unicode cmap drew from raw codes.

    A macOS-exported subset carries only a Macintosh (1,0) subtable. The
    best-cmap lookup finds no Unicode table, and the character codes then
    fell through as glyph ids -- indices into a subset whose order has
    nothing to do with them, so every glyph on the page was wrong.
    """
    fixture = score_bench_pdf("fhhd0346-p009.pdf")
    program = tt_font_for_data(
        internal_first_embedded_truetype(fixture.read_bytes()), None, use_cmap=True
    )
    assert program.cmap, "a Macintosh-only cmap must still resolve character codes"

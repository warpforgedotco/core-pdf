# SPDX-License-Identifier: AGPL-3.0-only
"""Both legacy text projections resolve predefined CMap names the same way.

pypdf maps a handful of predefined CMap names straight onto a Python codec
rather than loading the CMap. The pypdf facade emulates pypdf, and the
LlamaIndex facade emulates llama_index's PDFReader, which wraps pypdf -- so
the two facades have to agree, and both have to agree with pypdf.

They did not. The table was copied into each facade, and the LlamaIndex copy
lost the four Uni*-UTF16-* families, so a CJK font named by one of them fell
through to a 256-entry Latin table and decoded its two-byte codes as single
bytes. One shared table, pinned here against the vendored pypdf source.
"""

from __future__ import annotations

import re

from core_pdf.api.compat._text_state import internal_PREDEFINED_ENCODING_CODECS
from core_pdf.api.compat.llamaindex import _operator_text as llamaindex_text
from core_pdf.api.compat.pypdf import _text as pypdf_text
from tests.helpers.paths import FIXTURES


def internal_upstream_predefined_cmap() -> dict[str, str]:
    """pypdf's own _predefined_cmap, read from the vendored upstream source."""
    source = (FIXTURES / "pypdf" / "pypdf" / "_cmap.py").read_text()
    block = re.search(r"_predefined_cmap[^=]*=\s*\{(.*?)\n\}", source, re.S)
    assert block is not None, "pypdf._cmap no longer defines _predefined_cmap"
    # Core normalizes names, so the upstream keys lose their leading slash.
    return dict(re.findall(r'"/([^"]+)":\s*"([^"]+)"', block.group(1)))


def test_both_facades_share_one_table() -> None:
    assert (
        llamaindex_text.internal_PREDEFINED_ENCODING_CODECS is internal_PREDEFINED_ENCODING_CODECS
    )
    assert pypdf_text.internal_PREDEFINED_ENCODING_CODECS is internal_PREDEFINED_ENCODING_CODECS


def test_table_matches_upstream_pypdf() -> None:
    assert internal_PREDEFINED_ENCODING_CODECS == internal_upstream_predefined_cmap()


def test_the_utf16_cmap_families_are_present() -> None:
    """The entries the LlamaIndex copy had dropped."""
    assert internal_PREDEFINED_ENCODING_CODECS["UniGB-UTF16-H"] == "gb18030"
    assert internal_PREDEFINED_ENCODING_CODECS["UniGB-UTF16-V"] == "gb18030"
    assert internal_PREDEFINED_ENCODING_CODECS["UniCNS-UTF16-H"] == "utf-16-be"
    assert internal_PREDEFINED_ENCODING_CODECS["UniJIS-UTF16-H"] == "utf-16-be"

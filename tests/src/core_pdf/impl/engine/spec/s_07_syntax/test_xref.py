# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_syntax.primitives import PdfName
from core_pdf.impl.engine.spec.s_07_syntax.xref import XRefScanner, key_for


def build_xref_prev_chain(section_count: int) -> tuple[bytes, int]:
    data = bytearray(b"%PDF-1.4\n")
    prev: int | None = None
    for index in range(section_count):
        obj_num = index + 1
        offset = len(data)
        data.extend(f"xref\n{obj_num} 1\n0000000000 00000 n \ntrailer\n<< ".encode())
        data.extend(f"/Size {section_count + 1}".encode())
        if prev is not None:
            data.extend(f" /Prev {prev}".encode())
        data.extend(b" >>\n")
        prev = offset

    assert prev is not None
    return bytes(data), prev


def test_xref_prev_chain_loads_iteratively() -> None:
    data, start = build_xref_prev_chain(1500)

    entries, trailer = XRefScanner.load_section_chain(data, start, set())

    assert len(entries) == 1500
    assert key_for(1) in entries
    assert key_for(1500) in entries
    assert trailer[PdfName.of(b"Size")] == 1501

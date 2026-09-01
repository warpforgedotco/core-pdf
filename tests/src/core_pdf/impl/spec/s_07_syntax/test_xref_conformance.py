# SPDX-License-Identifier: AGPL-3.0-only
"""Cross-reference conformance rules from ISO 32000-1 clause 7.5.

Each test names the clause it pins. The deletion case is the load-bearing one:
a free entry that fails to shadow the object it deletes leaves content that a
conforming reader reports as null fully readable, which defeats redaction
performed the way 7.5.6 prescribes.
"""

from __future__ import annotations

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.xref import XRefScanner, XRefTable

BASE_OBJECTS = {
    1: b"<< /Type /Catalog /Pages 2 0 R /Custom 5 0 R >>",
    2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>",
    5: b"<< /Secret (BASE-OBJECT-5) >>",
}


def internal_base_revision() -> tuple[bytes, int, dict[int, int]]:
    out = bytearray(b"%PDF-1.7\n")
    offsets: dict[int, int] = {}
    for number in sorted(BASE_OBJECTS):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + BASE_OBJECTS[number] + b"\nendobj\n"
    xref_offset = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n"
    for number in (1, 2, 3):
        out += b"%010d 00000 n \n" % offsets[number]
    out += b"0000000000 65535 f \n"
    out += b"%010d 00000 n \n" % offsets[5]
    out += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref_offset
    return bytes(out), xref_offset, offsets


def internal_freeing_update() -> bytes:
    """A second revision that deletes object 5 exactly as 7.5.4/7.5.6 prescribe."""
    base, base_xref, _ = internal_base_revision()
    out = bytearray(base)
    update_xref = len(out)
    # 7.5.4: the free entry carries the generation to use *next*, so 1, not 0.
    out += b"xref\n0 1\n0000000000 65535 f \n5 1\n0000000000 00001 f \n"
    out += b"trailer\n<< /Size 6 /Root 1 0 R /Prev %d >>\nstartxref\n%d\n%%%%EOF\n" % (
        base_xref,
        update_xref,
    )
    return bytes(out)


def internal_merged_table(data: bytes) -> XRefTable:
    start = XRefScanner.find_startxref(data)
    assert start is not None
    entries, _ = XRefScanner.load_section_chain(data, start, set())
    return entries


def test_object_freed_by_an_incremental_update_is_not_reachable() -> None:
    """7.5.6: the deleted body stays in the file, but the newest xref wins.

    7.5.4 makes the free entry's generation the *next* one, so it cannot be
    keyed by generation and still shadow the entry it supersedes.
    """
    entries = internal_merged_table(internal_freeing_update())

    live = {key >> 16 for key, entry in entries.items() if entry.in_use}
    assert 5 not in live, "the freed object is still reachable through the xref"

    freed = [entry for key, entry in entries.items() if key >> 16 == 5]
    assert freed
    assert all(not entry.in_use for entry in freed)


def test_base_revision_alone_still_resolves_the_object() -> None:
    """Control: without the freeing update, object 5 is live."""
    base, _, _ = internal_base_revision()

    entries = internal_merged_table(base)

    live = {key >> 16 for key, entry in entries.items() if entry.in_use}
    assert 5 in live


def test_unknown_xref_stream_entry_type_yields_null_not_a_parse_error() -> None:
    """7.5.8.3: "Any other value shall be interpreted as a reference to the null object"."""
    rows = bytearray()
    for number in range(4):
        rows += bytes((1,)) + (100 + number).to_bytes(4, "big") + (0).to_bytes(2, "big")
    # A forward-compatible type the current spec does not define.
    rows += bytes((3,)) + (0).to_bytes(4, "big") + (0).to_bytes(2, "big")

    stream = internal_xref_stream(bytes(rows), size=5)
    entries, _ = XRefScanner.parse_stream(stream)

    entry = entries[(4 << 16)]
    assert not entry.in_use


def internal_xref_stream(rows: bytes, *, size: int) -> PdfStream:
    return PdfStream(
        {"Type": "XRef", "Size": size, "W": [1, 4, 2]},
        rows,
        decoded_data=rows,
    )

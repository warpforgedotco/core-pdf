import base64

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 3276, 3277, 4096, 4097, 4098, 4099])
@pytest.mark.parametrize("zero_groups", [False, True])
@pytest.mark.parametrize("whitespace", [False, True])
def test_ascii85_stream_decoding(length: int, zero_groups: bool, whitespace: bool) -> None:
    # Both sides of the bulk decoder boundary, all final-tuple lengths, and
    # zero-quad abbreviations must produce the same raw stream bytes.
    unit = b"\x00\x00\x00\x00PDF\xff" if zero_groups else b"PDF\xff"
    payload = (unit * ((length + len(unit) - 1) // len(unit)))[:length]
    encoded = base64.a85encode(payload) + b"~>"
    if whitespace:
        encoded = b"\x00\t\r\n\x0c ".join(
            encoded[index : index + 1] for index in range(len(encoded))
        )
    with real_pymupdf.open() as fixture:
        fixture.new_page()
        xref = fixture.get_new_xref()
        fixture.update_object(xref, "<< >>")
        fixture.update_stream(xref, encoded, compress=False)
        fixture.xref_set_key(xref, "Filter", "/ASCII85Decode")
        source = fixture.tobytes()

    with (
        real_pymupdf.open(stream=source) as reference,
        compat_pymupdf.open(stream=source) as actual,
    ):
        assert actual.xref_stream(xref) == reference.xref_stream(xref) == payload

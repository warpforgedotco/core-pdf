from io import BytesIO
from typing import Any

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

real_pymupdf = pytest.importorskip("pymupdf")
real_pypdf = pytest.importorskip("pypdf")
pytestmark = pytest.mark.compat_differential


def internal_compare_metadata(writer: Any) -> None:
    stream = BytesIO()
    writer.write(stream)
    with (
        real_pymupdf.open(stream=stream.getvalue()) as expected,
        compat_pymupdf.open(stream=stream.getvalue()) as actual,
    ):
        assert actual.metadata == expected.metadata


@pytest.mark.parametrize("header", [b"%PDF-1.3 ", b"%PDF-1.4", b"%PDF-1.7"])
@pytest.mark.parametrize("catalog_version", [None, "1.2", "2.0"])
def test_effective_metadata_version(header: bytes, catalog_version: str | None) -> None:
    writer = real_pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=300)
    writer.pdf_header = header
    if catalog_version is not None:
        writer.root_object[real_pypdf.generic.NameObject("/Version")] = (
            real_pypdf.generic.NameObject("/" + catalog_version)
        )
    internal_compare_metadata(writer)


@pytest.mark.parametrize(
    "raw",
    [
        b"a\0b",
        b"\xfe\xff\0a\0\0\0b",
        b"\xff\xfea\0\0\0b\0",
        b"\xef\xbb\xbfa\0b",
        b"abc\0",
        b"\x9f\xad",
    ],
)
def test_metadata_null_characters_follow_string_encoding(raw: bytes) -> None:
    writer = real_pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=300)
    writer._info.get_object()[real_pypdf.generic.NameObject("/Title")] = (
        real_pypdf.generic.ByteStringObject(raw)
    )
    internal_compare_metadata(writer)


@pytest.mark.parametrize("algorithm", ["RC4-40", "RC4-128", "AES-128", "AES-256-R5", "AES-256"])
def test_authenticated_metadata_describes_encryption(algorithm: str) -> None:
    writer = real_pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=300)
    writer.encrypt("", owner_password="owner", algorithm=algorithm)
    internal_compare_metadata(writer)


def test_aes_metadata_uses_crypt_filter_key_length_when_unspecified() -> None:
    writer = real_pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=300)
    writer.encrypt("", owner_password="owner", algorithm="AES-128")
    del writer._encrypt_entry[real_pypdf.generic.NameObject("/Length")]
    internal_compare_metadata(writer)

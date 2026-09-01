# SPDX-License-Identifier: AGPL-3.0-only
"""Dogfood the security specifications and compare their text with the code.

The source PDFs are copyrighted and intentionally gitignored. The public Adobe
documents used in CI are fetched with ``scripts/fetch_pdf_specs.sh security``;
the sponsored ISO 32000-2, ISO/TS 32003, and ISO/TS 32004 documents are checked
when a local copy is available.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.primitives import PdfName
from core_pdf.impl.spec.s_07_security.ciphers import (
    internal_AES_GCM_IV_BYTES,
    internal_AES_GCM_KEY_BYTES,
    internal_AES_GCM_MAX_PLAINTEXT_BYTES,
    internal_AES_GCM_TAG_BYTES,
)
from core_pdf.impl.spec.s_07_security.standard import (
    internal_parse_config,
    internal_parse_crypt_filters,
    internal_RESERVED_ONE_PERMISSION_BITS,
    internal_RESERVED_ONE_PERMISSION_MASK,
    internal_RESERVED_ZERO_PERMISSION_BITS,
    internal_RESERVED_ZERO_PERMISSION_MASK,
    internal_supported_revisions,
)
from core_pdf.impl.spec.s_07_syntax.types import PdfDict

internal_SPECS = Path(__file__).resolve().parents[5] / "fixtures" / "specifications" / "PDF"
internal_ADOBE_PDF_17 = "PDFReference-1.7-Adobe-2006.pdf"
internal_ISO_32000_1 = "ISO32000-1-2008-PDF-1.7.pdf"
internal_ISO_32000_2 = "ISO32000-2-2020-PDF-2.0-EC3.pdf"
internal_ISO_TS_32003 = "ISO-TS-32003-2023-AES-GCM.pdf"
internal_ISO_TS_32004 = "ISO-TS-32004-2024-Integrity-Protection.pdf"

internal_PUBLIC_SPEC_SHA256 = {
    internal_ADOBE_PDF_17: "4aa598e7e2cc88867565fae793db38c140a10ca2b5551dccdb45e2124282ce29",
    internal_ISO_32000_1: "9de0ca9e8570d6209e8bd48a355be8eb6ec376acfc3fc3ae97cd8730351417ff",
}


def internal_specification(name: str) -> Path:
    path = internal_SPECS / name
    if not path.is_file():
        if name in internal_PUBLIC_SPEC_SHA256:
            pytest.skip(f"fetch {name} with scripts/fetch_pdf_specs.sh security")
        pytest.skip(f"provide the sponsored {name} source locally; see the fixture README")
    expected_digest = internal_PUBLIC_SPEC_SHA256.get(name)
    if expected_digest is not None:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        assert digest == expected_digest, f"unexpected source revision for {name}"
    return path


def internal_page_text(document: PdfDocument, *page_indexes: int) -> str:
    return " ".join(
        " ".join(document.pages[page_index].extract().text.split()) for page_index in page_indexes
    )


def internal_mask(bit_positions: tuple[int, ...]) -> int:
    return sum(1 << (bit_position - 1) for bit_position in bit_positions)


def test_pdf_reference_1_7_permission_rows_match_the_implementation() -> None:
    """Adobe PDF Reference 1.7, sixth edition (November 2006), Table 3.20."""
    with PdfDocument.open(internal_specification(internal_ADOBE_PDF_17)) as document:
        assert len(document.pages) == 1310
        text = internal_page_text(document, 122, 123)

    assert "TABLE 3.20 User access permissions" in text
    rows = [
        (int(match[0]), int(match[1]), int(match[2]))
        for match in re.findall(
            r"(\d+)\s*[–-]\s*(\d+)\s+"
            r"(?:\(Revision 3 or greater\)\s+)?Reserved;\s+must be ([01])\.",
            text,
        )
    ]
    assert rows == [(1, 2, 0), (7, 8, 1), (13, 32, 1)]

    documented_zero_bits = tuple(
        bit for first, last, required in rows if required == 0 for bit in range(first, last + 1)
    )
    documented_one_bits = tuple(
        bit for first, last, required in rows if required == 1 for bit in range(first, last + 1)
    )
    assert internal_RESERVED_ZERO_PERMISSION_BITS == documented_zero_bits
    assert internal_RESERVED_ONE_PERMISSION_BITS == documented_one_bits
    assert internal_RESERVED_ZERO_PERMISSION_MASK == internal_mask(documented_zero_bits)
    assert internal_RESERVED_ONE_PERMISSION_MASK == internal_mask(documented_one_bits)


def test_iso_32000_1_security_rules_and_its_own_encryption_are_parseable() -> None:
    """ISO 32000-1:2008, 7.6.3.2 and Tables 21-22."""
    # This authorized ISO copy is itself encrypted with V=1, R=3, and P=-28.
    # Opening it therefore guards the tolerant reader behavior separately from
    # the prose assertions below.
    with PdfDocument.open(internal_specification(internal_ISO_32000_1)) as document:
        assert len(document.pages) == 756
        table_21 = internal_page_text(document, 67)
        table_22 = internal_page_text(document, 68)

    assert "PDF 32000-1:2008" in table_21
    assert re.search(r"2\s*if the document is encrypted with a V value less than 2", table_21)
    assert "3 if the document is encrypted with a V value of 2 or 3" in table_21
    assert "access permissions set to 0" in table_21
    assert "reserved high-order flag bits" in table_22
    assert "required to be 1" in table_22


def test_iso_32000_2_permission_rows_are_parseable_when_available() -> None:
    """ISO 32000-2:2020, Errata Collection 3, 7.6.4.2 and Table 22."""
    with PdfDocument.open(internal_specification(internal_ISO_32000_2)) as document:
        assert len(document.pages) == 1023
        text = internal_page_text(document, 93, 94)

    assert "Table 22 — Standard security handler user access permissions" in text
    assert "1 - 2 Reserved. Must be zero (0)." in text
    assert "7 - 8 Reserved. Must be 1." in text
    assert "13 - 32 (Security handlers of revision 3 or greater) Reserved. Must be 1." in text
    assert "(Security handlers of revision 2) Print the document." in text


def test_iso_ts_32003_aes_gcm_rules_match_the_implementation_when_available() -> None:
    """ISO/TS 32003:2023, Tables 2-4 and 5.2."""
    with PdfDocument.open(internal_specification(internal_ISO_TS_32003)) as document:
        assert len(document.pages) == 13
        tables = internal_page_text(document, 9)
        object_encryption = internal_page_text(document, 10)

    assert "introduced a value of 6 for V which supports AES-GCM" in tables
    assert "declares at least one crypt filter using the AESV4 method" in tables
    assert "R number" in tables
    assert "(Required) 7 (ISO/TS 32003)" in tables
    assert "CFM name AESV4" in tables
    assert "same manner as for AESV3" in tables
    assert "32-byte crypt filter encryption key" in tables
    assert "initialization vector (IV) shall be 12 bytes" in tables
    assert "block size parameter shall be set to 16 bytes" in tables

    assert "AAD input to the AES-GCM algorithm shall be nil" in object_encryption
    assert "first 12 bytes of encrypted output" in object_encryption
    assert "16-byte GCM authentication tag" in object_encryption
    assert "(2³⁹ - 256) bytes of plaintext" in object_encryption
    assert "password algorithms used shall be the same" in object_encryption
    assert "standard security handler of revision 6" in object_encryption

    assert internal_supported_revisions(6) == (7,)
    assert internal_AES_GCM_KEY_BYTES == 32
    assert internal_AES_GCM_IV_BYTES == 12
    assert internal_AES_GCM_TAG_BYTES == 16
    assert internal_AES_GCM_MAX_PLAINTEXT_BYTES == (1 << 39) - 256

    params: PdfDict = {
        "CF": {
            "StdCF": {
                "CFM": PdfName.of("AESV4"),
                "AuthEvent": PdfName.of("DocOpen"),
                "Length": 32,
            }
        },
        "StmF": PdfName.of("StdCF"),
        "StrF": PdfName.of("StdCF"),
    }
    _, stream_filter, string_filter, _, crypt_filters = internal_parse_crypt_filters(params, 6)
    assert stream_filter == string_filter == "StdCF"
    assert crypt_filters == {"StdCF": "AESV4"}


def test_iso_ts_32004_bit_13_extension_fails_closed_when_available() -> None:
    """ISO/TS 32004:2024, 5.1.2 and Table 3."""
    with PdfDocument.open(internal_specification(internal_ISO_TS_32004)) as document:
        assert len(document.pages) == 25
        text = internal_page_text(document, 10)

    assert "Table 3 — Additions to ISO 32000-2:2020, Table 22" in text
    assert "13 When zero, indicates that a PDF MAC token is required" in text
    assert "unless bit 13 is zero in all revisions" in text

    # core-pdf does not implement ISO/TS 32004:2024 AuthCode validation yet.
    # Reject its bit-13 signal instead of silently accepting unverified content.
    permissions_with_bit_13_clear = 0xFFFFFFFC & ~(1 << 12)
    with pytest.raises(ValueError, match="reserved encryption permission bits must be one"):
        internal_parse_config(
            [b"document-id"],
            {"R": 6, "P": permissions_with_bit_13_clear},
            5,
            (5, 6),
        )

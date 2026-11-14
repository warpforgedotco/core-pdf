# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 2: the specifications core implements, and the marker citing one from code.

Every rule whose meaning depends on the PDF version, and every feature one edition, extension
or external standard adds, carries a `cites` marker naming the specification and clause. A
module that implements one specification starts its docstring with the same id and clause,
for instance "ISO 32000-2 7.8: ...". `scripts/spec_index.py` reads both from the source and
writes docs/spec-index.md, the code by specification and by PDF version.

Profiles are cited by their identifier in `core_pdf.impl.spec.standards.STANDARD_PROFILES`,
and extensions as "PREFIX/base version/level" of `IMPLEMENTED_EXTENSIONS`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from core_pdf.impl.spec.standards import IMPLEMENTED_EXTENSIONS, PdfVersion, get_standard_profile
from core_records import GeneratedRecord

Cited = TypeVar("Cited", bound=Callable[..., object])


class Specification(GeneratedRecord):
    title: str
    edition: str
    reference_url: str
    # What it is: the PDF standard itself, an ISO extension of it, or a standard PDF embeds.
    kind: str


SPECIFICATIONS: dict[str, Specification] = {
    "PDF Reference": Specification(
        "Adobe PDF Reference (PDF 1.0 to 1.6)",
        "first to fifth editions, 1993-2004",
        "https://pdfa.org/resource/pdf-specification-archive/",
        "pdf",
    ),
    "ISO 32000-1": Specification(
        "Document management - Portable document format - Part 1: PDF 1.7",
        "ISO 32000-1:2008",
        "https://pdfa.org/resource/iso-32000-1/",
        "pdf",
    ),
    "ISO 32000-2": Specification(
        "Document management - Portable document format - Part 2: PDF 2.0",
        "ISO 32000-2:2020, with the errata of PDF_2_0_BASELINE",
        "https://pdfa.org/resource/iso-32000-2/",
        "pdf",
    ),
    "ISO 32003": Specification(
        "Adding support of AES-GCM in PDF 2.0",
        "ISO 32003:2023",
        "https://www.iso.org/standard/45876.html",
        "extension",
    ),
    "ISO 32004": Specification(
        "Integrity protection in encrypted documents in PDF 2.0",
        "ISO 32004:2024",
        "https://www.iso.org/standard/45877.html",
        "extension",
    ),
    "ITU-T T.88": Specification(
        "JBIG2 bi-level image compression",
        "ITU-T T.88 (2018)",
        "https://www.itu.int/rec/T-REC-T.88",
        "embedded",
    ),
    "ITU-T T.4": Specification(
        "CCITT Group 3 facsimile coding",
        "ITU-T T.4 (2003)",
        "https://www.itu.int/rec/T-REC-T.4",
        "embedded",
    ),
    "ITU-T T.6": Specification(
        "CCITT Group 4 facsimile coding",
        "ITU-T T.6 (1988)",
        "https://www.itu.int/rec/T-REC-T.6",
        "embedded",
    ),
    "ISO 10918-1": Specification(
        "JPEG (DCT) image compression",
        "ISO/IEC 10918-1:1994",
        "https://www.iso.org/standard/18902.html",
        "embedded",
    ),
    "ISO 15444-1": Specification(
        "JPEG 2000 image coding",
        "ISO/IEC 15444-1:2019",
        "https://www.iso.org/standard/78321.html",
        "embedded",
    ),
    "RFC 1950": Specification(
        "ZLIB compressed data format",
        "RFC 1950",
        "https://www.rfc-editor.org/rfc/rfc1950",
        "embedded",
    ),
    "RFC 1951": Specification(
        "DEFLATE compressed data format",
        "RFC 1951",
        "https://www.rfc-editor.org/rfc/rfc1951",
        "embedded",
    ),
    "RFC 2083": Specification(
        "PNG predictors", "RFC 2083", "https://www.rfc-editor.org/rfc/rfc2083", "embedded"
    ),
    "TIFF 6.0": Specification(
        "TIFF predictor 2 and LZW",
        "TIFF Revision 6.0 (1992)",
        "https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf",
        "embedded",
    ),
    "RFC 5652": Specification(
        "Cryptographic Message Syntax (CMS)",
        "RFC 5652",
        "https://www.rfc-editor.org/rfc/rfc5652",
        "embedded",
    ),
    "FIPS 197": Specification(
        "Advanced Encryption Standard (AES)",
        "FIPS 197 (2023 update)",
        "https://csrc.nist.gov/pubs/fips/197/final",
        "embedded",
    ),
    "Adobe Type 1": Specification(
        "Adobe Type 1 Font Format",
        "version 1.1 (1990)",
        "https://adobe-type-tools.github.io/font-tech-notes/pdfs/T1_SPEC.pdf",
        "embedded",
    ),
    "Adobe TN 5176": Specification(
        "The Compact Font Format Specification",
        "Technical Note 5176 (2003)",
        "https://adobe-type-tools.github.io/font-tech-notes/pdfs/5176.CFF.pdf",
        "embedded",
    ),
    "Adobe TN 5177": Specification(
        "The Type 2 Charstring Format",
        "Technical Note 5177 (2000)",
        "https://adobe-type-tools.github.io/font-tech-notes/pdfs/5177.Type2.pdf",
        "embedded",
    ),
    "Adobe TN 5014": Specification(
        "Adobe CMap and CIDFont Files Specification",
        "Technical Note 5014 (1996)",
        "https://adobe-type-tools.github.io/font-tech-notes/pdfs/5014.CIDFont_Spec.pdf",
        "embedded",
    ),
    "Adobe TN 5099": Specification(
        "Developing CMap Resources for CID-Keyed Fonts",
        "Technical Note 5099 (2012)",
        "https://adobe-type-tools.github.io/font-tech-notes/pdfs/5099.CMapResources.pdf",
        "embedded",
    ),
    "Adobe Glyph List": Specification(
        "Adobe Glyph List Specification",
        "AGL 2.0 and AGLFN 1.7",
        "https://github.com/adobe-type-tools/agl-specification",
        "embedded",
    ),
    "Adobe AFM": Specification(
        "Adobe Font Metrics File Format Specification",
        "Technical Note 5004 (1998)",
        "https://adobe-type-tools.github.io/font-tech-notes/pdfs/5004.AFM_Spec.pdf",
        "embedded",
    ),
    "PostScript LRM": Specification(
        "PostScript Language Reference",
        "third edition (1999)",
        "https://www.adobe.com/jp/print/postscript/pdfs/PLRM.pdf",
        "embedded",
    ),
    "ICC.1": Specification(
        "Image technology colour management - ICC profile format",
        "ICC.1:2022",
        "https://www.color.org/specification/ICC.1-2022-05.pdf",
        "embedded",
    ),
}


def implemented_extension_ids() -> frozenset[str]:
    """The extensions a citation can name: "PREFIX/base version/level"."""
    return frozenset(
        f"{coverage.prefix}/{coverage.base_version}/{coverage.extension_level}"
        for coverage in IMPLEMENTED_EXTENSIONS
    )


def cites(
    specification: str,
    clause: str = "",
    *,
    since: str | None = None,
    until: str | None = None,
    introduced: str | None = None,
    extension: str | None = None,
    profile: str | None = None,
    rule: str = "",
) -> Callable[[Cited], Cited]:
    """Mark the decorated function as implementing a clause of a specification.

    since and until bound the PDF versions a rule the function decides applies to (from
    since, before until); introduced names the version that added a feature the function
    does not gate on the version; extension and profile name what adds the rule. The
    arguments are literals, so scripts/spec_index.py reads them without importing anything,
    and the function is returned unchanged.
    """
    if specification not in SPECIFICATIONS:
        raise ValueError(f"unknown specification {specification!r}")
    if not isinstance(clause, str) or not isinstance(rule, str):
        raise TypeError("a citation's clause and rule are strings")
    for version in (since, until, introduced):
        if version is not None:
            PdfVersion.parse(version)
    if extension is not None and extension not in implemented_extension_ids():
        raise ValueError(f"no implemented extension {extension!r}")
    if profile is not None and get_standard_profile(profile) is None:
        raise ValueError(f"unknown profile {profile!r}")

    def mark(function: Cited) -> Cited:
        return function

    return mark


__all__ = ("SPECIFICATIONS", "Specification", "cites", "implemented_extension_ids")

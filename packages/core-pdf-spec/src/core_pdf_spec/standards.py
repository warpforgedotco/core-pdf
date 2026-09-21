# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class PdfVersion:
    major: int
    minor: int

    def __post_init__(self) -> None:
        if type(self.major) is not int or type(self.minor) is not int:
            raise ValueError("PDF version components must be integers")
        if self.major < 0 or self.minor < 0:
            raise ValueError("PDF version components must be non-negative")

    @classmethod
    def parse(cls, value: str) -> PdfVersion:
        if not isinstance(value, str) or re.fullmatch(r"[0-9]\.[0-9]", value) is None:
            raise ValueError("invalid PDF version; expected a single-digit major.minor")
        return cls(int(value[0]), int(value[2]))

    @property
    def recognized(self) -> bool:
        return (self.major == 1 and self.minor <= 7) or (self.major == 2 and self.minor == 0)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True, slots=True)
class SpecificationBaseline:
    edition: str
    reference_url: str
    errata_revision: str | None = None
    errata_url: str | None = None


PDF_2_0_BASELINE = SpecificationBaseline(
    edition="ISO 32000-2:2020",
    reference_url="https://pdfa.org/resource/iso-32000-2/",
    errata_revision="223821c58055f6ed2b93a5f40304c25ffa86459f",
    errata_url=(
        "https://github.com/pdf-association/pdf-issues/tree/"
        "223821c58055f6ed2b93a5f40304c25ffa86459f"
    ),
)


@dataclass(frozen=True, slots=True)
class PdfExtension:
    prefix: str
    base_version: PdfVersion
    extension_level: int
    url: str | None = None
    extension_revision: str | None = None


@dataclass(frozen=True, slots=True)
class ProfileClaim:
    identifier: str | None
    family: str
    source: str
    properties: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class StandardsDiagnostic:
    code: str
    message: str
    source: str


@dataclass(frozen=True, slots=True)
class SemanticContext:
    version: PdfVersion | None
    extensions: tuple[PdfExtension, ...] = ()
    baseline: SpecificationBaseline = PDF_2_0_BASELINE


@dataclass(frozen=True, slots=True)
class DocumentStandards:
    header_version: PdfVersion | None = None
    catalog_version: PdfVersion | None = None
    effective_version: PdfVersion | None = None
    header_declaration: str | None = None
    catalog_declaration: str | None = None
    extensions: tuple[PdfExtension, ...] = ()
    profile_claims: tuple[ProfileClaim, ...] = ()
    diagnostics: tuple[StandardsDiagnostic, ...] = ()

    @property
    def context(self) -> SemanticContext:
        return SemanticContext(self.effective_version, self.extensions)


@dataclass(frozen=True, slots=True)
class StandardProfile:
    identifier: str
    family: str
    edition: str
    base_version: PdfVersion
    reference_url: str


STANDARD_PROFILES: tuple[StandardProfile, ...] = (
    *(
        StandardProfile(f"pdfa-{part}{level}", "PDF/A", edition, base, url)
        for part, levels, edition, base, url in (
            (
                "1",
                ("a", "b"),
                "ISO 19005-1:2005",
                PdfVersion(1, 4),
                "https://pdfa.org/resource/iso-19005-pdfa/",
            ),
            (
                "2",
                ("a", "b", "u"),
                "ISO 19005-2:2011",
                PdfVersion(1, 7),
                "https://pdfa.org/resource/iso-19005-pdfa/",
            ),
            (
                "3",
                ("a", "b", "u"),
                "ISO 19005-3:2012",
                PdfVersion(1, 7),
                "https://pdfa.org/resource/iso-19005-pdfa/",
            ),
            (
                "4",
                ("", "e", "f"),
                "ISO 19005-4:2020",
                PdfVersion(2, 0),
                "https://pdfa.org/resource/iso-19005-pdfa/",
            ),
        )
        for level in levels
    ),
    StandardProfile(
        "pdfua-1",
        "PDF/UA",
        "ISO 14289-1:2014",
        PdfVersion(1, 7),
        "https://pdfa.org/resource/iso-14289-pdfua/",
    ),
    StandardProfile(
        "pdfua-2",
        "PDF/UA",
        "ISO 14289-2:2024",
        PdfVersion(2, 0),
        "https://pdfa.org/resource/iso-14289-pdfua/",
    ),
    *(
        StandardProfile(
            f"wtpdf-1.0-{level}",
            "WTPDF",
            "WTPDF 1.0:2024",
            PdfVersion(2, 0),
            "https://pdfa.org/wtpdf/",
        )
        for level in ("reuse", "accessibility")
    ),
    *(
        StandardProfile(
            f"pdfx-{suffix}", "PDF/X", edition, base, "https://pdfa.org/resource/iso-15930-pdfx/"
        )
        for suffixes, edition, base in (
            (("1a:2001",), "ISO 15930-1:2001", PdfVersion(1, 3)),
            (("1a:2003",), "ISO 15930-4:2003", PdfVersion(1, 4)),
            (("3:2002",), "ISO 15930-3:2002", PdfVersion(1, 3)),
            (("3:2003",), "ISO 15930-6:2003", PdfVersion(1, 4)),
            (("4", "4p"), "ISO 15930-7:2010", PdfVersion(1, 6)),
            (("5g", "5pg", "5n"), "ISO 15930-8:2010", PdfVersion(1, 6)),
            (("6", "6p", "6n"), "ISO 15930-9:2020", PdfVersion(2, 0)),
        )
        for suffix in suffixes
    ),
    StandardProfile(
        "pdfe-1",
        "PDF/E",
        "ISO 24517-1:2008",
        PdfVersion(1, 6),
        "https://www.iso.org/standard/42274.html",
    ),
    *(
        StandardProfile(
            f"pdfvt-{part}",
            "PDF/VT",
            "ISO 16612-2:2010",
            PdfVersion(1, 6),
            "https://www.iso.org/standard/46428.html",
        )
        for part in (1, 2)
    ),
    StandardProfile(
        "pdfvt-3",
        "PDF/VT",
        "ISO 16612-3:2020",
        PdfVersion(2, 0),
        "https://www.iso.org/standard/75218.html",
    ),
    StandardProfile(
        "pdfr-1",
        "PDF/R",
        "ISO 23504-1:2020",
        PdfVersion(1, 7),
        "https://pdfa.org/resource/iso-23504-pdfr/",
    ),
)


def get_standard_profile(identifier: str) -> StandardProfile | None:
    return next(
        (profile for profile in STANDARD_PROFILES if profile.identifier == identifier), None
    )


@dataclass(frozen=True, slots=True)
class ExtensionCoverage:
    prefix: str
    base_version: PdfVersion
    extension_level: int
    extension_revision: str | None
    features: tuple[str, ...]
    reference_url: str


IMPLEMENTED_EXTENSIONS = (
    ExtensionCoverage(
        "ADBE",
        PdfVersion(1, 7),
        3,
        None,
        ("AES-256 revision-5 decryption",),
        "https://pdfa.org/resource/pdf-specification-archive/",
    ),
    ExtensionCoverage(
        "ADBE",
        PdfVersion(1, 7),
        5,
        None,
        ("ColorDodge and ColorBurn revised component blending",),
        "https://pdfa.org/resource/pdf-specification-archive/",
    ),
    ExtensionCoverage(
        "ISO_",
        PdfVersion(2, 0),
        32003,
        ":2023",
        ("AES-256-GCM revision-7 decryption",),
        "https://www.iso.org/standard/45876.html",
    ),
    ExtensionCoverage(
        "ISO_",
        PdfVersion(2, 0),
        32004,
        ":2024",
        ("standalone PDF MAC verification",),
        "https://www.iso.org/standard/45877.html",
    ),
)


def get_extension_coverage(extension: PdfExtension) -> ExtensionCoverage | None:
    return next(
        (
            coverage
            for coverage in IMPLEMENTED_EXTENSIONS
            if (
                coverage.prefix,
                coverage.base_version,
                coverage.extension_level,
                coverage.extension_revision,
            )
            == (
                extension.prefix,
                extension.base_version,
                extension.extension_level,
                extension.extension_revision,
            )
        ),
        None,
    )


__all__ = (
    "DocumentStandards",
    "ExtensionCoverage",
    "IMPLEMENTED_EXTENSIONS",
    "PDF_2_0_BASELINE",
    "PdfExtension",
    "PdfVersion",
    "ProfileClaim",
    "STANDARD_PROFILES",
    "SemanticContext",
    "SpecificationBaseline",
    "StandardProfile",
    "StandardsDiagnostic",
    "get_extension_coverage",
    "get_standard_profile",
)

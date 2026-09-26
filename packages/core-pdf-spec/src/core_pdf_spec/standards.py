# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from typing import ClassVar

from core_records import Record, frozen_setattr


class PdfVersion(Record):
    __slots__ = ("major", "minor")

    major: int
    minor: int

    __fields__: ClassVar[tuple[str, ...]] = ("major", "minor")
    __match_args__ = ("major", "minor")

    def __init__(self, major: int, minor: int) -> None:
        frozen_setattr(self, "major", major)
        frozen_setattr(self, "minor", minor)
        self._post_init()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.major == other.major and self.minor == other.minor

    def __hash__(self) -> int:
        return hash((self.major, self.minor))

    def __lt__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.major, self.minor) < (other.major, other.minor)

    def __le__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.major, self.minor) <= (other.major, other.minor)

    def __gt__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.major, self.minor) > (other.major, other.minor)

    def __ge__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.major, self.minor) >= (other.major, other.minor)

    def _post_init(self) -> None:
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


class SpecificationBaseline(Record):
    __slots__ = ("edition", "reference_url", "errata_revision", "errata_url")

    edition: str
    reference_url: str
    errata_revision: str | None
    errata_url: str | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "edition",
        "reference_url",
        "errata_revision",
        "errata_url",
    )
    __match_args__ = ("edition", "reference_url", "errata_revision", "errata_url")

    def __init__(
        self,
        edition: str,
        reference_url: str,
        errata_revision: str | None = None,
        errata_url: str | None = None,
    ) -> None:
        frozen_setattr(self, "edition", edition)
        frozen_setattr(self, "reference_url", reference_url)
        frozen_setattr(self, "errata_revision", errata_revision)
        frozen_setattr(self, "errata_url", errata_url)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.edition == other.edition
            and self.reference_url == other.reference_url
            and self.errata_revision == other.errata_revision
            and self.errata_url == other.errata_url
        )

    def __hash__(self) -> int:
        return hash((self.edition, self.reference_url, self.errata_revision, self.errata_url))


PDF_2_0_BASELINE = SpecificationBaseline(
    edition="ISO 32000-2:2020",
    reference_url="https://pdfa.org/resource/iso-32000-2/",
    errata_revision="223821c58055f6ed2b93a5f40304c25ffa86459f",
    errata_url=(
        "https://github.com/pdf-association/pdf-issues/tree/"
        "223821c58055f6ed2b93a5f40304c25ffa86459f"
    ),
)


class PdfExtension(Record):
    __slots__ = ("prefix", "base_version", "extension_level", "url", "extension_revision")

    prefix: str
    base_version: PdfVersion
    extension_level: int
    url: str | None
    extension_revision: str | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "prefix",
        "base_version",
        "extension_level",
        "url",
        "extension_revision",
    )
    __match_args__ = ("prefix", "base_version", "extension_level", "url", "extension_revision")

    def __init__(
        self,
        prefix: str,
        base_version: PdfVersion,
        extension_level: int,
        url: str | None = None,
        extension_revision: str | None = None,
    ) -> None:
        frozen_setattr(self, "prefix", prefix)
        frozen_setattr(self, "base_version", base_version)
        frozen_setattr(self, "extension_level", extension_level)
        frozen_setattr(self, "url", url)
        frozen_setattr(self, "extension_revision", extension_revision)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.prefix == other.prefix
            and self.base_version == other.base_version
            and self.extension_level == other.extension_level
            and self.url == other.url
            and self.extension_revision == other.extension_revision
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.prefix,
                self.base_version,
                self.extension_level,
                self.url,
                self.extension_revision,
            )
        )


class ProfileClaim(Record):
    __slots__ = ("identifier", "family", "source", "properties")

    identifier: str | None
    family: str
    source: str
    properties: tuple[tuple[str, str], ...]

    __fields__: ClassVar[tuple[str, ...]] = ("identifier", "family", "source", "properties")
    __match_args__ = ("identifier", "family", "source", "properties")

    def __init__(
        self,
        identifier: str | None,
        family: str,
        source: str,
        properties: tuple[tuple[str, str], ...] = (),
    ) -> None:
        frozen_setattr(self, "identifier", identifier)
        frozen_setattr(self, "family", family)
        frozen_setattr(self, "source", source)
        frozen_setattr(self, "properties", properties)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.identifier == other.identifier
            and self.family == other.family
            and self.source == other.source
            and self.properties == other.properties
        )

    def __hash__(self) -> int:
        return hash((self.identifier, self.family, self.source, self.properties))


class StandardsDiagnostic(Record):
    __slots__ = ("code", "message", "source")

    code: str
    message: str
    source: str

    __fields__: ClassVar[tuple[str, ...]] = ("code", "message", "source")
    __match_args__ = ("code", "message", "source")

    def __init__(self, code: str, message: str, source: str) -> None:
        frozen_setattr(self, "code", code)
        frozen_setattr(self, "message", message)
        frozen_setattr(self, "source", source)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.code == other.code
            and self.message == other.message
            and self.source == other.source
        )

    def __hash__(self) -> int:
        return hash((self.code, self.message, self.source))


class SemanticContext(Record):
    __slots__ = ("version", "extensions", "baseline")

    version: PdfVersion | None
    extensions: tuple[PdfExtension, ...]
    baseline: SpecificationBaseline

    __fields__: ClassVar[tuple[str, ...]] = ("version", "extensions", "baseline")
    __match_args__ = ("version", "extensions", "baseline")

    def __init__(
        self,
        version: PdfVersion | None,
        extensions: tuple[PdfExtension, ...] = (),
        baseline: SpecificationBaseline = PDF_2_0_BASELINE,
    ) -> None:
        frozen_setattr(self, "version", version)
        frozen_setattr(self, "extensions", extensions)
        frozen_setattr(self, "baseline", baseline)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.version == other.version
            and self.extensions == other.extensions
            and self.baseline == other.baseline
        )

    def __hash__(self) -> int:
        return hash((self.version, self.extensions, self.baseline))


class DocumentStandards(Record):
    __slots__ = (
        "header_version",
        "catalog_version",
        "effective_version",
        "header_declaration",
        "catalog_declaration",
        "extensions",
        "profile_claims",
        "diagnostics",
    )

    header_version: PdfVersion | None
    catalog_version: PdfVersion | None
    effective_version: PdfVersion | None
    header_declaration: str | None
    catalog_declaration: str | None
    extensions: tuple[PdfExtension, ...]
    profile_claims: tuple[ProfileClaim, ...]
    diagnostics: tuple[StandardsDiagnostic, ...]

    __fields__: ClassVar[tuple[str, ...]] = (
        "header_version",
        "catalog_version",
        "effective_version",
        "header_declaration",
        "catalog_declaration",
        "extensions",
        "profile_claims",
        "diagnostics",
    )
    __match_args__ = (
        "header_version",
        "catalog_version",
        "effective_version",
        "header_declaration",
        "catalog_declaration",
        "extensions",
        "profile_claims",
        "diagnostics",
    )

    def __init__(
        self,
        header_version: PdfVersion | None = None,
        catalog_version: PdfVersion | None = None,
        effective_version: PdfVersion | None = None,
        header_declaration: str | None = None,
        catalog_declaration: str | None = None,
        extensions: tuple[PdfExtension, ...] = (),
        profile_claims: tuple[ProfileClaim, ...] = (),
        diagnostics: tuple[StandardsDiagnostic, ...] = (),
    ) -> None:
        frozen_setattr(self, "header_version", header_version)
        frozen_setattr(self, "catalog_version", catalog_version)
        frozen_setattr(self, "effective_version", effective_version)
        frozen_setattr(self, "header_declaration", header_declaration)
        frozen_setattr(self, "catalog_declaration", catalog_declaration)
        frozen_setattr(self, "extensions", extensions)
        frozen_setattr(self, "profile_claims", profile_claims)
        frozen_setattr(self, "diagnostics", diagnostics)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.header_version == other.header_version
            and self.catalog_version == other.catalog_version
            and self.effective_version == other.effective_version
            and self.header_declaration == other.header_declaration
            and self.catalog_declaration == other.catalog_declaration
            and self.extensions == other.extensions
            and self.profile_claims == other.profile_claims
            and self.diagnostics == other.diagnostics
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.header_version,
                self.catalog_version,
                self.effective_version,
                self.header_declaration,
                self.catalog_declaration,
                self.extensions,
                self.profile_claims,
                self.diagnostics,
            )
        )

    @property
    def context(self) -> SemanticContext:
        return SemanticContext(self.effective_version, self.extensions)


class StandardProfile(Record):
    __slots__ = ("identifier", "family", "edition", "base_version", "reference_url")

    identifier: str
    family: str
    edition: str
    base_version: PdfVersion
    reference_url: str

    __fields__: ClassVar[tuple[str, ...]] = (
        "identifier",
        "family",
        "edition",
        "base_version",
        "reference_url",
    )
    __match_args__ = ("identifier", "family", "edition", "base_version", "reference_url")

    def __init__(
        self,
        identifier: str,
        family: str,
        edition: str,
        base_version: PdfVersion,
        reference_url: str,
    ) -> None:
        frozen_setattr(self, "identifier", identifier)
        frozen_setattr(self, "family", family)
        frozen_setattr(self, "edition", edition)
        frozen_setattr(self, "base_version", base_version)
        frozen_setattr(self, "reference_url", reference_url)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.identifier == other.identifier
            and self.family == other.family
            and self.edition == other.edition
            and self.base_version == other.base_version
            and self.reference_url == other.reference_url
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.identifier,
                self.family,
                self.edition,
                self.base_version,
                self.reference_url,
            )
        )


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


class ExtensionCoverage(Record):
    __slots__ = (
        "prefix",
        "base_version",
        "extension_level",
        "extension_revision",
        "features",
        "reference_url",
    )

    prefix: str
    base_version: PdfVersion
    extension_level: int
    extension_revision: str | None
    features: tuple[str, ...]
    reference_url: str

    __fields__: ClassVar[tuple[str, ...]] = (
        "prefix",
        "base_version",
        "extension_level",
        "extension_revision",
        "features",
        "reference_url",
    )
    __match_args__ = (
        "prefix",
        "base_version",
        "extension_level",
        "extension_revision",
        "features",
        "reference_url",
    )

    def __init__(
        self,
        prefix: str,
        base_version: PdfVersion,
        extension_level: int,
        extension_revision: str | None,
        features: tuple[str, ...],
        reference_url: str,
    ) -> None:
        frozen_setattr(self, "prefix", prefix)
        frozen_setattr(self, "base_version", base_version)
        frozen_setattr(self, "extension_level", extension_level)
        frozen_setattr(self, "extension_revision", extension_revision)
        frozen_setattr(self, "features", features)
        frozen_setattr(self, "reference_url", reference_url)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.prefix == other.prefix
            and self.base_version == other.base_version
            and self.extension_level == other.extension_level
            and self.extension_revision == other.extension_revision
            and self.features == other.features
            and self.reference_url == other.reference_url
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.prefix,
                self.base_version,
                self.extension_level,
                self.extension_revision,
                self.features,
                self.reference_url,
            )
        )


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

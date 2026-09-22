# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from collections.abc import Callable
from copy import replace
from functools import partial
from typing import cast

from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import fromstring as defused_fromstring

from core_pdf.impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl.exceptions import PdfError, PdfParseError
from core_pdf_spec.s_07_document.metadata import catalog_metadata_stream, info_dictionary
from core_pdf_spec.s_07_document.standards import (
    effective_pdf_version,
    parse_catalog_version,
    parse_extension,
    parse_header_version,
)
from core_pdf_spec.s_07_syntax.resolution import resolve_reference_chain
from core_pdf_spec.s_07_syntax.types import Decipher, PdfDict, PdfValueResolver
from core_pdf_spec.s_07_syntax.xref import (
    PdfXRefEntry,
    XRefRevision,
    XRefScanner,
    iter_xref_revisions,
    overlay_xref_entries,
)
from core_pdf_spec.standards import (
    DocumentStandards,
    PdfExtension,
    PdfVersion,
    ProfileClaim,
    SemanticContext,
    StandardsDiagnostic,
    get_extension_coverage,
    get_standard_profile,
)
from core_pdf_spec.types import PdfByteBuffer, PdfName, PdfReference

internal_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
internal_IDENTIFICATION_NAMESPACES = {
    "http://www.aiim.org/pdfa/ns/id/": "PDF/A",
    "http://www.aiim.org/pdfua/ns/id/": "PDF/UA",
    "http://www.npes.org/pdfx/ns/id/": "PDF/X",
    "http://www.npes.org/pdfvt/ns/id/": "PDF/VT",
    "http://www.aiim.org/pdfe/ns/id/": "PDF/E",
}
internal_DECLARATIONS_NAMESPACES = (
    "http://pdfa.org/declarations/",
    "https://pdfa.org/declarations/",
)
internal_WTPDF_PROFILES = {
    "http://pdfa.org/declarations/wtpdf#reuse1.0": "wtpdf-1.0-reuse",
    "http://pdfa.org/declarations/wtpdf#reuse1.0-validated": "wtpdf-1.0-reuse",
    "http://pdfa.org/declarations/wtpdf#accessibility1.0": "wtpdf-1.0-accessibility",
    "http://pdfa.org/declarations/wtpdf#accessibility1.0-validated": "wtpdf-1.0-accessibility",
}


def discover_header_standards(data: PdfByteBuffer) -> DocumentStandards:
    prefix = bytes(data[:1024])
    offset = prefix.find(b"%PDF-")
    if offset < 0:
        return DocumentStandards(
            diagnostics=(StandardsDiagnostic("missing-header", "PDF header is missing", "header"),)
        )
    declaration = (
        bytes(data[offset : offset + 1024]).split(b"\n", 1)[0].split(b"\r", 1)[0].decode("latin-1")
    )
    diagnostics: list[StandardsDiagnostic] = []
    version: PdfVersion | None = None
    try:
        version = parse_header_version(data[offset : offset + 9])
    except PdfError, ValueError:
        diagnostics.append(StandardsDiagnostic("invalid-header", "Invalid PDF version", "header"))
    if offset:
        diagnostics.append(
            StandardsDiagnostic("displaced-header", f"PDF header starts at byte {offset}", "header")
        )
    if version is not None and not version.recognized:
        diagnostics.append(
            StandardsDiagnostic("unknown-version", f"Unrecognized PDF version {version}", "header")
        )
    return DocumentStandards(
        header_version=version,
        effective_version=version,
        header_declaration=declaration,
        diagnostics=tuple(diagnostics),
    )


def internal_resolve_catalog(resolver: PdfValueResolver, trailer: PdfDict) -> PdfDict | None:
    value = resolve_reference_chain(trailer.get("Root"), resolver.resolve)
    return cast(PdfDict, value) if isinstance(value, dict) else None


def discover_document_standards(
    header: DocumentStandards, resolver: PdfValueResolver, trailer: PdfDict
) -> DocumentStandards:
    diagnostics = list(header.diagnostics)
    try:
        catalog = internal_resolve_catalog(resolver, trailer)
        if catalog is None:
            raise ValueError("missing catalog")
    except PdfError, RecursionError, ValueError:
        return replace(
            header,
            diagnostics=(
                *diagnostics,
                StandardsDiagnostic("invalid-catalog", "Cannot read catalog", "catalog"),
            ),
        )
    version: PdfVersion | None = None
    declaration: str | None = None
    if "Version" in catalog:
        try:
            value = resolver.resolve(catalog["Version"])
            declaration = str(value) if isinstance(value, PdfName) else repr(value)
            version = parse_catalog_version(value)
        except PdfError, RecursionError, ValueError:
            diagnostics.append(
                StandardsDiagnostic(
                    "invalid-catalog-version", "Invalid catalog /Version", "catalog/Version"
                )
            )
        else:
            if not version.recognized:
                diagnostics.append(
                    StandardsDiagnostic(
                        "unknown-version", f"Unrecognized PDF version {version}", "catalog/Version"
                    )
                )
            if header.header_version is not None and version < header.header_version:
                diagnostics.append(
                    StandardsDiagnostic(
                        "catalog-version-downgrade",
                        "Catalog version cannot lower the header version",
                        "catalog/Version",
                    )
                )
    extensions = internal_discover_extensions(
        catalog.get("Extensions"), resolver.resolve, diagnostics
    )
    return replace(
        header,
        catalog_version=version,
        catalog_declaration=declaration,
        effective_version=effective_pdf_version(header.header_version, version),
        extensions=extensions,
        diagnostics=tuple(diagnostics),
    )


class internal_VersionProbeResolver(ObjectResolver):
    __slots__ = ()

    def xref_entry(self, ref: PdfReference) -> PdfXRefEntry | None:
        entry = super().xref_entry(ref)
        if entry is not None and entry.object_stream is not None:
            raise PdfParseError("version bootstrap requires an uncompressed object")
        return entry


def bootstrap_security_context(
    header: DocumentStandards,
    data: PdfByteBuffer,
    xref: dict[int, PdfXRefEntry],
    trailer: PdfDict,
) -> SemanticContext | None:
    resolver = internal_VersionProbeResolver(data, xref)
    try:
        catalog = resolver.resolve(trailer.get("Root"))
        if not isinstance(catalog, dict):
            return None
        catalog = cast(PdfDict, catalog)
        version = None
        if "Version" in catalog:
            version = parse_catalog_version(resolver.resolve(catalog["Version"]))
        current = replace(
            header, effective_version=effective_pdf_version(header.header_version, version)
        )
    except PdfError, ValueError, RecursionError:
        return None
    finally:
        resolver.close()
    current = preserve_historical_version(
        current,
        data,
        trailer,
        None,
        recovered=False,
        trailer_context=header.context,
        version_probe=True,
    )
    if any(
        item.code in {"incomplete-version-history", "invalid-historical-version"}
        for item in current.diagnostics
    ):
        return None
    return current.context


def internal_discover_extensions(
    value: object,
    resolve: Callable[[object], object],
    diagnostics: list[StandardsDiagnostic],
) -> tuple[PdfExtension, ...]:
    try:
        value = resolve(value)
        if value is None:
            return ()
        if not isinstance(value, dict):
            raise ValueError("invalid Extensions dictionary")
    except PdfError, RecursionError, ValueError:
        diagnostics.append(
            StandardsDiagnostic(
                "invalid-extensions", "Cannot read extension declarations", "catalog/Extensions"
            )
        )
        return ()
    extensions: list[PdfExtension] = []
    for prefix, declarations in value.items():
        if prefix == "Type":
            if not isinstance(declarations, PdfName) or declarations.value != "Extensions":
                diagnostics.append(
                    StandardsDiagnostic(
                        "invalid-extensions-type",
                        "Invalid Extensions /Type",
                        "catalog/Extensions/Type",
                    )
                )
            continue
        source = f"catalog/Extensions/{prefix}"
        try:
            declarations = resolve(declarations)
            items = declarations if isinstance(declarations, list) else [declarations]
            if not items:
                raise ValueError("empty developer extension array")
            for index, item in enumerate(items):
                item_source = f"{source}/{index}" if isinstance(declarations, list) else source
                try:
                    resolved = resolve(item)
                    if isinstance(resolved, dict):
                        resolved = {key: resolve(field) for key, field in resolved.items()}
                    extension = parse_extension(str(prefix), resolved)
                except PdfError, RecursionError, ValueError:
                    diagnostics.append(
                        StandardsDiagnostic(
                            "invalid-extension", "Invalid extension declaration", item_source
                        )
                    )
                else:
                    extensions.append(extension)
        except PdfError, RecursionError, ValueError:
            diagnostics.append(
                StandardsDiagnostic(
                    "invalid-extension", "Cannot resolve extension declaration", source
                )
            )
    return tuple(extensions)


def internal_extension_diagnostics(standards: DocumentStandards) -> DocumentStandards:
    diagnostics = list(standards.diagnostics)
    for extension in standards.extensions:
        source = f"catalog/Extensions/{extension.prefix}"
        if get_extension_coverage(extension) is None:
            diagnostics.append(
                StandardsDiagnostic(
                    "unknown-extension",
                    "No implementation coverage is recorded for this exact extension",
                    source,
                )
            )
        if standards.effective_version is not None:
            if extension.base_version > standards.effective_version:
                diagnostics.append(
                    StandardsDiagnostic(
                        "extension-version-mismatch",
                        "Extension BaseVersion exceeds effective PDF version",
                        source,
                    )
                )
            if standards.effective_version >= PdfVersion(2, 0) and extension.url is None:
                diagnostics.append(
                    StandardsDiagnostic(
                        "missing-extension-url",
                        "PDF 2.0 extension declaration is missing URL",
                        source,
                    )
                )
    return replace(standards, diagnostics=tuple(diagnostics))


def preserve_historical_version(
    standards: DocumentStandards,
    data: PdfByteBuffer,
    trailer: PdfDict,
    decipher: Decipher | None,
    *,
    recovered: bool,
    trailer_context: SemanticContext | None = None,
    version_probe: bool = False,
) -> DocumentStandards:
    diagnostics = list(standards.diagnostics)
    if recovered:
        diagnostics.append(
            StandardsDiagnostic(
                "recovered-version-history",
                "Recovered cross-references may omit earlier version declarations",
                "xref",
            )
        )
    previous = trailer.get("Prev")
    if previous is None:
        return internal_extension_diagnostics(replace(standards, diagnostics=tuple(diagnostics)))
    sections: list[XRefRevision] = []
    try:
        if type(previous) is not int:
            raise ValueError("invalid revision chain")
        read_section = partial(XRefScanner.parse_section_at, data, semantic_context=trailer_context)
        for revision in iter_xref_revisions(previous, read_section):
            if len(sections) >= 1024:
                raise ValueError("invalid or excessive revision chain")
            sections.append(revision)
    except PdfError, ValueError, RecursionError, struct.error, OSError:
        diagnostics.append(
            StandardsDiagnostic(
                "incomplete-version-history",
                "Cannot read the complete version declaration history",
                "xref/Prev",
            )
        )

    merged: dict[int, PdfXRefEntry] = {}
    floor = standards.effective_version
    for revision in reversed(sections):
        overlay_xref_entries(merged, revision.entries)
        resolver_type = internal_VersionProbeResolver if version_probe else ObjectResolver
        resolver = resolver_type(data, merged, decipher=decipher)
        try:
            catalog = resolver.resolve(revision.trailer.get("Root"))
            if not isinstance(catalog, dict):
                raise ValueError("missing historical catalog")
            catalog = cast(PdfDict, catalog)
            if "Version" in catalog:
                version = parse_catalog_version(resolver.resolve(catalog["Version"]))
                floor = effective_pdf_version(floor, version)
        except PdfError, ValueError, RecursionError:
            diagnostics.append(
                StandardsDiagnostic(
                    "invalid-historical-version",
                    "Cannot read a historical catalog version",
                    f"xref/{revision.offset}/Root",
                )
            )
        finally:
            resolver.close()
    if floor != standards.effective_version:
        diagnostics.append(
            StandardsDiagnostic(
                "historical-version-downgrade",
                f"Preserved an earlier PDF {floor} upgrade",
                "catalog/Version",
            )
        )
    return internal_extension_diagnostics(
        replace(standards, effective_version=floor, diagnostics=tuple(diagnostics))
    )


def discover_profile_claims(
    standards: DocumentStandards, resolver: PdfValueResolver, trailer: PdfDict
) -> DocumentStandards:
    diagnostics = list(standards.diagnostics)
    claims: list[ProfileClaim] = []
    try:
        catalog = internal_resolve_catalog(resolver, trailer)
        stream = catalog_metadata_stream(resolver, catalog) if catalog is not None else None
        if stream is not None and stream.data:
            claims.extend(internal_xmp_claims(stream.data, diagnostics))
    except PdfError, RecursionError, ValueError, ET.ParseError, DefusedXmlException:
        diagnostics.append(
            StandardsDiagnostic(
                "invalid-xmp", "Cannot read profile claims from XMP", "catalog/Metadata"
            )
        )
    try:
        info = info_dictionary(resolver, trailer)
        if info is not None and "GTS_PDFXVersion" in info:
            properties = tuple(
                (key, text)
                for key in ("GTS_PDFXVersion", "GTS_PDFXConformance")
                if (text := resolver.resolve_str(info.get(key))) is not None
            )
            claims.append(internal_profile_claim("PDF/X", "trailer/Info", properties, diagnostics))
    except PdfError, RecursionError, ValueError:
        diagnostics.append(
            StandardsDiagnostic(
                "invalid-profile-info", "Cannot read profile claims from Info", "trailer/Info"
            )
        )
    return replace(standards, profile_claims=tuple(claims), diagnostics=tuple(diagnostics))


def internal_xmp_claims(raw: bytes, diagnostics: list[StandardsDiagnostic]) -> list[ProfileClaim]:
    root = defused_fromstring(raw)
    groups: dict[str, list[tuple[str, str]]] = {}
    claims: list[ProfileClaim] = []
    if root.tag == f"{{{internal_RDF}}}RDF":
        document_rdf = [root]
    elif root.tag == "{adobe:ns:meta/}xmpmeta":
        document_rdf = root.findall(f"{{{internal_RDF}}}RDF")
    else:
        return []
    for rdf in document_rdf:
        for description in rdf:
            if description.tag != f"{{{internal_RDF}}}Description":
                continue
            if (
                description.get(f"{{{internal_RDF}}}about", "") != ""
                or description.get(f"{{{internal_RDF}}}nodeID") is not None
            ):
                continue
            properties = list(description.attrib.items())
            properties.extend((child.tag, (child.text or "").strip()) for child in description)
            for key, value in properties:
                for namespace, family in internal_IDENTIFICATION_NAMESPACES.items():
                    if key.startswith(f"{{{namespace}}}"):
                        groups.setdefault(family, []).append((key, value))
            for namespace in internal_DECLARATIONS_NAMESPACES:
                for declarations in description.findall(f"{{{namespace}}}declarations"):
                    for entry in declarations.iter(f"{{{namespace}}}conformsTo"):
                        value = (entry.text or "").strip()
                        if not value.startswith("http://pdfa.org/declarations/wtpdf#"):
                            continue
                        identifier = internal_WTPDF_PROFILES.get(value)
                        if identifier is None:
                            diagnostics.append(
                                StandardsDiagnostic(
                                    "unknown-profile-claim",
                                    "Unknown WTPDF declaration target",
                                    "catalog/Metadata",
                                )
                            )
                        claims.append(
                            ProfileClaim(
                                identifier, "WTPDF", "catalog/Metadata", ((entry.tag, value),)
                            )
                        )
    for family, properties in groups.items():
        claims.append(
            internal_profile_claim(family, "catalog/Metadata", tuple(properties), diagnostics)
        )
    return claims


def internal_profile_claim(
    family: str,
    source: str,
    properties: tuple[tuple[str, str], ...],
    diagnostics: list[StandardsDiagnostic],
) -> ProfileClaim:
    fields: dict[str, str] = {}
    conflicting = False
    for key, value in properties:
        name = key.rsplit("}", 1)[-1]
        if name in fields and fields[name] != value:
            conflicting = True
        fields[name] = value
    identifier: str | None = None
    if not conflicting:
        part = fields.get("part", "")
        conformance = fields.get("conformance", "").lower()
        if family == "PDF/A":
            identifier = f"pdfa-{part}{conformance}"
            if part == "4" and fields.get("rev", "2020") != "2020":
                identifier = None
        elif family == "PDF/UA":
            identifier = f"pdfua-{part}"
            if part == "2" and fields.get("rev", "2024") != "2024":
                identifier = None
        elif family == "PDF/X":
            value = fields.get("GTS_PDFXConformance") or fields.get("GTS_PDFXVersion", "")
            identifier = value.lower().replace("pdf/x-", "pdfx-", 1)
        elif family == "PDF/VT":
            identifier = fields.get("GTS_PDFVTVersion", "").lower().replace("pdf/vt-", "pdfvt-", 1)
            if identifier == "pdfvt-3" and fields.get("rev", "2020") != "2020":
                identifier = None
        elif family == "PDF/E":
            identifier = fields.get("ISO_PDFEVersion", "").lower().replace("pdf/e-", "pdfe-", 1)
        if identifier is not None and get_standard_profile(identifier) is None:
            identifier = None
    if identifier is None:
        diagnostics.append(
            StandardsDiagnostic(
                "conflicting-profile-claim" if conflicting else "unknown-profile-claim",
                f"Cannot identify a supported {family} profile from these declarations",
                source,
            )
        )
    return ProfileClaim(identifier, family, source, properties)

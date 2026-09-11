# SPDX-License-Identifier: AGPL-3.0-only
"""Automatic version context and unvalidated profile claims preserve reader policy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.document import document as engine_document
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl._impl.document.standards import preserve_historical_version
from core_pdf.impl.exceptions import PdfDocumentClosedError
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfValueResolver
from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf_spec.standards import DocumentStandards, PdfVersion, SemanticContext
from core_pdf_spec.types import PdfName, PdfString


def internal_pdf(
    *,
    header: bytes = b"%PDF-1.7\n",
    catalog: bytes = b"",
    xmp: bytes | None = None,
    info: bytes | None = None,
) -> tuple[bytes, int]:
    metadata_ref = b" /Metadata 3 0 R" if xmp is not None else b""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R " + catalog + metadata_ref + b" >>",
        b"<< /Type /Pages /Kids [] /Count 0 >>",
    ]
    if xmp is not None:
        objects.append(
            b"<< /Type /Metadata /Subtype /XML /Length "
            + str(len(xmp)).encode()
            + b" >>\nstream\n"
            + xmp
            + b"\nendstream"
        )
    info_ref = b""
    if info is not None:
        objects.append(info)
        info_ref = f" /Info {len(objects)} 0 R".encode()
    data = header
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    data += f"trailer\n<< /Size {len(offsets)} /Root 1 0 R".encode() + info_ref + b" >>\n"
    return data + f"startxref\n{xref}\n%%EOF\n".encode(), xref


def internal_revision(data: bytes, previous: int, version: bytes | None) -> tuple[bytes, int]:
    version_entry = b" /Version /" + version if version is not None else b""
    root = len(data)
    data += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R" + version_entry + b" >>\nendobj\n"
    xref = len(data)
    data += f"xref\n1 1\n{root:010} 00000 n \n".encode()
    data += f"trailer\n<< /Size 3 /Root 1 0 R /Prev {previous} >>\n".encode()
    return data + f"startxref\n{xref}\n%%EOF\n".encode(), xref


@pytest.mark.parametrize(("version", "key", "name"), [("1.1", "A#42", "C#44"), ("1.2", "AB", "CD")])
def test_document_reparses_bootstrap_names_using_the_effective_version(
    version: str, key: str, name: str
) -> None:
    data, _ = internal_pdf(header=f"%PDF-{version}\n".encode(), catalog=b"/A#42 /C#44")
    with PdfDocument(data) as document:
        assert document.catalog()[key] == PdfName.of(name)
        assert document.standards.effective_version == PdfVersion.parse(version)


def test_catalog_upgrade_can_itself_use_newer_name_escapes() -> None:
    data, _ = internal_pdf(header=b"%PDF-1.1\n", catalog=b"/V#65rsion /1.4 /A#42 /C#44")
    with PdfDocument(data) as document:
        assert document.standards.effective_version == PdfVersion(1, 4)
        assert document.catalog()["AB"] == PdfName.of("CD")


@pytest.mark.parametrize("version", [PdfVersion(1, 1), PdfVersion(1, 2), None, PdfVersion(9, 0)])
def test_reader_keeps_nul_whitespace_recovery_with_explicit_context(
    version: PdfVersion | None,
) -> None:
    from core_pdf.impl._impl.document.recovery.lexer import PdfLexer

    lexer = PdfLexer(b"\0[1\0 2] <4\0 1>", semantic_context=SemanticContext(version))
    try:
        assert lexer.parse_object() == [1, 2]
        assert lexer.parse_object() == PdfString(b"A", is_literal=False)
    finally:
        lexer.close()


def internal_xmp(descriptions: bytes) -> bytes:
    return (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        b'xmlns:a="http://www.aiim.org/pdfa/ns/id/" '
        b'xmlns:u="http://www.aiim.org/pdfua/ns/id/" '
        b'xmlns:p="http://www.npes.org/pdfx/ns/id/" '
        b'xmlns:d="http://pdfa.org/declarations/">' + descriptions + b"</rdf:RDF></x:xmpmeta>"
    )


@pytest.mark.parametrize("version", [*(f"1.{minor}" for minor in range(8)), "2.0"])
def test_document_versions_share_the_reader_and_expose_immutable_context(version: str) -> None:
    data, ignored = internal_pdf(header=f"%PDF-{version}\n".encode())
    with PdfDocument(data) as document:
        standards = document.standards
        assert standards.header_version == PdfVersion.parse(version)
        assert standards.effective_version == standards.header_version
        assert standards.context == document.resolver.semantic_context
        assert standards.header_declaration == f"%PDF-{version}"
        assert not standards.diagnostics
        assert document.page_count() == 0
        assert document.standards is standards
        with pytest.raises(FrozenInstanceError):
            setattr(standards, "effective_version", PdfVersion(1, 0))  # noqa: B010
        with pytest.raises(AttributeError):
            setattr(document, "standards", DocumentStandards())  # noqa: B010
    with pytest.raises(PdfDocumentClosedError):
        assert document.standards


@pytest.mark.parametrize(
    ("header", "catalog", "effective", "diagnostic"),
    [
        (b"%PDF-1.4\n", b"/Version /2.0", PdfVersion(2, 0), None),
        (b"%PDF-2.0\n", b"/Version /1.4", PdfVersion(2, 0), "catalog-version-downgrade"),
        (b"%PDF-1.7\n", b"/Version (2.0)", PdfVersion(1, 7), "invalid-catalog-version"),
        (b"%PDF-9.0\n", b"", PdfVersion(9, 0), "unknown-version"),
        (b"%PDF-1.7\n", b"/Version /9.0", PdfVersion(9, 0), "unknown-version"),
        (b"%PDF-bad\n", b"", None, "invalid-header"),
        (b"", b"", None, "missing-header"),
        (b"prefix\n%PDF-1.7\n", b"", PdfVersion(1, 7), "displaced-header"),
        (b" " * 1019 + b"%PDF-1.7\n", b"", PdfVersion(1, 7), "displaced-header"),
    ],
)
def test_version_declarations_do_not_interrupt_reader_recovery(
    header: bytes, catalog: bytes, effective: PdfVersion | None, diagnostic: str | None
) -> None:
    data, ignored = internal_pdf(header=header, catalog=catalog)
    with PdfDocument(data) as document:
        assert document.standards.effective_version == effective
        if diagnostic is not None:
            assert diagnostic in {item.code for item in document.standards.diagnostics}
        assert document.page_count() == 0


@pytest.mark.parametrize("latest", [b"1.4", None, b"2.0"])
def test_incremental_catalog_cannot_erase_an_earlier_upgrade(latest: bytes | None) -> None:
    data, previous = internal_pdf(header=b"%PDF-1.0\n")
    data, previous = internal_revision(data, previous, b"1.7")
    data, ignored = internal_revision(data, previous, latest)
    with PdfDocument(data) as document:
        standards = document.standards
        assert standards.header_version == PdfVersion(1, 0)
        assert standards.catalog_version == (PdfVersion.parse(latest.decode()) if latest else None)
        assert standards.effective_version == (
            PdfVersion(2, 0) if latest == b"2.0" else PdfVersion(1, 7)
        )
        assert document.resolver.semantic_context == standards.context
        codes = {item.code for item in standards.diagnostics}
        assert ("historical-version-downgrade" in codes) == (latest != b"2.0")


def test_unreadable_history_reports_uncertainty_without_discarding_current_version() -> None:
    data, ignored = internal_pdf()
    standards = preserve_historical_version(
        DocumentStandards(header_version=PdfVersion(1, 7), effective_version=PdfVersion(1, 7)),
        data,
        {"Prev": 999999},
        None,
        recovered=True,
    )
    assert standards.effective_version == PdfVersion(1, 7)
    assert {item.code for item in standards.diagnostics} == {
        "recovered-version-history",
        "incomplete-version-history",
    }


def test_historical_hybrid_and_classic_prev_branches_can_converge() -> None:
    data, previous = internal_pdf(header=b"%PDF-1.0\n")
    root_offset = len(data)
    data += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /Version /1.7 >>\nendobj\n"
    stream_offset = len(data)
    row = b"\x01" + root_offset.to_bytes(4) + b"\x00\x00"
    data += (
        b"6 0 obj\n<< /Type /XRef /Size 7 /Index [1 1] /W [1 4 2] /Length 7"
        + f" /Prev {previous} >>\nstream\n".encode()
        + row
        + b"\nendstream\nendobj\n"
    )
    historical_xref = len(data)
    data += f"xref\n6 1\n{stream_offset:010} 00000 n \n".encode()
    data += (
        f"trailer\n<< /Size 7 /Root 1 0 R /Prev {previous} /XRefStm {stream_offset} >>\n"
        f"startxref\n{historical_xref}\n%%EOF\n"
    ).encode()
    standards = preserve_historical_version(
        DocumentStandards(header_version=PdfVersion(1, 0), effective_version=PdfVersion(1, 0)),
        data,
        {"Prev": historical_xref},
        None,
        recovered=False,
    )
    assert standards.effective_version == PdfVersion(1, 7)
    assert {item.code for item in standards.diagnostics} == {"historical-version-downgrade"}


def test_extension_declarations_are_independent_and_unknown_levels_stay_visible() -> None:
    data, ignored = internal_pdf(
        header=b"%PDF-2.0\n",
        catalog=(
            b"/Extensions << /Type /Extensions "
            b"/ADBE << /BaseVersion /1.7 /ExtensionLevel 3 /URL (known) >> "
            b"/FUTR [<< /BaseVersion /2.0 /ExtensionLevel 99 /URL (future) >> "
            b"<< /BaseVersion /2.0 /ExtensionLevel 100 /URL (future2) >>] "
            b"/BAD << /BaseVersion /2.0 /ExtensionLevel (3) >> >>"
        ),
    )
    with PdfDocument(data) as document:
        standards = document.standards
        assert [(item.prefix, item.extension_level) for item in standards.extensions] == [
            ("ADBE", 3),
            ("FUTR", 99),
            ("FUTR", 100),
        ]
        assert standards.extensions[1].url == "future"
        assert [item.code for item in standards.diagnostics].count("unknown-extension") == 2
        assert [item.code for item in standards.diagnostics].count("invalid-extension") == 1


def test_profile_claims_keep_namespaces_and_multiple_independent_targets() -> None:
    xmp = internal_xmp(
        b'<rdf:Description rdf:about="" a:part="2" a:conformance="B">'
        b"<u:part>1</u:part><p:GTS_PDFXVersion>PDF/X-4</p:GTS_PDFXVersion>"
        b'<d:declarations><rdf:Bag><rdf:li rdf:parseType="Resource">'
        b"<d:conformsTo>http://pdfa.org/declarations/wtpdf#reuse1.0</d:conformsTo>"
        b"</rdf:li></rdf:Bag></d:declarations></rdf:Description>"
    )
    data, ignored = internal_pdf(xmp=xmp, info=b"<< /GTS_PDFXVersion (PDF/X-3:2003) >>")
    with PdfDocument(data) as document:
        claims = document.standards.profile_claims
        assert {claim.identifier for claim in claims} == {
            "pdfa-2b",
            "pdfua-1",
            "pdfx-4",
            "pdfx-3:2003",
            "wtpdf-1.0-reuse",
        }
        pdfa = next(claim for claim in claims if claim.family == "PDF/A")
        assert ("{http://www.aiim.org/pdfa/ns/id/}part", "2") in pdfa.properties
        assert document.metadata["xmp"] is not None


def test_unrelated_namespace_and_non_document_resources_cannot_claim_profiles() -> None:
    xmp = internal_xmp(
        b'<rdf:Description rdf:about="" xmlns:fake="https://example.test/" '
        b'fake:part="1" fake:conformance="B"/>'
        b'<rdf:Description rdf:about="another-resource" a:part="1" a:conformance="B"/>'
        b'<rdf:Description rdf:nodeID="resource" u:part="1"/>'
    )
    data, ignored = internal_pdf(xmp=xmp)
    with PdfDocument(data) as document:
        assert not document.standards.profile_claims


@pytest.mark.parametrize("bare_rdf", [False, True])
def test_profile_claims_accept_only_document_level_rdf(bare_rdf: bool) -> None:
    xmp = internal_xmp(
        b'<rdf:Description a:part="2" a:conformance="B"/>'
        b'<rdf:Description rdf:about="another-resource"><rdf:RDF>'
        b'<rdf:Description u:part="1"/></rdf:RDF></rdf:Description>'
    )
    if bare_rdf:
        xmp = xmp.removeprefix(b'<x:xmpmeta xmlns:x="adobe:ns:meta/">').removesuffix(
            b"</x:xmpmeta>"
        )
    data, ignored = internal_pdf(xmp=xmp)
    with PdfDocument(data) as document:
        assert [item.identifier for item in document.standards.profile_claims] == ["pdfa-2b"]


def test_rdf_inside_an_unrelated_wrapper_does_not_claim_document_profiles() -> None:
    xmp = b"<unrelated>" + internal_xmp(b'<rdf:Description u:part="1"/>') + b"</unrelated>"
    data, ignored = internal_pdf(xmp=xmp)
    with PdfDocument(data) as document:
        assert not document.standards.profile_claims


@pytest.mark.parametrize(
    "namespace",
    ["http://pdfa.org/declarations/", "https://pdfa.org/declarations/"],
)
@pytest.mark.parametrize(
    ("declaration", "identifier"),
    [
        ("reuse1.0", "wtpdf-1.0-reuse"),
        ("reuse1.0-validated", "wtpdf-1.0-reuse"),
        ("accessibility1.0", "wtpdf-1.0-accessibility"),
        ("accessibility1.0-validated", "wtpdf-1.0-accessibility"),
        ("reuse2.0", None),
    ],
)
def test_wtpdf_registry_forms_remain_unverified_claims_with_original_uris(
    namespace: str, declaration: str, identifier: str | None
) -> None:
    uri = f"http://pdfa.org/declarations/wtpdf#{declaration}"
    xmp = internal_xmp(
        (
            f'<rdf:Description xmlns:d="{namespace}"><d:declarations>'
            '<rdf:Bag><rdf:li rdf:parseType="Resource">'
            f"<d:conformsTo>{uri}</d:conformsTo>"
            "</rdf:li></rdf:Bag></d:declarations></rdf:Description>"
        ).encode()
    )
    data, ignored = internal_pdf(xmp=xmp)
    with PdfDocument(data) as document:
        assert len(document.standards.profile_claims) == 1
        claim = document.standards.profile_claims[0]
        assert claim.identifier == identifier
        assert claim.family == "WTPDF"
        assert claim.properties == ((f"{{{namespace}}}conformsTo", uri),)
        assert (
            "unknown-profile-claim" in {item.code for item in document.standards.diagnostics}
        ) == (identifier is None)


def test_split_identification_properties_combine_but_conflicting_parts_remain_unidentified() -> (
    None
):
    xmp = internal_xmp(
        b'<rdf:Description rdf:about="" a:part="2"/>'
        b'<rdf:Description rdf:about="" a:conformance="B"/>'
        b'<rdf:Description rdf:about="" u:part="1"/>'
        b'<rdf:Description rdf:about="" u:part="2"/>'
    )
    data, ignored = internal_pdf(xmp=xmp)
    with PdfDocument(data) as document:
        claims = document.standards.profile_claims
        assert claims[0].identifier == "pdfa-2b"
        assert claims[1].identifier is None
        assert len(claims[1].properties) == 2
        assert "conflicting-profile-claim" in {item.code for item in document.standards.diagnostics}


@pytest.mark.parametrize(
    "xmp",
    [b"<broken", b'<!DOCTYPE x [<!ENTITY secret "expanded">]><x>&secret;</x>'],
)
def test_invalid_or_entity_expanding_xmp_is_an_optional_declaration_error(xmp: bytes) -> None:
    data, ignored = internal_pdf(xmp=xmp)
    with PdfDocument(data) as document:
        assert not document.standards.profile_claims
        assert "invalid-xmp" in {item.code for item in document.standards.diagnostics}
        assert document.page_count() == 0


def test_profile_claims_are_resolved_lazily_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = engine_document.discover_profile_claims

    def record(
        standards: DocumentStandards, resolver: PdfValueResolver, trailer: PdfDict
    ) -> DocumentStandards:
        nonlocal calls
        calls += 1
        return original(standards, resolver, trailer)

    monkeypatch.setattr(engine_document, "discover_profile_claims", record)
    data, ignored = internal_pdf()
    with PdfDocument(data) as document:
        assert calls == 0
        snapshot = document.standards
        assert document.standards is snapshot
        assert calls == 1


@pytest.mark.parametrize("version", ["1.4", "1.7", "2.0", "9.0"])
def test_core_keeps_tolerant_utf8_text_decoding_with_automatic_context(version: str) -> None:
    data, ignored = internal_pdf(header=f"%PDF-{version}\n".encode())
    text = b"\xef\xbb\xbfhello"
    with PdfDocument(data) as document:
        assert document.resolver.resolve_str(PdfString(text)) == "hello"
        if version in ("1.4", "1.7"):
            assert decode_pdf_text_string(text, context=document.standards.context) != "hello"


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.2", "1.7", "2.0", "9.0"])
def test_core_keeps_tolerant_utf16_text_decoding_with_automatic_context(version: str) -> None:
    data, ignored = internal_pdf(header=f"%PDF-{version}\n".encode())
    text = b"\xfe\xff\x00h\x00e\x00l\x00l\x00o"
    with PdfDocument(data) as document:
        assert document.resolver.resolve_str(PdfString(text)) == "hello"
        if version in ("1.0", "1.1"):
            assert decode_pdf_text_string(text, context=document.standards.context) != "hello"


def test_mac_authenticated_extensions_are_discovered_after_security(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = engine_document.discover_document_standards
    calls = 0

    def guarded(
        header: DocumentStandards, resolver: ObjectResolver, trailer: PdfDict
    ) -> DocumentStandards:
        nonlocal calls
        calls += 1
        assert resolver.decipher is not None
        return original(header, resolver, trailer)

    monkeypatch.setattr(engine_document, "discover_document_standards", guarded)
    path = Path(__file__).resolve().parents[4] / "tests/fixtures/security_interop/pdf_mac"
    with PdfDocument(path / "aes-256-r6-cbc-mac.pdf", password="user-mac-cbc") as document:
        assert document.standards.effective_version == PdfVersion(2, 0)
        assert any(item.prefix == "ISO_" for item in document.standards.extensions)
        assert calls == 1


@pytest.mark.parametrize(
    "declarations",
    [
        b'<rdf:Description a:part="4" a:rev="2099"/>',
        b'<rdf:Description u:part="2" u:rev="2099"/>',
        b'<rdf:Description a:part="9" a:conformance="B"/>',
        b'<rdf:Description a:part="2"/>',
    ],
)
def test_unknown_profile_editions_and_incomplete_claims_do_not_guess_a_target(
    declarations: bytes,
) -> None:
    data, ignored = internal_pdf(xmp=internal_xmp(declarations))
    with PdfDocument(data) as document:
        assert len(document.standards.profile_claims) == 1
        assert document.standards.profile_claims[0].identifier is None
        assert document.standards.profile_claims[0].properties
        assert "unknown-profile-claim" in {item.code for item in document.standards.diagnostics}


def test_invalid_extension_shapes_and_base_versions_are_diagnosed_independently() -> None:
    data, ignored = internal_pdf(
        catalog=(
            b"/Extensions << /Type /Wrong /EMPTY [] "
            b"/FUTR << /BaseVersion /2.0 /ExtensionLevel 1 >> >>"
        ),
    )
    with PdfDocument(data) as document:
        assert len(document.standards.extensions) == 1
        assert {item.code for item in document.standards.diagnostics} >= {
            "invalid-extensions-type",
            "invalid-extension",
            "extension-version-mismatch",
        }


def test_engineering_and_variable_printing_claims_preserve_exact_family_properties() -> None:
    xmp = internal_xmp(
        b'<rdf:Description xmlns:v="http://www.npes.org/pdfvt/ns/id/" '
        b'xmlns:e="http://www.aiim.org/pdfe/ns/id/">'
        b"<v:GTS_PDFVTVersion>PDF/VT-3</v:GTS_PDFVTVersion><v:rev>2020</v:rev>"
        b"<e:ISO_PDFEVersion>PDF/E-1</e:ISO_PDFEVersion></rdf:Description>"
    )
    data, ignored = internal_pdf(xmp=xmp)
    with PdfDocument(data) as document:
        assert {item.identifier for item in document.standards.profile_claims} == {
            "pdfvt-3",
            "pdfe-1",
        }

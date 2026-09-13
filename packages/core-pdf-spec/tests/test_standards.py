# SPDX-License-Identifier: AGPL-3.0-only
"""Version recognition stays distinct from strict contextual rules and coverage."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_07_document.standards import (
    effective_pdf_version,
    parse_catalog_version,
    parse_extension,
    parse_extensions,
    parse_header_version,
)
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf_spec.standards import (
    PDF_2_0_BASELINE,
    STANDARD_PROFILES,
    DocumentStandards,
    PdfExtension,
    PdfVersion,
    ProfileClaim,
    SemanticContext,
    get_extension_coverage,
    get_standard_profile,
)
from core_pdf_spec.types import PdfName, PdfReference, PdfString


@pytest.mark.parametrize("version", [*(f"1.{minor}" for minor in range(8)), "2.0"])
def test_recognizes_published_versions(version: str) -> None:
    parsed = PdfVersion.parse(version)
    assert parsed.recognized
    assert str(parsed) == version
    assert parse_header_version(f"%PDF-{version}\r\n".encode()) == parsed
    assert parse_catalog_version(PdfName.of(version)) == parsed


@pytest.mark.parametrize("version", ["1", "1.10", "01.7", "1.7 ", " 2.0", "/2.0", "１.７"])
def test_rejects_malformed_version_names(version: str) -> None:
    with pytest.raises(ValueError, match="invalid PDF version"):
        PdfVersion.parse(version)


@pytest.mark.parametrize("header", [b"%PDF-1.7", b" %PDF-1.7\n", b"%PDF-1.7x\n", b"%PDF-1.x\n"])
def test_header_is_strict_and_does_not_search_or_repair(header: bytes) -> None:
    with pytest.raises(ValueError, match="header"):
        parse_header_version(header)


def test_unknown_versions_are_preserved_and_numeric_ordering_is_not_lexical() -> None:
    assert parse_header_version(memoryview(b"%PDF-2.1\n")) == PdfVersion(2, 1)
    assert not PdfVersion(2, 1).recognized
    assert not PdfVersion(1, 8).recognized
    assert PdfVersion(1, 10) > PdfVersion(1, 9)
    assert effective_pdf_version(PdfVersion(1, 7), PdfVersion(3, 0)) == PdfVersion(3, 0)


def test_catalog_version_requires_a_name() -> None:
    with pytest.raises(ValueError, match="PDF name"):
        parse_catalog_version(PdfString(b"2.0"))


def test_catalog_upgrade_and_incremental_update_cannot_downgrade() -> None:
    # ISO 32000-2:2020 7.5.6 errata explicitly includes preceding catalog versions.
    assert effective_pdf_version(PdfVersion(1, 4), PdfVersion(1, 7)) == PdfVersion(1, 7)
    assert effective_pdf_version(PdfVersion(1, 7), PdfVersion(1, 4)) == PdfVersion(1, 7)
    assert effective_pdf_version(PdfVersion(1, 4), None, previous=PdfVersion(2, 0)) == PdfVersion(
        2, 0
    )
    assert effective_pdf_version(None, None) is None


@pytest.mark.parametrize("level", [True, False, 1.0, 0, -1, PdfReference(3)])
def test_extension_level_is_a_positive_direct_integer(level: object) -> None:
    with pytest.raises(ValueError, match="ExtensionLevel"):
        parse_extension("ADBE", {"BaseVersion": PdfName.of("1.7"), "ExtensionLevel": level})


def test_extensions_keep_independent_levels_prefixes_and_revisions() -> None:
    extensions = parse_extensions(
        {
            "Type": PdfName.of("Extensions"),
            "ADBE": {"BaseVersion": PdfName.of("1.7"), "ExtensionLevel": 3},
            "ISO_": [
                {
                    "BaseVersion": PdfName.of("2.0"),
                    "ExtensionLevel": 32004,
                    "ExtensionRevision": PdfString(b":2024"),
                    "URL": PdfString(b"https://www.iso.org/standard/45877.html"),
                },
                {"BaseVersion": PdfName.of("2.0"), "ExtensionLevel": 32003},
            ],
            "FUTR": {"BaseVersion": PdfName.of("3.0"), "ExtensionLevel": 1},
        }
    )
    assert [(ext.prefix, ext.extension_level) for ext in extensions] == [
        ("ADBE", 3),
        ("ISO_", 32004),
        ("ISO_", 32003),
        ("FUTR", 1),
    ]
    assert extensions[1].extension_revision == ":2024"
    assert extensions[-1].base_version == PdfVersion(3, 0)
    assert get_extension_coverage(extensions[1]) is not None
    assert get_extension_coverage(extensions[-1]) is None
    assert get_extension_coverage(PdfExtension("ADBE", PdfVersion(1, 7), 4)) is None
    assert (
        get_extension_coverage(
            PdfExtension("ISO_", PdfVersion(2, 0), 32004, extension_revision=":2025")
        )
        is None
    )


def test_contextual_extension_constraints_and_valid_defaults() -> None:
    assert parse_extensions(None) == ()
    assert parse_extensions({}) == ()
    declaration = {"BaseVersion": PdfName.of("1.7"), "ExtensionLevel": 1}
    assert parse_extensions({"TEST": declaration}, context=SemanticContext(PdfVersion(1, 7)))
    with pytest.raises(ValueError, match="require URL"):
        parse_extensions({"TEST": declaration}, context=SemanticContext(PdfVersion(2, 0)))
    with pytest.raises(ValueError, match="exceeds"):
        parse_extensions({"TEST": declaration}, context=SemanticContext(PdfVersion(1, 6)))
    with pytest.raises(ValueError, match="arrays require"):
        parse_extensions({"TEST": [declaration]}, context=SemanticContext(PdfVersion(1, 7)))
    declaration["URL"] = PdfString(b"https://example.invalid/extension")
    assert parse_extensions({"TEST": [declaration]}, context=SemanticContext(PdfVersion(2, 0)))


@pytest.mark.parametrize(
    "declarations",
    [
        1,
        {"Type": PdfString(b"Extensions")},
        {"ADBE": PdfReference(3)},
        {"ADBE": []},
        {"ADBE": {"BaseVersion": PdfReference(2), "ExtensionLevel": 1}},
        {"ADBE": {"BaseVersion": PdfName.of("1.7"), "ExtensionLevel": 1, "URL": PdfReference(2)}},
    ],
)
def test_extensions_do_not_resolve_or_skip_invalid_entries(declarations: object) -> None:
    with pytest.raises(ValueError):
        parse_extensions(declarations)


def test_text_encoding_changes_only_with_pdf_2_0() -> None:
    # ISO 32000-1:2008 7.9.2.2 vs ISO 32000-2:2020 7.9.2.2.1.
    utf8 = b"\xef\xbb\xbfHello"
    assert decode_pdf_text_string(utf8, context=SemanticContext(PdfVersion(1, 7))) == "ï»¿Hello"
    assert decode_pdf_text_string(utf8, context=SemanticContext(PdfVersion(2, 0))) == "Hello"
    assert decode_pdf_text_string(utf8) == "Hello"
    for version in (PdfVersion(1, 7), PdfVersion(2, 0)):
        context = SemanticContext(version)
        assert decode_pdf_text_string(b"\xfe\xff\x00H", context=context) == "H"
        assert decode_pdf_text_string(memoryview(b"A\x80B"), context=context) == "A•B"


def test_unknown_context_and_malformed_unicode_fail_strictly() -> None:
    for version in (None, PdfVersion(2, 1)):
        with pytest.raises(PdfUnsupportedError):
            decode_pdf_text_string(b"text", context=SemanticContext(version))
    with pytest.raises(ValueError, match="UTF-8"):
        decode_pdf_text_string(b"\xef\xbb\xbf\xff", context=SemanticContext(PdfVersion(2, 0)))
    assert (
        decode_pdf_text_string(b"\xef\xbb\xbf\xff", context=SemanticContext(PdfVersion(1, 7)))
        == "ï»¿ÿ"
    )


def test_object_resolver_accepts_context_after_bootstrap() -> None:
    resolver = ObjectResolver(b"", {}, semantic_context=SemanticContext(PdfVersion(1, 7)))
    try:
        assert resolver.resolve_str(PdfString(b"\xef\xbb\xbfHello")) == "ï»¿Hello"
        resolver.semantic_context = SemanticContext(PdfVersion(2, 0))
        assert resolver.resolve_str(PdfString(b"\xef\xbb\xbfHello")) == "Hello"
    finally:
        resolver.close()


def test_declaration_projection_is_immutable_and_preserves_claim_properties() -> None:
    claim = ProfileClaim("pdfa-4", "PDF/A", "xmp", (("{pdfaid}rev", "2020"),))
    document = DocumentStandards(effective_version=PdfVersion(2, 0), profile_claims=(claim,))
    assert document.context.version == PdfVersion(2, 0)
    assert document.context.baseline == PDF_2_0_BASELINE
    assert len(PDF_2_0_BASELINE.errata_revision or "") == 40
    with pytest.raises(FrozenInstanceError):
        setattr(document, "effective_version", PdfVersion(1, 7))  # noqa: B010


def test_profile_catalog_separates_families_editions_and_unknown_targets() -> None:
    assert len({profile.identifier for profile in STANDARD_PROFILES}) == len(STANDARD_PROFILES)
    assert get_standard_profile("pdfa-4") is not None
    profile = get_standard_profile("pdfua-2")
    assert profile is not None
    assert profile.edition == "ISO 14289-2:2024"
    assert profile.base_version == PdfVersion(2, 0)
    assert get_standard_profile("pdfx-1a:2003") is not None
    assert get_standard_profile("wtpdf-1.0-reuse") is not None
    assert get_standard_profile("wtpdf-1.0-accessibility") is not None
    assert get_standard_profile("wtpdf-1.0") is None
    assert get_standard_profile("pdfa-99") is None


@pytest.mark.parametrize(
    ("identifier", "family", "edition", "base_version"),
    [
        ("pdfe-1", "PDF/E", "ISO 24517-1:2008", PdfVersion(1, 6)),
        ("pdfvt-1", "PDF/VT", "ISO 16612-2:2010", PdfVersion(1, 6)),
        ("pdfvt-2", "PDF/VT", "ISO 16612-2:2010", PdfVersion(1, 6)),
        ("pdfvt-3", "PDF/VT", "ISO 16612-3:2020", PdfVersion(2, 0)),
        ("pdfr-1", "PDF/R", "ISO 23504-1:2020", PdfVersion(1, 7)),
    ],
)
def test_additional_family_reference_catalog(
    identifier: str, family: str, edition: str, base_version: PdfVersion
) -> None:
    # These identities do not advertise validation support. In particular,
    # PDF/VT part names do not map one-to-one to ISO publication part numbers.
    profile = get_standard_profile(identifier)
    assert profile is not None
    assert (profile.family, profile.edition, profile.base_version) == (
        family,
        edition,
        base_version,
    )


def test_aes_gcm_extension_coverage_requires_published_revision_identity() -> None:
    # ISO/TS 32003:2023, clause 4 Table 1 includes the colon in :2023.
    extension = parse_extension(
        "ISO_",
        {
            "BaseVersion": PdfName.of("2.0"),
            "ExtensionLevel": 32003,
            "ExtensionRevision": PdfString(b":2023"),
            "URL": PdfString(b"https://www.iso.org/standard/45876.html"),
        },
    )
    coverage = get_extension_coverage(extension)
    assert coverage is not None
    assert coverage.features == ("AES-256-GCM revision-7 decryption",)
    assert get_extension_coverage(PdfExtension("ISO_", PdfVersion(2, 0), 32003)) is None
    assert (
        get_extension_coverage(
            PdfExtension("ISO_", PdfVersion(2, 0), 32003, extension_revision="2023")
        )
        is None
    )


def test_revised_blend_coverage_requires_exact_adobe_extension_identity() -> None:
    # Adobe PDF 1.7 ExtensionLevel 5 supplement, 3.1: revised separable modes.
    coverage = get_extension_coverage(PdfExtension("ADBE", PdfVersion(1, 7), 5))
    assert coverage is not None
    assert coverage.features == ("ColorDodge and ColorBurn revised component blending",)
    for extension in (
        PdfExtension("ADBE", PdfVersion(1, 7), 6),
        PdfExtension("ADBE", PdfVersion(2, 0), 5),
        PdfExtension("TEST", PdfVersion(1, 7), 5),
        PdfExtension("ADBE", PdfVersion(1, 7), 5, extension_revision="future"),
    ):
        assert get_extension_coverage(extension) is None

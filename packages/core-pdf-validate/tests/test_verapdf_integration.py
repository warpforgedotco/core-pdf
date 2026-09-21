# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from core_pdf_validate import VeraPdfBackend, validate
from core_pdf_validate.verapdf import internal_parse_report

internal_POSITIVES = Path(__file__).parent / "fixtures/positive"
internal_MANIFEST = json.loads((internal_POSITIVES / "manifest.json").read_text())
internal_POSITIVE_CASES: tuple[tuple[str, str, str], ...] = tuple(
    (profile, fixture["filename"], fixture["sha256"])
    for fixture in internal_MANIFEST["fixtures"]
    for profile in fixture["profiles"]
)


def test_positive_fixture_provenance_and_complete_profile_coverage() -> None:
    profiles = [profile for profile, filename, sha256 in internal_POSITIVE_CASES]
    assert len(profiles) == len(set(profiles))
    assert set(profiles) == {support.identifier for support in VeraPdfBackend.supported_profiles}
    for fixture in internal_MANIFEST["fixtures"]:
        data = (internal_POSITIVES / fixture["filename"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == fixture["sha256"]
        if fixture["filename"] != "pdfa-3a-actualtext.pdf":
            assert fixture["sha256"] == fixture["source_sha256"]
    source = (internal_POSITIVES / "pdfa-2a-actualtext.pdf").read_bytes()
    derived = (internal_POSITIVES / "pdfa-3a-actualtext.pdf").read_bytes()
    assert source.count(b'pdfaid:part="2"') == 1
    assert derived == source.replace(b'pdfaid:part="2"', b'pdfaid:part="3"')


@pytest.mark.parametrize(
    "profile", [profile for profile, filename, sha256 in internal_POSITIVE_CASES]
)
def test_recorded_real_pass_for_every_advertised_profile(profile: str) -> None:
    raw = (internal_POSITIVES / f"verapdf-1.30.2-{profile}-pass.xml").read_bytes()
    result = internal_parse_report(raw, profile=profile, returncode=0, stderr=b"")
    assert result.execution_status == "completed", result.diagnostics
    assert result.conformance == "pass"
    assert result.engine_version == "1.30.2"
    assert result.raw_report == raw
    assert not result.rules


@pytest.fixture(scope="module")
def installed_backend() -> VeraPdfBackend:
    executable = os.environ.get("CORE_PDF_VERAPDF")
    if not executable:
        if os.environ.get("CORE_PDF_REQUIRE_VERAPDF") == "1":
            pytest.fail("CORE_PDF_VERAPDF must name the pinned engine in the veraPDF CI job")
        pytest.skip("set CORE_PDF_VERAPDF to run installed-engine integration tests")
    return VeraPdfBackend(executable, timeout=60)


def internal_fixture(relative: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    source = root / "tests/fixtures" / relative
    assert source.is_file(), f"Initialize reference submodules: missing {source}"
    return source


@pytest.mark.parametrize(("profile", "filename", "sha256"), internal_POSITIVE_CASES)
def test_installed_engine_passes_every_advertised_profile(
    installed_backend: VeraPdfBackend, profile: str, filename: str, sha256: str
) -> None:
    source = internal_POSITIVES / filename
    original = source.read_bytes()
    assert hashlib.sha256(original).hexdigest() == sha256
    report = validate(source, profiles=profile, backend=installed_backend)
    result = report.results[0]
    assert result.execution_status == "completed", result.diagnostics
    assert result.conformance == "pass"
    assert result.engine_version == "1.30.2"
    assert result.profile_edition == next(
        support.edition
        for support in installed_backend.supported_profiles
        if support.identifier == profile
    )
    assert result.raw_report
    assert not result.rules
    assert report.source_sha256 == sha256
    assert source.read_bytes() == original
    if profile.startswith(("pdfua-", "wtpdf-")):
        assert any("Human review" in limitation for limitation in result.limitations)


def test_installed_engine_reports_real_pdfa_pass(installed_backend: VeraPdfBackend) -> None:
    source = internal_fixture("pypdf/sample-files/021-pdfa/crazyones-pdfa.pdf")
    original = source.read_bytes()
    report = validate(source, profiles="pdfa-1b", backend=installed_backend)
    result = report.results[0]
    assert result.execution_status == "completed", result.diagnostics
    assert result.conformance == "pass"
    assert result.engine_version == "1.30.2"
    assert result.profile_edition == "ISO 19005-1:2005"
    assert result.raw_report
    assert report.source_sha256 == hashlib.sha256(original).hexdigest()
    assert source.read_bytes() == original


def test_installed_engine_passes_all_three_original_accessibility_declarations(
    installed_backend: VeraPdfBackend,
) -> None:
    source = internal_POSITIVES / "pdfua-2-wtpdf-unicode.pdf"
    original = source.read_bytes()
    report = validate(source, profiles="declared", backend=installed_backend)
    assert set(report.targets) == {"pdfua-2", "wtpdf-1.0-reuse", "wtpdf-1.0-accessibility"}
    assert len(report.results) == 3
    assert all(result.execution_status == "completed" for result in report.results)
    assert all(result.conformance == "pass" for result in report.results)
    assert report.source_sha256 == hashlib.sha256(original).hexdigest()
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "profile", [support.identifier for support in VeraPdfBackend.supported_profiles]
)
def test_installed_engine_recognizes_every_advertised_profile(
    installed_backend: VeraPdfBackend, profile: str
) -> None:
    source = internal_fixture("pdf20examples/Simple PDF 2.0 file.pdf")
    result = validate(source, profiles=profile, backend=installed_backend).results[0]
    assert result.execution_status == "completed", result.diagnostics
    assert result.conformance == "fail"
    assert result.engine_version == "1.30.2"
    assert result.profile_edition == next(
        support.edition
        for support in installed_backend.supported_profiles
        if support.identifier == profile
    )
    assert result.rules
    assert all(rule.specification and rule.clause and rule.test_number for rule in result.rules)
    assert any(rule.locations for rule in result.rules)
    assert result.raw_report

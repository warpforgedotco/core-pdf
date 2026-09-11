# SPDX-License-Identifier: AGPL-3.0-only
"""Adapter behavior using recorded real reports and local executable stand-ins."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core_pdf_validate import ProfileResult, ProfileSupport, VeraPdfBackend, validate
from core_pdf_validate.verapdf import internal_parse_report

internal_FIXTURES = Path(__file__).parent / "fixtures"


def internal_report(name: str) -> bytes:
    return (internal_FIXTURES / f"verapdf-1.30.2-{name}.xml").read_bytes()


def internal_executable(
    directory: Path,
    *,
    report: bytes,
    code: int = 0,
    delay: float = 0,
) -> Path:
    executable = directory / "validator with spaces"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys, time\n"
        f"time.sleep({delay!r})\n"
        "source = pathlib.Path(sys.argv[-1])\n"
        f"pathlib.Path({str(directory / 'received.pdf')!r}).write_bytes(source.read_bytes())\n"
        f"pathlib.Path({str(directory / 'args.json')!r}).write_text(json.dumps(sys.argv[1:]))\n"
        f"sys.stdout.buffer.write({report!r})\n"
        "sys.stderr.buffer.write(b'engine log\\n')\n"
        f"sys.exit({code})\n"
    )
    executable.chmod(0o700)
    return executable


def internal_pdf(metadata: bytes | None = None, *, info: bytes | None = None) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R"
        + (b" /Metadata 4 0 R" if metadata is not None else b"")
        + b" >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] >>",
    ]
    if metadata is not None:
        objects.append(
            b"<< /Type /Metadata /Subtype /XML /Length "
            + str(len(metadata)).encode()
            + b" >>\nstream\n"
            + metadata
            + b"\nendstream"
        )
    if info is not None:
        objects.append(info)
    output = bytearray(b"%PDF-2.0\n")
    offsets = [0]
    for number, value in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode() + value + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    trailer = f"trailer\n<< /Size {len(offsets)} /Root 1 0 R"
    if info is not None:
        trailer += f" /Info {len(objects)} 0 R"
    output.extend((trailer + f" >>\nstartxref\n{xref}\n%%EOF\n").encode())
    return bytes(output)


def internal_xmp(properties: bytes) -> bytes:
    return (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description rdf:about="" xmlns:a="http://www.aiim.org/pdfa/ns/id/" '
        b'xmlns:u="http://www.aiim.org/pdfua/ns/id/" ' + properties + b"/></rdf:RDF></x:xmpmeta>"
    )


class RecordingBackend:
    name = "test"
    supported_profiles = (
        ProfileSupport("pdfa-1b", "ISO 19005-1:2005"),
        ProfileSupport("pdfua-2", "ISO 14289-2:2024"),
    )

    def __init__(self) -> None:
        self.received: list[tuple[str, bytes]] = []

    def validate(self, source: Path, *, profile: str) -> ProfileResult:
        self.received.append((profile, source.read_bytes()))
        source.write_bytes(b"simulate an external backend modifying its private copy")
        return ProfileResult(profile, "completed", "pass", self.name)


def test_explicit_validation_passes_original_bytes_without_core_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core_pdf

    def forbidden_parse(*args: object, **kwargs: object) -> None:
        raise AssertionError("explicit validation must not parse with core")

    monkeypatch.setattr(core_pdf, "PdfDocument", forbidden_parse)
    original = b"intentionally unparseable as a core PDF\x00\xff"
    engine = internal_executable(tmp_path, report=internal_report("pass"))
    report = validate(original, profiles="pdfa-1b", backend=VeraPdfBackend(engine))
    assert report.source_sha256 == hashlib.sha256(original).hexdigest()
    assert (tmp_path / "received.pdf").read_bytes() == original
    args = json.loads((tmp_path / "args.json").read_text())
    assert args[:-1] == ["--format", "xml", "--flavour", "1b", "--maxfailures", "-1"]
    assert not Path(args[-1]).exists()
    result = report.results[0]
    assert result.execution_status == "completed"
    assert result.conformance == "pass"
    assert result.engine_version == "1.30.2"
    assert result.profile_edition == "ISO 19005-1:2005"
    assert result.raw_report == internal_report("pass")
    assert result.stderr == b"engine log\n"


def test_nonconforming_exit_is_completed_with_rule_references_and_locations(tmp_path: Path) -> None:
    engine = internal_executable(tmp_path, report=internal_report("fail"), code=1)
    result = validate(b"original", profiles="pdfua-2", backend=VeraPdfBackend(engine)).results[0]
    assert result.execution_status == "completed"
    assert result.conformance == "fail"
    assert len(result.rules) == 8
    assert result.rules[0].specification == "ISO 14289-2:2024"
    assert result.rules[0].clause == "5"
    assert result.rules[0].test_number == "1"
    assert result.rules[0].locations == (
        "root/document[0]/metadata[0](2 0 obj PDMetadata)/XMPPackage[0]",
    )
    assert any("Human review" in limit for limit in result.limitations)


def test_profiles_are_independent_and_source_path_is_never_given_to_backend(tmp_path: Path) -> None:
    source = tmp_path / "original.pdf"
    original = internal_pdf()
    source.write_bytes(original)
    backend = RecordingBackend()
    report = validate(
        source,
        profiles=["pdfa-1b", "pdfx-4", "pdfua-2", "pdfa-1b"],
        backend=backend,
    )
    assert report.targets == ("pdfa-1b", "pdfx-4", "pdfua-2")
    assert backend.received == [("pdfa-1b", original), ("pdfua-2", original)]
    assert source.read_bytes() == original
    assert [result.conformance for result in report.results] == ["pass", "not_checked", "pass"]
    assert report.results[1].execution_status == "unsupported_profile"


def test_declared_profiles_request_all_claims_even_with_different_base_versions() -> None:
    original = internal_pdf(internal_xmp(b'a:part="1" a:conformance="B" u:part="2" u:rev="2024"'))
    backend = RecordingBackend()
    report = validate(original, profiles="declared", backend=backend)
    assert set(report.targets) == {"pdfa-1b", "pdfua-2"}
    assert len(report.results) == 2
    assert all(data == original for ignored, data in backend.received)


def test_declared_pdfx_reports_unsupported() -> None:
    source = internal_pdf(info=b"<< /GTS_PDFXVersion (PDF/X-4) >>")
    report = validate(source, profiles="declared", backend=RecordingBackend())
    assert report.targets == ("pdfx-4",)
    assert report.results[0].execution_status == "unsupported_profile"


@pytest.mark.parametrize("metadata", [None, b"<invalid", internal_xmp(b'a:part="9"')])
def test_missing_malformed_or_unknown_claims_never_select_a_default(metadata: bytes | None) -> None:
    backend = RecordingBackend()
    report = validate(internal_pdf(metadata), profiles="declared", backend=backend)
    assert not report.targets
    assert not report.results
    assert "no_target_identified" in report.diagnostics
    assert not backend.received
    if metadata == b"<invalid":
        assert any("invalid-xmp" in item for item in report.diagnostics)
    elif metadata is not None:
        assert any("unresolved_profile_claim" in item for item in report.diagnostics)


def test_unparseable_declared_input_explains_absent_targets() -> None:
    report = validate(b"this is not a PDF", profiles="declared", backend=RecordingBackend())
    assert not report.targets
    assert "no_target_identified" in report.diagnostics
    assert any(
        "declaration_read_failed" in diagnostic or "missing-header" in diagnostic
        for diagnostic in report.diagnostics
    )


def test_missing_engine_remains_not_checked(tmp_path: Path) -> None:
    result = validate(
        b"PDF",
        profiles="pdfa-1b",
        backend=VeraPdfBackend(tmp_path / "missing"),
    ).results[0]
    assert result.execution_status == "engine_unavailable"
    assert result.conformance == "not_checked"


def test_timeout_remains_not_checked(tmp_path: Path) -> None:
    engine = internal_executable(tmp_path, report=internal_report("pass"), delay=2)
    result = validate(b"PDF", profiles="pdfa-1b", backend=VeraPdfBackend(engine, 0.05)).results[0]
    assert result.execution_status == "timeout"
    assert result.conformance == "not_checked"


@pytest.mark.parametrize(
    ("old", "new", "status"),
    [
        (b"1.30.2", b"1.31.0", "unsupported_engine"),
        (b"PDF/A-1b", b"PDF/A-2b", "invalid_report"),
        (b'jobEndStatus="normal"', b'jobEndStatus="maxfailures"', "incomplete"),
        (b'jobEndStatus="normal"', b"", "incomplete"),
        (b'passedChecks="3825"', b'passedChecks="0"', "incomplete"),
        (b'outOfMemory="0"', b'outOfMemory="1"', "incomplete"),
        (b'failedJobs="0"', b'failedJobs="1"', "incomplete"),
        (b'failedRules="0"', b'failedRules="1"', "invalid_report"),
        (b'isCompliant="true"', b'isCompliant="false"', "invalid_report"),
        (b"<report>", b'<report xmlns="unknown">', "invalid_report"),
    ],
)
def test_untrusted_report_cannot_produce_conformance(old: bytes, new: bytes, status: str) -> None:
    raw = internal_report("pass").replace(old, new)
    result = internal_parse_report(raw, profile="pdfa-1b", returncode=0, stderr=b"")
    assert result.execution_status == status
    assert result.conformance == "not_checked"
    assert result.raw_report == raw


@pytest.mark.parametrize(
    "raw",
    [
        b"not XML",
        b"<report>",
        b"<report/>",
        '<!DOCTYPE report [<!ENTITY x "expansion">]><report>&x;</report>'.encode("utf-16"),
    ],
)
def test_malformed_or_entity_report_is_rejected(raw: bytes) -> None:
    result = internal_parse_report(raw, profile="pdfa-1b", returncode=0, stderr=b"")
    assert result.execution_status == "invalid_report"
    assert result.conformance == "not_checked"


@pytest.mark.parametrize(
    ("code", "status"), [(1, "invalid_report"), (3, "engine_error"), (-9, "engine_error")]
)
def test_exit_status_cannot_be_ignored(code: int, status: str) -> None:
    result = internal_parse_report(
        internal_report("pass"),
        profile="pdfa-1b",
        returncode=code,
        stderr=b"",
    )
    assert result.execution_status == status
    assert result.conformance == "not_checked"


def test_truncated_locations_are_reported_as_coverage_limitation() -> None:
    raw = internal_report("fail").replace(
        b'failedChecks="1" tags="metadata"', b'failedChecks="2" tags="metadata"'
    )
    result = internal_parse_report(raw, profile="pdfua-2", returncode=1, stderr=b"")
    assert result.conformance == "fail"
    assert any("omitted" in value for value in result.limitations)


def test_results_are_immutable() -> None:
    report = validate(b"source", profiles="pdfa-1b", backend=RecordingBackend())
    targets_field = "targets"
    conformance_field = "conformance"
    with pytest.raises(FrozenInstanceError):
        setattr(report, targets_field, ())
    with pytest.raises(FrozenInstanceError):
        setattr(report.results[0], conformance_field, "fail")


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_timeout_requires_positive_finite_number(timeout: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        VeraPdfBackend(timeout=timeout)


@pytest.mark.parametrize("profiles", [[], "", ["declared"], ["pdfa-1b", ""]])
def test_invalid_target_arguments_raise(profiles: str | list[str]) -> None:
    with pytest.raises(ValueError, match="canonical IDs"):
        validate(b"source", profiles=profiles)

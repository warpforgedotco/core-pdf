# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from core_pdf_validate import ProfileResult, ProfileSupport, VeraPdfBackend, validate
from core_pdf_validate.verapdf import parse_report

FIXTURES = Path(__file__).parent / "fixtures"


def internal_report(name: str) -> bytes:
    return (FIXTURES / f"verapdf-1.30.2-{name}.xml").read_bytes()


def executable(
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


def pdf(metadata: bytes | None = None, *, info: bytes | None = None) -> bytes:
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


def xmp(properties: bytes) -> bytes:
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
    engine = executable(tmp_path, report=internal_report("pass"))
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
    engine = executable(tmp_path, report=internal_report("fail"), code=1)
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
    original = pdf()
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
    original = pdf(xmp(b'a:part="1" a:conformance="B" u:part="2" u:rev="2024"'))
    backend = RecordingBackend()
    report = validate(original, profiles="declared", backend=backend)
    assert set(report.targets) == {"pdfa-1b", "pdfua-2"}
    assert len(report.results) == 2
    assert all(data == original for ignored, data in backend.received)


def test_declared_pdfx_reports_unsupported() -> None:
    source = pdf(info=b"<< /GTS_PDFXVersion (PDF/X-4) >>")
    report = validate(source, profiles="declared", backend=RecordingBackend())
    assert report.targets == ("pdfx-4",)
    assert report.results[0].execution_status == "unsupported_profile"


@pytest.mark.parametrize("metadata", [None, b"<invalid", xmp(b'a:part="9"')])
def test_missing_malformed_or_unknown_claims_never_select_a_default(metadata: bytes | None) -> None:
    backend = RecordingBackend()
    report = validate(pdf(metadata), profiles="declared", backend=backend)
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
    engine = executable(tmp_path, report=internal_report("pass"), delay=2)
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
    result = parse_report(raw, profile="pdfa-1b", returncode=0, stderr=b"")
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
    result = parse_report(raw, profile="pdfa-1b", returncode=0, stderr=b"")
    assert result.execution_status == "invalid_report"
    assert result.conformance == "not_checked"


@pytest.mark.parametrize(
    ("code", "status"), [(1, "invalid_report"), (3, "engine_error"), (-9, "engine_error")]
)
def test_exit_status_cannot_be_ignored(code: int, status: str) -> None:
    result = parse_report(
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
    result = parse_report(raw, profile="pdfua-2", returncode=1, stderr=b"")
    assert result.conformance == "fail"
    assert any("omitted" in value for value in result.limitations)


def test_results_are_immutable() -> None:
    report = validate(b"source", profiles="pdfa-1b", backend=RecordingBackend())
    targets_field = "targets"
    conformance_field = "conformance"
    with pytest.raises(AttributeError):
        setattr(report, targets_field, ())
    with pytest.raises(AttributeError):
        setattr(report.results[0], conformance_field, "fail")


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_timeout_requires_positive_finite_number(timeout: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        VeraPdfBackend(timeout=timeout)


@pytest.mark.parametrize("profiles", [[], "", ["declared"], ["pdfa-1b", ""]])
def test_invalid_target_arguments_raise(profiles: str | list[str]) -> None:
    with pytest.raises(ValueError, match="canonical IDs"):
        validate(b"source", profiles=profiles)


@pytest.mark.parametrize(
    ("path", "attribute", "value", "status", "diagnostic"),
    [
        (".//details", "passedRules", "-1", "invalid_report", "invalid details/@passedRules"),
        (".//details", "passedChecks", "１２", "invalid_report", "invalid details/@passedChecks"),
        (".//validationReport", "isCompliant", "unknown", "invalid_report", "isCompliant"),
        (".//batchSummary", "totalJobs", "2", "invalid_report", "one source document"),
        (".//batchSummary", "outOfMemory", "1", "incomplete", "incomplete processing"),
        (".//validationReports", "failedJobs", "1", "incomplete", "failed validation jobs"),
    ],
)
def test_report_rejects_invalid_counts_and_incomplete_jobs(
    path: str,
    attribute: str,
    value: str,
    status: str,
    diagnostic: str,
) -> None:
    from xml.etree import ElementTree

    root = ElementTree.fromstring(internal_report("pass"))
    node = root.find(path)
    assert node is not None
    node.set(attribute, value)
    raw = ElementTree.tostring(root)
    result = parse_report(raw, profile="pdfa-1b", returncode=0, stderr=b"engine log")
    assert result.execution_status == status
    assert result.conformance == "not_checked"
    assert diagnostic in result.diagnostics[0]
    assert result.raw_report == raw
    assert result.stderr == b"engine log"


@pytest.mark.parametrize(
    ("path", "attribute", "value", "diagnostic"),
    [
        (".//rule", "status", "unknown", "Unknown rule status"),
        (".//rule", "clause", "", "Missing rule reference"),
        (".//check", "status", "unknown", "Unknown check status"),
        (".//rule", "failedChecks", "0", "Inconsistent failed rule counts"),
        (".//rule", "failedChecks", "", "Missing or invalid"),
    ],
)
def test_report_rejects_malformed_rule_details(
    path: str,
    attribute: str,
    value: str,
    diagnostic: str,
) -> None:
    from xml.etree import ElementTree

    root = ElementTree.fromstring(internal_report("fail"))
    node = root.find(path)
    assert node is not None
    node.set(attribute, value)
    result = parse_report(ElementTree.tostring(root), profile="pdfua-2", returncode=1, stderr=b"")
    assert result.execution_status == "invalid_report"
    assert diagnostic in result.diagnostics[0]


def test_passed_rule_details_are_retained_without_failure_counts() -> None:
    from xml.etree import ElementTree

    root = ElementTree.fromstring(internal_report("pass"))
    details = root.find(".//details")
    assert details is not None
    rule = ElementTree.SubElement(
        details,
        "rule",
        specification="ISO 19005-1:2005",
        clause="6.1",
        testNumber="1",
        status="passed",
    )
    check = ElementTree.SubElement(rule, "check", status="passed")
    ElementTree.SubElement(check, "context")
    result = parse_report(ElementTree.tostring(root), profile="pdfa-1b", returncode=0, stderr=b"")
    assert result.execution_status == "completed"
    assert result.rules[0].status == "passed"
    assert result.rules[0].locations == ("",)


def test_missing_core_release_cannot_claim_engine_version() -> None:
    raw = internal_report("pass").replace(b'id="core"', b'id="other"')
    result = parse_report(raw, profile="pdfa-1b", returncode=0, stderr=b"")
    assert result.execution_status == "unsupported_engine"
    assert result.engine_version is None


def test_backend_rejects_empty_executable_and_unsupported_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="executable must not be empty"):
        VeraPdfBackend("")
    result = VeraPdfBackend("not-an-executable").validate(
        tmp_path / "missing.pdf", profile="unknown"
    )
    assert result.execution_status == "unsupported_profile"
    assert result.profile_edition is None


def test_execution_permission_error_remains_not_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    from core_pdf_validate import verapdf

    def denied(*args: object, **kwargs: object) -> None:
        raise PermissionError("execution denied")

    monkeypatch.setattr(verapdf.subprocess, "Popen", denied)
    result = VeraPdfBackend().validate(Path("missing.pdf"), profile="pdfa-1b")
    assert result.execution_status == "engine_error"
    assert result.conformance == "not_checked"
    assert "execution denied" in result.diagnostics[0]


def test_non_posix_timeout_kills_process_and_retains_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import subprocess
    from types import SimpleNamespace
    from typing import Self

    from core_pdf_validate import verapdf

    class Process:
        killed = False
        closed = False

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            self.closed = True

        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            if timeout is not None:
                raise subprocess.TimeoutExpired("validator", timeout)
            assert self.killed
            return b"partial report", b"timeout log"

        def kill(self) -> None:
            self.killed = True

    process = Process()
    monkeypatch.setattr(verapdf, "os", SimpleNamespace(name="nt", fspath=os.fspath))
    monkeypatch.setattr(verapdf.subprocess, "Popen", lambda *args, **kwargs: process)
    result = VeraPdfBackend().validate(Path("missing.pdf"), profile="pdfa-1b")
    assert result.execution_status == "timeout"
    assert result.raw_report == b"partial report"
    assert result.stderr == b"timeout log"
    assert process.killed
    assert process.closed


def test_declaration_reader_failures_produce_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    import core_pdf

    def unreadable(*args: object) -> None:
        raise OSError("source cannot be read")

    monkeypatch.setattr(core_pdf, "PdfDocument", unreadable)
    result = validate(b"source", profiles="declared")
    assert result.targets == ()
    assert result.results == ()
    assert result.diagnostics == (
        "declaration_read_failed: source cannot be read",
        "no_target_identified",
    )


def test_backend_profile_mismatch_raises_and_removes_snapshot() -> None:
    class WrongBackend(RecordingBackend):
        path: Path

        def validate(self, source: Path, *, profile: str) -> ProfileResult:
            self.path = source
            return ProfileResult("pdfua-2", "completed", "pass", self.name)

    backend = WrongBackend()
    with pytest.raises(ValueError, match="does not match"):
        validate(b"source", profiles="pdfa-1b", backend=backend)
    assert not backend.path.exists()


@pytest.mark.parametrize(
    ("report", "old", "new", "profile", "code", "diagnostic"),
    [
        ("pass", b'compliant="1"', b'compliant="0"', "pdfa-1b", 0, "Batch summary disagrees"),
        ("pass", b'nonCompliant="0"', b'nonCompliant="1"', "pdfa-1b", 0, "Batch summary disagrees"),
        (
            "pass",
            b'failedJobs="0">1<',
            b'failedJobs="0">2<',
            "pdfa-1b",
            0,
            "Batch summary disagrees",
        ),
        ("fail", b'failedRules="8"', b'failedRules="9"', "pdfua-2", 1, "failed rules disagree"),
    ],
)
def test_summary_must_agree_with_document_and_individual_rules(
    report: str,
    old: bytes,
    new: bytes,
    profile: str,
    code: int,
    diagnostic: str,
) -> None:
    original = internal_report(report)
    assert old in original
    raw = original.replace(old, new)
    result = parse_report(raw, profile=profile, returncode=code, stderr=b"")
    assert result.execution_status == "invalid_report"
    assert diagnostic in result.diagnostics[0]


def test_duplicate_declarations_validate_each_profile_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    import core_pdf

    claim = SimpleNamespace(identifier="pdfa-1b", family="PDF/A", source="XMP")
    document = SimpleNamespace(
        standards=SimpleNamespace(diagnostics=(), profile_claims=(claim, claim))
    )
    monkeypatch.setattr(core_pdf, "PdfDocument", lambda source: nullcontext(document))
    backend = RecordingBackend()
    report = validate(b"original", profiles="declared", backend=backend)
    assert report.targets == ("pdfa-1b",)
    assert backend.received == [("pdfa-1b", b"original")]

# SPDX-License-Identifier: AGPL-3.0-only
"""Local veraPDF adapter for the pinned 1.30.2 XML machine-readable report.

CLI contract: https://docs.verapdf.org/cli/validation/
Report contract: veraPDF/veraPDF-library tag v1.30.2, core/src/main/java/
org/verapdf/processor/reports/{ValidationReportImpl,ValidationDetailsImpl}.java.
Exit status contract: veraPDF/veraPDF-apps tag v1.30.2, tests/exit-status.sh.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Literal
from xml.etree import ElementTree

from defusedxml import ElementTree as SafeElementTree
from defusedxml.common import DefusedXmlException

from core_pdf_validate.models import (
    ExecutionStatus,
    ProfileResult,
    ProfileSupport,
    RuleResult,
)

__all__ = ["VeraPdfBackend"]

internal_PROFILES: dict[str, tuple[str, str, str]] = {
    "pdfa-1a": ("1a", "PDF/A-1a validation profile", "ISO 19005-1:2005"),
    "pdfa-1b": ("1b", "PDF/A-1b validation profile", "ISO 19005-1:2005"),
    "pdfa-2a": ("2a", "PDF/A-2a validation profile", "ISO 19005-2:2011"),
    "pdfa-2b": ("2b", "PDF/A-2b validation profile", "ISO 19005-2:2011"),
    "pdfa-2u": ("2u", "PDF/A-2u validation profile", "ISO 19005-2:2011"),
    "pdfa-3a": ("3a", "PDF/A-3a validation profile", "ISO 19005-3:2012"),
    "pdfa-3b": ("3b", "PDF/A-3b validation profile", "ISO 19005-3:2012"),
    "pdfa-3u": ("3u", "PDF/A-3u validation profile", "ISO 19005-3:2012"),
    "pdfa-4": ("4", "PDF/A-4 validation profile", "ISO 19005-4:2020"),
    "pdfa-4e": ("4e", "PDF/A-4e validation profile", "ISO 19005-4:2020"),
    "pdfa-4f": ("4f", "PDF/A-4f validation profile", "ISO 19005-4:2020"),
    "pdfua-1": ("ua1", "PDF/UA-1 validation profile", "ISO 14289-1:2014"),
    "pdfua-2": ("ua2", "PDF/UA-2 + Tagged PDF validation profile", "ISO 14289-2:2024"),
    "wtpdf-1.0-reuse": ("wt1r", "WTPDF 1.0 Reuse validation profile", "WTPDF 1.0:2024"),
    "wtpdf-1.0-accessibility": (
        "wt1a",
        "WTPDF 1.0 Accessibility validation profile",
        "WTPDF 1.0:2024",
    ),
}
internal_VERSION = "1.30.2"
internal_MACHINE_LIMIT = "Only the pinned engine's machine-checkable profile rules were evaluated."


def internal_number(element: ElementTree.Element, name: str) -> int:
    value = element.get(name)
    if value is None or not value.isascii() or not value.isdigit():
        raise ValueError(f"Missing or invalid {element.tag}/@{name}")
    return int(value)


def internal_one(element: ElementTree.Element, path: str) -> ElementTree.Element:
    found = element.findall(path)
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {path}")
    return found[0]


def internal_rules(details: ElementTree.Element) -> tuple[tuple[RuleResult, ...], bool]:
    rules: list[RuleResult] = []
    truncated = False
    for element in details.findall("rule"):
        status = element.get("status")
        if status not in ("passed", "failed"):
            raise ValueError("Unknown rule status")
        rule_status: Literal["passed", "failed"] = "passed" if status == "passed" else "failed"
        specification = element.get("specification")
        clause = element.get("clause")
        test = element.get("testNumber")
        if not specification or not clause or not test:
            raise ValueError("Missing rule reference")
        checks = element.findall("check")
        locations = tuple(
            context.text or "" for check in checks for context in check.findall("context")
        )
        for check in checks:
            if check.get("status") not in ("passed", "failed"):
                raise ValueError("Unknown check status")
        if status == "failed":
            failed_checks = internal_number(element, "failedChecks")
            displayed = sum(check.get("status") == "failed" for check in checks)
            if failed_checks == 0 or displayed > failed_checks:
                raise ValueError("Inconsistent failed rule counts")
            truncated |= displayed < failed_checks
        rules.append(
            RuleResult(
                specification,
                clause,
                test,
                rule_status,
                element.findtext("description", default=""),
                locations,
            )
        )
    return tuple(rules), truncated


def internal_parse_report(
    raw: bytes,
    *,
    profile: str,
    returncode: int,
    stderr: bytes,
) -> ProfileResult:
    version: str | None = None

    def failed(status: ExecutionStatus, diagnostic: str) -> ProfileResult:
        return ProfileResult(
            profile,
            status,
            "not_checked",
            "veraPDF",
            version,
            diagnostics=(diagnostic,),
            raw_report=raw,
            stderr=stderr,
        )

    try:
        root = SafeElementTree.fromstring(raw, forbid_dtd=True)
        if root.tag != "report":
            raise ValueError("Unknown XML report root or namespace")
        releases = internal_one(root, "buildInformation").findall("releaseDetails")
        core_versions = [item.get("version") for item in releases if item.get("id") == "core"]
        model_versions = [
            item.get("version") for item in releases if item.get("id") == "validation-model"
        ]
        if len(core_versions) == 1:
            version = core_versions[0]
        if core_versions != [internal_VERSION] or model_versions != [internal_VERSION]:
            return failed("unsupported_engine", "Expected veraPDF core and model version 1.30.2")
        if returncode not in (0, 1):
            return failed("engine_error", f"veraPDF exited with status {returncode}")
        job = internal_one(root, "jobs/job")
        report = internal_one(job, "validationReport")
        if report.get("profileName", "").casefold() != internal_PROFILES[profile][1].casefold():
            raise ValueError("Report profile does not match the explicit target")
        if report.get("jobEndStatus") != "normal":
            return failed("incomplete", "Validation did not finish normally")
        compliant = report.get("isCompliant")
        if compliant not in ("true", "false"):
            raise ValueError("Missing or invalid isCompliant flag")
        details = internal_one(report, "details")
        passed_rules = internal_number(details, "passedRules")
        failed_rules = internal_number(details, "failedRules")
        passed_checks = internal_number(details, "passedChecks")
        failed_checks = internal_number(details, "failedChecks")
        if passed_rules + failed_rules == 0 or passed_checks + failed_checks == 0:
            return failed("incomplete", "Report contains no evaluated rules or checks")
        passes = compliant == "true"
        if passes != (failed_rules == 0) or passes != (failed_checks == 0):
            raise ValueError("Inconsistent conformance and failure counts")
        if returncode != (0 if passes else 1):
            raise ValueError("Exit status contradicts report conformance")
        summary = internal_one(root, "batchSummary")
        if internal_number(summary, "totalJobs") != 1:
            raise ValueError("Report is not for one source document")
        if any(
            internal_number(summary, field) != 0
            for field in (
                "failedToParse",
                "encrypted",
                "outOfMemory",
                "veraExceptions",
            )
        ):
            return failed("incomplete", "Batch summary reports incomplete processing")
        validation_summary = internal_one(summary, "validationReports")
        if internal_number(validation_summary, "failedJobs") != 0:
            return failed("incomplete", "Batch summary reports failed validation jobs")
        if (
            internal_number(validation_summary, "compliant") != int(passes)
            or internal_number(validation_summary, "nonCompliant") != int(not passes)
            or (validation_summary.text or "").strip() != "1"
        ):
            raise ValueError("Batch summary disagrees with document result")
        rules, truncated = internal_rules(details)
        if sum(rule.status == "failed" for rule in rules) != failed_rules:
            raise ValueError("Reported failed rules disagree with summary")
        limitations = [internal_MACHINE_LIMIT]
        if profile.startswith(("pdfua-", "wtpdf-")):
            limitations.append(
                "Human review is still required for semantic correctness and accessibility."
            )
        if truncated:
            limitations.append("The engine omitted some individual failed-check locations.")
        return ProfileResult(
            profile,
            "completed",
            "pass" if passes else "fail",
            "veraPDF",
            version,
            rules=rules,
            raw_report=raw,
            stderr=stderr,
            limitations=tuple(limitations),
        )
    except (ElementTree.ParseError, DefusedXmlException, ValueError) as error:
        return failed("invalid_report", str(error))


@dataclass(frozen=True, slots=True)
class VeraPdfBackend:
    """Run a locally installed veraPDF 1.30.2; never install or fetch an engine.

    ``executable`` is one executable path or command name, not a shell command.
    ``timeout`` limits each target independently. Custom engine versions require
    a separately tested adapter, not a version-check override.
    """

    executable: str | os.PathLike[str] = "verapdf"
    timeout: float = 60.0

    name: ClassVar[str] = "veraPDF"
    supported_engine_versions: ClassVar[tuple[str, ...]] = (internal_VERSION,)
    supported_profiles: ClassVar[tuple[ProfileSupport, ...]] = tuple(
        ProfileSupport(identifier, values[2]) for identifier, values in internal_PROFILES.items()
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number of seconds")
        if not os.fspath(self.executable):
            raise ValueError("executable must not be empty")

    def validate(self, source: Path, *, profile: str) -> ProfileResult:
        result = self.internal_validate(source, profile=profile)
        capability = internal_PROFILES.get(profile)
        return replace(result, profile_edition=capability[2] if capability is not None else None)

    def internal_validate(self, source: Path, *, profile: str) -> ProfileResult:
        if profile not in internal_PROFILES:
            return ProfileResult(
                profile,
                "unsupported_profile",
                "not_checked",
                self.name,
                diagnostics=(f"Unsupported validation target: {profile}",),
            )
        command = [
            os.fspath(self.executable),
            "--format",
            "xml",
            "--flavour",
            internal_PROFILES[profile][0],
            "--maxfailures",
            "-1",
            str(source.resolve()),
        ]
        try:
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                start_new_session=os.name == "posix",
            ) as process:
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    # veraPDF's launcher can spawn Java. On POSIX terminate the
                    # whole new process group, including children holding pipes.
                    if os.name == "posix":
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    stdout, stderr = process.communicate()
                    return ProfileResult(
                        profile,
                        "timeout",
                        "not_checked",
                        self.name,
                        diagnostics=(f"veraPDF exceeded {self.timeout:g} seconds",),
                        raw_report=stdout,
                        stderr=stderr,
                    )
                return internal_parse_report(
                    stdout,
                    profile=profile,
                    returncode=process.returncode,
                    stderr=stderr,
                )
        except FileNotFoundError as error:
            return ProfileResult(
                profile,
                "engine_unavailable",
                "not_checked",
                self.name,
                diagnostics=(f"veraPDF executable not found: {error.filename}",),
            )
        except OSError as error:
            return ProfileResult(
                profile,
                "engine_error",
                "not_checked",
                self.name,
                diagnostics=(f"Could not run veraPDF: {error}",),
            )

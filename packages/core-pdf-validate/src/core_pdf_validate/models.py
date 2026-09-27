# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from core_records import GeneratedRecord

__all__ = [
    "Conformance",
    "ExecutionStatus",
    "ProfileResult",
    "ProfileSupport",
    "RuleResult",
    "ValidationBackend",
    "ValidationReport",
]

type Conformance = Literal["pass", "fail", "not_checked"]
type ExecutionStatus = Literal[
    "completed",
    "unsupported_profile",
    "engine_unavailable",
    "timeout",
    "engine_error",
    "unsupported_engine",
    "invalid_report",
    "incomplete",
]


class ProfileSupport(GeneratedRecord):
    identifier: str
    edition: str


class RuleResult(GeneratedRecord):
    specification: str
    clause: str
    test_number: str
    status: Literal["passed", "failed"]
    description: str
    locations: tuple[str, ...] = ()


class ProfileResult(GeneratedRecord):
    profile: str
    execution_status: ExecutionStatus
    conformance: Conformance
    engine: str
    engine_version: str | None = None
    rules: tuple[RuleResult, ...] = ()
    diagnostics: tuple[str, ...] = ()
    raw_report: bytes = b""
    stderr: bytes = b""
    limitations: tuple[str, ...] = ()
    profile_edition: str | None = None


class ValidationReport(GeneratedRecord):
    source_sha256: str
    targets: tuple[str, ...]
    results: tuple[ProfileResult, ...]
    diagnostics: tuple[str, ...] = ()


class ValidationBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def supported_profiles(self) -> tuple[ProfileSupport, ...]: ...

    def validate(self, source: Path, *, profile: str) -> ProfileResult: ...

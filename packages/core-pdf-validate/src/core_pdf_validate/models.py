# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable validator capabilities and machine-check reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

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


@dataclass(frozen=True, slots=True)
class ProfileSupport:
    """One exact target and its standards edition understood by an adapter."""

    identifier: str
    edition: str


@dataclass(frozen=True, slots=True)
class RuleResult:
    """A rule summary; locations retain the validator's original context syntax."""

    specification: str
    clause: str
    test_number: str
    status: Literal["passed", "failed"]
    description: str
    locations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Execution and machine conformance for a single requested profile.

    ``pass`` means the engine's machine checks passed, not that requirements
    outside the engine's coverage or requiring human review were satisfied.
    """

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


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Independent results for an immutable snapshot of the original input."""

    source_sha256: str
    targets: tuple[str, ...]
    results: tuple[ProfileResult, ...]
    diagnostics: tuple[str, ...] = ()


class ValidationBackend(Protocol):
    """Local adapters advertise exact capabilities and never repair the source."""

    @property
    def name(self) -> str: ...

    @property
    def supported_profiles(self) -> tuple[ProfileSupport, ...]: ...

    def validate(self, source: Path, *, profile: str) -> ProfileResult: ...

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal, Protocol

from core_records import Record, frozen_setattr

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


class ProfileSupport(Record):
    __slots__ = ("identifier", "edition")

    identifier: str
    edition: str

    __fields__: ClassVar[tuple[str, ...]] = ("identifier", "edition")
    __match_args__ = ("identifier", "edition")

    def __init__(self, identifier: str, edition: str) -> None:
        frozen_setattr(self, "identifier", identifier)
        frozen_setattr(self, "edition", edition)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.identifier == other.identifier and self.edition == other.edition

    def __hash__(self) -> int:
        return hash((self.identifier, self.edition))


class RuleResult(Record):
    __slots__ = ("specification", "clause", "test_number", "status", "description", "locations")

    specification: str
    clause: str
    test_number: str
    status: Literal["passed", "failed"]
    description: str
    locations: tuple[str, ...]

    __fields__: ClassVar[tuple[str, ...]] = (
        "specification",
        "clause",
        "test_number",
        "status",
        "description",
        "locations",
    )
    __match_args__ = (
        "specification",
        "clause",
        "test_number",
        "status",
        "description",
        "locations",
    )

    def __init__(
        self,
        specification: str,
        clause: str,
        test_number: str,
        status: Literal["passed", "failed"],
        description: str,
        locations: tuple[str, ...] = (),
    ) -> None:
        frozen_setattr(self, "specification", specification)
        frozen_setattr(self, "clause", clause)
        frozen_setattr(self, "test_number", test_number)
        frozen_setattr(self, "status", status)
        frozen_setattr(self, "description", description)
        frozen_setattr(self, "locations", locations)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.specification == other.specification
            and self.clause == other.clause
            and self.test_number == other.test_number
            and self.status == other.status
            and self.description == other.description
            and self.locations == other.locations
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.specification,
                self.clause,
                self.test_number,
                self.status,
                self.description,
                self.locations,
            )
        )


class ProfileResult(Record):
    __slots__ = (
        "profile",
        "execution_status",
        "conformance",
        "engine",
        "engine_version",
        "rules",
        "diagnostics",
        "raw_report",
        "stderr",
        "limitations",
        "profile_edition",
    )

    profile: str
    execution_status: ExecutionStatus
    conformance: Conformance
    engine: str
    engine_version: str | None
    rules: tuple[RuleResult, ...]
    diagnostics: tuple[str, ...]
    raw_report: bytes
    stderr: bytes
    limitations: tuple[str, ...]
    profile_edition: str | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "profile",
        "execution_status",
        "conformance",
        "engine",
        "engine_version",
        "rules",
        "diagnostics",
        "raw_report",
        "stderr",
        "limitations",
        "profile_edition",
    )
    __match_args__ = (
        "profile",
        "execution_status",
        "conformance",
        "engine",
        "engine_version",
        "rules",
        "diagnostics",
        "raw_report",
        "stderr",
        "limitations",
        "profile_edition",
    )

    def __init__(
        self,
        profile: str,
        execution_status: ExecutionStatus,
        conformance: Conformance,
        engine: str,
        engine_version: str | None = None,
        rules: tuple[RuleResult, ...] = (),
        diagnostics: tuple[str, ...] = (),
        raw_report: bytes = b"",
        stderr: bytes = b"",
        limitations: tuple[str, ...] = (),
        profile_edition: str | None = None,
    ) -> None:
        frozen_setattr(self, "profile", profile)
        frozen_setattr(self, "execution_status", execution_status)
        frozen_setattr(self, "conformance", conformance)
        frozen_setattr(self, "engine", engine)
        frozen_setattr(self, "engine_version", engine_version)
        frozen_setattr(self, "rules", rules)
        frozen_setattr(self, "diagnostics", diagnostics)
        frozen_setattr(self, "raw_report", raw_report)
        frozen_setattr(self, "stderr", stderr)
        frozen_setattr(self, "limitations", limitations)
        frozen_setattr(self, "profile_edition", profile_edition)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.profile == other.profile
            and self.execution_status == other.execution_status
            and self.conformance == other.conformance
            and self.engine == other.engine
            and self.engine_version == other.engine_version
            and self.rules == other.rules
            and self.diagnostics == other.diagnostics
            and self.raw_report == other.raw_report
            and self.stderr == other.stderr
            and self.limitations == other.limitations
            and self.profile_edition == other.profile_edition
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.profile,
                self.execution_status,
                self.conformance,
                self.engine,
                self.engine_version,
                self.rules,
                self.diagnostics,
                self.raw_report,
                self.stderr,
                self.limitations,
                self.profile_edition,
            )
        )


class ValidationReport(Record):
    __slots__ = ("source_sha256", "targets", "results", "diagnostics")

    source_sha256: str
    targets: tuple[str, ...]
    results: tuple[ProfileResult, ...]
    diagnostics: tuple[str, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("source_sha256", "targets", "results", "diagnostics")
    __match_args__ = ("source_sha256", "targets", "results", "diagnostics")

    def __init__(
        self,
        source_sha256: str,
        targets: tuple[str, ...],
        results: tuple[ProfileResult, ...],
        diagnostics: tuple[str, ...] = (),
    ) -> None:
        frozen_setattr(self, "source_sha256", source_sha256)
        frozen_setattr(self, "targets", targets)
        frozen_setattr(self, "results", results)
        frozen_setattr(self, "diagnostics", diagnostics)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.source_sha256 == other.source_sha256
            and self.targets == other.targets
            and self.results == other.results
            and self.diagnostics == other.diagnostics
        )

    def __hash__(self) -> int:
        return hash((self.source_sha256, self.targets, self.results, self.diagnostics))


class ValidationBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def supported_profiles(self) -> tuple[ProfileSupport, ...]: ...

    def validate(self, source: Path, *, profile: str) -> ProfileResult: ...

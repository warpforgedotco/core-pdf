# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, NoReturn, Protocol, Self

internal_frozen_setattr = object.__setattr__


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


class ProfileSupport:
    __slots__ = ("identifier", "edition")

    identifier: str
    edition: str

    __fields__: ClassVar[tuple[str, ...]] = ("identifier", "edition")
    __match_args__ = ("identifier", "edition")

    def __init__(self, identifier: str, edition: str) -> None:
        internal_frozen_setattr(self, "identifier", identifier)
        internal_frozen_setattr(self, "edition", edition)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"identifier={self.identifier!r}, "
            f"edition={self.edition!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.identifier == other.identifier and self.edition == other.edition

    def __hash__(self) -> int:
        return hash((self.identifier, self.edition))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        identifier = changes.pop("identifier", self.identifier)
        edition = changes.pop("edition", self.edition)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(identifier, edition)


class RuleResult:
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
        internal_frozen_setattr(self, "specification", specification)
        internal_frozen_setattr(self, "clause", clause)
        internal_frozen_setattr(self, "test_number", test_number)
        internal_frozen_setattr(self, "status", status)
        internal_frozen_setattr(self, "description", description)
        internal_frozen_setattr(self, "locations", locations)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"specification={self.specification!r}, "
            f"clause={self.clause!r}, "
            f"test_number={self.test_number!r}, "
            f"status={self.status!r}, "
            f"description={self.description!r}, "
            f"locations={self.locations!r}"
            ")"
        )

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        specification = changes.pop("specification", self.specification)
        clause = changes.pop("clause", self.clause)
        test_number = changes.pop("test_number", self.test_number)
        status = changes.pop("status", self.status)
        description = changes.pop("description", self.description)
        locations = changes.pop("locations", self.locations)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(specification, clause, test_number, status, description, locations)


class ProfileResult:
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
        internal_frozen_setattr(self, "profile", profile)
        internal_frozen_setattr(self, "execution_status", execution_status)
        internal_frozen_setattr(self, "conformance", conformance)
        internal_frozen_setattr(self, "engine", engine)
        internal_frozen_setattr(self, "engine_version", engine_version)
        internal_frozen_setattr(self, "rules", rules)
        internal_frozen_setattr(self, "diagnostics", diagnostics)
        internal_frozen_setattr(self, "raw_report", raw_report)
        internal_frozen_setattr(self, "stderr", stderr)
        internal_frozen_setattr(self, "limitations", limitations)
        internal_frozen_setattr(self, "profile_edition", profile_edition)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"profile={self.profile!r}, "
            f"execution_status={self.execution_status!r}, "
            f"conformance={self.conformance!r}, "
            f"engine={self.engine!r}, "
            f"engine_version={self.engine_version!r}, "
            f"rules={self.rules!r}, "
            f"diagnostics={self.diagnostics!r}, "
            f"raw_report={self.raw_report!r}, "
            f"stderr={self.stderr!r}, "
            f"limitations={self.limitations!r}, "
            f"profile_edition={self.profile_edition!r}"
            ")"
        )

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        profile = changes.pop("profile", self.profile)
        execution_status = changes.pop("execution_status", self.execution_status)
        conformance = changes.pop("conformance", self.conformance)
        engine = changes.pop("engine", self.engine)
        engine_version = changes.pop("engine_version", self.engine_version)
        rules = changes.pop("rules", self.rules)
        diagnostics = changes.pop("diagnostics", self.diagnostics)
        raw_report = changes.pop("raw_report", self.raw_report)
        stderr = changes.pop("stderr", self.stderr)
        limitations = changes.pop("limitations", self.limitations)
        profile_edition = changes.pop("profile_edition", self.profile_edition)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            profile,
            execution_status,
            conformance,
            engine,
            engine_version,
            rules,
            diagnostics,
            raw_report,
            stderr,
            limitations,
            profile_edition,
        )


class ValidationReport:
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
        internal_frozen_setattr(self, "source_sha256", source_sha256)
        internal_frozen_setattr(self, "targets", targets)
        internal_frozen_setattr(self, "results", results)
        internal_frozen_setattr(self, "diagnostics", diagnostics)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"source_sha256={self.source_sha256!r}, "
            f"targets={self.targets!r}, "
            f"results={self.results!r}, "
            f"diagnostics={self.diagnostics!r}"
            ")"
        )

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        source_sha256 = changes.pop("source_sha256", self.source_sha256)
        targets = changes.pop("targets", self.targets)
        results = changes.pop("results", self.results)
        diagnostics = changes.pop("diagnostics", self.diagnostics)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(source_sha256, targets, results, diagnostics)


class ValidationBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def supported_profiles(self) -> tuple[ProfileSupport, ...]: ...

    def validate(self, source: Path, *, profile: str) -> ProfileResult: ...

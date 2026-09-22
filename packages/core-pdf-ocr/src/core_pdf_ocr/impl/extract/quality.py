# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections import Counter
from typing import Any, ClassVar, NamedTuple, NoReturn, Self

import numpy

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.extract.quality import (
    TextAnalysis as TextAnalysis,
)
from core_pdf.impl.extract.quality import (
    internal_analyze_text as internal_analyze_text,
)

internal_frozen_setattr = object.__setattr__


class internal_CandidateMetrics:
    __slots__ = (
        "characters",
        "alphanumeric_characters",
        "tokens",
        "line_count",
        "mean_confidence",
        "symbol_ratio",
        "utility",
        "median_text_height",
    )

    characters: int
    alphanumeric_characters: int
    tokens: int
    line_count: int
    mean_confidence: float
    symbol_ratio: float
    utility: float
    median_text_height: float

    __fields__: ClassVar[tuple[str, ...]] = (
        "characters",
        "alphanumeric_characters",
        "tokens",
        "line_count",
        "mean_confidence",
        "symbol_ratio",
        "utility",
        "median_text_height",
    )
    __match_args__ = (
        "characters",
        "alphanumeric_characters",
        "tokens",
        "line_count",
        "mean_confidence",
        "symbol_ratio",
        "utility",
        "median_text_height",
    )

    def __init__(
        self,
        characters: int,
        alphanumeric_characters: int,
        tokens: int,
        line_count: int,
        mean_confidence: float,
        symbol_ratio: float,
        utility: float,
        median_text_height: float = 0.0,
    ) -> None:
        internal_frozen_setattr(self, "characters", characters)
        internal_frozen_setattr(self, "alphanumeric_characters", alphanumeric_characters)
        internal_frozen_setattr(self, "tokens", tokens)
        internal_frozen_setattr(self, "line_count", line_count)
        internal_frozen_setattr(self, "mean_confidence", mean_confidence)
        internal_frozen_setattr(self, "symbol_ratio", symbol_ratio)
        internal_frozen_setattr(self, "utility", utility)
        internal_frozen_setattr(self, "median_text_height", median_text_height)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"characters={self.characters!r}, "
            f"alphanumeric_characters={self.alphanumeric_characters!r}, "
            f"tokens={self.tokens!r}, "
            f"line_count={self.line_count!r}, "
            f"mean_confidence={self.mean_confidence!r}, "
            f"symbol_ratio={self.symbol_ratio!r}, "
            f"utility={self.utility!r}, "
            f"median_text_height={self.median_text_height!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.characters == other.characters
            and self.alphanumeric_characters == other.alphanumeric_characters
            and self.tokens == other.tokens
            and self.line_count == other.line_count
            and self.mean_confidence == other.mean_confidence
            and self.symbol_ratio == other.symbol_ratio
            and self.utility == other.utility
            and self.median_text_height == other.median_text_height
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.characters,
                self.alphanumeric_characters,
                self.tokens,
                self.line_count,
                self.mean_confidence,
                self.symbol_ratio,
                self.utility,
                self.median_text_height,
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
        characters = changes.pop("characters", self.characters)
        alphanumeric_characters = changes.pop(
            "alphanumeric_characters", self.alphanumeric_characters
        )
        tokens = changes.pop("tokens", self.tokens)
        line_count = changes.pop("line_count", self.line_count)
        mean_confidence = changes.pop("mean_confidence", self.mean_confidence)
        symbol_ratio = changes.pop("symbol_ratio", self.symbol_ratio)
        utility = changes.pop("utility", self.utility)
        median_text_height = changes.pop("median_text_height", self.median_text_height)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            characters,
            alphanumeric_characters,
            tokens,
            line_count,
            mean_confidence,
            symbol_ratio,
            utility,
            median_text_height,
        )


class internal_Candidate:
    __slots__ = ("mode", "observations", "metrics", "symbols", "recognition_status")

    mode: int
    observations: ObservationBatch
    metrics: internal_CandidateMetrics
    symbols: ObservationBatch
    recognition_status: str

    __fields__: ClassVar[tuple[str, ...]] = (
        "mode",
        "observations",
        "metrics",
        "symbols",
        "recognition_status",
    )
    __match_args__ = ("mode", "observations", "metrics", "symbols", "recognition_status")

    def __init__(
        self,
        mode: int,
        observations: ObservationBatch,
        metrics: internal_CandidateMetrics,
        symbols: ObservationBatch | None = None,
        recognition_status: str = "not-run",
    ) -> None:
        internal_frozen_setattr(self, "mode", mode)
        internal_frozen_setattr(self, "observations", observations)
        internal_frozen_setattr(self, "metrics", metrics)
        internal_frozen_setattr(
            self, "symbols", ObservationBatch.empty() if symbols is None else symbols
        )
        internal_frozen_setattr(self, "recognition_status", recognition_status)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"mode={self.mode!r}, "
            f"observations={self.observations!r}, "
            f"metrics={self.metrics!r}, "
            f"symbols={self.symbols!r}, "
            f"recognition_status={self.recognition_status!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.mode == other.mode
            and self.observations == other.observations
            and self.metrics == other.metrics
            and self.symbols == other.symbols
            and self.recognition_status == other.recognition_status
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.mode,
                self.observations,
                self.metrics,
                self.symbols,
                self.recognition_status,
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
        mode = changes.pop("mode", self.mode)
        observations = changes.pop("observations", self.observations)
        metrics = changes.pop("metrics", self.metrics)
        symbols = changes.pop("symbols", self.symbols)
        recognition_status = changes.pop("recognition_status", self.recognition_status)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(mode, observations, metrics, symbols, recognition_status)


class internal_TextUtility(NamedTuple):
    nonspace: int
    alphanumeric: int
    utility: float


def internal_text_utility_stats(text: str, confidence: float) -> internal_TextUtility:
    stripped = "".join(text.split())
    nonspace = len(stripped)
    if not nonspace:
        return internal_TextUtility(0, 0, 0.0)
    alphanumeric = sum(map(str.isalnum, stripped))
    counts = Counter(map(str.casefold, stripped))
    symbols = nonspace - alphanumeric
    symbol_credit = min(symbols, max(2.0, alphanumeric * 0.5)) * 0.30
    confidence_factor = 0.25 + 0.75 * min(100.0, max(0.0, confidence)) / 100.0
    repetition_penalty = 1.0
    if nonspace >= 6:
        dominant_ratio = max(counts.values()) / nonspace
        if dominant_ratio > 0.60:
            repetition_penalty = max(0.20, 1.0 - (dominant_ratio - 0.60) * 2.0)
    utility = (alphanumeric + symbol_credit) * confidence_factor * repetition_penalty
    return internal_TextUtility(nonspace, alphanumeric, utility)


def internal_candidate(
    mode: int,
    observations: ObservationBatch,
    *,
    symbols: ObservationBatch | None = None,
    recognition_status: str = "not-run",
    median_text_height: float = 0.0,
) -> internal_Candidate:
    confidences = observations.confidence
    finite_confidences = confidences[numpy.isfinite(confidences)]
    mean_confidence = float(numpy.mean(finite_confidences)) if len(finite_confidences) else 0.0
    characters = max(0, len(observations) - 1)
    nonspace_characters = 0
    alphanumeric = 0
    tokens = 0
    utility = 0.0
    for text, confidence in zip(
        observations.text,
        observations.confidence,
        strict=True,
    ):
        characters += len(text)
        tokens += len(text.split())
        nonspace, text_alphanumeric, text_utility = internal_text_utility_stats(
            text,
            float(confidence),
        )
        nonspace_characters += nonspace
        alphanumeric += text_alphanumeric
        utility += text_utility
    symbol_characters = nonspace_characters - alphanumeric
    return internal_Candidate(
        mode,
        observations,
        internal_CandidateMetrics(
            characters=characters,
            alphanumeric_characters=alphanumeric,
            tokens=tokens,
            line_count=len(observations),
            mean_confidence=mean_confidence,
            symbol_ratio=symbol_characters / max(1, nonspace_characters),
            utility=utility,
            median_text_height=median_text_height,
        ),
        symbols=symbols if symbols is not None else ObservationBatch.empty(),
        recognition_status=recognition_status,
    )

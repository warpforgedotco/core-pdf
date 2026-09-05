# SPDX-License-Identifier: AGPL-3.0-only
"""Recognition candidate quality and scoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy

from core_pdf.impl._impl.extract.contracts import ObservationBatch
from core_pdf.impl._impl.extract.quality import (
    TextAnalysis as TextAnalysis,
)
from core_pdf.impl._impl.extract.quality import (
    internal_analyze_text as internal_analyze_text,
)


@dataclass(frozen=True, slots=True)
class internal_CandidateMetrics:
    """Quality signals used to choose between extraction candidates."""

    characters: int
    alphanumeric_characters: int
    tokens: int
    line_count: int
    mean_confidence: float
    symbol_ratio: float
    utility: float
    median_text_height: float = 0.0


@dataclass(frozen=True, slots=True)
class internal_Candidate:
    mode: int
    observations: ObservationBatch
    metrics: internal_CandidateMetrics
    symbols: ObservationBatch = field(default_factory=ObservationBatch.empty)
    recognition_status: str = "not-run"


class internal_TextUtility(NamedTuple):
    nonspace: int
    alphanumeric: int
    utility: float


def internal_text_utility_stats(text: str, confidence: float) -> internal_TextUtility:
    """Return non-space count, alphanumeric count, and utility in one character scan."""
    # str.split() removes exactly the isspace() characters, and Counter over
    # a map counts in C; per-character casefold keeps expanding folds (one
    # count under a multi-character key) identical to the previous loop.
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

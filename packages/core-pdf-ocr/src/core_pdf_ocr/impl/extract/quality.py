# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

import numpy

from core_pdf.impl.extract_capture import (
    TextAnalysis as TextAnalysis,
)
from core_pdf.impl.extract_capture import (
    analyze_text as analyze_text,
)
from core_pdf.impl.extract_contracts import ObservationBatch
from core_pdf.impl.types import GeneratedRecord, frozen_setattr


class CandidateMetrics(GeneratedRecord):
    characters: int
    alphanumeric_characters: int
    tokens: int
    line_count: int
    mean_confidence: float
    symbol_ratio: float
    utility: float
    median_text_height: float = 0.0


class Candidate(GeneratedRecord):
    mode: int
    observations: ObservationBatch
    metrics: CandidateMetrics
    symbols: ObservationBatch
    recognition_status: str

    def __init__(
        self,
        mode: int,
        observations: ObservationBatch,
        metrics: CandidateMetrics,
        symbols: ObservationBatch | None = None,
        recognition_status: str = "not-run",
    ) -> None:
        frozen_setattr(self, "mode", mode)
        frozen_setattr(self, "observations", observations)
        frozen_setattr(self, "metrics", metrics)
        frozen_setattr(self, "symbols", ObservationBatch.empty() if symbols is None else symbols)
        frozen_setattr(self, "recognition_status", recognition_status)


class TextUtility(NamedTuple):
    nonspace: int
    alphanumeric: int
    utility: float


def text_utility_stats(text: str, confidence: float) -> TextUtility:
    stripped = "".join(text.split())
    nonspace = len(stripped)
    if not nonspace:
        return TextUtility(0, 0, 0.0)
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
    return TextUtility(nonspace, alphanumeric, utility)


def make_candidate(
    mode: int,
    observations: ObservationBatch,
    *,
    symbols: ObservationBatch | None = None,
    recognition_status: str = "not-run",
    median_text_height: float = 0.0,
) -> Candidate:
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
        nonspace, text_alphanumeric, text_utility = text_utility_stats(
            text,
            float(confidence),
        )
        nonspace_characters += nonspace
        alphanumeric += text_alphanumeric
        utility += text_utility
    symbol_characters = nonspace_characters - alphanumeric
    return Candidate(
        mode,
        observations,
        CandidateMetrics(
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

from __future__ import annotations

import random

import pytest

from core_pdf.impl.engine.extraction.common import observation_resolver
from core_pdf.impl.engine.extraction.common.observation_resolver import (
    ResolvedTextLine,
    best_observation_geometry_match,
    observation_coverage_ratio,
    observation_geometry_resolution,
    observation_useful_new_token_count,
    resolve_observation_append,
    resolve_text_lines,
)
from core_pdf.impl.engine.extraction.common.page_geometry import PageObservation


def observation(text: str, x0: float) -> PageObservation:
    return PageObservation(
        kind="native_line",
        source="native_text",
        bbox=(x0, 0.0, x0 + 10.0, 10.0),
        text=text,
    )


def test_resolve_text_lines_updates_tokens_without_retokenizing_growing_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = observation_resolver.normalized_text_tokens

    def record_tokens(text: str) -> list[str]:
        calls.append(text)
        return original(text)

    monkeypatch.setattr(observation_resolver, "normalized_text_tokens", record_tokens)
    lines = tuple(
        ResolvedTextLine(text, observation(text, index * 20.0))
        for index, text in enumerate(("Álpha one", "Beta two", "Gamma three"))
    )

    resolved = resolve_text_lines(lines, existing_text="Seed value")

    assert [line.text for line in resolved] == ["Álpha one", "Beta two", "Gamma three"]
    assert all("\n" not in text for text in calls)
    assert max(map(len, calls)) <= len("Gamma three")


def test_useful_token_count_accepts_pre_normalized_tokens_without_changing_duplicates() -> None:
    candidate = observation("new new old", 0.0)

    assert observation_useful_new_token_count(candidate, "OLD") == 2
    assert (
        observation_useful_new_token_count(
            candidate,
            "ignored",
            existing_tokens={"old"},
        )
        == 2
    )


def test_resolution_keeps_first_match_and_accumulates_geometry_coverage() -> None:
    first = observation("First layer", 0.0)
    tied = observation("Tied layer", 0.0)
    duplicate = observation("First layer", 0.0)

    resolution = resolve_observation_append(duplicate, (first, tied))

    assert resolution.action == "skip"
    assert resolution.matched is first
    assert resolution.geometry_score == pytest.approx(1.0)
    assert resolution.coverage_ratio == 1.0


def test_resolution_preserves_center_match_for_nearby_short_box_without_overlap() -> None:
    tall = PageObservation(
        kind="native_line",
        source="native_text",
        bbox=(0.0, 0.0, 10.0, 10.0),
        text="Tall label",
    )
    far = PageObservation(
        kind="native_line",
        source="native_text",
        bbox=(0.0, 30.0, 10.0, 31.0),
        text="Far label",
    )
    short = PageObservation(
        kind="native_line",
        source="native_text",
        bbox=(0.0, 10.1, 10.0, 10.6),
        text="Short label",
    )

    resolution = resolve_observation_append(short, (far, tall))

    assert resolution.matched is tall
    assert resolution.geometry_score > 0.45
    assert resolution.coverage_ratio == 0.0


def test_geometry_broad_phase_matches_exhaustive_resolution() -> None:
    randomizer = random.Random(7)
    for _ in range(200):
        candidate_x0 = randomizer.uniform(-100.0, 100.0)
        candidate_y0 = randomizer.uniform(-100.0, 100.0)
        candidate_width = randomizer.uniform(0.1, 80.0)
        candidate_height = randomizer.uniform(0.1, 80.0)
        candidate = PageObservation(
            kind="native_line",
            source="native_text",
            bbox=(
                candidate_x0,
                candidate_y0,
                candidate_x0 + candidate_width,
                candidate_y0 + candidate_height,
            ),
            text="candidate",
        )
        accepted = tuple(
            PageObservation(
                kind="native_line",
                source="native_text",
                bbox=(x0, y0, x0 + width, y0 + height),
                text=f"accepted {index}",
            )
            for index in range(20)
            for x0, y0, width, height in (
                (
                    randomizer.uniform(-100.0, 100.0),
                    randomizer.uniform(-100.0, 100.0),
                    randomizer.uniform(0.1, 80.0),
                    randomizer.uniform(0.1, 80.0),
                ),
            )
        )

        expected_match, expected_score = best_observation_geometry_match(candidate, accepted)
        expected_coverage = observation_coverage_ratio(candidate, accepted)
        match, score, coverage = observation_geometry_resolution(candidate, accepted)

        assert match is expected_match
        assert score == pytest.approx(expected_score)
        assert coverage == pytest.approx(expected_coverage)

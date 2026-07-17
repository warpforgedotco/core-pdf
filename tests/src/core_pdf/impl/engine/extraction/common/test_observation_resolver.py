from __future__ import annotations

import pytest

from core_pdf.impl.engine.extraction.common import observation_resolver
from core_pdf.impl.engine.extraction.common.observation_resolver import (
    ResolvedTextLine,
    observation_useful_new_token_count,
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

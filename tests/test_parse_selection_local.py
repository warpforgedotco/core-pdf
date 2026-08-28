from __future__ import annotations

from dataclasses import replace
from threading import RLock
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl.engine.parse import (
    CapturedPage,
    ObservationBatch,
    ObservationSource,
    PageEvidence,
)
from core_pdf.impl.engine.parse import pipeline as parse_pipeline


def internal_observations(text: str) -> ObservationBatch:
    return ObservationBatch.from_columns(
        (text,),
        ((0.0, 0.0, 10.0, 10.0),),
        source=ObservationSource.NATIVE,
    )


def internal_base_capture(page: object, decoder: object, text: str) -> CapturedPage:
    program = SimpleNamespace(
        products=SimpleNamespace(
            glyphs=(SimpleNamespace(font_decoder=decoder),),
        )
    )
    return CapturedPage(
        page=page,
        program=cast(Any, program),
        observations=internal_observations(text),
        runs=(),
        drawings=(),
        grid_lines=(),
        inline_images=(),
        evidence=PageEvidence(
            page_area=100.0,
            native_characters=len(text),
            visible_native_characters=len(text),
            suspicious_characters=0,
            image_count=0,
            image_area_ratio=0.0,
            vector_complexity=0,
        ),
    )


def test_selection_local_enrichment_is_history_independent_and_does_not_replace_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = object()
    pages = tuple(
        SimpleNamespace(
            page_number=page_number,
            extraction_cache={},
            internal_page_lock=RLock(),
        )
        for page_number in (1, 2, 3)
    )
    captures = tuple(
        internal_base_capture(page, decoder, f"base-{page.page_number}") for page in pages
    )
    for page, capture in zip(pages, captures, strict=True):
        parse_pipeline.page_extraction(page).replace_capture(capture)

    def enrich_capture(
        page: object,
        program: object,
        *,
        learned_unicode: parse_pipeline.LearnedUnicodeMap,
    ) -> CapturedPage:
        del program
        mapping = learned_unicode[decoder]
        marker = mapping[b"x"]
        page_number = int(getattr(page, "page_number"))
        return replace(
            captures[page_number - 1],
            observations=internal_observations(f"{marker}-{page_number}"),
        )

    monkeypatch.setattr(parse_pipeline, "internal_capture_from_program", enrich_capture)

    def selection_text(
        selected_pages: tuple[object, ...],
        selected_captures: tuple[CapturedPage, ...],
        marker: str,
    ) -> tuple[parse_pipeline.internal_PageExtraction, ...]:
        enrichment = parse_pipeline.internal_FontEnrichment(
            learned_unicode={decoder: {b"x": marker}}
        )
        return parse_pipeline.internal_apply_font_enrichment(
            selected_pages,
            selected_captures,
            enrichment,
        )

    first = selection_text(pages[:2], captures[:2], "first")
    second = selection_text(pages[1:], captures[1:], "second")
    assert first[1].page is second[0].page
    assert first[1] is not second[0]
    assert first[1].capture().observations.text == ("first-2",)
    assert second[0].capture().observations.text == ("second-2",)

    second_before_first = selection_text(pages[1:], captures[1:], "second")
    first_after_second = selection_text(pages[:2], captures[:2], "first")
    assert tuple(item.capture().observations.text[0] for item in first_after_second) == (
        "first-1",
        "first-2",
    )
    assert tuple(item.capture().observations.text[0] for item in second_before_first) == (
        "second-2",
        "second-3",
    )

    assert tuple(
        parse_pipeline.page_extraction(page).capture().observations.text[0] for page in pages
    ) == ("base-1", "base-2", "base-3")
    assert first[1].capture().observations.text == ("first-2",)

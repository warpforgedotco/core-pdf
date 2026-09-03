from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core_pdf.impl.extract import pipeline as parse_pipeline
from core_pdf.impl.extract import selection as parse_selection
from core_pdf.impl.extract.contracts import CapturedPage, ObservationBatch
from tests.helpers.extract_fakes import capture as make_capture
from tests.helpers.extract_fakes import observations, page_evidence
from tests.helpers.ocr_fakes import FakeDocumentPage


def internal_observations(text: str) -> ObservationBatch:
    return observations([(text, (0.0, 0.0, 10.0, 10.0))])


def internal_base_capture(page: object, decoder: object, text: str) -> CapturedPage:
    program = SimpleNamespace(
        products=SimpleNamespace(
            glyphs=(SimpleNamespace(font_decoder=decoder),),
        )
    )
    return make_capture(
        page_evidence(
            page_area=100.0, native_characters=len(text), visible_native_characters=len(text)
        ),
        page=page,
        program=program,
        batch=internal_observations(text),
    )


def test_document_font_seed_selection_requires_repeated_decoder_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoders = tuple(object() for _ in range(3))
    captures = tuple(make_capture(page=object()) for _ in range(4))
    counts = (
        Counter({decoders[0]: 20, decoders[1]: 8}),
        Counter({decoders[0]: 16}),
        Counter({decoders[1]: 30}),
        Counter({decoders[2]: 100}),
    )
    counts_by_capture_id = dict(zip(map(id, captures), counts, strict=True))
    monkeypatch.setattr(
        parse_selection,
        "internal_unknown_decoder_counts",
        lambda capture: counts_by_capture_id[id(capture)],
    )

    assert parse_selection.internal_document_font_seed_indexes(captures) == (2, 0, 1)


def test_document_font_votes_only_resolve_consistent_high_confidence_alignment() -> None:
    decoder = object()
    glyphs = tuple(
        SimpleNamespace(
            visible=True,
            code_bytes=bytes((code,)),
            text="\x00",
            rotation_angle=0,
            ink_bbox=(float(index), 0.0, float(index + 1), 1.0),
            seqno=index,
            font_decoder=decoder,
            unicode_source="fallback_nul",
        )
        for index, code in enumerate((1, 2, 3))
    )
    capture = make_capture(
        program=SimpleNamespace(products=SimpleNamespace(glyphs=glyphs)),
    )

    consistent = observations([("CAT", (0.0, 0.0, 3.0, 1.0))], confidence=95.0)
    conflict = observations([("BAT", (0.0, 0.0, 3.0, 1.0))], confidence=95.0)
    votes: dict[object, dict[bytes, Counter[str]]] = {}
    for batch in (consistent, consistent, conflict):
        parse_selection.internal_merge_font_mapping_votes(
            votes,
            parse_selection.internal_font_mapping_votes(capture, batch),
        )

    assert parse_selection.internal_resolve_document_font_mappings(votes) == {
        decoder: {b"\x02": "A", b"\x03": "T"}
    }


def test_selection_local_enrichment_is_history_independent_and_does_not_replace_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = object()
    pages = tuple(FakeDocumentPage(page_number=page_number) for page_number in (1, 2, 3))
    captures = tuple(
        internal_base_capture(page, decoder, f"base-{page.page_number}") for page in pages
    )
    for page, capture in zip(pages, captures, strict=True):
        parse_pipeline.page_extraction(page).replace_capture(capture)

    def enrich_capture(
        page: object,
        program: object,
        *,
        learned_unicode: parse_selection.LearnedUnicodeMap,
    ) -> CapturedPage:
        del program
        mapping = learned_unicode[decoder]
        marker = mapping[b"x"]
        page_number = int(getattr(page, "page_number"))
        return replace(
            captures[page_number - 1],
            observations=internal_observations(f"{marker}-{page_number}"),
        )

    monkeypatch.setattr(parse_selection, "internal_capture_from_program", enrich_capture)

    def selection_text(
        selected_pages: tuple[object, ...],
        selected_captures: tuple[CapturedPage, ...],
        marker: str,
    ) -> tuple[parse_pipeline.internal_PageExtraction, ...]:
        enrichment = parse_selection.internal_FontEnrichment(
            learned_unicode={decoder: {b"x": marker}}
        )
        return parse_selection.internal_apply_font_enrichment(
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

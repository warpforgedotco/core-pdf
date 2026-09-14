from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl._impl.capture.program import CapturedProgram, PageProgram
from core_pdf.impl._impl.extract.contracts import GlyphEvidence, ObservationBatch
from core_pdf.impl._impl.model.glyphs import GlyphObservation
from core_pdf.impl._impl.runtime.execution import ExtractionScope, internal_ExtractionCancelled
from core_pdf_ocr.impl.extract import selection
from core_pdf_ocr.impl.extract.contracts import (
    PageAnalysis,
    PageRoute,
    RecognitionResult,
    StrokedVectorTextEvidence,
    WorkPlan,
)


@pytest.mark.parametrize("sufficient", [False, True])
def test_cross_page_stroke_learning_uses_richest_seed_and_falls_back_when_needed(
    ocr_capture: PageAnalysis,
    monkeypatch: pytest.MonkeyPatch,
    sufficient: bool,
) -> None:
    captures = tuple(
        replace(
            ocr_capture,
            evidence=replace(
                ocr_capture.evidence,
                stroked_vector_text=StrokedVectorTextEvidence(trusted=True, candidate_paths=count),
            ),
        )
        for count in (10, 20)
    )
    signature = cast(Any, (1, 2))
    recognition = RecognitionResult(
        ObservationBatch.empty(), stroked_vector_alphabet=((signature, "A"),)
    )
    calls: list[int] = []

    def recognize(index: int, context: ExtractionScope) -> RecognitionResult:
        calls.append(index)
        return recognition

    extractions = tuple(
        SimpleNamespace(
            recognition_result=None,
            stroked_profile=object(),
            recognize=lambda context, index=index: recognize(index, context),
        )
        for index in range(2)
    )

    def decode(profile: object, alphabet: dict[Any, str]) -> Any:
        assert alphabet == {signature: "A"}
        return SimpleNamespace(
            observations=(object(),) * 20,
            decoded_candidate_runs=20,
            candidate_run_coverage=0.70 if sufficient else 0.69,
            candidate_glyph_coverage=0.70,
        )

    decoded_recognition = RecognitionResult(ObservationBatch.empty())
    monkeypatch.setattr(selection, "decode_stroked_text_profile_with_alphabet", decode)
    monkeypatch.setattr(
        selection, "internal_document_stroked_recognition", lambda decoded: decoded_recognition
    )
    results = selection.internal_prepare_document_stroked_mappings(
        cast(Any, extractions), captures, ExtractionScope()
    )
    assert calls == ([1] if sufficient else [1, 0])
    assert results[1] is recognition
    assert results[0] is (decoded_recognition if sufficient else recognition)
    assert not selection.internal_prepare_document_stroked_mappings(
        cast(Any, extractions[:1]), captures[:1], ExtractionScope()
    )


def captured_glyphs(base: PageAnalysis, decoder: object, count: int = 16) -> PageAnalysis:
    glyphs = tuple(
        GlyphObservation(
            "?",
            (i % 8 * 10, i // 8 * 20, i % 8 * 10 + 8, i // 8 * 20 + 10),
            (0, 0, 0, 0),
            i,
            code_bytes=bytes([i % 8]),
            font_decoder=decoder,
        )
        for i in range(count)
    )
    return replace(
        base,
        program=PageProgram(CapturedProgram(glyphs=glyphs)),
        evidence=replace(
            base.evidence, glyphs=GlyphEvidence(glyph_count=count, unknown_glyphs=count)
        ),
    )


def test_cross_page_font_learning_votes_and_seed_reuse(ocr_capture: PageAnalysis) -> None:
    decoder = object()
    captures = (
        captured_glyphs(ocr_capture, decoder),
        captured_glyphs(ocr_capture, decoder),
        captured_glyphs(ocr_capture, decoder, 2),
    )
    observations = ObservationBatch.from_columns(
        ("ABCDEFGH", "ABCDEFGH"), ((0, 0, 78, 10), (0, 20, 78, 30)), source=1, confidence=(99, 99)
    )
    recognition = RecognitionResult(observations)
    calls: list[int] = []

    def recognize(index: int, context: ExtractionScope) -> RecognitionResult:
        calls.append(index)
        return recognition

    extractions = tuple(
        SimpleNamespace(recognize=lambda context, index=index: recognize(index, context))
        for index in range(3)
    )
    font = selection.internal_prepare_document_font_mappings(
        cast(Any, extractions), captures, ExtractionScope()
    )
    assert calls == [0, 1]
    assert dict(font.recognition_by_index) == {0: recognition, 1: recognition}
    assert dict(font.learned_unicode[decoder]) == {
        bytes([i]): char for i, char in enumerate("ABCDEFGH")
    }
    with pytest.raises(TypeError):
        cast(Any, font.learned_unicode)[decoder] = {}
    with pytest.raises(TypeError):
        cast(Any, font.learned_unicode[decoder])[b"\x00"] = "X"
    # Another selection must not inherit mappings or recognition results.
    isolated = selection.internal_prepare_document_font_mappings(
        cast(Any, extractions[2:]), captures[2:], ExtractionScope()
    )
    assert not isolated.learned_unicode
    assert not isolated.recognition_by_index
    assert calls == [0, 1]


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({}, None),
        ({"A": 1}, None),
        ({"A": 9, "B": 1}, "A"),
        ({"A": 8, "B": 2}, None),
        ({"A": 2, "B": 2}, None),
    ],
)
def test_font_mapping_requires_repeated_unambiguous_votes(
    counts: dict[str, int], expected: str | None
) -> None:
    decoder = object()
    result = selection.internal_resolve_document_font_mappings({decoder: {b"a": Counter(counts)}})
    assert result.get(decoder, {}).get(b"a") == expected


def test_font_vote_geometry_and_confidence_filters(ocr_capture: PageAnalysis) -> None:
    decoder = object()
    capture = captured_glyphs(ocr_capture, decoder, 8)
    for text, box, confidence in [
        ("ABCDEFGH", (0, 0, 78, 10), 89),
        ("ABCDEFGH", (0, 0, 78, 10), float("nan")),
        ("AB", (0, 0, 18, 10), 99),
        ("ABCDEFG", (0, 0, 78, 10), 99),
        ("ABCDEFGH", (0, 100, 78, 110), 99),
    ]:
        ocr = ObservationBatch.from_columns((text,), (box,), source=1, confidence=(confidence,))
        assert not selection.internal_font_mapping_votes(capture, ocr)
    known = replace(
        capture,
        program=PageProgram(
            CapturedProgram(
                glyphs=tuple(
                    replace(g, text="Z", unicode_source="to_unicode")
                    for g in capture.program.glyphs
                )
            )
        ),
    )
    ocr = ObservationBatch.from_columns(
        ("ABCDEFGH",), ((0, 0, 78, 10),), source=1, confidence=(99,)
    )
    assert not selection.internal_font_mapping_votes(known, ocr)


def test_enrichment_only_rebuilds_changed_pages_and_reuses_seed_recognition(
    ocr_capture: PageAnalysis, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoder = object()
    captures = (
        captured_glyphs(ocr_capture, decoder),
        captured_glyphs(ocr_capture, decoder, 2),
        ocr_capture,
    )
    recognition = RecognitionResult(ObservationBatch.empty())
    bases = tuple(
        SimpleNamespace(
            page=object(),
            capture=capture,
            plan=WorkPlan(PageRoute.NATIVE),
            internal_structure=None,
            internal_hidden_layers=frozenset(),
            internal_stroked_profile=None,
        )
        for capture in captures
    )
    rebuilt: list[dict[str, Any]] = []
    overlays: list[Any] = []

    def build(page: object, **kwargs: Any) -> Any:
        rebuilt.append(kwargs)
        return SimpleNamespace(page=page, **kwargs)

    def recapture(page: object, program: PageProgram, **kwargs: Any) -> PageAnalysis:
        overlays.append(kwargs["learned_unicode"])
        assert program is captures[1].program
        return captures[1]

    monkeypatch.setattr(selection, "internal_PageExtraction", build)
    monkeypatch.setattr(selection, "internal_capture_from_program", recapture)
    font = selection.internal_FontEnrichment(
        seed_indexes=(0,),
        learned_unicode={decoder: {b"a": "A"}},
        recognition_by_index={0: recognition},
    )
    result = selection.internal_apply_font_enrichment(cast(Any, bases), captures, font)
    assert result[2] is bases[2]
    assert rebuilt[0]["recognition"] is recognition
    assert "recognition" not in rebuilt[1]
    assert "plan" not in rebuilt[1]
    assert overlays == [font.learned_unicode]
    assert all(base.capture is capture for base, capture in zip(bases, captures, strict=True))
    assert selection.internal_apply_stroked_enrichment(cast(Any, bases), {}) is bases
    stroked = selection.internal_apply_stroked_enrichment(cast(Any, bases), {1: recognition})
    assert stroked[0] is bases[0]
    assert stroked[2] is bases[2]
    assert rebuilt[-1]["recognition"] is recognition


def test_selection_cancellation_stops_before_second_seed(ocr_capture: PageAnalysis) -> None:
    decoder = object()
    captures = (captured_glyphs(ocr_capture, decoder),) * 2
    calls: list[int] = []

    def recognize(context: ExtractionScope) -> RecognitionResult:
        calls.append(1)
        return RecognitionResult(ObservationBatch.empty())

    extractions = (SimpleNamespace(recognize=recognize),) * 2
    with pytest.raises(internal_ExtractionCancelled):
        selection.internal_prepare_document_font_mappings(
            cast(Any, extractions), captures, ExtractionScope(cancelled=lambda: bool(calls))
        )
    assert calls == [1]
    with pytest.raises(internal_ExtractionCancelled):
        selection.internal_capture_document_pages(
            cast(Any, extractions), ExtractionScope(cancelled=lambda: True)
        )


def test_stroked_alphabet_conflicts_remain_excluded() -> None:
    alphabet: dict[Any, str] = {}
    ambiguous: set[Any] = set()
    signature = cast(Any, (1, 2))
    selection.internal_merge_document_stroked_alphabet(alphabet, ambiguous, ((signature, "A"),))
    selection.internal_merge_document_stroked_alphabet(alphabet, ambiguous, ((signature, "B"),))
    selection.internal_merge_document_stroked_alphabet(alphabet, ambiguous, ((signature, "A"),))
    assert alphabet == {}
    assert ambiguous == {signature}


def test_cancelled_stroke_enrichment_stops_even_with_cached_recognition(
    ocr_capture: PageAnalysis,
) -> None:
    capture = replace(
        ocr_capture,
        evidence=replace(
            ocr_capture.evidence, stroked_vector_text=StrokedVectorTextEvidence(trusted=True)
        ),
    )
    recognition = RecognitionResult(ObservationBatch.empty())
    extractions = (SimpleNamespace(recognition_result=recognition),) * 2
    with pytest.raises(internal_ExtractionCancelled):
        selection.internal_prepare_document_stroked_mappings(
            cast(Any, extractions), (capture, capture), ExtractionScope(cancelled=lambda: True)
        )

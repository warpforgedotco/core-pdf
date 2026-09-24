from collections import Counter
from copy import replace
from types import SimpleNamespace
from typing import Any

import pytest

from core_pdf.impl.capture.program import CapturedProgram, PageProgram
from core_pdf.impl.extract.contracts import GlyphEvidence, ObservationBatch
from core_pdf.impl.model.glyphs import GlyphObservation
from core_pdf.impl.runtime.execution import ExtractionCancelled, ExtractionScope
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
    signature = (1, 2)
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
        selection, "document_stroked_recognition", lambda decoded: decoded_recognition
    )
    results = selection.prepare_document_stroked_mappings(extractions, captures, ExtractionScope())  # ty: ignore[invalid-argument-type]
    assert calls == ([1] if sufficient else [1, 0])
    assert results[1] is recognition
    assert results[0] is (decoded_recognition if sufficient else recognition)
    assert not selection.prepare_document_stroked_mappings(
        extractions[:1],  # ty: ignore[invalid-argument-type]
        captures[:1],
        ExtractionScope(),
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
    font = selection.prepare_document_font_mappings(extractions, captures, ExtractionScope())  # ty: ignore[invalid-argument-type]
    assert calls == [0, 1]
    assert dict(font.recognition_by_index) == {0: recognition, 1: recognition}
    assert dict(font.learned_unicode[decoder]) == {
        bytes([i]): char for i, char in enumerate("ABCDEFGH")
    }
    with pytest.raises(TypeError):
        font.learned_unicode[decoder] = {}  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        font.learned_unicode[decoder][b"\x00"] = "X"  # ty: ignore[invalid-assignment]
    isolated = selection.prepare_document_font_mappings(
        extractions[2:],  # ty: ignore[invalid-argument-type]
        captures[2:],
        ExtractionScope(),
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
    result = selection.resolve_document_font_mappings({decoder: {b"a": Counter(counts)}})
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
        assert not selection.font_mapping_votes(capture, ocr)
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
    assert not selection.font_mapping_votes(known, ocr)


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
            structure_value=None,
            hidden_layer_names=frozenset(),
            stroked_profile_of=None,
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

    monkeypatch.setattr(selection, "PageExtraction", build)
    monkeypatch.setattr(selection, "capture_from_program", recapture)
    font = selection.FontEnrichment(
        learned_unicode={decoder: {b"a": "A"}},
        recognition_by_index={0: recognition},
    )
    result = selection.apply_font_enrichment(bases, captures, font)  # ty: ignore[invalid-argument-type]
    assert result[2] is bases[2]
    assert rebuilt[0]["recognition"] is recognition
    assert "recognition" not in rebuilt[1]
    assert "plan" not in rebuilt[1]
    assert overlays == [font.learned_unicode]
    assert all(base.capture is capture for base, capture in zip(bases, captures, strict=True))
    assert selection.apply_stroked_enrichment(bases, {}) is bases  # ty: ignore[invalid-argument-type]
    stroked = selection.apply_stroked_enrichment(bases, {1: recognition})
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
    with pytest.raises(ExtractionCancelled):
        selection.prepare_document_font_mappings(
            extractions,  # ty: ignore[invalid-argument-type]
            captures,
            ExtractionScope(cancelled=lambda: bool(calls)),
        )
    assert calls == [1]
    with pytest.raises(ExtractionCancelled):
        selection.capture_document_pages(extractions, ExtractionScope(cancelled=lambda: True))  # ty: ignore[invalid-argument-type]


def test_stroked_alphabet_conflicts_remain_excluded() -> None:
    alphabet: dict[Any, str] = {}
    ambiguous: set[Any] = set()
    signature = (1, 2)
    selection.merge_document_stroked_alphabet(alphabet, ambiguous, ((signature, "A"),))  # ty: ignore[invalid-argument-type]
    selection.merge_document_stroked_alphabet(alphabet, ambiguous, ((signature, "B"),))  # ty: ignore[invalid-argument-type]
    selection.merge_document_stroked_alphabet(alphabet, ambiguous, ((signature, "A"),))  # ty: ignore[invalid-argument-type]
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
    with pytest.raises(ExtractionCancelled):
        selection.prepare_document_stroked_mappings(
            extractions,  # ty: ignore[invalid-argument-type]
            (capture, capture),
            ExtractionScope(cancelled=lambda: True),
        )


def test_unknown_decoder_counts_ignore_unusable_and_already_mapped_glyphs(ocr_capture) -> None:
    decoder = object()
    capture = captured_glyphs(ocr_capture, decoder, 1)
    valid = capture.program.glyphs[0]
    variants = (
        valid,
        replace(valid, font_decoder=None),
        replace(valid, visible=False),
        replace(valid, text=""),
        replace(valid, text=" "),
        replace(valid, code_bytes=b""),
        replace(valid, text="A", unicode_source="to_unicode"),
    )
    capture = replace(capture, program=PageProgram(CapturedProgram(glyphs=variants)))
    assert selection.unknown_decoder_counts(capture) == Counter({decoder: 1})
    assert not selection.unknown_decoder_counts(ocr_capture)


def test_small_decoder_samples_do_not_seed_document_learning(ocr_capture) -> None:
    decoder = object()
    captures = (captured_glyphs(ocr_capture, decoder, 7), captured_glyphs(ocr_capture, decoder, 24))
    assert selection.document_font_seed_indexes(captures) == ()


def test_font_votes_skip_empty_glyphs_and_nonprintable_predictions(ocr_capture) -> None:
    assert not selection.font_mapping_votes(ocr_capture, ObservationBatch.empty())
    decoder = object()
    capture = captured_glyphs(ocr_capture, decoder, 3)
    observations = ObservationBatch.from_columns(
        ("A\x00C",), ((0, 0, 28, 10),), source=1, confidence=(99,)
    )
    votes = selection.font_mapping_votes(capture, observations)
    assert dict(votes[decoder]) == {b"\x00": Counter(A=1), b"\x02": Counter(C=1)}


def test_structural_decode_becomes_positioned_recognition_with_shared_alphabet() -> None:
    from core_pdf_ocr.impl.extract.ocr.strokes import StrokedTextDecode, StrokedTextObservation

    alphabet = ((((1, 2)), "A"),)
    decoded = StrokedTextDecode(
        observations=(StrokedTextObservation("AB", (10, 20, 30, 40), 7, 8),),
        alphabet=alphabet,  # ty: ignore[invalid-argument-type]
    )
    result = selection.document_stroked_recognition(decoded)
    assert result.observations.text == ("AB",)
    assert result.observations.bbox.tolist() == [[10, 20, 30, 40]]
    assert result.observations.sequence.tolist() == [7]
    assert result.stroked_vector_alphabet is alphabet


def test_cached_stroke_results_are_reused_without_recognition(ocr_capture) -> None:
    capture = replace(
        ocr_capture,
        evidence=replace(
            ocr_capture.evidence,
            stroked_vector_text=StrokedVectorTextEvidence(trusted=True, candidate_paths=20),
        ),
    )
    recognition = RecognitionResult(ObservationBatch.empty())
    extractions = tuple(SimpleNamespace(recognition_result=recognition) for _ in range(2))
    result = selection.prepare_document_stroked_mappings(
        extractions,  # ty: ignore[invalid-argument-type]
        (capture, capture),
        ExtractionScope(),
    )
    assert result[0] is result[1] is recognition


@pytest.mark.parametrize("count", [1, 2])
def test_document_selection_captures_only_for_multiple_pages_and_keeps_exact_order(
    ocr_capture, monkeypatch, count
) -> None:
    document = object()
    pages = tuple(object() for _ in range(count))
    captures = []
    assembled = object()

    class Extraction:
        @property
        def capture(self):
            captures.append(self)
            return ocr_capture

    extractions = tuple(Extraction() for _ in pages)
    context = ExtractionScope()

    def prepare(actual_document, actual_pages, factory):
        assert actual_document is document
        assert actual_pages == pages
        return extractions

    def assemble(actual_document, actual_extractions, actual_context):
        assert actual_document is document
        assert actual_extractions == extractions
        assert actual_context is context
        return assembled

    monkeypatch.setattr(selection, "prepare_document_pages", prepare)
    monkeypatch.setattr(selection, "assemble_document", assemble)
    assert selection.extract_document(document, context, pages) is assembled  # ty: ignore[invalid-argument-type]
    if count == 1:
        assert captures == []
    else:
        assert captures[:2] == list(extractions)
        assert all(extraction in extractions for extraction in captures)


def test_stroked_enrichment_replaces_only_selected_page_and_preserves_context(ocr_capture) -> None:
    base = SimpleNamespace(
        page=object(),
        capture=ocr_capture,
        plan=WorkPlan(PageRoute.OCR),
        structure_value=None,
        hidden_layer_names=frozenset(),
        stroked_profile_of=None,
    )
    recognition = RecognitionResult(ObservationBatch.empty())
    result = selection.apply_stroked_enrichment(((base, base)), {1: recognition})  # ty: ignore[invalid-argument-type]
    assert result[0] is base
    assert result[1].capture.program is ocr_capture.program
    assert result[1].capture.observations is ocr_capture.observations
    assert result[1].recognition_result is recognition
    assert result[1].page is base.page
    assert result[1].plan is base.plan

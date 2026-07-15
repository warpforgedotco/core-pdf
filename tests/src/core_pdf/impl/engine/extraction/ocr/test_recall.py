from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl.engine.extraction.common import page_geometry
from core_pdf.impl.engine.extraction.common.observation_resolver import ResolvedTextLine
from core_pdf.impl.engine.extraction.ocr import postprocess
from core_pdf.impl.engine.extraction.ocr.candidates import OcrCandidate, OcrPageTextResult
from core_pdf.impl.engine.extraction.ocr.types import OcrImage, OcrTextResult
from core_pdf.impl.engine.extraction.page_text import mixin
from core_pdf.impl.engine.extraction.page_text.policy import (
    sparse_drawing_schematic_should_yield_to_ocr,
)


def test_sparse_vector_page_triggers_full_page_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        postprocess,
        "page_has_vector_stroke_text_candidates",
        lambda _page: True,
    )
    monkeypatch.setattr(
        postprocess,
        "native_text_layer_looks_reliable_enough",
        lambda *_args, **_kwargs: False,
    )

    assert postprocess.should_ocr_fallback(cast(Any, object()), "A B C")


def test_vector_page_with_substantial_native_title_skips_full_page_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        postprocess,
        "page_has_vector_stroke_text_candidates",
        lambda _page: True,
    )
    monkeypatch.setattr(
        postprocess,
        "native_text_layer_looks_reliable_enough",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        postprocess,
        "layout_geometry_should_trigger_ocr",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        postprocess.ocr_text_analysis,
        "sparse_text_looks_noisy",
        lambda _text: False,
    )
    monkeypatch.setattr(
        postprocess.ocr_page_analysis,
        "has_dominant_page_image",
        lambda _page: False,
    )
    monkeypatch.setattr(
        postprocess.ocr_page_analysis,
        "has_uninterpretable_type3_fonts",
        lambda _page: False,
    )

    native_title = (
        "SPUR REPORT TRANSPORTATION The Future of Transportation Harnessing private "
        "mobility services to support the public good JULY 2020"
    )

    assert not postprocess.should_ocr_fallback(cast(Any, object()), native_title)


def test_sparse_drawing_schematic_yields_to_cleaner_ocr() -> None:
    native_profile = SimpleNamespace(
        drawing_line_count=4_323,
        candidate_schematic_signals=149,
    )
    ocr_profile = SimpleNamespace(confidence=60)

    assert sparse_drawing_schematic_should_yield_to_ocr(
        "A B C D",
        "R1 100k R2 10k GND OUT IN VCC " * 8,
        text_tokens=4,
        ocr_tokens=64,
        native_profile=cast(Any, native_profile),
        ocr_profile=cast(Any, ocr_profile),
    )


def test_image_only_complex_layout_gets_alternate_ocr_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = SimpleNamespace(
        get_page_profile=lambda: SimpleNamespace(
            recommended_strategy="image_or_ocr",
            has_text_showing_ops=False,
        )
    )
    image = OcrImage(b"", 100, 100, 0, 0, source="full_page_encoded_image")
    result = OcrTextResult(
        "poster text",
        70,
        line_rows=tuple({} for _ in range(20)),
        word_rows=tuple({} for _ in range(100)),
    )
    candidate = OcrCandidate("full_page_simple", result)
    monkeypatch.setattr(mixin, "extracted_text_token_count", lambda _text: 486)
    monkeypatch.setattr(mixin, "text_ocr_quality_score", lambda _text: 0.22)

    assert mixin.should_try_image_only_layout_ocr_candidate(
        cast(Any, page),
        image,
        candidate,
    )


def test_non_image_page_does_not_get_alternate_layout_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = SimpleNamespace(
        get_page_profile=lambda: SimpleNamespace(
            recommended_strategy="image_or_ocr",
            has_text_showing_ops=False,
        )
    )
    image = OcrImage(b"", 100, 100, 0, 0, source="rendered_page_300dpi")
    candidate = OcrCandidate(
        "rendered_page_300dpi",
        OcrTextResult(
            "page text",
            70,
            line_rows=tuple({} for _ in range(20)),
            word_rows=tuple({} for _ in range(100)),
        ),
    )
    monkeypatch.setattr(mixin, "extracted_text_token_count", lambda _text: 486)
    monkeypatch.setattr(mixin, "text_ocr_quality_score", lambda _text: 0.22)

    assert not mixin.should_try_image_only_layout_ocr_candidate(
        cast(Any, page),
        image,
        candidate,
    )


def test_image_only_layout_supplements_high_confidence_top_line() -> None:
    page = SimpleNamespace(
        media_box=(0.0, 0.0, 100.0, 100.0),
        get_page_profile=lambda: SimpleNamespace(
            recommended_strategy="image_or_ocr",
            has_text_showing_ops=False,
        ),
    )
    body_observation = page_geometry.page_observation_from_bbox(
        (5.0, 50.0, 95.0, 60.0),
        source="full_page_simple",
        kind="ocr_textline",
        text="Known body",
        confidence=90,
    )
    assert body_observation is not None
    lines = (ResolvedTextLine("Known body", body_observation),)
    primary = OcrCandidate("full_page_simple", OcrTextResult("Known body", 90))
    verification = OcrCandidate(
        "verification_full_page_simple_psm4",
        OcrTextResult(
            "POW-WOW",
            95,
            line_rows=(
                {
                    "text": "POW-WOW",
                    "conf": 95,
                    "page_bbox": (20.0, 80.0, 80.0, 90.0),
                },
            ),
        ),
    )
    broad = OcrPageTextResult(
        "Known body",
        primary,
        (primary,),
        verification_candidates=(verification,),
    )

    supplemented = mixin.supplement_image_only_layout_top_lines(
        cast(Any, page),
        lines,
        broad_page_result=broad,
    )

    assert [line.text for line in supplemented] == ["POW-WOW", "Known body"]


def test_image_only_layout_supplements_prominent_consensus_title() -> None:
    page = SimpleNamespace(
        media_box=(0.0, 0.0, 100.0, 100.0),
        get_page_profile=lambda: SimpleNamespace(
            recommended_strategy="image_or_ocr",
            has_text_showing_ops=False,
        ),
    )
    body_observation = page_geometry.page_observation_from_bbox(
        (5.0, 50.0, 95.0, 60.0),
        source="full_page_simple",
        kind="ocr_textline",
        text="Known body",
        confidence=90,
    )
    assert body_observation is not None
    lines = (ResolvedTextLine("Known body", body_observation),)
    primary = OcrCandidate(
        "full_page_simple",
        OcrTextResult("Known body\nPOW-WOW", 70),
    )
    verification = OcrCandidate(
        "verification_full_page_simple_psm4",
        OcrTextResult(
            "POW-WOW",
            80,
            line_rows=(
                {
                    "text": "POW-WOW",
                    "conf": 46,
                    "page_bbox": (10.0, 82.0, 90.0, 96.0),
                },
            ),
        ),
    )
    broad = OcrPageTextResult(
        primary.result.text,
        primary,
        (primary,),
        verification_candidates=(verification,),
    )

    supplemented = mixin.supplement_image_only_layout_top_lines(
        cast(Any, page),
        lines,
        broad_page_result=broad,
    )

    assert [line.text for line in supplemented] == ["POW-WOW", "Known body"]

from copy import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl.capture.program import CapturedProgram, PageProgram
from core_pdf.impl.capture.records import CapturedDrawing, CapturedPath, CapturedSubpath
from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.render.display import DisplayList
from core_pdf.impl.render.model import RasterImage
from core_pdf_ocr.impl.extract.contracts import (
    ObservationSource,
    PageAnalysis,
    StrokedVectorTextEvidence,
)
from core_pdf_ocr.impl.extract.ocr import vector
from core_pdf_ocr.impl.extract.ocr.strokes import (
    StrokedTextDecode,
    StrokedTextObservation,
    StrokedTextProfile,
    StrokedTextRun,
    profile_stroked_text,
)
from core_pdf_ocr.impl.extract.ocr.types import (
    PackedStrokedTextRaster,
    Raster,
    StrokedTextCell,
)
from core_pdf_ocr.impl.extract.quality import internal_candidate


def internal_batch(
    text: tuple[str, ...],
    boxes: tuple[tuple[float, float, float, float], ...],
    *,
    sequences: tuple[int, ...] | None = None,
) -> ObservationBatch:
    return ObservationBatch.from_columns(
        text, boxes, source=ObservationSource.OCR, confidence=(95 for _ in text), sequence=sequences
    )


def internal_packed(
    cells: tuple[StrokedTextCell, ...],
) -> PackedStrokedTextRaster:
    return PackedStrokedTextRaster(
        Raster(RasterImage(b"\xff", 1, 1, 1), 72), (0, 0, 100, 100), cells
    )


def test_shelf_packing_orders_by_height_and_wraps_without_rescaling() -> None:
    runs = (
        StrokedTextRun((100, 200, 120, 204), (10,), 2),
        StrokedTextRun((0, 0, 30, 8), (20,), 3),
        StrokedTextRun((0, 50, 20, 54), (5,), 2),
    )
    cells, height = vector.pack_stroked_text_runs(runs, width=60)
    assert [c.drawing_indexes for c in cells] == [(20,), (5,), (10,)]
    assert [c.packed_box for c in cells] == [(8, 8, 38, 16), (8, 24, 28, 28), (36, 24, 56, 28)]
    assert height == 36
    for cell in cells:
        assert cell.source_box[2] - cell.source_box[0] == cell.packed_box[2] - cell.packed_box[0]
        assert cell.source_box[3] - cell.source_box[1] == cell.packed_box[3] - cell.packed_box[1]
    assert vector.pack_stroked_text_runs(()) == ((), 0)


def test_remapping_requires_unique_cell_and_preserves_observation_identity() -> None:
    reference = object()
    observations = ObservationBatch.from_columns(
        ("A1", "outside", "ambiguous"),
        ((11, 11, 15, 14), (80, 80, 85, 85), (29, 11, 31, 13)),
        source=ObservationSource.OCR,
        confidence=(91, 92, 93),
        references=(reference, None, None),
    )
    packed = internal_packed(
        (
            StrokedTextCell((100, 200, 120, 210), (10, 10, 30, 20), (7, 8)),
            StrokedTextCell((200, 300, 220, 310), (34, 10, 54, 20), (9,)),
        )
    )
    mapped, dropped = vector.remap_stroked_vector_observations(observations, packed)
    assert mapped.text == ("A1",)
    assert mapped.bbox.tolist() == [[101, 201, 105, 204]]
    assert mapped.confidence.tolist() == [91]
    assert mapped.sequence.tolist() == [7]
    assert mapped.font_size.tolist() == [3]
    assert mapped.line_break_before.tolist() == [True]
    assert mapped.references[0] is reference
    assert dropped == 2
    empty, count = vector.remap_stroked_vector_observations(observations, internal_packed(()))
    assert len(empty) == 0
    assert count == 3


@pytest.mark.parametrize(
    ("text", "valid"),
    [
        ("1", True),
        (" A2 ", True),
        ("1234", True),
        ("12345", False),
        ("A", False),
        ("/", False),
        ("", False),
    ],
)
def test_isolated_pin_labels_require_short_digit_bearing_text(text: str, valid: bool) -> None:
    assert vector.isolated_pin_label(text) is valid


def test_candidate_remapping_filters_words_but_retains_symbol_evidence() -> None:
    batch = internal_batch(("A", "12"), ((10, 10, 12, 12), (14, 10, 16, 12)))
    candidate = internal_candidate(6, batch, symbols=batch)
    packed = internal_packed((StrokedTextCell((100, 100, 120, 110), (8, 8, 28, 18), (5,)),))
    result, dropped = vector.remap_stroked_vector_candidate(
        candidate, packed, digit_bearing_only=True
    )
    unfiltered, _ = vector.remap_stroked_vector_candidate(candidate, packed)
    assert unfiltered.observations.text == ("A", "12")
    assert result.observations.text == ("12",)
    assert result.symbols.text == ("A", "12")
    assert result.mode == candidate.mode
    assert result.recognition_status == candidate.recognition_status
    assert dropped == 0


@pytest.mark.parametrize(
    ("left", "right", "maximum", "expected"),
    [
        ("", "", 2, 0),
        ("ab", "ab", 0, 0),
        ("ab", "ac", 1, 1),
        ("ab", "abc", 2, 1),
        ("abc", "ab", 2, 1),
        ("ab", "ba", 2, 2),
        ("a", "abcde", 2, 3),
        ("abcd", "wxyz", 2, 3),
    ],
)
def test_bounded_edit_distance_preserves_insertions_deletions_and_substitutions(
    left: str,
    right: str,
    maximum: int,
    expected: int,
) -> None:
    assert vector.bounded_edit_distance(left, right, maximum) == expected


@pytest.mark.parametrize(
    ("recognized", "decoded", "confidence", "overlap", "allowed"),
    [
        ("AB", "AC", 99, 0.55, True),
        ("abc", "aXY", 84, 0.9, True),
        ("abc", "aXY", 85, 0.9, False),
        ("abc", "aXY", 84, 0.89, False),
        ("abc", "XYZ", 80, 1, False),
        ("a", "abc", 80, 1, False),
        ("abc", "a", 80, 1, False),
        ("abcdefghijklm", "abc", 80, 1, False),
        ("abc", "abcdefghijklm", 80, 1, False),
        ("ab", "abcd", 80, 1, False),
        ("+--", "+..", 80, 1, False),
        ("+ab", "+..", 80, 1, False),
        ("abcde", "aWXYZ", 80, 1, False),
    ],
)
def test_vector_substitution_requires_confidence_overlap_and_lexical_agreement(
    recognized: str,
    decoded: str,
    confidence: float,
    overlap: float,
    allowed: bool,
) -> None:
    assert (
        vector.stroked_vector_substitution(
            recognized,
            decoded,
            confidence=confidence,
            overlap=overlap,
        )
        is allowed
    )


@pytest.mark.parametrize(
    ("cells", "aligned", "learned", "observations"),
    [(0, 4, 8, 8), (36, 9, 12, 12), (100, 12, 16, 16)],
)
def test_packed_decode_gate_checks_each_threshold(
    cells: int, aligned: int, learned: int, observations: int
) -> None:
    observation = StrokedTextObservation("AB", (0, 0, 5, 4), 0, 1)
    decoded = StrokedTextDecode(
        aligned_seeds=aligned,
        learned_signatures=learned,
        observations=(observation,) * observations,
    )
    assert vector.packed_stroked_vector_decode_gate(decoded, cells)
    assert not vector.packed_stroked_vector_decode_gate(
        replace(decoded, aligned_seeds=aligned - 1), cells
    )
    assert not vector.packed_stroked_vector_decode_gate(
        replace(decoded, learned_signatures=learned - 1), cells
    )
    assert not vector.packed_stroked_vector_decode_gate(
        replace(decoded, observations=decoded.observations[:-1]), cells
    )


@pytest.fixture
def vector_capture(ocr_capture: PageAnalysis) -> tuple[PageAnalysis, StrokedTextProfile]:
    drawings = tuple(
        CapturedDrawing(
            i,
            None,
            None,
            kind="stroke",
            stroke_color=(0.0,),
            line_width=0.2,
            line_cap=1,
            line_join=1,
            path=CapturedPath(
                [
                    CapturedSubpath(
                        [
                            (x + (i % 2) * 3, y + (i // 2) * 10)
                            for x, y in (
                                ((0, 4), (1, 0), (2, 4))
                                if i % 2 == 0
                                else ((0, 0), (2, 0), (2, 4), (0, 4))
                            )
                        ]
                    )
                ]
            ),
        )
        for i in range(4)
    )
    profile = profile_stroked_text(drawings, range(4))
    capture = replace(
        ocr_capture,
        page=cast(Any, SimpleNamespace(width=600, height=800, page_number=1)),
        program=PageProgram(CapturedProgram(drawings=drawings)),
        evidence=replace(
            ocr_capture.evidence,
            stroked_vector_text=StrokedVectorTextEvidence(
                trusted=True, drawing_indexes=(0, 1, 2, 3), bbox=(0, 0, 5, 14)
            ),
        ),
    )
    return capture, profile


@pytest.mark.parametrize("general", [False, True])
def test_packed_raster_contains_ink_and_respects_pixel_budget(
    vector_capture: tuple[PageAnalysis, StrokedTextProfile],
    general: bool,
) -> None:
    capture, profile = vector_capture
    if general:
        drawings = tuple(replace(d, line_cap=0) for d in capture.program.drawings)
        capture = replace(capture, program=PageProgram(CapturedProgram(drawings=drawings)))
    packed = vector.stroked_vector_text_raster(capture, 4, profile=profile, max_pixels=20000)
    assert packed is not None
    assert len(packed.cells) == 2
    assert packed.raster.width * packed.raster.height <= 20000
    assert min(packed.raster.image.pixels) < 200
    assert max(packed.raster.image.pixels) == 255
    assert packed.raster.resolution >= 70
    assert capture.program.drawings[0].rect == (0, 0, 2, 4)


def test_full_layer_raster_clips_padding_to_page_bounds(
    vector_capture: tuple[PageAnalysis, StrokedTextProfile],
) -> None:
    capture, _ = vector_capture
    region = vector.full_stroked_vector_text_raster(capture, 4, max_pixels=1000)
    assert region is not None
    assert region.page_box == (0, 0, 9, 18)
    assert region.raster.width * region.raster.height <= 1000
    assert min(region.raster.image.pixels) < 200


def test_untrusted_or_absent_vector_evidence_does_not_rasterize(
    vector_capture: tuple[PageAnalysis, StrokedTextProfile],
    ocr_capture: PageAnalysis,
) -> None:
    capture, profile = vector_capture
    assert vector.stroked_vector_text_raster(ocr_capture, 2, profile=profile) is None
    assert vector.stroked_vector_text_raster(capture, 2, profile=None) is None
    assert vector.stroked_vector_text_raster(capture, 2, profile=StrokedTextProfile()) is None
    assert vector.full_stroked_vector_text_raster(ocr_capture, 2) is None


@pytest.mark.parametrize("wide", [False, True])
def test_isolated_raster_boosts_small_glyphs_and_renders_both_orientations(
    vector_capture: tuple[PageAnalysis, StrokedTextProfile],
    wide: bool,
) -> None:
    capture, _ = vector_capture
    points = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0)] if wide else [(0.0, 0.0), (2.0, 0.0), (2.0, 4.0)]
    drawing = replace(
        capture.program.drawings[0], path=CapturedPath([CapturedSubpath(points)]), line_width=0
    )
    capture = replace(capture, program=PageProgram(CapturedProgram(drawings=(drawing,))))
    profile = profile_stroked_text((drawing,), (0,))
    packed = vector.stroked_vector_text_raster(
        capture, 1, profile=profile, variant="isolated", max_pixels=300000
    )
    assert packed is not None
    assert packed.raster.resolution > 72
    assert packed.raster.width * packed.raster.height <= 300000
    assert min(packed.raster.image.pixels) < 200
    assert len(packed.cells) == 1


def test_symbol_seeds_sort_characters_and_require_a_complete_known_run(
    vector_capture: tuple[PageAnalysis, StrokedTextProfile],
) -> None:
    _, profile = vector_capture
    symbols = internal_batch(
        ("B", "A", "X", "AB", "C"),
        ((3, 0, 5, 4), (0, 0, 2, 4), (0, 10, 2, 14), (3, 10, 5, 14), (0, 0, 1, 1)),
        sequences=(0, 0, 2, 2, 999),
    )
    seeds = vector.stroked_vector_symbol_seeds(profile, symbols)
    assert len(seeds) == 1
    assert seeds[0].text == "AB"
    assert seeds[0].sequence == 0
    assert seeds[0].bbox == (0, 0, 5, 4)
    assert seeds[0].confidence == 95
    assert vector.stroked_vector_symbol_seeds(profile, ObservationBatch.empty()) == ()


@pytest.mark.parametrize("with_symbols", [False, True])
def test_vector_adapter_learns_from_words_and_supplemental_symbols(
    vector_capture: tuple[PageAnalysis, StrokedTextProfile],
    with_symbols: bool,
) -> None:
    _, profile = vector_capture
    words = internal_batch(("AB", "AB"), ((0, 0, 5, 4), (0, 10, 5, 14)), sequences=(0, 2))
    symbols = internal_batch(("A", "B"), ((0, 10, 2, 14), (3, 10, 5, 14)), sequences=(2, 2))
    decoded = vector.decode_stroked_vector_text(profile, words, symbols if with_symbols else None)
    assert [o.text for o in decoded.observations] == ["AB", "AB"]
    assert decoded.learned_signatures == 2
    result, alphabet = vector.recover_stroked_vector_text(profile, words)
    assert result is words
    assert len(alphabet) == 2
    assert vector.decode_stroked_vector_text(None, words) == StrokedTextDecode()
    assert vector.recover_stroked_vector_text(None, words) == (words, ())
    assert (
        vector.decode_stroked_vector_text(profile, ObservationBatch.empty()) == StrokedTextDecode()
    )


def test_recovery_replaces_a_misread_word_and_adds_unrecognized_runs(
    vector_capture: tuple[PageAnalysis, StrokedTextProfile],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, profile = vector_capture
    original = internal_batch(("AX", "note"), ((0, 0, 5, 4), (40, 40, 50, 45)))
    decoded = StrokedTextDecode(
        observations=(
            StrokedTextObservation("AB", (0, 0, 5, 4), 0, 1),
            StrokedTextObservation("BA", (0, 10, 5, 14), 2, 3),
            StrokedTextObservation("AC", (0, 0, 5, 4), 4, 5),
        )
    )
    monkeypatch.setattr(vector, "decode_stroked_vector_text", lambda *_: decoded)
    result, _ = vector.recover_stroked_vector_text(profile, original)
    assert result.text == ("note", "AB", "BA")
    assert result.source.tolist() == [
        ObservationSource.OCR,
        ObservationSource.STRUCTURE,
        ObservationSource.STRUCTURE,
    ]
    assert result.sequence.tolist()[-2:] == [0, 2]
    assert result.bbox.tolist()[-2:] == [[0, 0, 5, 4], [0, 10, 5, 14]]
    assert original.text == ("AX", "note")


@pytest.mark.parametrize("budget", [0, -1])
def test_nonpositive_pixel_budgets_fail_before_allocating(
    vector_capture: tuple[PageAnalysis, StrokedTextProfile],
    budget: int,
) -> None:
    from core_pdf.impl.render.page import RenderedPage
    from core_pdf_ocr.impl.extract.ocr.raster import fit_raster_scale

    rendered = RenderedPage(
        page_number=1, width=100, height=100, rotate=0, display_list=DisplayList(100, 100)
    )
    with pytest.raises(ValueError, match="pixel budget"):
        fit_raster_scale(rendered, 2, budget)


@pytest.mark.parametrize("budget", [1, 17, 101])
def test_pixel_fitting_handles_single_pixel_axes(budget: int) -> None:
    from core_pdf.impl.render.page import RenderedPage
    from core_pdf_ocr.impl.extract.ocr.raster import fit_raster_scale

    rendered = RenderedPage(
        page_number=1, width=0.01, height=100, rotate=0, display_list=DisplayList(0.01, 100)
    )
    scale = fit_raster_scale(rendered, 2, budget)
    width, height = rendered.unrotated_raster_size(scale)
    assert width == 1
    assert 1 <= width * height <= budget


def test_dense_seed_montage_uses_tighter_vertical_spacing(
    vector_capture: tuple[PageAnalysis, StrokedTextProfile],
) -> None:
    capture, profile = vector_capture
    run = profile.seed_runs[0]
    dense = replace(profile, seed_runs=(run,) * 96)
    packed = vector.stroked_vector_text_raster(capture, 1, profile=dense)
    assert packed is not None
    assert len(packed.cells) == 96
    assert packed.cells[0].packed_box[1] == 4


def test_packed_raster_skips_missing_path_geometry(
    vector_capture: tuple[PageAnalysis, StrokedTextProfile],
) -> None:
    capture, profile = vector_capture
    drawings = tuple(replace(d, path=None, bbox=None) for d in capture.program.drawings)
    capture = replace(capture, program=PageProgram(CapturedProgram(drawings=drawings)))
    packed = vector.stroked_vector_text_raster(capture, 1, profile=profile)
    assert packed is not None
    assert min(packed.raster.image.pixels) == 255

from copy import replace
from typing import Any

import numpy
import pytest

from core_pdf.impl.capture_records import CapturedDrawing, CapturedPath, CapturedSubpath
from core_pdf_ocr._vendor.newstroke_data import NEWSTROKE_ASCII
from core_pdf_ocr.impl.extract.ocr import newstroke
from core_pdf_spec.s_08_graphics.matrix import Matrix


def line(points: list[tuple[float, float]]) -> CapturedDrawing:
    return CapturedDrawing(
        0,
        None,
        None,
        kind="stroke",
        stroke_color=(0.0,),
        line_width=0.2,
        path=CapturedPath([CapturedSubpath(points)]),
    )


def make_text(text: str, y: float = 0, angle: int = 0) -> tuple[CapturedDrawing, ...]:
    drawings = []
    cursor = 0.0
    rotation = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[angle]
    for char in text:
        if char == " ":
            cursor += 16
            continue
        encoded = NEWSTROKE_ASCII[ord(char) - 32]
        left, right = (ord(c) - ord("R") for c in encoded[:2])
        previous = None
        for i in range(2, len(encoded), 2):
            pair = encoded[i : i + 2]
            if pair == " R":
                previous = None
                continue
            x = (ord(pair[0]) - ord("R") - left + cursor) * 0.5
            v = (ord(pair[1]) - ord("R") - 8) * 0.5 + y
            a, b = rotation
            point = (x * a - v * b, x * b + v * a)
            if previous is not None:
                drawings.append(line([previous, point]))
            previous = point
        cursor += right - left
    return tuple(drawings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_segments", 9999),
        ("matched_segments", 9999),
        ("candidate_segments", 20000),
        ("characters", 999),
        ("sequences", 99),
        ("maximum_error", 0.10001),
    ],
)
def test_trust_requires_all_page_evidence(field: str, value: float) -> None:
    trusted = newstroke.NewstrokeDecode(
        candidate_segments=10000,
        matched_segments=10000,
        characters=1000,
        sequences=100,
        maximum_error=0.1,
    )
    assert trusted.trusted
    assert not replace(trusted, **{field: value}).trusted
    assert not newstroke.NewstrokeDecode().trusted
    assert newstroke.NewstrokeDecode().matched_coverage == 0


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_dense_vector_page_decodes_with_original_orientation(angle: int) -> None:
    text = "R1234567890"
    drawings = tuple(drawing for row in range(160) for drawing in make_text(text, row * 20, angle))
    decoded = newstroke.decode_newstroke_drawings(drawings)
    assert decoded.trusted
    assert decoded.candidate_segments == len(drawings)
    assert decoded.matched_segments == len(drawings)
    assert decoded.characters == len(text) * 160
    assert decoded.sequences == 160
    assert [run.text for run in decoded.runs] == [text] * 160
    assert {run.rotation_angle for run in decoded.runs} == {angle}
    for index, run in enumerate(decoded.runs):
        assert run.order == index
        assert run.font_name == "KiCad Newstroke"
        assert run.font_size == pytest.approx(10.5)
        assert run.space_width == pytest.approx(8)
        assert run.fill_color == (0.0,)
        assert run.confidence == pytest.approx(100)
        assert dict(run.provenance)["unicode_source"] == "newstroke-template"
        assert run.ink_bbox == run.advance_bbox
        assert run.x0 < run.x1
        assert run.y0 < run.y1


def test_small_vector_page_retains_candidate_evidence_without_decoding() -> None:
    drawings = make_text("ABC")
    decoded = newstroke.decode_newstroke_drawings(drawings)
    assert decoded.candidate_segments == len(drawings)
    assert decoded.runs == ()
    assert not decoded.trusted


@pytest.mark.parametrize(
    "drawing",
    [
        CapturedDrawing(0, None, None),
        line([]),
        line([(0, 0)]),
        line([(0, 0), (0, 0)]),
        line([(0, 0), (1, 1), (2, 2)]),
        replace(line([(0, 0), (1, 1)]), stroke_opacity=0),
        replace(line([(0, 0), (1, 1)]), line_width=0),
        replace(
            line([(0, 0), (1, 1)]),
            path=CapturedPath([CapturedSubpath([(0, 0), (1, 1)], closed=True)]),
        ),
        replace(
            line([(0, 0), (1, 1)]),
            path=CapturedPath(
                [CapturedSubpath([(0, 0), (1, 1)]), CapturedSubpath([(2, 2), (3, 3)])]
            ),
        ),
    ],
)
def test_non_candidate_drawings_remain_sequence_barriers(drawing: CapturedDrawing) -> None:
    valid = line([(0, 0), (1, 1)])
    segments, styles, count = newstroke.drawing_segments((valid, drawing, valid))
    assert count == 2
    assert len(styles) == 1
    assert segments[1] is None
    assert newstroke.segment_continuity(segments) == (False, False)


@pytest.mark.parametrize(("gap", "expected"), [(0, True), (0.019, True), (0.021, False)])
def test_segment_continuity_respects_line_width(gap: float, expected: bool) -> None:
    first = line([(0, 0), (1, 1)])
    second = line([(1 + gap, 1), (2, 2)])
    segments, _, _ = newstroke.drawing_segments((first, second))
    assert newstroke.segment_continuity(segments) == (expected,)
    second.stroke_color = (1.0,)
    segments, styles, _ = newstroke.drawing_segments((first, second))
    assert len(styles) == 2
    assert newstroke.segment_continuity(segments) == (False,)


def arrays(
    drawings: tuple[CapturedDrawing, ...],
) -> tuple[
    tuple[newstroke.Segment | None, ...],
    tuple[tuple[object, ...], ...],
    numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    numpy.ndarray[Any, numpy.dtype[numpy.int16]],
]:
    segments, styles, _ = newstroke.drawing_segments(drawings)
    points = numpy.asarray(
        [((s.x0, s.y0), (s.x1, s.y1)) if s is not None else ((0, 0), (0, 0)) for s in segments],
        dtype=numpy.float64,
    )
    style_ids = numpy.asarray(
        [s.style if s is not None else -1 for s in segments], dtype=numpy.int16
    )
    return segments, styles, points, style_ids


@pytest.mark.parametrize(
    "matrix",
    [
        ((0.001, 0), (0, 0.001)),
        ((10, 0), (0, 10)),
        ((0.5, 0), (0, 0.01)),
        ((0.5, 0), (0.4, 0.5)),
        ((0, 0), (0, 0.5)),
    ],
)
def test_fitted_glyph_rejects_implausible_size_aspect_and_shear(
    matrix: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    template = next(t for t in newstroke.make_templates().all if t.char == "R")
    transformed = template.segments @ numpy.asarray(matrix)
    drawings = tuple(line([tuple(left), tuple(right)]) for left, right in transformed)
    segments, _, points, styles = arrays(drawings)
    assert newstroke.fit_match(segments, 0, template, points, styles) is None


def test_fitted_and_fixed_glyph_reject_corrupted_shape_and_style() -> None:
    template = next(t for t in newstroke.make_templates().all if t.char == "R")
    drawings = make_text("R")
    segments, _, points, styles = arrays(drawings)
    fitted = newstroke.fit_match(segments, 0, template, points, styles)
    assert fitted is not None
    assert fitted.char == "R"
    assert fitted.error < 1e-12
    assert fitted.transform.scale == pytest.approx(0.5)
    continuity = newstroke.segment_continuity(segments)
    fixed = newstroke.fixed_template_match(
        segments,
        continuity,
        template,
        0,
        fitted.transform,
        0,
        points,
        styles,
    )
    assert fixed is not None
    assert fixed.error < 1e-12
    corrupted = points.copy()
    corrupted[-1, 1] += (1, -1)
    assert newstroke.fit_match(segments, 0, template, corrupted, styles) is None
    assert (
        newstroke.fixed_template_match(
            segments,
            continuity,
            template,
            0,
            fitted.transform,
            0,
            corrupted,
            styles,
        )
        is None
    )
    mismatched = styles.copy()
    mismatched[-1] = 1
    assert newstroke.fit_match(segments, 0, template, points, mismatched) is None
    assert (
        newstroke.fixed_template_match(
            segments,
            continuity,
            template,
            0,
            fitted.transform,
            0,
            points,
            mismatched,
        )
        is None
    )
    assert newstroke.fit_match(segments[:-1], 0, template, points[:-1], styles[:-1]) is None


@pytest.mark.parametrize("text", ["IR123", "I HR12  3", "RHPI"])
def test_decode_around_robust_seed_recovers_prefixes_spaces_and_longest_glyph(text: str) -> None:
    drawings = make_text(text)
    segments, styles, points, style_ids = arrays(drawings)
    templates = newstroke.make_templates()
    template = next(t for t in templates.robust if t.char == "R")
    start = len(make_text(text[: text.index("R")]))
    seed = newstroke.fit_match(segments, start, template, points, style_ids)
    assert seed is not None
    matches = newstroke.decode_around(
        segments,
        newstroke.segment_continuity(segments),
        templates,
        seed,
        0,
        points,
        style_ids,
    )
    assert newstroke.sequence_text(matches) == text
    run = newstroke.sequence_run(matches, segments, styles, 7)
    assert run.text == text
    assert run.order == 7
    assert matches[0].start == 0
    assert matches[-1].stop == len(segments)


@pytest.mark.parametrize(
    ("x", "y", "follows"),
    [
        (0, 0, True),
        (16, 0, True),
        (32, 0, True),
        (-0.76, 0, False),
        (0, 0.76, False),
        (33, 0, False),
        (8, 0, False),
    ],
)
def test_cursor_accepts_only_aligned_zero_one_or_two_spaces(
    x: float, y: float, follows: bool
) -> None:
    template = next(t for t in newstroke.make_templates().robust if t.char == "R")
    segments, _, points, styles = arrays(make_text("R"))
    previous = newstroke.fit_match(segments, 0, template, points, styles)
    assert previous is not None
    current = replace(
        previous,
        translation=previous.translation
        + numpy.asarray((previous.width + x, y)) @ previous.transform.matrix,
    )
    assert newstroke.cursor_follows(previous, current) is follows


def test_empty_match_sequence_has_no_text() -> None:
    assert newstroke.sequence_text(()) == ""


def test_many_unrelated_segments_cannot_become_trusted_text() -> None:
    drawing = line([(0, 0), (100, 100)])
    result = newstroke.decode_newstroke_drawings((drawing,) * 10000)
    assert result.candidate_segments == 10000
    assert result.matched_segments == 0
    assert result.runs == ()
    assert not result.trusted


def test_dense_page_preserves_barriers_and_separate_stroke_styles() -> None:
    text = "IR1234567890"
    drawings: list[CapturedDrawing] = []
    for row in range(160):
        drawings.append(CapturedDrawing(row, None, None))
        drawings.extend(
            replace(drawing, stroke_color=(float(row % 2),), line_width=0.2 + row % 2)
            for drawing in make_text(text, row * 20)
        )
    result = newstroke.decode_newstroke_drawings(tuple(drawings))
    assert result.trusted
    assert [run.text for run in result.runs] == [text] * 160
    for index, run in enumerate(result.runs):
        assert run.fill_color == (float(index % 2),)
        line = make_text(text, index * 20)
        points = [
            point
            for drawing in line
            if drawing.path
            for path in drawing.path.subpaths
            for point in path.points
        ]
        padding = (0.2 + index % 2) / 2
        assert run.ink_bbox == pytest.approx(
            (
                min(p[0] for p in points) - padding,
                min(p[1] for p in points) - padding,
                max(p[0] for p in points) + padding,
                max(p[1] for p in points) + padding,
            )
        )


def test_fixed_match_rejects_ambiguous_templates() -> None:
    templates = newstroke.make_templates()
    template = next(t for t in templates.robust if t.char == "R")
    segments, _, points, styles = arrays(make_text("R"))
    seed = newstroke.fit_match(segments, 0, template, points, styles)
    assert seed is not None
    delta = template.segments[0, 1] - template.segments[0, 0]
    ambiguous = newstroke.TemplateSet(
        (template, replace(template, char="X")),
        (template,),
        {(int(delta[0]), int(delta[1])): (template, replace(template, char="X"))},
    )
    assert (
        newstroke.fixed_match(
            segments,
            newstroke.segment_continuity(segments),
            ambiguous,
            0,
            seed.transform,
            0,
            points,
            styles,
        )
        is None
    )


def test_matchers_reject_barrier_at_start_and_stale_seed() -> None:
    templates = newstroke.make_templates()
    template = next(t for t in templates.robust if t.char == "R")
    segments, _, points, styles = arrays(make_text("R"))
    seed = newstroke.fit_match(segments, 0, template, points, styles)
    assert seed is not None
    blocked = (None, *segments[1:])
    continuity = newstroke.segment_continuity(blocked)
    assert newstroke.fit_match(blocked, 0, template, points, styles) is None
    assert (
        newstroke.fixed_match(blocked, continuity, templates, 0, seed.transform, 0, points, styles)
        is None
    )
    assert newstroke.decode_forward(blocked, continuity, templates, seed, points, styles) == ()
    assert newstroke.decode_around(blocked, continuity, templates, seed, 0, points, styles) == ()


def test_fixed_match_declines_delta_without_any_template() -> None:
    templates = newstroke.make_templates()
    template = next(t for t in templates.robust if t.char == "R")
    segments, _, points, styles = arrays(make_text("R"))
    seed = newstroke.fit_match(segments, 0, template, points, styles)
    assert seed is not None
    empty = replace(templates, by_first_delta={})
    assert (
        newstroke.fixed_match(
            segments,
            newstroke.segment_continuity(segments),
            empty,
            0,
            seed.transform,
            0,
            points,
            styles,
        )
        is None
    )


def test_backward_decode_stops_at_ambiguous_preceding_glyph() -> None:
    templates = newstroke.make_templates()
    template = next(t for t in templates.robust if t.char == "R")
    segments, _, points, styles = arrays(make_text("RR"))
    start = len(make_text("R"))
    seed = newstroke.fit_match(segments, start, template, points, styles)
    assert seed is not None
    ambiguous = replace(templates, all=(template, replace(template, char="X")))
    matches = newstroke.decode_around(
        segments, newstroke.segment_continuity(segments), ambiguous, seed, 0, points, styles
    )
    assert len(matches) == 1
    assert matches[0] is seed


def test_template_loader_accepts_repeated_pen_up_markers(monkeypatch) -> None:
    encoded = NEWSTROKE_ASCII[ord("R") - 32]
    repeated = encoded[:2] + " R R" + encoded[2:]
    monkeypatch.setattr(newstroke, "NEWSTROKE_ASCII_ALTERNATES", {"R": repeated})
    templates = newstroke.make_templates()
    variants = [template for template in templates.all if template.char == "R"]
    assert len(variants) == 2
    numpy.testing.assert_array_equal(variants[0].points, variants[1].points)


def test_isolated_glyphs_do_not_become_decoded_sequences_on_dense_page() -> None:
    per_glyph = len(make_text("R"))
    drawings = tuple(
        drawing
        for row in range(10000 // per_glyph + 1)
        for drawing in (*make_text("R", y=row * 20), CapturedDrawing(row, None, None))
    )
    result = newstroke.decode_newstroke_drawings(drawings)
    assert result.candidate_segments >= 10000
    assert result.runs == ()
    assert result.sequences == 0
    assert not result.trusted


def test_fixed_match_rejects_fractional_template_delta() -> None:
    templates = newstroke.make_templates()
    template = next(t for t in templates.robust if t.char == "R")
    segments, _, points, styles = arrays(make_text("R"))
    seed = newstroke.fit_match(segments, 0, template, points, styles)
    assert seed is not None
    first = segments[0]
    assert first is not None
    changed = (replace(first, x1=first.x1 + 0.15), *segments[1:])
    points = points.copy()
    points[0, 1, 0] += 0.15
    assert (
        newstroke.fixed_match(
            changed,
            newstroke.segment_continuity(changed),
            templates,
            0,
            seed.transform,
            first.style,
            points,
            styles,
        )
        is None
    )


def test_dense_page_learns_multiple_scales_for_same_style() -> None:
    text = "R1234567890"
    drawings = []
    for row in range(160):
        for drawing in make_text(text, row * 30):
            assert drawing.path is not None
            drawings.append(
                replace(drawing, path=drawing.path.transformed(Matrix(2, 0, 0, 2, 0, 0)))
                if row >= 80
                else drawing
            )
    result = newstroke.decode_newstroke_drawings(tuple(drawings))
    assert result.trusted
    assert [run.text for run in result.runs] == [text] * 160
    assert result.runs[0].font_size == pytest.approx(10.5)
    assert result.runs[-1].font_size == pytest.approx(21)

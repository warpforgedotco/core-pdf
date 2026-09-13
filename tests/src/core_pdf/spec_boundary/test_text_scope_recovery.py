# SPDX-License-Identifier: AGPL-3.0-only
"""Nested text scopes unwind without changing their caller's paint state."""

from types import SimpleNamespace
from typing import NoReturn

import numpy
import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.capture.program import CapturedProgram
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl._impl.render.commands import append_captured_program
from core_pdf.impl._impl.render.display import DisplayList
from core_pdf.impl._impl.render.model import PathPaintItem
from core_pdf.impl._impl.render.page import RenderedPage
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.operations import ContentOperands
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName

internal_BACKDROP = (51, 153, 204, 255)
internal_RED = b"0 0 d0 1 0 0 rg 0 0 1000 1000 re f"
internal_BLUE = b"0 0 d0 0 0 1 rg 0 0 1000 1000 re f"
internal_TEXT = b"BT /F 12 Tf 1 0 0 1 4 4 Tm (AB) Tj ET "


def internal_state(first: PdfStream | None = None, *, child: PdfStream | None = None) -> TextState:
    resolver = ObjectResolver(b"", {})
    state = TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))
    glyph_resources: PdfDict = {"XObject": {"Child": child}} if child is not None else {}
    font: PdfDict = {
        "Type": PdfName.of("Font"),
        "Subtype": PdfName.of("Type3"),
        "FontBBox": [0, 0, 1000, 1000],
        "FontMatrix": [0.001, 0, 0, 0.001, 0, 0],
        "FirstChar": 65,
        "LastChar": 66,
        "Widths": [0, 0],
        "Encoding": {"Differences": [65, PdfName.of("A"), PdfName.of("B")]},
        "CharProcs": {
            "A": first if first is not None else PdfStream(raw_data=internal_RED),
            "B": PdfStream(raw_data=internal_BLUE),
        },
        "Resources": glyph_resources,
    }
    state.resources = {"Font": {"F": font}}
    return state


def internal_consume(state: TextState, content: bytes) -> None:
    state.stream_executor.consume(PdfStream(raw_data=content), state.resources, IDENTITY_MATRIX, 0)


def internal_display(state: TextState, *, include_text: bool = True) -> DisplayList:
    program = CapturedProgram(
        runs=tuple(state.runs),
        glyphs=tuple(state.glyphs),
        drawings=tuple(state.drawings),
        inline_images=tuple(state.inline_images),
        text_boundaries=tuple(state.text_boundaries),
    )
    display = DisplayList(24, 20)
    append_captured_program(display, program, include_text=include_text)
    return display


def internal_pixels(state: TextState, *, include_text: bool = True) -> numpy.ndarray:
    page = RenderedPage(0, 24, 20, 0, internal_display(state, include_text=include_text))
    return page.rasterize(background=internal_BACKDROP).array().copy()


def internal_assert_balanced(state: TextState) -> None:
    assert not state.capture_text_open
    assert not state.capture_text_frames
    assert not state.capture_frames
    assert not state.capture_graphics_stack
    assert not state.stream_executor.active_streams
    assert not state.stack
    assert not state.in_text_object
    scopes: list[str] = []
    for item in internal_display(state).items:
        if item.kind in {"group-begin", "scope-begin", "state-push"}:
            scopes.append(item.kind)
        elif item.kind in {"group-end", "scope-end", "state-pop"}:
            assert (
                scopes.pop()
                == {
                    "group-end": "group-begin",
                    "scope-end": "scope-begin",
                    "state-pop": "state-push",
                }[item.kind]
            )
    assert not scopes


@pytest.mark.parametrize("stage", ["decode", "handler"])
@pytest.mark.parametrize("transparency_group", [False, True])
def test_failed_charproc_closes_markers_before_form_end_and_next_page_paint(
    stage: str, transparency_group: bool
) -> None:
    failing = PdfStream(raw_data=b"0 0 d0 BT q 0 0 50 50 re W n FAIL")
    state = internal_state(failing)
    triggered: list[str] = []

    def fail() -> NoReturn:
        triggered.append(stage)
        raise PdfParseError("failed CharProc")

    if stage == "decode":

        def decode(*args: object, **kwargs: object) -> bytes:
            fail()

        failing.decoder = decode
    else:

        def handler(operands: ContentOperands, depth: int) -> None:
            fail()

        state.operator_overrides["FAIL"] = handler

    dictionary: PdfDict = {
        "Subtype": PdfName.of("Form"),
        "BBox": [0, 0, 24, 20],
        "Resources": state.resources,
    }
    if transparency_group:
        dictionary["Group"] = {"S": PdfName.of("Transparency"), "K": True}
    form = PdfStream(raw_data=internal_TEXT, dictionary=dictionary)
    state.resources = {"XObject": {"Form": form}}
    internal_consume(state, b"/Form Do 0 1 0 rg 4 4 12 12 re f")
    assert triggered == [stage]
    kinds = [boundary.kind for boundary in state.text_boundaries]
    assert kinds.count("glyph-begin") == kinds.count("glyph-end") == 1
    internal_assert_balanced(state)
    pixels = internal_pixels(state)
    numpy.testing.assert_array_equal(pixels[6, 14], [0, 255, 0, 255])
    numpy.testing.assert_array_equal(pixels[0, 23], internal_BACKDROP)


@pytest.mark.parametrize("nested_form", [False, True])
@pytest.mark.parametrize("knockout", [False, True])
def test_missing_et_in_child_stream_does_not_finish_parent_text_object(
    nested_form: bool, knockout: bool
) -> None:
    def capture(*, terminate: bool) -> TextState:
        content = b"BT 1 0 0 rg 0 0 1000 1000 re f" + (b" ET" if terminate else b"")
        child = (
            PdfStream(
                raw_data=content,
                dictionary={
                    "Subtype": PdfName.of("Form"),
                    "BBox": [0, 0, 1000, 1000],
                    "Group": {"S": PdfName.of("Transparency"), "K": True},
                },
            )
            if nested_form
            else None
        )
        glyph = b"0 0 d0 /Child Do" if nested_form else b"0 0 d0 " + content
        state = internal_state(PdfStream(raw_data=glyph), child=child)
        state.graphics.fill_opacity = 0.5
        state.graphics.text_knockout = knockout
        internal_consume(state, internal_TEXT)
        internal_assert_balanced(state)
        return state

    actual = capture(terminate=False)
    reference = capture(terminate=True)
    numpy.testing.assert_array_equal(internal_pixels(actual), internal_pixels(reference))
    if knockout:
        alpha = 128 / 255
        expected = numpy.rint(numpy.array(internal_BACKDROP[:3]) * (1 - alpha) + [0, 0, 128])
        numpy.testing.assert_allclose(internal_pixels(actual)[6, 14, :3], expected, atol=1, rtol=0)


@pytest.mark.parametrize("knockout", [False, True])
@pytest.mark.parametrize("nested_group", [False, True])
def test_include_text_false_suppresses_charproc_and_nested_paint_but_keeps_next_vector(
    knockout: bool, nested_group: bool
) -> None:
    dictionary: PdfDict = {"Subtype": PdfName.of("Form"), "BBox": [0, 0, 1000, 1000]}
    if nested_group:
        dictionary["Group"] = {"S": PdfName.of("Transparency"), "K": True}
    child = PdfStream(raw_data=b"BT 1 0 0 rg 0 0 1000 1000 re f", dictionary=dictionary)
    glyph = PdfStream(raw_data=b"0 0 d0 q 0 0 500 500 re W n /Child Do Q " + internal_RED)
    state = internal_state(glyph, child=child)
    state.graphics.text_knockout = knockout
    vector = b"0 1 0 rg 18 4 4 12 re f"
    internal_consume(state, internal_TEXT + vector)
    reference = internal_state()
    internal_consume(reference, vector)
    actual = internal_pixels(state, include_text=False)
    numpy.testing.assert_array_equal(actual, internal_pixels(reference))
    assert not numpy.array_equal(actual, internal_pixels(state))
    numpy.testing.assert_array_equal(actual[10, 20], [0, 255, 0, 255])


@pytest.mark.parametrize("knockout", [False, True])
@pytest.mark.parametrize("alpha_is_shape", [False, True])
@pytest.mark.parametrize("blend", [None, "Multiply"])
def test_scalar_mask_is_inherited_by_glyphs_and_not_applied_again_at_text_group_exit(
    knockout: bool, alpha_is_shape: bool, blend: str | None
) -> None:
    # ISO 32000-2 9.3.8: implicit text groups inherit transparency; their
    # completed result composites with Normal, alpha=1 and no soft mask.
    state = internal_state()
    state.graphics.fill_opacity = 0.5
    state.graphics.text_knockout = knockout
    state.graphics.alpha_is_shape = alpha_is_shape
    state.graphics.blend_mode = blend
    state.group_alpha = 0.5
    internal_consume(state, internal_TEXT)
    internal_assert_balanced(state)
    assert state.group_alpha == 0.5
    paints = [drawing for drawing in state.drawings if drawing.kind == "fill"]
    assert len(paints) == 2
    assert all(paint.soft_mask_alpha == paint.fill_opacity == 0.5 for paint in paints)

    backdrop = numpy.array(internal_BACKDROP[:3], dtype=float)
    red = numpy.array([255, 0, 0], dtype=float)
    blue = numpy.array([0, 0, 255], dtype=float)
    alpha = 64 / 255
    first_blend = red if blend is None else backdrop * red / 255
    first = numpy.rint(backdrop * (1 - alpha) + first_blend * alpha)
    if knockout:
        second_blend = blue if blend is None else backdrop * blue / 255
        element = numpy.rint(backdrop * (1 - alpha) + second_blend * alpha)
        shape = alpha if alpha_is_shape else 1.0
        expected = element + (1 - shape) * (first - backdrop)
    else:
        second_blend = blue if blend is None else first * blue / 255
        expected = numpy.rint(first * (1 - alpha) + second_blend * alpha)
    numpy.testing.assert_allclose(internal_pixels(state)[6, 14, :3], expected, atol=1, rtol=0)


@pytest.mark.parametrize("knockout", [False, True])
@pytest.mark.parametrize("nested_group", [False, True])
@pytest.mark.parametrize("separate_strokes", [False, True])
def test_type3_glyph_preserves_separate_translucent_strokes_under_shape_tracking(
    knockout: bool, nested_group: bool, separate_strokes: bool
) -> None:
    # Each S is its own elementary object inside one non-knockout glyph
    # group. A single S with crossing subpaths still applies opacity once.
    strokes = (
        b"1 0 0 RG 300 w 0 500 m 1000 500 l "
        + (b"S " if separate_strokes else b"")
        + b"500 0 m 500 1000 l S"
    )
    child = (
        PdfStream(
            raw_data=b"/Half gs " + strokes,
            dictionary={
                "Subtype": PdfName.of("Form"),
                "BBox": [0, 0, 1000, 1000],
                "Group": {"S": PdfName.of("Transparency"), "K": False},
                "Resources": {"ExtGState": {"Half": {"CA": 0.5}}},
            },
        )
        if nested_group
        else None
    )
    glyph = b"0 0 d0 " + (b"/Child Do" if nested_group else strokes)
    state = internal_state(PdfStream(raw_data=glyph), child=child)
    state.graphics.stroke_opacity = 0.5
    state.graphics.text_knockout = knockout
    internal_consume(state, b"BT /F 12 Tf 1 0 0 1 4 4 Tm (A) Tj ET")
    internal_assert_balanced(state)
    assert sum(item.kind == "stroke" for item in internal_display(state).items) == (
        2 if separate_strokes else 1
    )
    alpha = 128 / 255
    if separate_strokes:
        alpha = 1 - (1 - alpha) ** 2
    expected = numpy.array(internal_BACKDROP[:3]) * (1 - alpha) + [255 * alpha, 0, 0]
    numpy.testing.assert_allclose(internal_pixels(state)[10, 10, :3], expected, atol=1, rtol=0)


def test_ordinary_strokes_still_coalesce_after_a_shape_tracked_text_group() -> None:
    state = internal_state()
    state.graphics.text_knockout = False
    internal_consume(
        state,
        b"BT /F 12 Tf (A) Tj ET 0 0 m 10 10 l S 0 10 m 10 0 l S",
    )
    strokes = [item for item in internal_display(state).items if item.kind == "stroke"]
    assert len(strokes) == 1
    assert isinstance(strokes[0], PathPaintItem)
    assert strokes[0].coalesced_path

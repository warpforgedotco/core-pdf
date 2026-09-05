from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content.operations import ContentOperands
from core_pdf.impl.spec.s_07_content.state import TextState
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.types import PdfName
from tests.helpers.resolvers import IdentityResolver


def internal_state() -> TextState:
    return TextState(cast(Any, SimpleNamespace(resolver=IdentityResolver())))


class internal_BrokenStream(PdfStream):
    @property
    def data(self) -> bytes:
        raise PdfParseError("cannot decode stream")


def test_failed_decoding_does_not_enter_or_register_the_stream() -> None:
    state = internal_state()
    state.ctm = Matrix(2, 0, 0, 3, 5, 7)
    state.resources = {"existing": 1}
    state.resources_id = id(state.resources)
    before = state.capture_stream_state()
    stream = internal_BrokenStream()

    for _ in range(2):
        with pytest.raises(PdfParseError, match="cannot decode stream"):
            state.consume_stream(stream, {}, IDENTITY_MATRIX, 0)
        assert state.capture_stream_state() == before
        assert not state.stream_executor.active_streams
        assert state.stream_order == -1


def test_undecodable_form_is_skipped_without_leaking_a_transparency_group() -> None:
    state = internal_state()
    state.fill_opacity = 0.4
    state.group_alpha = 0.6
    form = internal_BrokenStream(dictionary={"Subtype": "Form", "Group": {"S": "Transparency"}})
    resources: PdfDict = {"XObject": {"Child": form}}

    state.consume_stream(
        PdfStream(raw_data=b"/Child Do 0 0 1 1 re f /Child Do"),
        resources,
        IDENTITY_MATRIX,
        0,
    )

    assert [drawing.kind for drawing in state.drawings] == ["fill"]
    assert state.drawings[0].soft_mask_alpha == 0.6
    assert state.group_alpha == 0.6
    assert not state.stream_executor.active_streams


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_escaping_nested_error_unwinds_every_entered_parent(
    error_type: type[BaseException],
) -> None:
    state = internal_state()
    state.fill_opacity = 0.4
    state.group_alpha = 0.6
    state.op_q((), 0)
    state.line_width = 7
    state.ctm = Matrix(2, 0, 0, 3, 5, 7)
    state.op_BMC((), 0)
    before = state.capture_stream_state()
    before_stack = list(state.stack)
    before_marked_content = list(state.marked_content_stack)

    def fail(operands: ContentOperands, depth: int) -> None:
        raise error_type("nested failure")

    state.op_handlers["fail"] = fail
    inner = PdfStream(
        dictionary={"Subtype": "Form", "Group": {"S": "Transparency"}},
        raw_data=b"q 2 w /Tag BMC fail",
    )
    outer = PdfStream(
        dictionary={
            "Subtype": "Form",
            "Group": {"S": "Transparency"},
            "Resources": {"XObject": {"Inner": inner}},
        },
        raw_data=b"q 3 w /Tag BMC /Inner Do",
    )
    with pytest.raises(error_type, match="nested failure"):
        state.consume_stream(
            PdfStream(raw_data=b"q 4 w /Tag BMC /Outer Do"),
            {"XObject": {"Outer": outer}},
            IDENTITY_MATRIX,
            0,
        )

    assert state.capture_stream_state() == before
    assert state.stack == before_stack
    assert state.marked_content_stack == before_marked_content
    assert len(state.clip_scope_stack) == len(before_stack)
    assert not state.stream_executor.active_streams
    assert [drawing.kind for drawing in state.drawings] == [
        "group-begin",
        "group-begin",
        "group-end",
        "group-end",
    ]
    assert state.group_alpha == 0.6


def test_form_parse_error_restores_parent_and_balances_group_markers() -> None:
    state = internal_state()
    state.fill_opacity = 0.4
    state.group_alpha = 0.6

    def fail(operands: ContentOperands, depth: int) -> None:
        raise PdfParseError("bad Form operator")

    state.op_handlers["fail"] = fail
    form = PdfStream(
        dictionary={"Subtype": "Form", "Group": {"S": "Transparency"}},
        raw_data=b"q 3 w /Tag BMC fail",
    )

    state.consume_stream(
        PdfStream(raw_data=b"2 w /Child Do 0 0 1 1 re S"),
        {"XObject": {"Child": form}},
        IDENTITY_MATRIX,
        0,
    )

    assert [drawing.kind for drawing in state.drawings] == [
        "group-begin",
        "group-end",
        "stroke",
    ]
    stroke = state.drawings[-1]
    assert stroke.line_width == 2
    assert stroke.soft_mask_alpha == 0.6
    assert not state.stack
    assert not state.marked_content_stack
    assert not state.stream_executor.active_streams


def test_form_restores_resolved_fill_and_stroke_color_spaces() -> None:
    state = internal_state()
    form = PdfStream(
        dictionary={
            "Subtype": "Form",
            "Resources": {
                "ColorSpace": {
                    "Palette": [PdfName.of("Indexed"), PdfName.of("DeviceRGB"), 0, b"\x80\x80\x80"]
                }
            },
        },
        raw_data=b"/Palette cs /Palette CS 0 sc 0 SC 0 0 1 1 re B",
    )
    resources: PdfDict = {
        "ColorSpace": {
            "Palette": [PdfName.of("Indexed"), PdfName.of("DeviceRGB"), 0, b"\x00\xff\x00"]
        },
        "XObject": {"Child": form},
    }

    state.consume_stream(
        PdfStream(
            raw_data=(
                b"/Palette cs /Palette CS 0 sc 0 SC 0 0 1 1 re B /Child Do 0 sc 0 SC 0 0 1 1 re B"
            )
        ),
        resources,
        IDENTITY_MATRIX,
        0,
    )

    assert len(state.drawings) == 3
    before, child, after = state.drawings
    assert before.fill == after.fill == (0.0, 1.0, 0.0)
    assert before.stroke_color == after.stroke_color == (0.0, 1.0, 0.0)
    assert child.fill == child.stroke_color == pytest.approx((128 / 255,) * 3)


def test_q_Q_leaves_text_matrices_while_nested_stream_restores_them() -> None:
    state = internal_state()
    initial_text = Matrix(1, 0, 0, 1, 2, 3)
    initial_line = Matrix(1, 0, 0, 1, 4, 5)
    new_text = Matrix(2, 0, 0, 3, 6, 7)
    new_line = Matrix(1, 0, 0, 1, 8, 9)
    state.text_matrix = initial_text
    state.line_matrix = initial_line
    state.op_q((), 0)
    state.text_matrix = new_text
    state.line_matrix = new_line
    state.op_Q((), 0)

    assert state.text_matrix == new_text
    assert state.line_matrix == new_line
    state.consume_stream(
        PdfStream(raw_data=b"2 0 0 2 0 0 cm BT 1 0 0 1 20 30 Tm ET"),
        {},
        IDENTITY_MATRIX,
        0,
    )
    assert state.text_matrix == new_text
    assert state.line_matrix == new_line
    assert (state.combined_A, state.combined_B, state.combined_C, state.combined_D) == (
        2,
        0,
        0,
        3,
    )


def test_recursive_form_cycle_is_skipped_and_parent_resumes() -> None:
    state = internal_state()
    form = PdfStream(
        dictionary={"Subtype": "Form"},
        raw_data=b"0 0 1 1 re f /Self Do 1 1 1 1 re f",
    )
    state.consume_stream(
        PdfStream(raw_data=b"/Self Do"),
        {"XObject": {"Self": form}},
        IDENTITY_MATRIX,
        0,
    )

    assert len(state.drawings) == 2
    assert [drawing.xobject_depth for drawing in state.drawings] == [1, 1]
    assert not state.stream_executor.active_streams


def test_nested_form_depth_limit_keeps_the_first_ten_levels() -> None:
    state = internal_state()
    resources: PdfDict = {}
    for _ in range(12):
        form = PdfStream(
            dictionary={"Subtype": "Form", "Resources": resources},
            raw_data=b"0 0 1 1 re f /Child Do",
        )
        resources = {"XObject": {"Child": form}}

    state.consume_stream(PdfStream(raw_data=b"/Child Do"), resources, IDENTITY_MATRIX, 0)

    assert [drawing.xobject_depth for drawing in state.drawings] == list(range(1, 11))
    assert not state.stream_executor.active_streams


def test_reentrant_type3_execution_retains_the_active_parent_stream() -> None:
    state = internal_state()
    page = PdfStream(dictionary={"Subtype": "Form"}, raw_data=b"glyph 0 0 1 1 re f")
    resources: PdfDict = {"XObject": {"Parent": page}}
    decoder = FontDecoder(
        {
            "Subtype": "Type3",
            "CharProcs": {
                "A": PdfStream(raw_data=b"500 0 d0 /Parent Do 0 0 1 1 re f"),
            },
            "FontMatrix": [0.001, 0, 0, 0.001, 0, 0],
        }
    )
    decoder.type3_glyph_names = {65: "A"}
    state.font_widths = (500.0,) * 256

    def glyph(operands: ContentOperands, depth: int) -> None:
        state.internal_render_type3_glyphs(b"A", decoder)
        assert state.stream_executor.active_streams == {state.stream_executor.execution_key(page)}
        assert state.resources is resources

    state.op_handlers["glyph"] = glyph
    state.consume_stream(page, resources, IDENTITY_MATRIX, 0)

    assert [drawing.xobject_depth for drawing in state.drawings] == [1, 0]
    assert not state.stream_executor.active_streams


def test_caught_reentrant_failure_unwinds_only_the_inner_consume_call() -> None:
    state = internal_state()
    page = PdfStream(raw_data=b"nested 0 0 1 1 re S")

    def fail(operands: ContentOperands, depth: int) -> None:
        raise PdfParseError("inner failure")

    def nested(operands: ContentOperands, depth: int) -> None:
        before = state.capture_stream_state()
        with pytest.raises(PdfParseError, match="inner failure"):
            state.consume_stream(PdfStream(raw_data=b"q 3 w fail"), {}, IDENTITY_MATRIX, depth + 1)
        assert state.capture_stream_state() == before
        assert state.stream_executor.active_streams == {state.stream_executor.execution_key(page)}

    state.op_handlers.update(fail=fail, nested=nested)
    state.consume_stream(page, {}, IDENTITY_MATRIX, 0)

    assert [drawing.kind for drawing in state.drawings] == ["stroke"]
    assert state.drawings[0].line_width == 1
    assert not state.stream_executor.active_streams


@pytest.mark.parametrize("form_ending", [b"ET", b"fail"])
def test_parent_text_resumes_at_its_own_position_after_form_text(form_ending: bytes) -> None:
    state = internal_state()

    def fail(operands: ContentOperands, depth: int) -> None:
        raise PdfParseError("failure after captured text")

    state.op_handlers["fail"] = fail
    form = PdfStream(
        dictionary={"Subtype": "Form"},
        raw_data=b"BT 1 0 0 1 50 50 Tm (B) Tj " + form_ending,
    )
    state.consume_stream(
        PdfStream(raw_data=b"BT 1 0 0 1 0 10 Tm (A) Tj /Child Do (C) Tj ET"),
        {"XObject": {"Child": form}},
        IDENTITY_MATRIX,
        0,
    )

    assert [run.text for run in state.runs] == ["A", "B", "C"]
    assert [run.xobject_depth for run in state.runs] == [0, 1, 0]
    assert [(run.x0, run.y0) for run in state.runs] == [(0, 10), (50, 50), (12, 10)]
    assert not state.stream_executor.active_streams

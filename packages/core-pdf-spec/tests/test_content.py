# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.events import ContentSink
from core_pdf_spec.s_07_content.operations import dispatch_operations, iter_content_operations
from core_pdf_spec.s_07_content.state import TextState
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph
from core_pdf_spec.types import PdfName, PdfString


class EventSink:
    def __init__(self) -> None:
        self.colors: list[tuple[float, ...] | None] = []
        self.saved = 0

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

    def save_graphics(self, state: TextState) -> None:
        self.saved += 1

    def restore_graphics(self, state: TextState) -> None:
        self.saved -= 1

    def paint_path(self, state: TextState, *args: Any) -> None:
        self.colors.append(state.fill_color)


class EmptyTextFont:
    is_type3 = False
    is_vertical = False
    ascent = 800.0
    descent = -200.0
    fast_widths = (500.0,) * 256

    def decode_glyphs(self, data: bytes) -> tuple[DecodedFontGlyph, ...]:
        return tuple(DecodedFontGlyph(bytes([code]), code, code, code, "", code) for code in data)

    def text_advance_vector(self, data: Any, **kwargs: Any) -> tuple[float, float]:
        return (5.0 * len(data), 0.0)

    def glyph_width(self, code: int) -> float:
        return 500.0


def state_with_sink() -> tuple[TextState, EventSink]:
    sink = EventSink()
    resolver = ObjectResolver(b"", {}, {})
    font: Any = EmptyTextFont()
    return TextState(
        SimpleNamespace(resolver=resolver), cast(ContentSink, sink), lambda *args: font
    ), sink


@pytest.mark.parametrize(
    "content", [b"unknown", b"]", b">>", b"R", b"1 2 cm", b"(x) w", b"[1 /Bad] TJ", b"1 2", b"EX"]
)
def test_invalid_content_rejected(content: bytes) -> None:
    with pytest.raises(PdfParseError):
        list(iter_content_operations(PdfLexer(content)))


def test_compatibility_section_ignores_unknown_operators() -> None:
    operations = list(iter_content_operations(PdfLexer(b"BX 1 2 extension EX 3 w")))
    assert [name for name, _ in operations] == ["BX", "EX", "w"]


def test_indexed_components_remain_palette_indices() -> None:
    state, sink = state_with_sink()
    state.resources = {
        "ColorSpace": {
            "Palette": [
                PdfName.of("Indexed"),
                PdfName.of("DeviceRGB"),
                2,
                b"\0\0\0\xff\0\0\0\xff\0",
            ]
        }
    }
    dispatch_operations(
        PdfLexer(b"/Palette cs 2 sc 0 0 10 10 re f"), state.get_operation_handler, 0
    )
    assert state.fill_color == (2.0,)
    assert sink.colors == [(2.0,)]


def test_missing_font_resource_raises_without_substitution() -> None:
    state, _ = state_with_sink()
    with pytest.raises(PdfParseError, match="font resource"):
        dispatch_operations(PdfLexer(b"/Missing 12 Tf"), state.get_operation_handler, 0)


def test_failed_child_stream_restores_parent_graphics_state() -> None:
    state, sink = state_with_sink()
    child = PdfStream(
        raw_data=b"q 0.75 g invalid",
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
    )
    resources: PdfDict = {"XObject": {"Child": child}}
    initial = state.capture_stream_state()
    with pytest.raises(PdfParseError, match="unknown content"):
        state.consume_stream(
            PdfStream(raw_data=b"0.25 g /Child Do", dictionary={}), resources, IDENTITY_MATRIX, 0
        )
    assert state.capture_stream_state() == initial
    assert sink.saved == 0
    assert not state.stream_executor.active_streams


def test_compatibility_scope_survives_nested_stream_suspension() -> None:
    state, _ = state_with_sink()
    child = PdfStream(
        raw_data=b"", dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]}
    )
    state.consume_stream(
        PdfStream(raw_data=b"BX /Child Do extension EX", dictionary={}),
        {"XObject": {"Child": child}},
        IDENTITY_MATRIX,
        0,
    )


def test_empty_unicode_still_advances_text() -> None:
    state, _ = state_with_sink()
    font: Any = EmptyTextFont()
    state.current_decoder = font
    state.append_text(data=b"ab")
    assert state.text_matrix.e == 10.0


def test_spec_has_no_reader_form_depth_limit() -> None:
    state, sink = state_with_sink()
    objects: PdfDict = {}
    for depth in range(12):
        following = f"/Child{depth + 1} Do".encode() if depth < 11 else b""
        objects[f"Child{depth}"] = PdfStream(
            dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
            raw_data=b"0 0 10 10 re f " + following,
        )
    state.consume_stream(
        PdfStream(raw_data=b"/Child0 Do"), {"XObject": objects}, IDENTITY_MATRIX, 0
    )
    assert len(sink.colors) == 12


@pytest.mark.parametrize("content", [b"q", b"/Span BMC"])
def test_unbalanced_scopes_are_rejected_and_restored(content: bytes) -> None:
    state, sink = state_with_sink()
    with pytest.raises(PdfParseError, match="unbalanced"):
        state.consume_stream(PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0)
    assert not state.stack
    assert not state.marked_content_stack
    assert sink.saved == 0


def test_uncolored_type3_glyph_ignores_color_setting() -> None:
    state, _ = state_with_sink()
    state.type3_uncolored = True
    dispatch_operations(PdfLexer(b"1 0 0 rg /DeviceRGB cs"), state.get_operation_handler, 0)
    assert state.fill_color == (0.0,)
    assert state.fill_color_space == "DeviceGray"


@pytest.mark.parametrize(
    "content", [b"/Missing cs", b"/DeviceRGB cs 1 sc", b"/Pattern cs /Missing scn"]
)
def test_invalid_color_resources_and_operands_rejected(content: bytes) -> None:
    state, _ = state_with_sink()
    with pytest.raises(PdfParseError):
        dispatch_operations(PdfLexer(content), state.get_operation_handler, 0)


def test_inline_image_rejects_declared_length_mismatch() -> None:
    with pytest.raises(PdfParseError, match="data length"):
        list(iter_content_operations(PdfLexer(b"BI /W 3 /H 1 /BPC 8 /CS /G ID a EI")))


def test_lab_components_use_numeric_pdf_ranges() -> None:
    state, _ = state_with_sink()
    params: PdfDict = {"WhitePoint": [1, 1, 1], "Range": [-2, 2, -3, 3]}
    state.resources = {"ColorSpace": {"Test": [PdfName.of("Lab"), params]}}
    dispatch_operations(PdfLexer(b"/Test cs 150 4 -4 sc"), state.get_operation_handler, 0)
    assert state.fill_color == (100.0, 2.0, -3.0)
    params["Range"] = [PdfString(b"-2"), 2, -3, 3]
    dispatch_operations(PdfLexer(b"/Test cs"), state.get_operation_handler, 0)
    with pytest.raises(PdfParseError, match="PDF number"):
        dispatch_operations(PdfLexer(b"50 0 0 sc"), state.get_operation_handler, 0)

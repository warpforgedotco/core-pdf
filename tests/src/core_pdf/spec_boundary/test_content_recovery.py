# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.capture.recovery import iter_content_operations
from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.types import PdfName, PdfString


def new_state() -> TextState:
    resolver = ObjectResolver(b"", {})
    return TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))


def internal_execute(state: TextState, content: bytes) -> None:
    lexer = PdfLexer(content)
    try:
        for name, operands in iter_content_operations(lexer):
            assert state.execute_operation(name, operands, 0) is None
    finally:
        lexer.close()


class IntSubclass(int):
    pass


class BytesSubclass(bytes):
    pass


class StrSubclass(str):
    pass


def tj_state_with_recording(
    monkeypatch: pytest.MonkeyPatch, *, vertical: bool = False
) -> tuple[TextState, list[tuple[bytes, float, float]]]:
    state = new_state()
    state.graphics.current_decoder = cast(Any, SimpleNamespace(is_vertical=vertical))
    state.text_matrix = Matrix(2.0, 3.0, 5.0, 7.0, 11.0, 13.0)
    state.graphics.font_size = 100
    state.graphics.horizontal_scale = 100
    shown: list[tuple[bytes, float, float]] = []

    def append_text(*, data: bytes, decoder: object) -> None:
        shown.append((data, state.text_matrix.e, state.text_matrix.f))
        state.text_matrix = state.text_matrix._replace(
            e=state.text_matrix.e + 1.0, f=state.text_matrix.f + 2.0
        )

    monkeypatch.setattr(state, "append_text", append_text)
    return state, shown


@pytest.mark.parametrize(
    ("vertical", "positions", "final"),
    [
        (False, [(11, 13), (10, 12), (11.5, 14.75)], (12.5, 16.75)),
        (True, [(11, 13), (7, 8), (9.25, 11.75)], (10.25, 13.75)),
    ],
)
def test_reader_tj_recovers_strings_without_changing_adjustments(
    monkeypatch: pytest.MonkeyPatch,
    vertical: bool,
    positions: list[tuple[float, float]],
    final: tuple[float, float],
) -> None:
    state, shown = tj_state_with_recording(monkeypatch, vertical=vertical)
    state.append_tj_array(
        [
            PdfString(b"a"),
            b"b",
            10,
            "c",
            True,
            None,
            PdfName.of("Ignored"),
            IntSubclass(100),
            BytesSubclass(b"ignored"),
            StrSubclass("ignored"),
            -2.5,
            b"d",
        ]
    )
    assert [data for data, _, _ in shown] == [b"ab", b"c", b"d"]
    assert [(x, y) for _, x, y in shown] == positions
    assert (state.text_matrix.e, state.text_matrix.f) == final


@pytest.mark.parametrize("array", [None, b"text", "text", 1, [], ()])
def test_reader_ignores_nonarray_or_empty_tj_without_loading_font(array: object) -> None:
    state = new_state()
    state.append_tj_array(array)
    assert state.graphics.current_decoder is None
    assert not state.glyphs


def test_reader_tj_keeps_prior_text_when_later_string_encoding_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, shown = tj_state_with_recording(monkeypatch)
    with pytest.raises(UnicodeEncodeError):
        state.append_tj_array([b"a", 10, b"b", "€"])
    assert shown == [(b"a", 11, 13)]
    assert (state.text_matrix.e, state.text_matrix.f) == (12, 15)


@pytest.mark.parametrize("delimiter", [b"]", b">", b">>", b")", b"{", b"}"])
def test_reader_skips_unknown_operators_and_stray_delimiters(delimiter: bytes) -> None:
    state = new_state()
    internal_execute(state, delimiter + b" 1 invalid >> 3 w")
    assert state.graphics.line_width == 3


@pytest.mark.parametrize("content", [b"EX BX extension", b"q BX Q extension EX EX"])
def test_reader_keeps_tolerant_compatibility_scope_execution(content: bytes) -> None:
    state = new_state()
    state.stream_executor.consume(
        PdfStream(raw_data=content + b" 0 0 10 10 re f"), {}, IDENTITY_MATRIX, 0
    )
    assert len(state.drawings) == 1
    assert not state.stack
    assert state.compatibility_depth == 0


def test_reader_execution_tracks_and_clamps_compatibility_depth() -> None:
    state = new_state()
    internal_execute(state, b"BX BX EX")
    assert state.compatibility_depth == 1
    internal_execute(state, b"EX EX EX")
    assert state.compatibility_depth == 0


def test_reader_content_preserves_legacy_real_overflow_acceptance() -> None:
    state = new_state()
    internal_execute(state, b"9" * 400 + b".0 w")
    assert state.graphics.line_width == float("inf")


def test_reader_child_scope_failure_does_not_change_parent_recovery() -> None:
    state = new_state()
    child = PdfStream(
        raw_data=b"EX BX extension",
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
    )
    state.stream_executor.consume(
        PdfStream(raw_data=b"BX /Child Do extension EX 0 0 10 10 re f"),
        {"XObject": {"Child": child}},
        IDENTITY_MATRIX,
        0,
    )
    assert len(state.drawings) == 1
    assert state.compatibility_depth == 0
    assert not state.capture_frames


def test_reader_caps_operands_and_preserves_existing_first_values() -> None:
    lexer = PdfLexer(b" ".join([b"1"] * 20) + b" custom")
    try:
        assert list(iter_content_operations(lexer)) == [("custom", (1,) * 16)]
    finally:
        lexer.close()


def test_reader_supplies_missing_font() -> None:
    state = new_state()
    internal_execute(state, b"BT /Missing 12 Tf (hello) Tj ET")
    assert state.graphics.current_decoder is not None
    assert state.text_matrix.e > 0


def test_failed_form_unwinds_local_saves_and_resumes_parent() -> None:
    state = new_state()
    child = PdfStream(
        raw_data=b"Q q 0.75 g BI /W 1 /H 1 ID",
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
    )
    state.stream_executor.consume(
        PdfStream(raw_data=b"0.25 g /Child Do 0 0 10 10 re f", dictionary={}),
        {"XObject": {"Child": child}},
        IDENTITY_MATRIX,
        0,
    )
    assert state.drawings[-1].fill == (0.25,)
    assert not state.stack
    assert not state.capture_graphics_stack
    assert not state.capture_frames


def test_indexed_color_converts_at_capture_boundary() -> None:
    state = new_state()
    resources: PdfDict = {
        "ColorSpace": {
            "Palette": [
                PdfName.of("Indexed"),
                PdfName.of("DeviceRGB"),
                2,
                b"\0\0\0\xff\0\0\0\xff\0",
            ]
        }
    }
    state.resources = resources
    internal_execute(state, b"/Palette cs 2 sc 0 0 10 10 re f")
    assert state.graphics.fill_color == (2.0,)
    assert state.drawings[-1].fill == pytest.approx((0.0, 1.0, 0.0))


def test_spot_color_converts_at_capture_boundary() -> None:
    state = new_state()
    tint: PdfDict = {"FunctionType": 2, "Domain": [0, 1], "C0": [1, 1, 1], "C1": [1, 0, 0], "N": 1}
    state.resources = {
        "ColorSpace": {
            "Spot": [PdfName.of("Separation"), PdfName.of("Ink"), PdfName.of("DeviceRGB"), tint]
        }
    }
    internal_execute(state, b"/Spot cs 1 scn 0 0 10 10 re f")
    assert state.graphics.fill_color == (1.0,)
    assert state.drawings[-1].fill == pytest.approx((1.0, 0.0, 0.0))


def test_reader_applies_form_depth_limit() -> None:
    state = new_state()
    objects: PdfDict = {}
    for depth in range(12):
        following = f"/Child{depth + 1} Do".encode() if depth < 11 else b""
        objects[f"Child{depth}"] = PdfStream(
            dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
            raw_data=b"0 0 10 10 re f " + following,
        )
    state.stream_executor.consume(
        PdfStream(raw_data=b"/Child0 Do"), {"XObject": objects}, IDENTITY_MATRIX, 0
    )
    assert len(state.drawings) == 10
    assert not state.capture_graphics_stack


def test_reader_recovers_oversized_form_matrix() -> None:
    state = new_state()
    child = PdfStream(
        raw_data=b"0 0 10 10 re f",
        dictionary={
            "Subtype": PdfName.of("Form"),
            "BBox": [0, 0, 10, 10],
            "Matrix": [1, 0, 0, 1, 20, 0, 99],
        },
    )
    state.stream_executor.consume(
        PdfStream(raw_data=b"/Child Do"), {"XObject": {"Child": child}}, IDENTITY_MATRIX, 0
    )
    path = state.drawings[-1].path
    assert path is not None
    bbox = path.bbox()
    assert bbox is not None
    assert bbox[0] == 20


def test_reader_retains_inline_image_with_damaged_dimensions() -> None:
    state = new_state()
    internal_execute(state, b"BI /W 3 /H 1 /BPC 8 /CS /G ID a EI")
    assert state.inline_images[-1].data == b"a"

# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace

import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.capture.recovery import dispatch_operations
from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName


def new_state() -> TextState:
    resolver = ObjectResolver(b"", {}, {})
    return TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))


@pytest.mark.parametrize("delimiter", [b"]", b">", b">>", b")", b"{", b"}"])
def test_reader_skips_unknown_operators_and_stray_delimiters(delimiter: bytes) -> None:
    state = new_state()
    dispatch_operations(PdfLexer(delimiter + b" 1 invalid >> 3 w"), state.op_handlers.get, 0)
    assert state.line_width == 3


def test_reader_caps_operands_and_preserves_existing_first_values() -> None:
    seen = []
    dispatch_operations(
        PdfLexer(b" ".join([b"1"] * 20) + b" custom"),
        lambda op: lambda operands, depth: seen.append(operands),
        0,
    )
    assert seen == [(1,) * 16]


def test_reader_supplies_missing_font() -> None:
    state = new_state()
    dispatch_operations(PdfLexer(b"BT /Missing 12 Tf (hello) Tj ET"), state.op_handlers.get, 0)
    assert state.current_decoder is not None
    assert state.text_matrix.e > 0


def test_failed_form_unwinds_local_saves_and_resumes_parent() -> None:
    state = new_state()
    child = PdfStream(
        raw_data=b"Q q 0.75 g BI /W 1 /H 1 ID",
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
    )
    state.consume_stream(
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
    dispatch_operations(PdfLexer(b"/Palette cs 2 sc 0 0 10 10 re f"), state.op_handlers.get, 0)
    assert state.fill_color == (2.0,)
    assert state.drawings[-1].fill == pytest.approx((0.0, 1.0, 0.0))


def test_spot_color_converts_at_capture_boundary() -> None:
    state = new_state()
    tint: PdfDict = {"FunctionType": 2, "Domain": [0, 1], "C0": [1, 1, 1], "C1": [1, 0, 0], "N": 1}
    state.resources = {
        "ColorSpace": {
            "Spot": [PdfName.of("Separation"), PdfName.of("Ink"), PdfName.of("DeviceRGB"), tint]
        }
    }
    dispatch_operations(PdfLexer(b"/Spot cs 1 scn 0 0 10 10 re f"), state.op_handlers.get, 0)
    assert state.fill_color == (1.0,)
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
    state.consume_stream(
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
    state.consume_stream(
        PdfStream(raw_data=b"/Child Do"), {"XObject": {"Child": child}}, IDENTITY_MATRIX, 0
    )
    path = state.drawings[-1].path
    assert path is not None
    bbox = path.bbox()
    assert bbox is not None
    assert bbox[0] == 20


def test_reader_retains_inline_image_with_damaged_dimensions() -> None:
    state = new_state()
    dispatch_operations(PdfLexer(b"BI /W 3 /H 1 /BPC 8 /CS /G ID a EI"), state.op_handlers.get, 0)
    assert state.inline_images[-1].data == b"a"

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl.primitives import PdfName
from core_pdf.impl.spec.s_07_content.capture import type3_glyph_names
from core_pdf.impl.spec.s_07_content.state import TextState
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder
from tests.helpers.resolvers import IdentityResolver


def internal_type3_state(program: PdfStream) -> tuple[TextState, FontDecoder]:
    document = cast(
        Any,
        SimpleNamespace(
            resolver=IdentityResolver(),
        ),
    )
    state = TextState(document)
    state.font_widths = (500.0,) * 256
    font = {
        "Subtype": "Type3",
        "CharProcs": {"A": program},
        "FontMatrix": [0.001, 0, 0, 0.001, 0, 0],
    }
    decoder = FontDecoder(font)
    decoder.type3_glyph_names = {65: "A"}
    return state, decoder


def test_type3_names_include_base_encoding_and_differences() -> None:
    font = {
        "Subtype": "Type3",
        "Encoding": {
            "BaseEncoding": "StandardEncoding",
            "Differences": [65, PdfName(b"custom")],
        },
    }
    names = type3_glyph_names(font, FontDecoder(font))

    assert names[32] == "space"
    assert names[65] == "custom"
    assert 1 not in names


def test_type3_win_ansi_euro_char_proc_is_rendered() -> None:
    stream = PdfStream(raw_data=b"500 0 d0 0 0 1 1 re f")
    document = cast(
        Any,
        SimpleNamespace(
            resolver=IdentityResolver(),
        ),
    )
    state = TextState(document)
    state.font_widths = (500.0,) * 256
    font = {
        "Subtype": "Type3",
        "Encoding": "WinAnsiEncoding",
        "CharProcs": {"Euro": stream},
        "FontMatrix": [0.001, 0, 0, 0.001, 0, 0],
    }
    decoder = FontDecoder(font)

    state.internal_render_type3_glyphs(b"\x80", decoder)

    assert decoder.type3_glyph_names is not None
    assert 0 not in decoder.type3_glyph_names
    assert decoder.type3_glyph_names[128] == "Euro"
    assert len(state.drawings) == 1
    assert state.drawings[0].path is not None
    # ISO 32000-1 9.6.5 concatenates the font matrix with the text space in
    # effect, and 9.4.4 NOTE 2 puts Tfs in that matrix. The 1x1 glyph box is
    # scaled by FontMatrix (0.001) and then by the default Tfs of 12.
    assert state.drawings[0].path.bbox() == pytest.approx((0.0, 0.0, 0.012, 0.012))


def test_repeated_type3_char_proc_does_not_leak_stream_state() -> None:
    stream = PdfStream(raw_data=b"500 0 0 0 1 1 d1 q 0 0 m 1 0 l 1 1 l 0 1 l h f Q")
    state, decoder = internal_type3_state(stream)

    state.internal_render_type3_glyphs(b"AAA", decoder)

    assert len(state.drawings) == 3
    assert not state.active_streams
    assert not state.stack
    assert not state.clip_scope_stack


def test_type3_char_proc_with_unresolved_xobject_does_not_leak_stream_state() -> None:
    stream = PdfStream(raw_data=b"/Nested Do")
    state, decoder = internal_type3_state(stream)

    state.internal_render_type3_glyphs(b"AA", decoder)

    assert not state.drawings
    assert not state.active_streams


def test_type3_dash_operator_preserves_dash() -> None:
    stream = PdfStream(raw_data=b"500 0 d0 [3 2] 1 d 0 0 m 10 0 l S")
    state, decoder = internal_type3_state(stream)

    state.internal_render_type3_glyphs(b"A", decoder)

    assert len(state.drawings) == 1
    dash_pattern = state.drawings[0].dash_pattern
    assert dash_pattern is not None
    dash_array, dash_phase = dash_pattern
    # Same scaling as above: FontMatrix 0.001 then the default Tfs of 12.
    assert dash_array == pytest.approx([0.036, 0.024])
    assert dash_phase == pytest.approx(0.012)


@pytest.mark.parametrize(
    ("metrics_operator", "expected_fill"),
    [
        ("500 0 d0", (1.0, 0.0, 0.0)),
        ("500 0 0 0 1 1 d1", (0.0, 1.0, 0.0)),
    ],
)
def test_type3_colorized_and_uncolored_glyph_semantics(
    metrics_operator: str, expected_fill: tuple[float, ...]
) -> None:
    """Honor d0 colors while d1 glyphs inherit the text fill color."""
    stream = PdfStream(raw_data=(f"{metrics_operator} 1 0 0 rg 0 0 1 1 re f").encode("ascii"))
    state, decoder = internal_type3_state(stream)
    state.fill_color = (0.0, 1.0, 0.0)

    state.internal_render_type3_glyphs(b"A", decoder)

    assert len(state.drawings) == 1
    assert state.drawings[0].fill == expected_fill
    assert state.type3_uncolored is False


@pytest.mark.parametrize(
    ("operator", "operands", "attribute", "expected_space"),
    [
        ("op_g", (0,), "fill_color_space", "DeviceGray"),
        ("op_rg", (0, 0, 1), "fill_color_space", "DeviceRGB"),
        ("op_k", (0, 0, 0, 1), "fill_color_space", "DeviceCMYK"),
        ("op_G", (0,), "stroke_color_space", "DeviceGray"),
        ("op_RG", (0, 0, 1), "stroke_color_space", "DeviceRGB"),
        ("op_K", (0, 0, 0, 1), "stroke_color_space", "DeviceCMYK"),
    ],
)
def test_uncolored_type3_glyph_ignores_device_color_operators(
    operator: str, operands: tuple[float, ...], attribute: str, expected_space: str
) -> None:
    """9.6.5.2: colour operators are ignored inside a d1 glyph, label included.

    The colour setters already refuse to move the colour itself, so a colour
    space that moved anyway would be a label describing a colour it does not
    belong to. Nothing downstream reads the pair together today -- `scn` is
    blocked by the same flag, and the stream frame restores both on the way
    out -- so this pins an invariant rather than a visible defect.
    """
    state, _ = internal_type3_state(PdfStream(raw_data=b"500 0 d0"))
    state.fill_color = (0.0, 1.0, 0.0)
    state.stroke_color = (0.0, 1.0, 0.0)
    state.fill_color_space = "DeviceRGB"
    state.stroke_color_space = "DeviceRGB"
    handler = getattr(state, operator)

    state.type3_uncolored = True
    handler(cast(Any, list(operands)), 0)

    assert getattr(state, attribute) == "DeviceRGB"
    assert state.fill_color == (0.0, 1.0, 0.0)
    assert state.stroke_color == (0.0, 1.0, 0.0)

    state.type3_uncolored = False
    handler(cast(Any, list(operands)), 0)

    assert getattr(state, attribute) == expected_space


def test_uncolored_type3_glyph_ignores_the_color_space_operators() -> None:
    """`cs` and `CS` are colour operators too, and were never guarded."""
    state, _ = internal_type3_state(PdfStream(raw_data=b"500 0 d0"))
    state.fill_color_space = "DeviceRGB"
    state.stroke_color_space = "DeviceRGB"
    state.type3_uncolored = True

    state.op_cs(cast(Any, [PdfName(b"DeviceCMYK")]), 0)
    state.op_CS(cast(Any, [PdfName(b"DeviceCMYK")]), 0)

    assert state.fill_color_space == "DeviceRGB"
    assert state.stroke_color_space == "DeviceRGB"


def test_colored_type3_glyph_still_takes_its_own_color() -> None:
    """The d0 counterpart must keep working: a coloured glyph owns its colour."""
    stream = PdfStream(raw_data=b"500 0 d0 0 0 0 1 k 0 0 1 1 re f")
    state, decoder = internal_type3_state(stream)
    state.fill_color = (0.0, 1.0, 0.0)

    state.internal_render_type3_glyphs(b"A", decoder)

    assert len(state.drawings) == 1
    assert state.drawings[0].fill == (0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("font_size", "expected"),
    [(1.0, 0.001), (12.0, 0.012), (100.0, 0.1)],
)
def test_type3_glyph_ctm_scales_with_the_font_size(font_size: float, expected: float) -> None:
    """ISO 32000-1 9.6.5 concatenates the font matrix with the text space in
    effect, and 9.4.4 NOTE 2 puts Tfs (and Th, Trise) in that matrix.

    Tfs was absent from the glyph CTM and the font matrix was concatenated on
    the wrong side, so every Type 3 glyph painted at FontMatrix scale near the
    origin, the same size whatever the font size.
    """
    stream = PdfStream(raw_data=b"1000 0 d0 0 0 1 1 re f")
    state, decoder = internal_type3_state(stream)
    state.font_size = font_size

    state.internal_render_type3_glyphs(b"A", decoder)

    assert len(state.drawings) == 1
    path = state.drawings[0].path
    assert path is not None
    assert path.bbox() == pytest.approx((0.0, 0.0, expected, expected))


def test_type3_glyph_is_not_painted_in_render_mode_3() -> None:
    """9.3.6: "Only a value of 3 for text rendering mode shall have any effect
    on text displayed in a Type 3 font", and Table 106 makes mode 3 invisible.

    Mode 7 deliberately still paints here: for a Type 3 font the clause says
    only mode 3 has an effect.
    """
    stream = PdfStream(raw_data=b"1000 0 d0 0 0 1 1 re f")

    state, decoder = internal_type3_state(stream)
    state.render_mode = 3
    state.internal_render_type3_glyphs(b"A", decoder)
    assert state.drawings == []

    state, decoder = internal_type3_state(stream)
    state.render_mode = 7
    state.internal_render_type3_glyphs(b"A", decoder)
    assert len(state.drawings) == 1

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl.engine.spec.s_07_content.capture import type3_glyph_names
from core_pdf.impl.engine.spec.s_07_content.state import TextState
from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder, Type3CharProcProgram
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.primitives import PdfName


def internal_type3_state(program: PdfStream) -> tuple[TextState, FontDecoder]:
    resolver = SimpleNamespace(
        kw_cache={},
        resolve=lambda value: value,
        resolve_name=lambda value: None,
    )
    document = cast(Any, SimpleNamespace(resolver=resolver, decoder_cache={}))
    state = TextState(document, {})
    state.font_widths = (500.0,) * 256
    font = {
        "Subtype": "Type3",
        "CharProcs": {"A": program},
        "FontMatrix": [0.001, 0, 0, 0.001, 0, 0],
    }
    decoder = FontDecoder(font)
    decoder.type3_glyph_names = {65: "A"}
    return state, decoder


def internal_drawing_signature(state: TextState) -> list[tuple[object, ...]]:
    return [
        (
            drawing.kind,
            drawing.seqno,
            drawing.fill,
            drawing.fill_opacity,
            drawing.fill_rule,
            drawing.path.bbox() if drawing.path is not None else None,
            tuple((tuple(subpath.points), subpath.closed) for subpath in drawing.path.subpaths)
            if drawing.path is not None
            else (),
        )
        for drawing in state.drawings
    ]


def test_type3_names_include_base_encoding_and_differences() -> None:
    decoder = SimpleNamespace(base_encoding="StandardEncoding")
    names = type3_glyph_names(
        {"Encoding": {"BaseEncoding": "StandardEncoding", "Differences": [65, PdfName(b"custom")]}},
        decoder,
    )

    assert names[32] == "space"
    assert names[65] == "custom"
    assert 1 not in names


def test_type3_char_proc_compiles_once_and_replays_exactly() -> None:
    stream = PdfStream(raw_data=b"500 0 0 0 1 1 d1 q 0 0 m 1 0 l 1 1 l 0 1 l h f Q")
    compiled_state, compiled_decoder = internal_type3_state(stream)

    compiled_state._render_type3_glyphs_impl(b"AAA", compiled_decoder)

    fallback_state, fallback_decoder = internal_type3_state(stream)
    fallback_decoder.type3_charproc_cache[65] = Type3CharProcProgram(stream, None)
    fallback_state._render_type3_glyphs_impl(b"AAA", fallback_decoder)

    assert internal_drawing_signature(compiled_state) == internal_drawing_signature(fallback_state)
    assert compiled_decoder.type3_charproc_cache_misses == 1
    assert compiled_decoder.type3_charproc_cache_hits == 2
    assert compiled_decoder.type3_charproc_compiled_programs == 1
    assert compiled_decoder.type3_charproc_compiled_operations == 9
    assert compiled_decoder.type3_charproc_unsafe_fallbacks == 0
    assert fallback_decoder.type3_charproc_unsafe_fallbacks == 3
    assert not compiled_state.active_streams
    assert not compiled_state.stack
    assert not compiled_state.clip_scope_stack


def test_type3_char_proc_caches_unsupported_stream_fallback() -> None:
    stream = PdfStream(raw_data=b"/Nested Do")
    state, decoder = internal_type3_state(stream)

    state._render_type3_glyphs_impl(b"AA", decoder)

    cached = decoder.type3_charproc_cache[65]
    assert cached is not None
    assert cached.operations is None
    assert decoder.type3_charproc_cache_misses == 1
    assert decoder.type3_charproc_cache_hits == 1
    assert decoder.type3_charproc_compiled_programs == 0
    assert decoder.type3_charproc_unsafe_fallbacks == 2
    assert not state.active_streams


def test_type3_dash_operator_uses_safe_fallback_and_preserves_dash() -> None:
    # Dash operands contain an array and cannot use the direct replay representation.
    stream = PdfStream(raw_data=b"500 0 d0 [3 2] 1 d 0 0 m 10 0 l S")
    state, decoder = internal_type3_state(stream)

    state._render_type3_glyphs_impl(b"A", decoder)

    cached = decoder.type3_charproc_cache[65]
    assert cached is not None
    assert cached.operations is None
    assert decoder.type3_charproc_unsafe_fallbacks == 1
    assert len(state.drawings) == 1
    dash_pattern = state.drawings[0].dash_pattern
    assert dash_pattern is not None
    dash_array, dash_phase = dash_pattern
    assert dash_array == pytest.approx([0.003, 0.002])
    assert dash_phase == pytest.approx(0.001)


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

    state._render_type3_glyphs_impl(b"A", decoder)

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

    state._render_type3_glyphs_impl(b"A", decoder)

    assert len(state.drawings) == 1
    assert state.drawings[0].fill == (0.0, 0.0, 0.0, 1.0)

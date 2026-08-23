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

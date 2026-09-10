"""Shared valid semantics and reader-owned recovery for content values."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.capture.program import CapturedProgram
from core_pdf.impl._impl.capture.recovery import dispatch_operations
from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl._impl.fonts.decoder import FontDecoder
from core_pdf.impl._impl.render.commands import append_captured_program, internal_append_glyph_paint
from core_pdf.impl._impl.render.display import DisplayList
from core_pdf_spec.s_07_content.inline_images import InlineImage
from core_pdf_spec.s_07_content.marked_content import MarkedContentEntry
from core_pdf_spec.s_07_content.patterns import TilingPattern
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.color_spec import ImageColorSpec
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.types import PdfName, PdfReference


def internal_state() -> TextState:
    resolver = ObjectResolver(b"", {}, {})
    return TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))


def internal_execute(state: TextState, content: bytes) -> None:
    dispatch_operations(PdfLexer(content), state.op_handlers.get, 0)


def test_reader_matrix_resolution_retains_truncation_and_default_policies() -> None:
    state = internal_state()
    resolver = cast(ObjectResolver, state.document.resolver)
    resolver.objects[key_for(1, 0)] = [1, 0, 0, 1, PdfReference(2, 0), 6, 99]
    resolver.objects[key_for(2, 0)] = 5
    assert state.matrix_operand(PdfReference(1, 0), "form") == Matrix(1, 0, 0, 1, 5, 6)
    assert state.matrix_operand(PdfReference(1, 0), "pattern") == IDENTITY_MATRIX
    assert state.matrix_operand([1], "pattern") == IDENTITY_MATRIX
    with pytest.raises(ValueError, match="matrix"):
        state.matrix_operand([1], "form")


@pytest.mark.parametrize(
    "value", [None, [1], [1, 0, 0, 1, float("inf"), 0], [1, 0, 0, 1, 10**400, 0]]
)
def test_reader_font_matrix_uses_existing_default_for_invalid_values(value: object) -> None:
    font = FontDecoder.__new__(FontDecoder)
    font.font = {"FontMatrix": value}
    assert font.font_matrix == Matrix(0.001, 0, 0, 0.001, 0, 0)
    font.font["FontMatrix"] = [0.002, 0, 0, 0.003, 4, 5]
    assert font.font_matrix == Matrix(0.002, 0, 0, 0.003, 4, 5)


def test_reader_color_selection_inherits_reserved_names_and_initialization() -> None:
    state = internal_state()
    state.resources = {"ColorSpace": {"DeviceRGB": PdfName.of("DeviceGray")}}
    internal_execute(state, b"1 0 0 rg /DeviceRGB cs")
    assert state.fill_color_space == "DeviceRGB"
    assert state.fill_color == (0, 0, 0)
    internal_execute(state, b"/DeviceCMYK cs")
    assert state.fill_color == (0, 0, 0, 1)
    internal_execute(state, b"/Pattern cs")
    assert state.fill_color is None
    assert state.fill_pattern is None


def test_reader_still_tolerates_special_space_sc_and_wrong_component_counts() -> None:
    state = internal_state()
    state.fill_color_space = "Separation"
    state.fill_color_spec = ImageColorSpec("Separation", {})
    internal_execute(state, b"0.5 sc")
    assert state.fill_color == (0.5,)
    internal_execute(state, b"/DeviceRGB cs 0.5 sc /Unknown cs")
    assert state.fill_color == (0.5,)
    assert state.fill_color_space == "Unknown"


def test_reader_pattern_keeps_missing_painttype_and_matrix_fallbacks() -> None:
    state = internal_state()
    pattern = PdfStream(
        raw_data=b"",
        dictionary={
            "PatternType": 1,
            "BBox": [0, 0, 1, 1],
            "XStep": 1,
            "YStep": 1,
            "Matrix": [1],
        },
    )
    state.resources = {"Pattern": {"P": pattern}}
    internal_execute(state, b"/Pattern cs /P scn")
    assert isinstance(state.fill_pattern, TilingPattern)
    assert state.fill_pattern.paint_type == 1
    assert state.fill_pattern.matrix == IDENTITY_MATRIX


def test_reader_pattern_projects_the_retained_indexed_base_space() -> None:
    state = internal_state()
    pattern = PdfStream(
        raw_data=b"0 0 1 1 re f",
        dictionary={
            "PatternType": 1,
            "PaintType": 2,
            "TilingType": 1,
            "BBox": [0, 0, 1, 1],
            "XStep": 1,
            "YStep": 1,
            "Resources": {},
        },
    )
    state.resources = {
        "ColorSpace": {
            "P": [
                PdfName.of("Pattern"),
                [
                    PdfName.of("Indexed"),
                    PdfName.of("DeviceRGB"),
                    2,
                    b"\x00\x00\x00\xff\x00\x00\x00\xff\x00",
                ],
            ]
        },
        "Pattern": {"Tile": pattern},
    }
    internal_execute(state, b"/P cs 2 /Tile scn")
    assert isinstance(state.fill_pattern, TilingPattern)
    assert state.fill_pattern.base_color == (2,)
    captured = state.capture_pattern(state.fill_pattern)
    assert captured is not None
    assert cast(Any, captured).drawings[0].fill == (0, 1, 0)


def test_reader_vertical_tj_uses_shared_displacement() -> None:
    state = internal_state()
    state.current_decoder = cast(Any, SimpleNamespace(is_vertical=True))
    state.font_size, state.horizontal_scale = 10, 200
    state.update_text_scales()
    internal_execute(state, b"[100] TJ")
    assert (state.tm_e, state.tm_f) == (0, -1)


def test_reader_glyph_origins_preserve_spacing_at_zero_font_size() -> None:
    state = internal_state()
    font = FontDecoder({"Subtype": PdfName.of("Type1"), "BaseFont": PdfName.of("Helvetica")})
    state.current_decoder = font
    state.font_size, state.char_space, state.word_space = 0, 2, 3
    state.update_text_scales()
    state.update_font_metrics()
    state.append_text(data=b"A A", decoder=font)
    assert [glyph.advance_bbox[0] for glyph in state.glyphs] == [0, 2, 7]
    assert state.tm_e == 9


@pytest.mark.parametrize(
    ("spec", "values", "expected"),
    [
        (
            ImageColorSpec("Lab", {"Range": ["-100", "100", "-100", "100"]}),
            ("50", "20", "-20"),
            (50, 20, -20),
        ),
        (ImageColorSpec("Indexed", {}, hival=3), ("2.5",), (3,)),
    ],
)
def test_reader_coerces_components_before_applying_color_ranges(
    spec: ImageColorSpec, values: tuple[str, ...], expected: tuple[int, ...]
) -> None:
    state = internal_state()
    assert state.normalize_color_components(spec, values) == expected


@pytest.mark.parametrize(
    ("selection", "operator", "expected"),
    [
        (b"/Pattern cs", b"f", []),
        (b"/Pattern CS", b"S", []),
        (b"/Pattern cs", b"B", ["stroke"]),
        (b"/Pattern CS", b"B", ["fill"]),
        (b"/Pattern cs /Pattern CS", b"B", []),
        (b"/Pattern cs", b"S", ["stroke"]),
    ],
)
def test_initial_pattern_removes_only_its_path_paint(
    selection: bytes, operator: bytes, expected: list[str]
) -> None:
    state = internal_state()
    internal_execute(state, selection + b" 0 0 10 10 re " + operator)
    assert [drawing.kind for drawing in state.drawings] == expected


@pytest.mark.parametrize("mode", range(8))
@pytest.mark.parametrize("selection", [b"/Pattern cs", b"/Pattern CS", b"/Pattern cs /Pattern CS"])
def test_initial_pattern_removes_only_its_text_paint_without_losing_clipping(
    mode: int, selection: bytes
) -> None:
    state = internal_state()
    font = FontDecoder({"Subtype": PdfName.of("Type1"), "BaseFont": PdfName.of("Helvetica")})
    state.current_decoder = font
    state.update_font_metrics()
    state.render_mode = mode
    internal_execute(state, selection)
    state.append_text(data=b"A", decoder=font)
    fill = mode in {0, 2, 4, 6} and b" cs" not in selection
    stroke = mode in {1, 2, 5, 6} and b" CS" not in selection
    expected = (2 if fill and stroke else 0 if fill else 1 if stroke else 3) + (
        4 if mode >= 4 else 0
    )
    assert len(state.glyphs) == 1
    assert state.glyphs[0].text_render_mode == expected
    assert state.glyphs[0].visible is (expected not in {3, 7})
    assert state.glyphs[0].clip_glyph is (mode >= 4)
    assert state.render_mode == mode
    assert state.tm_e > 0


def test_nonpainting_pattern_text_retains_clip_and_hidden_text_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = internal_state()
    font = FontDecoder({"Subtype": PdfName.of("Type1"), "BaseFont": PdfName.of("Helvetica")})
    state.current_decoder = font
    state.update_font_metrics()
    state.render_mode = 4
    internal_execute(state, b"/Pattern cs")
    state.append_text(data=b"A", decoder=font)
    glyph = state.glyphs[0]
    glyph.font_decoder = cast(
        Any, SimpleNamespace(glyph_outline=lambda *args: (((0, 0), (1, 0), (1, 1)),))
    )
    display = DisplayList(10, 10)
    clips: list[Any] = []
    assert internal_append_glyph_paint(display, glyph, clips)
    assert not display.items
    assert len(clips) == 1
    assert glyph.visible is False
    assert dict(glyph.provenance)["text_render_mode"] == 4
    monkeypatch.setattr(state, "is_graphics_visible", lambda: False)
    state.append_text(data=b"A", decoder=font)
    hidden = state.glyphs[-1]
    hidden.font_decoder = glyph.font_decoder
    assert hidden.clip_glyph is False
    clips.clear()
    assert internal_append_glyph_paint(display, hidden, clips)
    assert not clips
    assert not display.items


@pytest.mark.parametrize("scope", ["graphics-save", "form"])
@pytest.mark.parametrize("hidden", [False, True])
def test_text_clip_restores_before_paint_outside_its_scope(scope: str, hidden: bool) -> None:
    state = internal_state()
    resources: PdfDict = {
        "Font": {"F": {"Subtype": PdfName.of("Type1"), "BaseFont": PdfName.of("Helvetica")}}
    }
    text = b"/Pattern cs BT /F 12 Tf 4 Tr (A) Tj ET"
    if hidden:
        state.hidden_layers = frozenset({"Hidden"})
        state.marked_content_stack.append(MarkedContentEntry(layer="Hidden"))
    if scope == "graphics-save":
        state.resources = resources
        internal_execute(state, b"q " + text + b" Q")
    else:
        form = PdfStream(
            raw_data=text,
            dictionary={
                "Subtype": PdfName.of("Form"),
                "BBox": [0, 0, 10, 10],
                "Resources": resources,
            },
        )
        state.consume_stream(
            PdfStream(raw_data=b"/F Do"), {"XObject": {"F": form}}, IDENTITY_MATRIX, 0
        )
    state.marked_content_stack.clear()
    internal_execute(state, b"0 0 10 10 re f")
    for glyph in state.glyphs:
        glyph.font_decoder = cast(
            Any, SimpleNamespace(glyph_outline=lambda *args: (((0, 0), (1, 0), (1, 1)),))
        )
    display = DisplayList(20, 20)
    append_captured_program(
        display,
        CapturedProgram(glyphs=tuple(state.glyphs), drawings=tuple(state.drawings)),
        include_text=True,
    )
    kinds = [item.kind for item in display.items]
    assert ("clip" in kinds) is not hidden
    if scope == "graphics-save":
        assert kinds == (["fill"] if hidden else ["state-push", "clip", "state-pop", "fill"])
    else:
        assert kinds[-1] == "fill"
        depth = 0
        for kind in kinds[:-1]:
            if kind == "state-push":
                depth += 1
            elif kind == "state-pop":
                depth -= 1
            elif kind == "clip":
                assert depth > 0
        assert depth == 0
    assert len(state.glyphs) == 1
    assert state.glyphs[0].clip_glyph is not hidden


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("stencil", [False, True])
def test_initial_pattern_suppresses_stencils_but_not_colored_images(
    inline: bool, stencil: bool
) -> None:
    state = internal_state()
    internal_execute(state, b"/Pattern cs")
    dictionary: PdfDict = {"Width": 1, "Height": 1, "BitsPerComponent": 1 if stencil else 8}
    dictionary.update({"ImageMask": True} if stencil else {"ColorSpace": PdfName.of("DeviceGray")})
    if inline:
        state.paint_inline_image(state, InlineImage(dictionary, b"\x00"))
        assert len(state.inline_images) == (0 if stencil else 1)
    else:
        state.paint_image(state, PdfStream(raw_data=b"\x00", dictionary=dictionary))
        assert len(state.drawings) == (0 if stencil else 1)


def test_selecting_pattern_after_initial_color_restores_paint() -> None:
    state = internal_state()
    state.resources = {
        "Pattern": {
            "P": PdfStream(
                raw_data=b"0 0 1 1 re f",
                dictionary={
                    "PatternType": 1,
                    "PaintType": 1,
                    "TilingType": 1,
                    "BBox": [0, 0, 1, 1],
                    "XStep": 1,
                    "YStep": 1,
                    "Resources": {},
                },
            )
        }
    }
    internal_execute(state, b"/Pattern cs 0 0 1 1 re f /P scn 0 0 1 1 re f")
    assert len(state.drawings) == 1
    assert state.drawings[0].kind == "fill"
    assert state.drawings[0].fill_pattern is not None

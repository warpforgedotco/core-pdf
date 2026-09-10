"""Shared matrix, text-displacement and color rules at strict content boundaries."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.patterns import TilingPattern
from core_pdf_spec.s_07_content.state import TextState
from core_pdf_spec.s_07_content.stream_execution import NestedStreamRequest
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.color import (
    color_component_count,
    initial_color_components,
    normalize_color_components,
)
from core_pdf_spec.s_08_graphics.color_spec import ImageColorSpec, color_spec_from_value
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.s_09_fonts.metrics import glyph_advance_vector, text_adjustment_vector
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph
from core_pdf_spec.types import PdfName, PdfReference, PdfString


class Sink:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


def internal_state() -> TextState:
    return TextState(
        SimpleNamespace(resolver=ObjectResolver(b"", {}, {})), cast(Any, Sink()), cast(Any, None)
    )


def internal_pattern(paint_type: int = 1, **entries: Any) -> PdfStream:
    return PdfStream(
        raw_data=b"0 0 1 1 re f",
        dictionary={
            "PatternType": 1,
            "PaintType": paint_type,
            "TilingType": 1,
            "BBox": [0, 0, 1, 1],
            "XStep": 1,
            "YStep": 1,
            "Resources": {},
            **entries,
        },
    )


@pytest.mark.parametrize("value", [True, "1", PdfString(b"1"), float("inf"), float("nan"), 10**400])
def test_matrix_requires_finite_pdf_numbers(value: object) -> None:
    with pytest.raises(ValueError, match="invalid matrix operand"):
        Matrix.from_operand([1, 0, 0, 1, value, 0])


@pytest.mark.parametrize("value", [None, [], [1] * 5, [1] * 7])
def test_matrix_requires_exactly_six_entries(value: object) -> None:
    with pytest.raises(ValueError, match="invalid matrix operand"):
        Matrix.from_operand(value)
    assert Matrix.from_operand([1, 0, 0.5, 1, 5, 6]) == Matrix(1, 0, 0.5, 1, 5, 6)


@pytest.mark.parametrize("indirect", [False, True])
def test_form_and_pattern_share_matrix_resolution(indirect: bool) -> None:
    # ISO 32000-1 7.3.10 and Tables 75/95: arrays and their entries may be indirect.
    state = internal_state()
    resolver = cast(ObjectResolver, state.document.resolver)
    resolver.objects[key_for(1, 0)] = [1, 0, 0, 1, PdfReference(2, 0), 6]
    resolver.objects[key_for(2, 0)] = 5
    matrix = PdfReference(1, 0) if indirect else [1, 0, 0, 1, 5, 6]
    state.resources = {
        "XObject": {
            "F": PdfStream(
                raw_data=b"",
                dictionary={
                    "Subtype": PdfName.of("Form"),
                    "BBox": [0, 0, 1, 1],
                    "Matrix": matrix,
                },
            )
        },
        "Pattern": {"P": internal_pattern(Matrix=matrix)},
    }
    with pytest.raises(NestedStreamRequest) as request:
        state.append_xobject(PdfName.of("F"), 0)
    pattern = state.resolve_pattern_color((PdfName.of("P"),))
    assert isinstance(pattern, TilingPattern)
    assert request.value.frame.ctm == pattern.matrix == Matrix(1, 0, 0, 1, 5, 6)
    assert state.matrix_operand(None, "form") == IDENTITY_MATRIX
    resolver.objects[key_for(3, 0)] = None
    assert state.matrix_operand(PdfReference(3, 0), "pattern") == IDENTITY_MATRIX


@pytest.mark.parametrize(("vertical", "expected"), [(False, (-2, 0)), (True, (0, -1))])
def test_tj_horizontal_scale_applies_only_horizontally(
    vertical: bool, expected: tuple[int, int]
) -> None:
    # ISO 32000-1 9.4.4: Th is absent from the vertical displacement equation.
    assert (
        text_adjustment_vector(100, vertical=vertical, font_size=10, horizontal_scale=200)
        == expected
    )
    state = internal_state()
    state.current_decoder = cast(Any, SimpleNamespace(is_vertical=vertical))
    state.font_size, state.horizontal_scale = 10, 200
    state.update_text_scales()
    state.text_matrix = Matrix(2, 3, 5, 7, 11, 13)
    state.append_tj_array([100])
    dx, dy = expected
    assert (state.tm_e, state.tm_f) == (11 + 2 * dx + 5 * dy, 13 + 3 * dx + 7 * dy)


class Font:
    is_type3 = True
    is_vertical = False
    font_matrix = Matrix(0.001, 0, 0, 0.001, 0, 0)
    font = {"CharProcs": {"A": PdfStream(raw_data=b""), "space": PdfStream(raw_data=b"")}}

    def glyph_name(self, code: int) -> str:
        return "space" if code == 32 else "A"

    def glyph_advance_vector(self, code: int, **kwargs: Any) -> tuple[float, float]:
        return glyph_advance_vector(500, vertical=False, **kwargs)

    def decode_glyphs(self, data: bytes) -> tuple[DecodedFontGlyph, ...]:
        return tuple(
            DecodedFontGlyph(bytes([code]), code, code, code, chr(code), code) for code in data
        )

    def text_advance_vector(self, data: bytes, **kwargs: Any) -> tuple[float, float]:
        kwargs.pop("glyphs")
        return (
            sum(
                self.glyph_advance_vector(code, encoded_space=code == 32, **kwargs)[0]
                for code in data
            ),
            0,
        )


@pytest.mark.parametrize("font_size", [0, 10])
def test_type3_glyphs_use_font_service_spacing(
    monkeypatch: pytest.MonkeyPatch, font_size: int
) -> None:
    state = internal_state()
    state.font_size, state.char_space, state.word_space = font_size, 2, 3
    state.update_text_scales()
    origins: list[float] = []
    monkeypatch.setattr(
        state, "consume_stream", lambda stream, resources, ctm, depth: origins.append(ctm.e)
    )
    state.append_text(data=b"A A", decoder=cast(Any, Font()))
    step = font_size / 2 + 2
    assert origins == [0, step, 2 * step + 3]
    assert state.tm_e == 3 * step + 3


@pytest.mark.parametrize("text", ["", "A"])
def test_text_show_updates_after_callback_and_emits_one_boundary(text: str) -> None:
    state = internal_state()
    events: list[tuple[str, float]] = []

    def show(*args: Any) -> None:
        events.append(("show", state.tm_e))
        state.tm_e = 1000

    state.sink = cast(
        Any,
        SimpleNamespace(
            show_text=show, text_boundary=lambda *args: events.append(("boundary", state.tm_e))
        ),
    )
    font = SimpleNamespace(
        is_type3=False,
        decode_glyphs=lambda data: (DecodedFontGlyph(b"A", 65, 65, 65, text, 65),),
        text_advance_vector=lambda *args, **kwargs: (5, 0),
    )
    state.append_text(data=b"A", decoder=cast(Any, font))
    assert events == ([("show", 0)] if text else []) + [("boundary", 5)]
    assert state.tm_e == 5


@pytest.mark.parametrize(
    ("kind", "channels", "expected"),
    [
        ("DeviceGray", 1, (0,)),
        ("DeviceRGB", 1, (0, 0, 0)),
        ("DeviceCMYK", 1, (0, 0, 0, 1)),
        ("CalGray", 1, (0,)),
        ("CalRGB", 1, (0, 0, 0)),
        ("Lab", 1, (0, 0, 0)),
        ("Indexed", 1, (0,)),
        ("Separation", 1, (1,)),
        ("DeviceN", 2, (1, 1)),
        ("ICCBased", 3, (0, 0, 0)),
        ("Pattern", 1, None),
    ],
)
def test_initial_color_and_component_count(
    kind: str, channels: int, expected: tuple[int, ...] | None
) -> None:
    spec = ImageColorSpec(kind, {}, channels=channels)
    assert initial_color_components(spec) == expected
    assert color_component_count(spec) == (len(expected) if expected is not None else 0)


@pytest.mark.parametrize(
    ("spec", "values", "expected", "initial"),
    [
        (ImageColorSpec("Lab", {"Range": [2, 4, -4, -2]}), (150, 1, 1), (100, 2, -2), (0, 2, -2)),
        (ImageColorSpec("ICCBased", {"Range": [2, 4]}, channels=1), (5,), (4,), (2,)),
        (ImageColorSpec("Indexed", {}, hival=4), (2.5,), (3,), (0,)),
    ],
)
def test_color_ranges_are_shared_for_components_and_initialization(
    spec: ImageColorSpec,
    values: tuple[float, ...],
    expected: tuple[int, ...],
    initial: tuple[int, ...],
) -> None:
    assert normalize_color_components(spec, values) == expected
    assert initial_color_components(spec) == initial


@pytest.mark.parametrize(
    "components", [(1,), (True, 0, 0), ("1", 0, 0), (float("inf"), 0, 0), (10**400, 0, 0)]
)
def test_color_components_reject_wrong_count_or_non_pdf_numbers(
    components: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError):
        normalize_color_components(ImageColorSpec("DeviceRGB", {}), components)


@pytest.mark.parametrize("ranges", [[1], [2, 1], [0, float("nan")], ["0", 1]])
def test_color_range_validation_precedes_initialization(ranges: list[object]) -> None:
    with pytest.raises(ValueError):
        initial_color_components(ImageColorSpec("ICCBased", {"Range": ranges}, channels=1))


@pytest.mark.parametrize("stroke", [False, True])
def test_color_space_selection_initializes_and_clears_pattern(stroke: bool) -> None:
    # ISO 32000-1 Table 74: reserved names cannot be shadowed by resources.
    state = internal_state()
    state.resources = {"ColorSpace": {"DeviceRGB": PdfName.of("DeviceGray")}}
    state.fill_pattern = state.stroke_pattern = cast(Any, object())
    state.execute_operation("CS" if stroke else "cs", (PdfName.of("DeviceRGB"),), 0)
    assert (state.stroke_color_space if stroke else state.fill_color_space) == "DeviceRGB"
    assert (state.stroke_color if stroke else state.fill_color) == (0, 0, 0)
    assert (state.stroke_pattern if stroke else state.fill_pattern) is None
    state.execute_operation("CS" if stroke else "cs", (PdfName.of("DeviceCMYK"),), 0)
    assert (state.stroke_color if stroke else state.fill_color) == (0, 0, 0, 1)
    state.execute_operation("CS" if stroke else "cs", (PdfName.of("Pattern"),), 0)
    assert (state.stroke_color if stroke else state.fill_color) is None


@pytest.mark.parametrize("kind", ["Separation", "DeviceN", "ICCBased"])
@pytest.mark.parametrize("stroke", [False, True])
def test_special_color_spaces_require_extended_operator(kind: str, stroke: bool) -> None:
    state = internal_state()
    state.fill_color_space = state.stroke_color_space = kind
    state.fill_color_spec = state.stroke_color_spec = ImageColorSpec(kind, {}, channels=1)
    with pytest.raises(PdfParseError, match="requires SCN"):
        state.execute_operation("SC" if stroke else "sc", (0.5,), 0)
    state.execute_operation("SCN" if stroke else "scn", (0.5,), 0)
    assert (state.stroke_color if stroke else state.fill_color) == (0.5,)


@pytest.mark.parametrize(
    ("base", "paint_type", "components", "valid"),
    [
        (None, 1, (), True),
        (None, 1, (0.2,), False),
        (None, 2, (), False),
        ("DeviceRGB", 2, (0.1, 0.2, 0.3), True),
        ("DeviceRGB", 2, (0.1,), False),
        ("DeviceRGB", 1, (0.1, 0.2, 0.3), False),
    ],
)
def test_pattern_selection_matches_underlying_space(
    base: str | None, paint_type: int, components: tuple[float, ...], valid: bool
) -> None:
    state = internal_state()
    space = [PdfName.of("Pattern")] + ([PdfName.of(base)] if base else [])
    state.resources = {
        "ColorSpace": {"P": space},
        "Pattern": {"Tile": internal_pattern(paint_type)},
    }
    state.execute_operation("cs", (PdfName.of("P"),), 0)
    if not valid:
        with pytest.raises(PdfParseError):
            state.execute_operation("scn", (*components, PdfName.of("Tile")), 0)
        assert state.fill_pattern is None
        return
    state.execute_operation("scn", (*components, PdfName.of("Tile")), 0)
    assert isinstance(state.fill_pattern, TilingPattern)
    assert state.fill_pattern.base_color == (components if base else None)


def test_pattern_retains_lab_base_and_public_positional_constructors() -> None:
    space = color_spec_from_value(["Pattern", ["Lab", {"Range": [-2, 2, -3, 3]}]])
    assert space.pattern_base is not None
    state = internal_state()
    state.resources = {"Pattern": {"P": internal_pattern(2)}}
    pattern = state.resolve_pattern_color((50, -5, 5, PdfName.of("P")), color_spec=space)
    assert isinstance(pattern, TilingPattern)
    assert pattern.base_color == (50, -2, 3)
    assert pattern.base_color_spec is space.pattern_base
    assert "pattern_base" not in ImageColorSpec.__match_args__
    assert "base_color_spec" not in TilingPattern.__match_args__
    assert (
        TilingPattern(
            (0, 0, 1, 1), 1, 1, internal_pattern(), {}, IDENTITY_MATRIX, 1, None
        ).base_color_spec
        is None
    )
    for value in (["Pattern", "Pattern"], ["Pattern", "DeviceRGB", 1]):
        with pytest.raises(ValueError, match="Pattern"):
            color_spec_from_value(value)

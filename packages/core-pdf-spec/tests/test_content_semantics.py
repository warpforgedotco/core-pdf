from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.model import TilingPattern
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.color import (
    initial_color_components,
    normalize_color_components,
)
from core_pdf_spec.s_08_graphics.color_spec import ColorSpace, parse_color_space
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.s_09_fonts.metrics import glyph_advance_vector, text_adjustment_vector
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph
from core_pdf_spec.types import PdfName, PdfReference, PdfString


class Sink:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


def make_state() -> ContentInterpreter:
    return ContentInterpreter(ObjectResolver(b"", {}), cast(Any, Sink()), cast(Any, None))


def make_pattern(paint_type: int = 1, **entries: Any) -> PdfStream:
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
    state = make_state()
    resolver = cast(ObjectResolver, state.resolver)
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
        "Pattern": {"P": make_pattern(Matrix=matrix)},
    }
    frame = state.append_xobject(PdfName.of("F"), 0)
    pattern = state.resolve_pattern_color(
        PdfName.of("P"), space=parse_color_space("Pattern"), base_components=()
    )
    assert isinstance(pattern, TilingPattern)
    assert frame is not None
    assert frame.ctm == pattern.matrix == Matrix(1, 0, 0, 1, 5, 6)
    assert state.matrix_operand(None, "form") == IDENTITY_MATRIX
    resolver.objects[key_for(3, 0)] = None
    assert state.matrix_operand(PdfReference(3, 0), "pattern") == IDENTITY_MATRIX


@pytest.mark.parametrize("indirect", [False, True])
@pytest.mark.parametrize(
    ("isolated", "opacity", "blend"),
    [
        (False, 1.0, None),
        (False, 1.0, "Normal"),
        (True, 1.0, None),
        (False, 0.4, None),
        (False, 1.0, "Multiply"),
    ],
)
def test_form_retains_transparency_transform_and_source_identity(
    indirect: bool, isolated: bool, opacity: float, blend: str | None
) -> None:
    state = make_state()
    bbox = [0, 0, 2, 3]
    form = PdfStream(
        raw_data=b"",
        dictionary={
            "Subtype": PdfName.of("Form"),
            "BBox": bbox,
            "Matrix": [1, 0, 0, 1, 5, 6],
            "Group": {"S": PdfName.of("Transparency"), "I": isolated, "ca": 0.1},
        },
    )
    cast(ObjectResolver, state.resolver).objects[key_for(7, 2)] = form
    state.resources = {"XObject": {"F": PdfReference(7, 2) if indirect else form}}
    state.graphics.ctm = Matrix(2, 0, 0, 3, 7, 11)
    state.graphics.fill_opacity = opacity
    state.graphics.blend_mode = blend
    frame = state.append_xobject(PdfName.of("F"), 2)
    assert frame is not None
    assert frame.stream is form
    assert frame.source_key == (("ref", 7, 2) if indirect else None)
    assert frame.form_bbox_operand is bbox
    assert frame.ctm == Matrix(2, 0, 0, 3, 17, 29)
    assert frame.clip_bbox == (17, 29, 21, 38)
    assert frame.depth == 3
    assert frame.group_alpha == opacity
    assert frame.group_isolated is isolated


@pytest.mark.parametrize("resources", [None, {}, {"Font": {}}])
def test_form_inherits_resources_only_when_absent(resources: dict | None) -> None:
    state = make_state()
    form = PdfStream(dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 1, 1]})
    if resources is not None:
        form.dictionary["Resources"] = resources
    state.resources = {"XObject": {"F": form}}
    frame = state.append_xobject(PdfName.of("F"), 0)
    assert frame is not None
    assert frame.resources is (state.resources if resources is None else resources)


@pytest.mark.parametrize(
    ("dictionary", "message"),
    [
        ({"Subtype": PdfName.of("Form")}, "requires a BBox"),
        ({"Subtype": PdfName.of("Unknown")}, "unsupported XObject subtype"),
        ({"BBox": [0, 0, 1, 1]}, "unsupported XObject subtype"),
    ],
)
def test_form_rejects_missing_bbox_and_invalid_subtypes(dictionary: dict, message: str) -> None:
    state = make_state()
    state.resources = {"XObject": {"F": PdfStream(dictionary=dictionary)}}
    with pytest.raises(PdfParseError, match=message):
        state.append_xobject(PdfName.of("F"), 0)
    assert not state.stream_executor.active_streams


@pytest.mark.parametrize(("vertical", "expected"), [(False, (-2, 0)), (True, (0, -1))])
def test_tj_horizontal_scale_applies_only_horizontally(
    vertical: bool, expected: tuple[int, int]
) -> None:
    assert (
        text_adjustment_vector(100, vertical=vertical, font_size=10, horizontal_scale=200)
        == expected
    )
    state = make_state()
    state.graphics.current_decoder = cast(Any, SimpleNamespace(is_vertical=vertical))
    state.graphics.font_size, state.graphics.horizontal_scale = 10, 200
    state.text_matrix = Matrix(2, 3, 5, 7, 11, 13)
    state.append_tj_array([100])
    dx, dy = expected
    assert (state.text_matrix.e, state.text_matrix.f) == (
        11 + 2 * dx + 5 * dy,
        13 + 3 * dx + 7 * dy,
    )


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
    state = make_state()
    state.graphics.font_size, state.graphics.char_space, state.graphics.word_space = font_size, 2, 3
    origins: list[float] = []
    monkeypatch.setattr(
        type(state.stream_executor),
        "consume",
        lambda executor, stream, resources, ctm, depth: origins.append(ctm.e),
    )
    state.append_text(data=b"A A", decoder=cast(Any, Font()))
    step = font_size / 2 + 2
    assert origins == [0, step, 2 * step + 3]
    assert state.text_matrix.e == 3 * step + 3


@pytest.mark.parametrize("text", ["", "A"])
def test_text_show_updates_after_callback_and_emits_one_boundary(text: str) -> None:
    state = make_state()
    events: list[tuple[str, float]] = []

    def show(*args: Any) -> None:
        events.append(("show", state.text_matrix.e))
        state.text_matrix = state.text_matrix._replace(e=1000)

    state.sink = cast(
        Any,
        SimpleNamespace(
            show_text=show,
            text_boundary=lambda *args: events.append(("boundary", state.text_matrix.e)),
        ),
    )
    font = SimpleNamespace(
        is_type3=False,
        decode_glyphs=lambda data: (DecodedFontGlyph(b"A", 65, 65, 65, text, 65),),
        text_advance_vector=lambda *args, **kwargs: (5, 0),
    )
    state.append_text(data=b"A", decoder=cast(Any, font))
    assert events == ([("show", 0)] if text else []) + [("boundary", 5)]
    assert state.text_matrix.e == 5


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
    spec = ColorSpace(kind, ((0.0, 1.0),) * (len(expected) if expected is not None else 0))
    assert initial_color_components(spec) == expected
    assert len(spec.component_ranges) == (len(expected) if expected is not None else 0)


@pytest.mark.parametrize(
    ("spec", "values", "expected", "initial"),
    [
        (
            parse_color_space(["Lab", {"WhitePoint": [0.9505, 1, 1.089], "Range": [2, 4, -4, -2]}]),
            (150, 1, 1),
            (100, 2, -2),
            (0, 2, -2),
        ),
        (
            parse_color_space(["ICCBased", PdfStream(dictionary={"N": 1, "Range": [2, 4]})]),
            (5,),
            (4,),
            (2,),
        ),
        (
            ColorSpace("Indexed", ((0.0, float(4)),), base=parse_color_space("DeviceRGB"), hival=4),
            (2.5,),
            (3,),
            (0,),
        ),
    ],
)
def test_color_ranges_are_shared_for_components_and_initialization(
    spec: ColorSpace,
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
        normalize_color_components(parse_color_space("DeviceRGB"), components)


@pytest.mark.parametrize("ranges", [[1], [2, 1], [0, float("nan")], ["0", 1]])
def test_color_range_validation_precedes_initialization(ranges: list[object]) -> None:
    with pytest.raises(ValueError):
        initial_color_components(
            parse_color_space(["ICCBased", PdfStream(dictionary={"N": 1, "Range": ranges})])
        )


@pytest.mark.parametrize("stroke", [False, True])
def test_color_space_selection_initializes_and_clears_pattern(stroke: bool) -> None:
    state = make_state()
    state.resources = {"ColorSpace": {"DeviceRGB": PdfName.of("DeviceGray")}}
    state.graphics.fill_pattern = state.graphics.stroke_pattern = cast(Any, object())
    state.execute_operation("CS" if stroke else "cs", (PdfName.of("DeviceRGB"),), 0)
    assert (
        state.graphics.stroke_space.kind if stroke else state.graphics.fill_space.kind
    ) == "DeviceRGB"
    assert (state.graphics.stroke_color if stroke else state.graphics.fill_color) == (0, 0, 0)
    assert (state.graphics.stroke_pattern if stroke else state.graphics.fill_pattern) is None
    state.execute_operation("CS" if stroke else "cs", (PdfName.of("DeviceCMYK"),), 0)
    assert (state.graphics.stroke_color if stroke else state.graphics.fill_color) == (0, 0, 0, 1)
    state.execute_operation("CS" if stroke else "cs", (PdfName.of("Pattern"),), 0)
    assert (state.graphics.stroke_color if stroke else state.graphics.fill_color) is None


@pytest.mark.parametrize("kind", ["Separation", "DeviceN", "ICCBased"])
@pytest.mark.parametrize("stroke", [False, True])
def test_special_color_spaces_require_extended_operator(kind: str, stroke: bool) -> None:
    state = make_state()
    state.graphics.fill_space = state.graphics.stroke_space = ColorSpace(kind, ((0.0, 1.0),) * 1)
    with pytest.raises(PdfParseError, match="requires SCN"):
        state.execute_operation("SC" if stroke else "sc", (0.5,), 0)
    state.execute_operation("SCN" if stroke else "scn", (0.5,), 0)
    assert (state.graphics.stroke_color if stroke else state.graphics.fill_color) == (0.5,)


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
    state = make_state()
    space = [PdfName.of("Pattern")] + ([PdfName.of(base)] if base else [])
    state.resources = {
        "ColorSpace": {"P": space},
        "Pattern": {"Tile": make_pattern(paint_type)},
    }
    state.execute_operation("cs", (PdfName.of("P"),), 0)
    if not valid:
        with pytest.raises(PdfParseError):
            state.execute_operation("scn", (*components, PdfName.of("Tile")), 0)
        assert state.graphics.fill_pattern is None
        return
    state.execute_operation("scn", (*components, PdfName.of("Tile")), 0)
    assert isinstance(state.graphics.fill_pattern, TilingPattern)
    assert state.graphics.fill_pattern.base_color == (components if base else None)


def test_pattern_retains_lab_base_and_public_positional_constructors() -> None:
    space = parse_color_space(
        ["Pattern", ["Lab", {"WhitePoint": [1, 1, 1], "Range": [-2, 2, -3, 3]}]]
    )
    assert space.base is not None
    state = make_state()
    state.resources = {"Pattern": {"P": make_pattern(2)}}
    pattern = state.resolve_pattern_color(PdfName.of("P"), space=space, base_components=(50, -2, 3))
    assert isinstance(pattern, TilingPattern)
    assert pattern.base_color == (50, -2, 3)
    assert pattern.base_color_spec is space.base
    assert "pattern_base" not in ColorSpace.__match_args__
    assert "base_color_spec" not in TilingPattern.__match_args__
    assert (
        TilingPattern(
            (0, 0, 1, 1), 1, 1, make_pattern(), {}, IDENTITY_MATRIX, 1, None
        ).base_color_spec
        is None
    )
    for value in (["Pattern", "Pattern"], ["Pattern", "DeviceRGB", 1]):
        with pytest.raises(ValueError, match="Pattern"):
            parse_color_space(value)

from types import SimpleNamespace

import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.model import ShadingPattern, TilingPattern
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.color_spec import DEVICE_RGB, ColorSpace


class RecordingState(TextState):
    errors: list[str]

    def handle_operand_error(self, error, context):
        self.errors.append(context)


@pytest.fixture
def state():
    resolver = ObjectResolver(b"", {})
    instance = RecordingState(SimpleNamespace(resolver=resolver))
    instance.errors = []
    yield instance
    resolver.close()


@pytest.mark.parametrize("failure", ["lookup", "resolve"])
def test_font_resource_errors_return_a_fallback_and_report_context(state, monkeypatch, failure):
    def fail(*args):
        raise PdfParseError("broken resource")

    state.graphics.current_font = "F"
    state.resources = {"Font": {"F": {"BaseFont": "Helvetica"}}}
    if failure == "lookup":
        monkeypatch.setattr(state, "lookup_page_resource", fail)
    else:
        monkeypatch.setattr(state, "lookup_page_resource", lambda *args: "font")
        monkeypatch.setattr(ObjectResolver, "resolve", fail)
    assert state.get_decoder().decode_glyphs(b"A")
    assert state.errors == ["font-resource" if failure == "lookup" else "font-resolution"]


@pytest.mark.parametrize(
    ("name", "kind"),
    [(None, "DeviceGray"), ("DeviceRGB", "DeviceRGB"), ("UnknownSpace", "UnknownSpace")],
)
def test_color_space_resolution_uses_builtin_or_named_fallback(state, name, kind):
    assert state.resolve_color_space(name).kind == kind


def test_invalid_named_color_space_retains_its_kind_and_reports_recovery(state):
    state.resources = {"ColorSpace": {"C": ["Indexed", "DeviceRGB", -1, b""]}}
    recovered = state.resolve_color_space("C")
    assert recovered.kind == "Indexed"
    assert recovered.component_ranges == ()
    assert state.errors == ["color-space"]


@pytest.mark.parametrize(
    ("components", "expected"),
    [([], None), ([2, -1], (1.0, 0.0)), (["0.5"], (0.5,)), ([None], None), ([float("nan")], None)],
)
def test_generic_color_recovery_clamps_valid_components_and_skips_invalid_ones(
    state, components, expected
):
    assert state.recover_color_components(components) == expected


@pytest.mark.parametrize("kind", ["Lab", "Indexed"])
def test_special_color_spaces_do_not_apply_generic_component_recovery(state, kind):
    assert state.normalize_color_components(ColorSpace(kind, ()), [0.5]) is None
    assert state.errors == ["color-components"]


@pytest.mark.parametrize("stroke", [False, True])
def test_invalid_color_initialization_keeps_the_previous_paint(state, stroke):
    state.graphics.fill_color = (0.25,)
    state.graphics.stroke_color = (0.75,)
    actual = state.initial_color_components(ColorSpace("Unknown", ()), stroke=stroke)
    assert actual == ((0.75,) if stroke else (0.25,))
    assert state.errors == ["color-space"]


@pytest.mark.parametrize(
    ("base", "operands", "expected"),
    [(None, (), None), (None, ("P",), ()), (DEVICE_RGB, (0.2, 0.3, 0.4, "P"), (0.2, 0.3, 0.4))],
)
def test_pattern_color_components_separate_the_pattern_name_from_base_color(
    state, base, operands, expected
):
    assert (
        state.prepare_color_components(
            ColorSpace("Pattern", (), base=base), operands, allow_special=True
        )
        == expected
    )


@pytest.mark.parametrize(
    "pattern",
    [
        None,
        {"PatternType": 99},
        {"PatternType": 1},
        {"PatternType": 2},
        {"PatternType": 2, "Shading": 4},
    ],
)
def test_unusable_pattern_resources_are_skipped(state, pattern):
    state.resources = {"Pattern": {"P": pattern}}
    assert (
        state.resolve_pattern_color("P", space=ColorSpace("Pattern", ()), base_components=())
        is None
    )


def test_shading_pattern_retains_shading_and_extended_graphics_state(state):
    shading = {"ShadingType": 2, "ColorSpace": "DeviceRGB"}
    extgstate = {"ca": 0.5}
    state.resources = {
        "Pattern": {"P": {"PatternType": 2, "Shading": shading, "ExtGState": extgstate}}
    }
    pattern = state.resolve_pattern_color("P", space=ColorSpace("Pattern", ()), base_components=())
    assert isinstance(pattern, ShadingPattern)
    assert pattern.dictionary == shading
    assert pattern.extgstate == extgstate


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("PaintType", 3),
        ("BBox", None),
        ("XStep", None),
        ("YStep", None),
        ("XStep", 0),
        ("YStep", 0),
    ],
)
def test_invalid_tiling_pattern_geometry_is_skipped(state, key, value):
    params = {"PatternType": 1, "BBox": [0, 0, 10, 10], "XStep": 10, "YStep": 10, key: value}
    state.resources = {"Pattern": {"P": PdfStream(params)}}
    assert (
        state.resolve_pattern_color("P", space=ColorSpace("Pattern", ()), base_components=())
        is None
    )


@pytest.mark.parametrize("paint_type", [1, 2])
def test_tiling_pattern_preserves_source_and_uses_base_color_only_when_uncolored(state, paint_type):
    stream = PdfStream(
        {
            "PatternType": 1,
            "PaintType": paint_type,
            "BBox": [0, 0, 10, 10],
            "XStep": -10,
            "YStep": 10,
        }
    )
    state.resources = {"Pattern": {"P": stream}}
    components = (0.2, 0.3, 0.4)
    pattern = state.resolve_pattern_color(
        "P", space=ColorSpace("Pattern", (), base=DEVICE_RGB), base_components=components
    )
    assert isinstance(pattern, TilingPattern)
    assert pattern.stream is stream
    assert pattern.x_step == -10
    assert pattern.y_step == 10
    assert pattern.base_color == (components if paint_type == 2 else None)
    assert pattern.base_color_spec is DEVICE_RGB

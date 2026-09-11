"""Shared resource preparation preserves strict content application semantics."""

from __future__ import annotations

from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.model import ShadingPattern, TilingPattern
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.color_spec import parse_color_space
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName, PdfReference


def internal_state() -> ContentInterpreter:
    return ContentInterpreter(ObjectResolver(b"", {}), cast(Any, None), cast(Any, None))


@pytest.mark.parametrize("entry_point", ["operator", "application"])
def test_extgstate_updates_only_present_fields_and_clamps_opacity(entry_point: str) -> None:
    state = internal_state()
    values = {"ca": -0.5, "CA": 1.5, "BM": [PdfName.of("Multiply"), PdfName.of("Screen")]}
    if entry_point == "operator":
        state.resources = {"ExtGState": {"G": values}}
        state.execute_operation("gs", (PdfName.of("G"),), 0)
    else:
        state.apply_extgstate(values)
    assert (
        state.graphics.fill_opacity,
        state.graphics.stroke_opacity,
        state.graphics.blend_mode,
    ) == (0.0, 1.0, "Multiply")
    state.apply_extgstate({"ca": None, "CA": None, "BM": []})
    assert (
        state.graphics.fill_opacity,
        state.graphics.stroke_opacity,
        state.graphics.blend_mode,
    ) == (0.0, 1.0, "Multiply")
    state.apply_extgstate({"CA": 0.25, "BM": PdfName.of("Screen")})
    assert (
        state.graphics.fill_opacity,
        state.graphics.stroke_opacity,
        state.graphics.blend_mode,
    ) == (0.0, 0.25, "Screen")


@pytest.mark.parametrize("entry_point", ["operator", "application"])
def test_extgstate_keeps_earlier_fields_when_a_later_field_is_invalid(entry_point: str) -> None:
    state = internal_state()
    state.graphics.stroke_opacity = 0.75
    state.graphics.blend_mode = "Screen"
    values = {"ca": 0.5, "CA": "invalid", "BM": PdfName.of("Multiply")}
    with pytest.raises(PdfParseError, match="numeric operand"):
        if entry_point == "operator":
            state.resources = {"ExtGState": {"G": values}}
            state.execute_operation("gs", (PdfName.of("G"),), 0)
        else:
            state.apply_extgstate(values)
    assert (
        state.graphics.fill_opacity,
        state.graphics.stroke_opacity,
        state.graphics.blend_mode,
    ) == (0.5, 0.75, "Screen")


def test_extgstate_application_retains_polymorphic_coercion_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = internal_state()
    calls: list[tuple[str, object]] = []

    def number(value: object) -> float:
        calls.append(("number", value))
        return 0.25 if value == "fill" else 0.75

    def name(value: object, *, allow_text: bool = False) -> str:
        calls.append(("name", value))
        return "Screen"

    monkeypatch.setattr(state, "as_float", number)
    monkeypatch.setattr(state, "named_value", name)
    state.apply_extgstate({"ca": "fill", "CA": "stroke", "BM": ["blend", "unused"]})
    assert calls == [("number", "fill"), ("number", "stroke"), ("name", "blend")]
    assert (
        state.graphics.fill_opacity,
        state.graphics.stroke_opacity,
        state.graphics.blend_mode,
    ) == (0.25, 0.75, "Screen")


@pytest.mark.parametrize("stream", [False, True], ids=["shading-dictionary", "tiling-stream"])
@pytest.mark.parametrize("indirect", [False, True])
def test_pattern_resource_lookup_preserves_source_identity_and_laziness(
    monkeypatch: pytest.MonkeyPatch, stream: bool, indirect: bool
) -> None:
    state = internal_state()
    resolver = cast(ObjectResolver, state.resolver)
    dictionary: PdfDict = {
        "PatternType": 1 if stream else 2,
        "PaintType": 1,
        "BBox": [0, 0, 1, 1],
        "XStep": 1,
        "YStep": 1,
        "Shading": {"ShadingType": 2},
    }

    def unexpected_decode(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("Pattern lookup must not decode the selected stream")

    source = PdfStream(dictionary=dictionary, decoder=unexpected_decode) if stream else dictionary
    reference = PdfReference(1, 0)
    resolver.objects[key_for(1, 0)] = source
    calls: list[tuple[str, str]] = []

    def lookup(category: str, name: str) -> object:
        calls.append((category, name))
        return reference if indirect else source

    monkeypatch.setattr(state, "lookup_page_resource", lookup)
    selected = state.resolve_pattern_resource(PdfName.of("P"))
    assert selected is not None
    assert selected[0] is source
    assert selected[1] is dictionary
    assert calls == [("Pattern", "P")]
    calls.clear()
    pattern = state.resolve_pattern_color(
        PdfName.of("P"), space=parse_color_space("Pattern"), base_components=()
    )
    assert calls == [("Pattern", "P")]
    if stream:
        assert isinstance(pattern, TilingPattern)
        assert pattern.stream is source
        assert pattern.matrix == IDENTITY_MATRIX
    else:
        assert isinstance(pattern, ShadingPattern)
        assert pattern.dictionary == {"ShadingType": 2}


@pytest.mark.parametrize("resource", [None, 42, [], PdfName.of("Invalid")])
def test_pattern_selection_rejects_resources_without_a_dictionary(resource: object) -> None:
    state = internal_state()
    state.resources = {"Pattern": {"P": resource}}
    assert state.resolve_pattern_resource(PdfName.of("P")) is None
    with pytest.raises(PdfParseError, match="invalid pattern resource"):
        state.resolve_pattern_color(
            PdfName.of("P"), space=parse_color_space("Pattern"), base_components=()
        )


def test_pattern_selection_still_requires_painttype_and_matching_color_space() -> None:
    state = internal_state()
    dictionary: PdfDict = {"PatternType": 1, "BBox": [0, 0, 1, 1], "XStep": 1, "YStep": 1}
    state.resources = {"Pattern": {"P": PdfStream(dictionary=dictionary)}}
    with pytest.raises(PdfParseError, match="invalid pattern"):
        state.resolve_pattern_color(
            PdfName.of("P"), space=parse_color_space("Pattern"), base_components=()
        )
    dictionary["PaintType"] = 1
    stencil_space = parse_color_space(["Pattern", "DeviceRGB"])
    with pytest.raises(PdfParseError, match="PaintType"):
        state.resolve_pattern_color(
            PdfName.of("P"), space=stencil_space, base_components=(0.1, 0.2, 0.3)
        )

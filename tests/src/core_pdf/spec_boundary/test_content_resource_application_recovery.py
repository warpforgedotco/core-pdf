"""Reader policy around shared content-resource preparation and application."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf_spec.s_07_content.model import ShadingPattern, TilingPattern
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.color_spec import parse_color_space
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName, PdfString


def internal_state() -> TextState:
    resolver = ObjectResolver(b"", {}, {})
    return TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))


def test_reader_extgstate_retains_numeric_and_name_coercion() -> None:
    state = internal_state()
    state.resources = {"ExtGState": {"G": {"ca": "0.25", "CA": "0.75", "BM": PdfString(b"Screen")}}}
    state.op_gs((PdfName.of("G"),), 0)
    assert (
        state.graphics.fill_opacity,
        state.graphics.stroke_opacity,
        state.graphics.blend_mode,
    ) == (0.25, 0.75, "Screen")


def test_reader_extgstate_keeps_partial_application_and_reports_error_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = internal_state()
    state.graphics.stroke_opacity = 0.75
    state.graphics.blend_mode = "Screen"
    state.resources = {"ExtGState": {"G": {"ca": "0.5", "CA": "bad", "BM": PdfName.of("Multiply")}}}
    errors: list[tuple[Exception, str]] = []
    monkeypatch.setattr(
        state, "handle_operand_error", lambda error, context: errors.append((error, context))
    )
    state.op_gs((PdfName.of("G"),), 0)
    assert (
        state.graphics.fill_opacity,
        state.graphics.stroke_opacity,
        state.graphics.blend_mode,
    ) == (0.5, 0.75, "Screen")
    assert len(errors) == 1
    assert isinstance(errors[0][0], ValueError)
    assert errors[0][1] == "extended-graphics-state"


@pytest.mark.parametrize("resource", [None, {}, 42, []])
def test_reader_skips_absent_or_invalid_extgstate_resources(resource: object) -> None:
    state = internal_state()
    state.graphics.fill_opacity = 0.25
    state.resources = {"ExtGState": {"G": resource}}
    state.op_gs((PdfName.of("G"),), 0)
    state.op_gs((), 0)
    assert state.graphics.fill_opacity == 0.25


@pytest.mark.parametrize("resource", [None, 42, [], PdfName.of("Invalid")])
def test_reader_keeps_missing_pattern_resource_recovery(resource: object) -> None:
    state = internal_state()
    state.resources = {"Pattern": {"P": resource}}
    assert state.resolve_pattern_resource(PdfName.of("P")) is None
    assert (
        state.resolve_pattern_color(
            PdfName.of("P"), space=parse_color_space("Pattern"), base_components=()
        )
        is None
    )
    assert (
        state.resolve_pattern_color(None, space=parse_color_space("Pattern"), base_components=())
        is None
    )


def test_reader_pattern_lookup_preserves_defaults_and_mismatched_selection() -> None:
    state = internal_state()
    dictionary: PdfDict = {
        "PatternType": 1,
        "BBox": [0, 0, 1, 1],
        "XStep": 1,
        "YStep": 1,
        "Matrix": [1],
    }
    source = PdfStream(dictionary=dictionary)
    state.resources = {"Pattern": {"P": source}}
    space = parse_color_space(["Pattern", "DeviceRGB"])
    pattern = state.resolve_pattern_color(
        PdfName.of("P"), space=space, base_components=(0.1, 0.2, 0.3)
    )
    assert isinstance(pattern, TilingPattern)
    assert pattern.stream is source
    assert pattern.paint_type == 1
    assert pattern.matrix == IDENTITY_MATRIX


def test_reader_shading_pattern_keeps_mismatched_stencil_context() -> None:
    state = internal_state()
    state.resources = {"Pattern": {"P": {"PatternType": 2, "Shading": {"ShadingType": 2}}}}
    space = parse_color_space(["Pattern", "DeviceRGB"])
    pattern = state.resolve_pattern_color(
        PdfName.of("P"), space=space, base_components=(0.1, 0.2, 0.3)
    )
    assert isinstance(pattern, ShadingPattern)
    assert pattern.dictionary == {"ShadingType": 2}

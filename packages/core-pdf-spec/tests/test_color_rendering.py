# SPDX-License-Identifier: AGPL-3.0-only
"""ISO 32000-2 8.6.5.8–9, Tables 57/69/87: colour rendering parameters."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
    ColorRendering,
    override_color_rendering,
    parse_black_point_compensation,
    parse_rendering_intent,
    use_black_point_compensation,
)
from core_pdf_spec.s_08_graphics.image_spec import image_color_rendering
from core_pdf_spec.types import PdfName, PdfReference


@pytest.mark.parametrize(
    "name", ["Perceptual", "Saturation", "RelativeColorimetric", "AbsoluteColorimetric"]
)
def test_rendering_intent_names_and_mandated_unknown_fallback(name: str) -> None:
    assert parse_rendering_intent(PdfName.of(name)) == name
    assert parse_rendering_intent(PdfName.of("Unrecognized")) == "RelativeColorimetric"


@pytest.mark.parametrize("value", [None, 1, [], {}])
def test_rendering_intent_requires_a_name(value: object) -> None:
    with pytest.raises(ValueError, match="PDF name"):
        parse_rendering_intent(value)


@pytest.mark.parametrize("name", ["Default", "ON", "OFF"])
@pytest.mark.parametrize("default", [False, True])
def test_black_point_controls_and_absolute_override(name: str, default: bool) -> None:
    setting = parse_black_point_compensation(PdfName.of(name))
    relative = ColorRendering(black_point_compensation=setting)
    assert use_black_point_compensation(relative, default=default) == (
        default if name == "Default" else name == "ON"
    )
    absolute = ColorRendering("AbsoluteColorimetric", setting)
    assert not use_black_point_compensation(absolute, default=default)


@pytest.mark.parametrize("value", [None, 1, "on", "Off", "Unknown"])
def test_black_point_rejects_nonpermitted_values(value: object) -> None:
    with pytest.raises(ValueError, match="UseBlackPtComp"):
        parse_black_point_compensation(value)


def test_extgstate_absent_and_resolved_null_entries_preserve_state() -> None:
    resolver = ObjectResolver(b"", {})
    state = ContentInterpreter(
        resolver,
        cast(Any, SimpleNamespace(save_graphics=lambda *_: None, restore_graphics=lambda *_: None)),
        cast(Any, None),
    )
    assert state.graphics.color_rendering == DEFAULT_COLOR_RENDERING
    state.apply_extgstate({"RI": PdfName.of("Saturation"), "UseBlackPtComp": PdfName.of("ON")})
    previous = state.graphics.color_rendering
    state.execute_operation("q", (), 0)
    state.execute_operation("ri", (PdfName.of("AbsoluteColorimetric"),), 0)
    state.apply_extgstate({"UseBlackPtComp": PdfName.of("OFF")})
    assert state.graphics.color_rendering == ColorRendering("AbsoluteColorimetric", "OFF")
    state.execute_operation("Q", (), 0)
    assert state.graphics.color_rendering == previous
    state.apply_extgstate({"ca": 0.5})
    assert state.graphics.color_rendering == previous
    # 7.3.7 and 7.3.9: null values, including undefined references, mean absent.
    state.apply_extgstate({"RI": PdfReference(99, 0), "UseBlackPtComp": None, "ca": 0.75})
    state.apply_extgstate({"UseBlackPtComp": PdfReference(99, 0), "RI": None})
    assert state.graphics.color_rendering == previous
    assert state.graphics.fill_opacity == 0.75
    resolver.objects[key_for(10, 0)] = PdfName.of("OFF")
    state.apply_extgstate({"UseBlackPtComp": PdfReference(10, 0)})
    assert state.graphics.black_point_compensation == "OFF"
    state.execute_operation("ri", (PdfName.of("Unknown"),), 0)
    assert state.graphics.render_intent == "RelativeColorimetric"


def test_pattern_overrides_and_image_intent_preserve_unmentioned_parameters() -> None:
    current = ColorRendering("Saturation", "ON")
    assert override_color_rendering({}, current) is not None
    assert override_color_rendering({"RI": None}, current) == current
    assert override_color_rendering({"UseBlackPtComp": "OFF"}, current) == ColorRendering(
        "Saturation", "OFF"
    )
    assert image_color_rendering({}, current) == current
    assert image_color_rendering({"Intent": "AbsoluteColorimetric"}, current) == ColorRendering(
        "AbsoluteColorimetric", "ON"
    )
    assert image_color_rendering({"Intent": "Unknown"}, current) == ColorRendering(
        "RelativeColorimetric", "ON"
    )
    # Table 87: stencil images ignore Intent, including malformed values.
    assert image_color_rendering({"ImageMask": True, "Intent": 123}, current) == current

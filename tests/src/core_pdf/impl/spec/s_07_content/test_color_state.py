from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl.spec.s_07_content.capture import ShadingPattern
from core_pdf.impl.spec.s_07_content.state import TextState
from core_pdf.impl.types import PdfName
from tests.helpers.resolvers import IdentityResolver


def internal_indexed_state(*, stroke: bool) -> TextState:
    state = TextState(cast(Any, SimpleNamespace(resolver=IdentityResolver())))
    state.resources = cast(
        Any,
        {"ColorSpace": {"Palette": ["Indexed", "DeviceRGB", 1, b"\xff\x00\x00\x00\xff\x00"]}},
    )
    handler = state.op_CS if stroke else state.op_cs
    handler((PdfName.of("Palette"),), 0)
    return state


@pytest.mark.parametrize("stroke", [False, True])
@pytest.mark.parametrize(
    ("operator", "space", "components"),
    [
        ("g", "DeviceGray", (0.25,)),
        ("rg", "DeviceRGB", (0.25, 0.5, 0.75)),
        ("k", "DeviceCMYK", (0.25, 0.5, 0.75, 0.0)),
    ],
)
def test_device_operator_replaces_the_previous_color_spec(
    stroke: bool, operator: str, space: str, components: tuple[float, ...]
) -> None:
    state = internal_indexed_state(stroke=stroke)
    prefix = "stroke" if stroke else "fill"
    setattr(state, f"{prefix}_pattern", ShadingPattern({}))
    handler = getattr(state, f"op_{operator.upper() if stroke else operator}")

    handler(components, 0)
    # A following SC/sc must use device components, not reinterpret the first
    # component as an index in the previous palette.
    color_handler = state.op_SC if stroke else state.op_sc
    color_handler(components, 0)

    assert getattr(state, f"{prefix}_color_space") == space
    assert getattr(state, f"{prefix}_color_spec") is None
    assert getattr(state, f"{prefix}_color") == components
    assert getattr(state, f"{prefix}_pattern") is None


@pytest.mark.parametrize("stroke", [False, True])
def test_invalid_device_color_keeps_the_previous_selection(stroke: bool) -> None:
    state = internal_indexed_state(stroke=stroke)
    prefix = "stroke" if stroke else "fill"
    fields = tuple(f"{prefix}_{part}" for part in ("color_space", "color_spec", "color", "pattern"))
    before = tuple(getattr(state, name) for name in fields)

    handler = state.op_RG if stroke else state.op_rg
    handler((0.0, PdfName.of("Invalid"), 0.0), 0)

    assert tuple(getattr(state, name) for name in fields) == before


@pytest.mark.parametrize("stroke", [False, True])
def test_named_palette_selection_resolves_name_and_components_together(stroke: bool) -> None:
    state = internal_indexed_state(stroke=stroke)
    handler = state.op_SCN if stroke else state.op_scN

    handler((1,), 0)

    prefix = "stroke" if stroke else "fill"
    assert getattr(state, f"{prefix}_color_space") == "Indexed"
    assert getattr(state, f"{prefix}_color") == (0.0, 1.0, 0.0)

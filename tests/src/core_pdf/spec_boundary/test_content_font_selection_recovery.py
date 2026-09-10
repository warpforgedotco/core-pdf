# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName


def internal_state() -> TextState:
    resolver = ObjectResolver(b"", {}, {})
    state = TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))
    state.resources = {"Font": {"F": {}, "Other": {}}}
    state.resources_id = id(state.resources)
    return state


@pytest.mark.parametrize("reuse_operand", [False, True])
def test_reader_tf_recovers_bad_size_without_changing_font(
    monkeypatch: pytest.MonkeyPatch, reuse_operand: bool
) -> None:
    state = internal_state()
    name = PdfName.of("F")
    state.op_Tf((name, 12), 0)
    decoder = state.current_decoder
    errors: list[str] = []
    monkeypatch.setattr(
        state, "handle_operand_error", lambda error, context: errors.append(context)
    )
    state.op_Tf((name if reuse_operand else PdfName.of("Other"), "bad"), 0)
    assert errors == ["font-size"]
    assert state.current_decoder is decoder
    assert state.current_font == "F"
    assert state.font_size == 12
    assert state.font_operand is name
    assert state.font_size_operand == 12


def test_reader_tf_preserves_arity_and_numeric_string_recovery() -> None:
    state = internal_state()
    name = PdfName.of("F")
    state.op_Tf((), 0)
    state.op_Tf((name,), 0)
    assert state.current_font is None
    state.op_Tf((name, "12", PdfName.of("Ignored")), 0)
    decoder = state.current_decoder
    assert decoder is not None
    assert state.font_size == 12
    state.op_Tf((name, "20"), 0)
    assert state.current_decoder is decoder
    assert state.font_size == 20


def test_reader_tf_ignores_invalid_name_before_bad_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = internal_state()
    errors: list[str] = []
    monkeypatch.setattr(
        state, "handle_operand_error", lambda error, context: errors.append(context)
    )
    state.op_Tf((99, "bad"), 0)
    assert state.current_font is None
    assert errors == []


@pytest.mark.parametrize("error_type", [ValueError, TypeError])
def test_reader_tf_does_not_swallow_provider_errors(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    state = internal_state()
    errors: list[str] = []
    monkeypatch.setattr(
        state, "handle_operand_error", lambda error, context: errors.append(context)
    )

    def fail(*args: Any) -> Any:
        raise error_type("provider failed")

    state.font_provider = fail
    with pytest.raises(error_type, match="provider failed"):
        state.op_Tf((PdfName.of("F"), 12), 0)
    assert errors == []
    assert state.current_font == "F"
    assert state.font_size == state.font_size_operand == 12
    assert state.current_decoder is None


@pytest.mark.parametrize("scope", ["absent", "empty", "local"])
def test_reader_type3_keeps_explicit_resource_scope(
    monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    state = internal_state()
    outer = PdfStream(dictionary={"Subtype": PdfName.of("Image")}, raw_data=b"outer")
    local = PdfStream(dictionary={"Subtype": PdfName.of("Image")}, raw_data=b"local")
    state.resources = {"XObject": {"Image": outer}}
    font: PdfDict = {"CharProcs": {"A": PdfStream(raw_data=b"0 0 d0 /Image Do")}}
    if scope == "empty":
        font["Resources"] = {}
    elif scope == "local":
        font["Resources"] = {"XObject": {"Image": local}}
    decoder = SimpleNamespace(
        font=font,
        font_matrix=IDENTITY_MATRIX,
        glyph_name=lambda code: "A",
        glyph_advance_vector=lambda code, **kwargs: (0.0, 0.0),
    )
    images: list[PdfStream] = []
    monkeypatch.setattr(state, "paint_image", lambda state, image: images.append(image))
    state.internal_render_type3_glyphs(b"A", cast(Any, decoder))
    assert images == ([] if scope == "empty" else [outer if scope == "absent" else local])

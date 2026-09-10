# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName, PdfReference, PdfString


def internal_state() -> TextState:
    resolver = ObjectResolver(b"", {}, {})
    state = TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))
    state.resources = {"Font": {"F": {}, "Other": {}}}
    return state


@pytest.mark.parametrize("reuse_operand", [False, True])
def test_reader_tf_recovers_bad_size_without_changing_font(
    monkeypatch: pytest.MonkeyPatch, reuse_operand: bool
) -> None:
    state = internal_state()
    name = PdfName.of("F")
    state.op_Tf((name, 12), 0)
    decoder = state.graphics.current_decoder
    errors: list[str] = []
    monkeypatch.setattr(
        state, "handle_operand_error", lambda error, context: errors.append(context)
    )
    state.op_Tf((name if reuse_operand else PdfName.of("Other"), "bad"), 0)
    assert errors == ["font-size"]
    assert state.graphics.current_decoder is decoder
    assert state.graphics.current_font == "F"
    assert state.graphics.font_size == 12


def test_reader_tf_preserves_arity_and_numeric_string_recovery() -> None:
    state = internal_state()
    name = PdfName.of("F")
    state.op_Tf((), 0)
    state.op_Tf((name,), 0)
    assert state.graphics.current_font is None
    state.op_Tf((name, "12", PdfName.of("Ignored")), 0)
    decoder = state.graphics.current_decoder
    assert decoder is not None
    assert state.graphics.font_size == 12
    state.op_Tf((name, "20"), 0)
    assert state.graphics.current_decoder is decoder
    assert state.graphics.font_size == 20


def test_reader_tf_ignores_invalid_name_before_bad_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = internal_state()
    errors: list[str] = []
    monkeypatch.setattr(
        state, "handle_operand_error", lambda error, context: errors.append(context)
    )
    state.op_Tf((99, "bad"), 0)
    assert state.graphics.current_font is None
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
    assert state.graphics.current_font == "F"
    assert state.graphics.font_size == 12
    assert state.graphics.current_decoder is None


@pytest.mark.parametrize("operator", ["Tj", "'", '"'])
@pytest.mark.parametrize("indirect", [False, True])
def test_reader_text_operators_keep_recovered_literal_text(
    monkeypatch: pytest.MonkeyPatch, operator: str, indirect: bool
) -> None:
    state = internal_state()
    operand: Any = "€"
    if indirect:
        operand = PdfReference(9, 0)
        cast(ObjectResolver, state.resolver).objects[key_for(9, 0)] = PdfString(b"\xfe\xff\x20\xac")
    shown: list[tuple[str, bytes]] = []
    monkeypatch.setattr(
        state,
        "show_text",
        lambda current, text, data, *args: shown.append((text, bytes(data))),
    )
    operands = (1, 2, operand) if operator == '"' else (operand,)
    assert state.execute_operation(operator, operands, 0) is None
    # Reader text recovery preserves the Unicode string even when Latin-1
    # replacement supplies a different encoded glyph to the font decoder.
    assert shown == [("€", b"?")]


@pytest.mark.parametrize("select_font", [False, True])
def test_reader_uncached_missing_font_keeps_zero_capture_metrics(
    monkeypatch: pytest.MonkeyPatch, select_font: bool
) -> None:
    state = internal_state()
    if select_font:
        state.op_Tf((PdfName.of("Missing"), 12), 0)
    decoder = state.graphics.current_decoder
    metrics: list[tuple[float, float]] = []
    capture = state.record_glyph_observations

    def record(*args: Any, **kwargs: Any) -> Any:
        metrics.append((kwargs["font_ascent"], kwargs["font_descent"]))
        return capture(*args, **kwargs)

    monkeypatch.setattr(state, "record_glyph_observations", record)
    state.op_Tj((PdfString(b"A"),), 0)
    state.run_accumulator.flush()
    if select_font:
        assert decoder is not None
        assert metrics == [(decoder.ascent * 0.012, decoder.descent * 0.012)]
        assert state.runs[0].space_width == decoder.glyph_width(32) * 12 * 0.001
    else:
        assert metrics == [(0.0, 0.0)]
        assert state.runs[0].space_width == 0.0
        assert state.graphics.current_decoder is None


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

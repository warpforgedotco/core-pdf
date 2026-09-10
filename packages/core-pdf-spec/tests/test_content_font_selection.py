# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.operations import ContentOperands
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName, PdfReference


class Sink:
    def __init__(self) -> None:
        self.images: list[PdfStream] = []

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

    def paint_image(self, state: ContentInterpreter, image: PdfStream) -> None:
        self.images.append(image)


class RecordingState(ContentInterpreter):
    def __init__(self) -> None:
        self.events: list[str] = []

        def provide(font: dict[str, Any], resources: dict[str, Any]) -> Any:
            self.events.append("load")
            return SimpleNamespace(
                font=font,
                ascent=800.0,
                descent=-200.0,
                fast_widths=(),
                glyph_width=lambda code: 500.0,
            )

        super().__init__(ObjectResolver(b"", {}, {}), cast(Any, Sink()), provide)
        self.resources = {"Font": {"F": {}, "Other": {"BaseFont": PdfName.of("Other")}}}

    def resolve_font_name(self, value: object) -> str | None:
        self.events.append("name")
        return super().resolve_font_name(value)

    def parse_font_size(self, value: object) -> float | None:
        self.events.append("size")
        return super().parse_font_size(value)


def test_tf_reuses_decoder_for_equal_font_names_and_changed_sizes() -> None:
    state = RecordingState()
    state.op_Tf((PdfName.of("F"), 12.5), 0)
    assert state.events == ["name", "size", "load"]
    decoder = state.graphics.current_decoder

    for size in (12.5, 20.0, 30.0):
        state.events.clear()
        state.op_Tf((PdfName.of("F"), size), 0)
        assert state.events == ["name", "size"]
        assert state.graphics.current_decoder is decoder
        assert state.graphics.font_size == size


def test_tf_reloads_same_name_only_when_selected_in_new_resources() -> None:
    state = RecordingState()
    name = PdfName.of("F")
    state.op_Tf((name, 12), 0)
    original_decoder = state.graphics.current_decoder
    original_resources = state.resources
    # Equal dictionaries still represent distinct resource scopes.
    state.resources = dict(state.resources)
    state.events.clear()

    # A Form inherits its selected font until Tf selects a name in the new scope.
    assert state.get_decoder() is original_decoder
    assert state.graphics.decoder_resources is original_resources
    assert state.events == []
    state.op_Tf((name, 12), 0)
    assert state.events == ["name", "size", "load"]
    assert state.graphics.current_decoder is not original_decoder
    assert state.graphics.decoder_resources is state.resources


def test_tf_graphics_restore_recovers_selected_font_size_and_resource_scope() -> None:
    state = RecordingState()
    name, size = PdfName.of("F"), 12.5
    state.op_Tf((name, size), 0)
    decoder = state.graphics.current_decoder
    resources = state.graphics.decoder_resources
    state.op_q((), 0)
    state.op_Tf((PdfName.of("Other"), 20), 0)
    state.op_Q((), 0)
    assert state.graphics.current_decoder is decoder
    assert state.graphics.current_font == "F"
    assert state.graphics.font_size == size
    assert state.graphics.decoder_resources is resources
    state.events.clear()
    state.op_Tf((name, size), 0)
    assert state.events == ["name", "size"]


@pytest.mark.parametrize("operands", [(), (PdfName.of("F"),), (PdfName.of("F"), 12, 99)])
def test_tf_rejects_wrong_arity_before_resolving(operands: ContentOperands) -> None:
    state = RecordingState()
    with pytest.raises(PdfParseError, match="Tf requires two operands"):
        state.op_Tf(operands, 0)
    assert state.events == []


@pytest.mark.parametrize("reuse_operand", [False, True])
def test_tf_invalid_size_preserves_existing_selection(reuse_operand: bool) -> None:
    state = RecordingState()
    name = PdfName.of("F")
    state.op_Tf((name, 12), 0)
    decoder = state.graphics.current_decoder
    state.events.clear()
    with pytest.raises(PdfParseError):
        state.op_Tf((name if reuse_operand else PdfName.of("Other"), "bad"), 0)
    assert state.events == ["name", "size"]
    assert state.graphics.current_decoder is decoder
    assert state.graphics.current_font == "F"
    assert state.graphics.font_size == 12


def test_tf_invalid_name_precedes_size_parsing() -> None:
    state = RecordingState()
    with pytest.raises(PdfParseError, match="Tf requires a font name"):
        state.op_Tf((99, "bad"), 0)
    assert state.events == ["name"]


def test_tf_provider_error_occurs_after_selection_update() -> None:
    state = RecordingState()
    name = PdfName.of("F")

    def fail(*args: Any) -> Any:
        raise ValueError("provider failed")

    state.font_provider = fail
    with pytest.raises(ValueError, match="provider failed"):
        state.op_Tf((name, 12), 0)
    assert state.graphics.current_font == "F"
    assert state.graphics.font_size == 12
    assert state.graphics.current_decoder is None
    assert state.events == ["name", "size"]


@pytest.mark.parametrize("scope", ["absent", "empty", "indirect-empty", "local"])
def test_type3_resource_scope_distinguishes_absent_and_empty(scope: str) -> None:
    # ISO 32000-1 Table 112: only an absent Resources entry inherits page resources.
    state = RecordingState()
    outer = PdfStream(dictionary={"Subtype": PdfName.of("Image")}, raw_data=b"outer")
    local = PdfStream(dictionary={"Subtype": PdfName.of("Image")}, raw_data=b"local")
    state.resources = {"XObject": {"Image": outer}}
    font: PdfDict = {
        "CharProcs": {"A": PdfStream(raw_data=b"0 0 d0 /Image Do")},
    }
    if scope == "empty":
        font["Resources"] = {}
    elif scope == "indirect-empty":
        font["Resources"] = PdfReference(1, 0)
        resolver = cast(ObjectResolver, state.resolver)
        resolver.objects[key_for(1, 0)] = {}
    elif scope == "local":
        font["Resources"] = {"XObject": {"Image": local}}
    decoder = SimpleNamespace(
        font=font,
        font_matrix=IDENTITY_MATRIX,
        glyph_name=lambda code: "A",
        glyph_advance_vector=lambda code, **kwargs: (0.0, 0.0),
    )
    if "empty" in scope:
        with pytest.raises(PdfParseError, match="XObject resource must be a stream"):
            state.internal_render_type3_glyphs(b"A", cast(Any, decoder))
        assert cast(Sink, state.sink).images == []
    else:
        state.internal_render_type3_glyphs(b"A", cast(Any, decoder))
        assert cast(Sink, state.sink).images == [outer if scope == "absent" else local]

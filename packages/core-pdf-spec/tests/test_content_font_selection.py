# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.operations import ContentOperands
from core_pdf_spec.s_07_content.state import TextState
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

    def paint_image(self, state: TextState, image: PdfStream) -> None:
        self.images.append(image)


class RecordingState(TextState):
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

        super().__init__(
            SimpleNamespace(resolver=ObjectResolver(b"", {}, {})), cast(Any, Sink()), provide
        )
        self.resources = {"Font": {"F": {}, "Other": {"BaseFont": PdfName.of("Other")}}}
        self.resources_id = id(self.resources)

    def resolve_font_name(self, value: object) -> str | None:
        self.events.append("name")
        return super().resolve_font_name(value)

    def parse_font_size(self, value: object) -> float | None:
        self.events.append("size")
        return super().parse_font_size(value)

    def update_font_metrics(self) -> None:
        self.events.append("metrics")
        super().update_font_metrics()


def test_tf_preserves_identity_and_equal_name_fast_paths() -> None:
    state = RecordingState()
    name = PdfName.of("F")
    size = float("12.5")
    state.op_Tf((name, size), 0)
    assert state.events == ["name", "size", "load", "metrics"]
    decoder = state.current_decoder

    state.events.clear()
    state.op_Tf((name, size), 0)
    assert state.events == []

    equal_size = float("12.5")
    assert equal_size is not size
    state.op_Tf((name, equal_size), 0)
    assert state.events == ["size"]
    assert state.font_size_operand is equal_size

    state.events.clear()
    equal_name = PdfName.of("F")
    state.op_Tf((equal_name, 20), 0)
    assert state.events == ["name", "size", "metrics"]
    assert state.current_decoder is decoder
    assert state.font_operand is equal_name
    assert state.font_ascent == 16

    state.events.clear()
    state.op_Tf((equal_name, 30), 0)
    assert state.events == ["size", "metrics"]
    assert state.font_ascent == 24


def test_tf_reloads_same_name_only_when_selected_in_new_resources() -> None:
    state = RecordingState()
    name = PdfName.of("F")
    state.op_Tf((name, 12), 0)
    original_decoder = state.current_decoder
    state.resources = {"Font": {"F": {"BaseFont": PdfName.of("Child")}}}
    state.resources_id = id(state.resources)
    state.events.clear()

    # A Form inherits its selected font until Tf selects a name in the new scope.
    assert state.get_decoder() is original_decoder
    assert state.events == []
    state.op_Tf((name, 12), 0)
    assert state.events == ["name", "size", "load", "metrics"]
    assert state.current_decoder is not original_decoder
    assert state.current_decoder_resources_id == state.resources_id


def test_tf_graphics_restore_recovers_font_and_operand_cache() -> None:
    state = RecordingState()
    name, size = PdfName.of("F"), float("12.5")
    state.op_Tf((name, size), 0)
    decoder = state.current_decoder
    state.op_q((), 0)
    state.op_Tf((PdfName.of("Other"), 20), 0)
    state.op_Q((), 0)
    assert state.current_decoder is decoder
    assert state.font_size == size
    assert state.font_ascent == 10
    state.events.clear()
    state.op_Tf((name, size), 0)
    assert state.events == []


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
    decoder = state.current_decoder
    state.events.clear()
    with pytest.raises(PdfParseError):
        state.op_Tf((name if reuse_operand else PdfName.of("Other"), "bad"), 0)
    assert state.events == (["size"] if reuse_operand else ["name", "size"])
    assert state.current_decoder is decoder
    assert state.current_font == "F"
    assert state.font_size == 12
    assert state.font_operand is name
    assert state.font_size_operand == 12


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
    assert state.current_font == "F"
    assert state.font_operand is name
    assert state.font_size == state.font_size_operand == 12
    assert state.current_decoder is None
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
        resolver = cast(ObjectResolver, state.document.resolver)
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

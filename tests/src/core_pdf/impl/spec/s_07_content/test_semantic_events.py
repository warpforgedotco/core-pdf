# SPDX-License-Identifier: AGPL-3.0-only
"""PDF execution remains usable without capture products or rendering policy."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content.events import ContentSink
from core_pdf.impl.spec.s_07_content.paths import PathCommand, PdfPath
from core_pdf.impl.spec.s_07_content.state import TextState
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf.impl.spec.s_09_fonts.service import DecodedFontGlyph, FontProvider
from core_pdf.impl.types import PdfName, PdfReference
from tests.helpers.resolvers import IdentityResolver


class EventSink:
    def __init__(self) -> None:
        self.shows: list[tuple[str, float, float]] = []
        self.paths: list[tuple[PathCommand, ...]] = []

    def show_text(
        self,
        state: TextState,
        text: str,
        data: object,
        glyphs: object,
        decoder: object,
        advance_x: float,
        advance_y: float,
    ) -> None:
        self.shows.append((text, advance_x, advance_y))

    def paint_path(self, state: TextState, path: PdfPath, kind: str, fill_rule: str) -> None:
        self.paths.append(tuple(path.commands))

    def graphics_visible(self, state: TextState) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


def state_with_events() -> tuple[TextState, EventSink]:
    sink = EventSink()
    font = SimpleNamespace(
        is_type3=False,
        is_vertical=False,
        ascent=800.0,
        descent=-200.0,
        fast_widths=(600.0,) * 256,
        glyph_width=lambda code: 600.0,
        decode_glyphs=lambda data: tuple(
            DecodedFontGlyph(bytes((code,)), code, code, None, chr(code), code) for code in data
        ),
        text_advance_vector=lambda data, **values: (
            len(data) * 600.0 * values["font_size"] / 1000.0,
            0.0,
        ),
    )
    state = TextState(
        SimpleNamespace(resolver=IdentityResolver()),
        sink=cast(ContentSink, sink),
        font_provider=cast(FontProvider, lambda font_dict, resources: font),
    )
    return state, sink


@pytest.mark.parametrize("render_mode", [0, 3])
def test_text_execution_reports_source_and_advances_independently_of_capture_visibility(
    render_mode: int,
) -> None:
    state, sink = state_with_events()
    state.consume_stream(
        PdfStream(raw_data=f"BT /F1 .05 Tf {render_mode} Tr (A) Tj".encode()),
        {},
        IDENTITY_MATRIX,
        0,
    )
    assert sink.shows == [("A", 0.03, 0.0)]
    assert not hasattr(state, "runs")
    assert not hasattr(state, "glyphs")


def test_path_execution_reports_exact_cubic_control_points() -> None:
    state, sink = state_with_events()
    state.consume_stream(PdfStream(raw_data=b"1 2 m 3 4 5 6 7 8 c S"), {}, IDENTITY_MATRIX, 0)
    assert [(command.operator, command.operands) for command in sink.paths[0]] == [
        ("m", (1.0, 2.0)),
        ("c", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)),
    ]


def test_semantic_resource_lookup_rejects_wrong_type_without_recovery() -> None:
    state, _ = state_with_events()
    state.resources = {"Font": []}
    with pytest.raises(PdfParseError, match="resource value must be a dictionary"):
        state.lookup_page_resource("Font", "F1")


@pytest.mark.parametrize("content", [b"Tc", b"(bad) Tc", b"J", b"/F1 (bad) Tf", b"[1 /bad] 0 d"])
def test_semantic_numeric_errors_require_explicit_recovery(content: bytes) -> None:
    state, _ = state_with_events()
    with pytest.raises(PdfParseError, match="numeric operand"):
        state.consume_stream(PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0)


def test_semantic_font_resolution_error_is_not_replaced_by_a_missing_font() -> None:
    class BrokenResolver(IdentityResolver):
        def resolve(self, ref: object) -> object:
            if isinstance(ref, PdfReference):
                raise PdfParseError("broken font object")
            return ref

    state, _ = state_with_events()
    state.document = SimpleNamespace(resolver=BrokenResolver())
    state.current_font = "F1"
    state.resources = {"Font": {"F1": PdfReference(7, 0)}}
    with pytest.raises(PdfParseError, match="broken font object"):
        state.get_decoder()


def test_semantic_form_matrix_is_not_truncated() -> None:
    state, _ = state_with_events()
    form = PdfStream(
        {"Subtype": PdfName.of("Form"), "Matrix": [1, 0, 0, 1, 10, 20, 99]},
        b"0 0 m 1 1 l S",
    )
    with pytest.raises(ValueError, match="invalid matrix operand"):
        state.consume_stream(
            PdfStream(raw_data=b"/Form Do"), {"XObject": {"Form": form}}, IDENTITY_MATRIX, 0
        )


def test_semantic_pattern_rejects_invalid_matrix_instead_of_using_identity() -> None:
    state, _ = state_with_events()
    state.resources = {
        "Pattern": {
            "P1": PdfStream(
                {
                    "PatternType": 1,
                    "PaintType": 1,
                    "BBox": [0, 0, 10, 10],
                    "XStep": 10,
                    "YStep": 10,
                    "Matrix": [1, 2],
                },
                b"0 0 10 10 re f",
            )
        }
    }
    with pytest.raises(ValueError, match="invalid matrix operand"):
        state.resolve_pattern_color((PdfName.of("P1"),))

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl.spec.s_07_content.glyph_capture import glyph_bitmap_dimensions
from core_pdf.impl.spec.s_07_content.state import TextState
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder
from tests.helpers.resolvers import IdentityResolver


def test_glyph_bitmap_dimensions_derive_from_bbox_aspect_and_font_size() -> None:
    assert glyph_bitmap_dimensions((0.0, -0.2, 0.6, 0.8), 12.0) == (18, 30)


def test_glyph_bitmap_dimensions_preserves_degenerate_fallback() -> None:
    assert glyph_bitmap_dimensions((1.0, 1.0, 1.0, 2.0), 12.0) == (24, 32)


def internal_font(text: str) -> dict[str, Any]:
    mapping = text.encode("utf-16-be").hex().encode("ascii")
    return {
        "Subtype": "Type1",
        "BaseFont": "CaptureFixture",
        "FirstChar": 65,
        "LastChar": 65,
        "Widths": [600],
        "FontDescriptor": {"Ascent": 800, "Descent": -200, "FontBBox": [0, -200, 600, 800]},
        "ToUnicode": PdfStream(
            raw_data=b"1 begincodespacerange <00> <ff> endcodespacerange\n"
            b"1 beginbfchar <41> <" + mapping + b"> endbfchar"
        ),
    }


def internal_state() -> TextState:
    resolver = IdentityResolver()
    document = cast(
        Any, SimpleNamespace(resolver=resolver, resolve=resolver.resolve, raster_font_provider=None)
    )
    return TextState(document)


@pytest.mark.parametrize(
    ("text", "fragments", "captures_bitmap"),
    [
        ("A", ["A"], True),
        ("fi", ["f", "i"], False),
        ("XY", ["XY"], False),
        ("A;", ["A;"], True),
    ],
)
def test_glyph_mappings_share_transformed_stroke_and_cluster_metadata(
    text: str, fragments: list[str], captures_bitmap: bool
) -> None:
    state = internal_state()
    state.consume_stream(
        PdfStream(
            raw_data=b"2 0 0 2 0 0 cm 3 w 2 J 1 j [4 2] 1 d "
            b"0.2 0.3 0.4 RG BT /F1 10 Tf 1 Tr (A) Tj 30 0 Td (A) Tj ET"
        ),
        cast(PdfDict, {"Font": {"F1": internal_font(text)}}),
        IDENTITY_MATRIX,
        0,
    )

    assert [glyph.text for glyph in state.glyphs] == fragments * 2
    assert len(state.glyph_clusters) == 2
    for cluster_id, cluster in enumerate(state.glyph_clusters):
        assert cluster.cluster_id == cluster_id
        assert cluster.text == text
        assert [glyph.paint_glyph for glyph in cluster.glyphs] == [
            index == 0 for index in range(len(fragments))
        ]
        for glyph in cluster.glyphs:
            assert glyph.cluster_key == (cluster_id, cluster_id)
            assert glyph.line_width == 6.0
            assert glyph.line_cap == 2
            assert glyph.line_join == 1
            assert glyph.dash_pattern == ([8.0, 4.0], 2.0)
            assert glyph.stroke_color == (0.2, 0.3, 0.4)
            assert glyph.text_render_mode == 1
            assert glyph.glyph_transform == cluster.glyphs[0].glyph_transform
            assert glyph.provenance is cluster.glyphs[0].provenance
            assert bool(glyph.bitmap_width and glyph.bitmap_height) is captures_bitmap
            assert (glyph.bitmap_code is not None) is captures_bitmap
        x0 = 60.0 * cluster_id
        assert cluster.advance_bbox == (x0, -4.0, x0 + 12.0, 16.0)


def test_glyph_capture_uses_decoded_inputs_and_returns_owned_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = internal_state()
    decoder = FontDecoder(internal_font("fi"))
    state.current_decoder = decoder
    state.font_size = 10.0
    state.update_text_scales()
    state.update_font_metrics()
    decoded = decoder.decode_glyphs(b"A")

    def unexpected_decode(*args: object, **kwargs: object) -> None:
        pytest.fail("The recorder must consume the glyphs already decoded for the text run")

    monkeypatch.setattr(FontDecoder, "decode_glyphs", unexpected_decode)
    result = state.record_glyph_observations(
        "fi",
        decoder,
        0,
        True,
        glyphs=decoded,
        text_basis=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        effective_font_size=10.0,
        effective_font_height=10.0,
    )

    assert [glyph.text for glyph in result.glyphs] == ["f", "i"]
    assert result.clusters[0].glyphs == tuple(result.glyphs)
    assert result.geometry.started
    assert result.geometry.advance == (0.0, -2.0, 6.0, 8.0)
    assert result.geometry.ink == result.geometry.advance
    assert result.geometry.confidence == 1.0
    assert not state.glyphs
    assert not state.glyph_clusters

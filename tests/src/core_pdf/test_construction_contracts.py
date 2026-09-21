import dataclasses

from core_pdf.impl._impl.fonts.decoder import DecodedGlyph
from core_pdf.impl._impl.model.glyphs import GlyphCluster
from core_pdf.impl.types import TextWord


def test_decoded_glyph_field_order_is_pinned():
    assert tuple(field.name for field in dataclasses.fields(DecodedGlyph)) == (
        "code_bytes",
        "char_code",
        "cid",
        "gid",
        "unicode",
        "width_code",
        "unicode_source",
        "alternates",
        "bitmap_code",
        "split_unicode",
    )


def test_glyph_cluster_field_order_is_pinned():
    assert tuple(field.name for field in dataclasses.fields(GlyphCluster)) == (
        "cluster_id",
        "text",
        "glyphs",
        "advance_bbox",
        "ink_bbox",
        "baseline",
        "confidence",
    )


def test_text_word_field_order_is_pinned():
    assert tuple(field.name for field in dataclasses.fields(TextWord)) == (
        "text",
        "bbox",
        "line_index",
        "word_index",
        "block_index",
        "page_number",
        "source",
    )

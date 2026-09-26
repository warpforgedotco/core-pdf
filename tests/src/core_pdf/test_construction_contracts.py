from core_pdf.impl.fonts_decoder import DecodedGlyph
from core_pdf.impl.glyphs import GlyphCluster
from core_pdf.impl.types import TextWord


def test_decoded_glyph_field_order_is_pinned():
    assert DecodedGlyph.__fields__ == (
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
    assert GlyphCluster.__fields__ == (
        "cluster_id",
        "text",
        "glyphs",
        "advance_bbox",
        "ink_bbox",
        "baseline",
        "confidence",
    )


def test_text_word_field_order_is_pinned():
    assert TextWord.__fields__ == (
        "text",
        "bbox",
        "line_index",
        "word_index",
        "block_index",
        "page_number",
        "source",
    )

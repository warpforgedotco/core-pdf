from core_glyph.impl.glyph_recognizer import GlyphBitmapCatalog, glyph_bitmap_item_repairs
from core_glyph.impl.recognition import bitmap_to_pgm, parse_tesseract_symbol


class FakeRecognizer:
    def recognize(self, bitmap: tuple[int, ...], width: int, height: int):
        assert bitmap
        assert width == 2
        assert height == 2
        return "A", 0.99


def test_unmapped_glyph_uses_injected_recognizer() -> None:
    repairs = glyph_bitmap_item_repairs(
        [
            {
                "glyph_index": 4,
                "text": "�",
                "bitmap": (3, 3),
                "bitmap_width": 2,
                "bitmap_height": 2,
                "font_name": "Unknown",
            }
        ],
        recognizer=FakeRecognizer(),
    )

    assert repairs == {4: "A"}


def test_repair_supports_multi_character_corrupt_cmap_run() -> None:
    repairs = glyph_bitmap_item_repairs(
        [
            {
                "glyph_index": 4,
                "text": 'r"!',
                "bitmap": (3, 3),
                "bitmap_width": 2,
                "bitmap_height": 2,
                "font_name": "Unknown",
            }
        ],
        recognizer=FakeRecognizer(),
    )

    assert repairs == {4: "A"}


def test_catalog_reuses_labeled_shape_across_pages() -> None:
    catalog = GlyphBitmapCatalog()
    glyph_bitmap_item_repairs(
        [
            {
                "glyph_index": 1,
                "text": "A",
                "bitmap": (3, 3),
                "bitmap_width": 2,
                "font_name": "SharedFont",
            }
        ],
        catalog=catalog,
    )
    repairs = glyph_bitmap_item_repairs(
        [
            {
                "glyph_index": 9,
                "text": "\ufffd",
                "bitmap": (3, 3),
                "bitmap_width": 2,
                "font_name": "SharedFont",
            }
        ],
        catalog=catalog,
    )
    assert repairs == {9: "A"}


def test_catalog_reuses_labeled_shape_at_a_different_rendered_size() -> None:
    catalog = GlyphBitmapCatalog()
    catalog.observe(
        [
            {
                "glyph_index": 1,
                "text": "A",
                "bitmap": (1, 3),
                "bitmap_width": 2,
                "bitmap_height": 2,
                "font_name": "SharedFont",
            }
        ]
    )
    repairs = glyph_bitmap_item_repairs(
        [
            {
                "glyph_index": 9,
                "text": "�",
                "bitmap": (3, 3, 15, 15),
                "bitmap_width": 4,
                "bitmap_height": 4,
                "font_name": "SharedFont",
            }
        ],
        catalog=catalog,
    )

    assert repairs == {9: "A"}


def test_bitmap_to_pgm_has_expected_dimensions() -> None:
    image = bitmap_to_pgm((1, 2), 2, 2, scale=2, border=1)

    assert image.startswith(b"P5\n6 6\n255\n")
    assert len(image.split(b"\n", 4)[-1]) == 36


def test_parse_tesseract_symbol_rejects_low_confidence() -> None:
    data = (
        b"level\tpage\tblock\tpar\tline\tword\tleft\ttop\twidth\theight\tconf\ttext\n"
        b"5\t1\t1\t1\t1\t1\t0\t0\t2\t2\t40.0\tA\n"
    )

    assert parse_tesseract_symbol(data, 55.0) is None

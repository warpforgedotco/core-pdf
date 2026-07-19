from core_ocr.impl.text_analysis import uninterpretable_char_count


def test_braille_block_is_detected_as_uninterpretable_overlay_text() -> None:
    assert uninterpretable_char_count("DESCRIPTION ⠾ LAUNCH") == 1


def test_normal_technical_unicode_is_not_marked_uninterpretable() -> None:
    assert uninterpretable_char_count("Temperature −12 °C ± 2") == 0

"""PDF symbolic TrueType code selection and Unicode scalar semantics."""


def symbol_character_code(codepoint: int) -> int:
    """PDF 9.6.6.4 selects symbolic TrueType glyphs with a single-byte code."""
    return codepoint & 0xFF if 0xF000 <= codepoint <= 0xF2FF else codepoint


def is_unicode_scalar(codepoint: int) -> bool:
    return 0 <= codepoint < 0x110000 and not 0xD800 <= codepoint <= 0xDFFF


__all__ = ["symbol_character_code", "is_unicode_scalar"]

from core_pdf.api.compat.llamaindex._operator_text import (
    internal_difference_text,
    internal_TextState,
)
from core_pdf.impl.objects import PdfName


def test_unknown_adobe_glyph_names_remain_visible() -> None:
    assert internal_difference_text("i255", 0) == "/i255"
    assert internal_difference_text("17", 4) == "/17"


def test_notdef_uses_the_adobe_missing_glyph_marker() -> None:
    assert internal_difference_text(".notdef", 10) == "□"


def test_name_text_operand_preserves_its_pdf_lexical_form() -> None:
    state = internal_TextState({})

    state.show_name(PdfName.of("BOGUS"))

    assert state.text == "/BOGUS"

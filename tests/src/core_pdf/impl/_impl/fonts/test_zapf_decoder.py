import pytest

from core_pdf.impl._impl.fonts.decoder import FontDecoder


def test_a_font_naming_dingbats_in_differences_decodes_and_measures() -> None:
    decoder = FontDecoder(
        {
            "Subtype": "Type1",
            "BaseFont": "ZapfDingbats",
            "Encoding": {"Differences": [110, "a71", 111, "a79"]},
        }
    )
    assert decoder.decode(bytes([110])) == "●"
    assert decoder.decode(bytes([111])) == "❖"
    assert decoder.fast_widths[110] == pytest.approx(791.0)
    assert decoder.fast_widths[111] == pytest.approx(784.0)

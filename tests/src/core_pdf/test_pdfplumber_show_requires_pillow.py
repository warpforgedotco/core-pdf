import sys
from io import BytesIO

import pytest

from core_pdf.api.compat import pdfplumber as compat


@pytest.fixture
def pillow_absent(monkeypatch):
    """Make ``from PIL import Image`` raise, as it would without the extra installed."""
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)


def test_show_without_pillow_names_the_extra(text_pdf_bytes, pillow_absent):
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        image = pdf.pages[0].to_image()
        with pytest.raises(ImportError, match=r"core-pdf\[pdfplumber\]") as raised:
            image.show()
    assert "PageImage.show() requires Pillow" in str(raised.value)


def test_rendering_does_not_require_pillow(text_pdf_bytes, pillow_absent):
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        image = pdf.pages[0].to_image()
        image.draw_rect((20, 30, 50, 60), fill="red", stroke="blue")
        output = BytesIO()
        image.save(output)
        png = output.getvalue()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert image._repr_png_() == png

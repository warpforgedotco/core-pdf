from io import BytesIO
from typing import Any

import pytest
from PIL import Image

from core_pdf.api.compat import pdfplumber as compat

reference = pytest.importorskip("pdfplumber")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"resolution": 144},
        {"width": 100},
        {"height": 300},
        {"force_mediabox": True},
        {"antialias": True},
    ],
)
@pytest.mark.parametrize("cropped", [False, True])
def test_rendered_and_saved_image_dimensions_match_reference(
    text_pdf_bytes: bytes,
    options: dict[str, Any],
    cropped: bool,
) -> None:
    snapshots = []
    for library in (reference, compat):
        with library.open(BytesIO(text_pdf_bytes)) as pdf:
            page = pdf.pages[0]
            if cropped:
                page = page.crop((10, 20, 110, 170))
            image = page.to_image(**options)
            output = BytesIO()
            image.save(output, format="PNG", quantize=False)
            with Image.open(BytesIO(output.getvalue())) as saved:
                snapshots.append((image.original.size, saved.size, image.resolution))
                assert saved.format == "PNG"
    assert snapshots[1] == snapshots[0]


@pytest.mark.parametrize(
    "options",
    [
        {"resolution": 100, "width": 100},
        {"resolution": 100, "height": 100},
        {"width": 100, "height": 100},
    ],
)
def test_image_resolution_controls_are_mutually_exclusive(
    text_pdf_bytes: bytes,
    options: dict[str, Any],
) -> None:
    for library in (reference, compat):
        with library.open(BytesIO(text_pdf_bytes)) as pdf:
            with pytest.raises(ValueError):
                pdf.pages[0].to_image(**options)


def test_copying_cropped_image_preserves_size_and_independent_annotations(
    text_pdf_bytes: bytes,
) -> None:
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        image = pdf.pages[0].crop((10, 20, 110, 170)).to_image()
        copied = image.copy()
        assert (copied.width, copied.height) == (image.width, image.height)
        copied.draw_rect((20, 30, 50, 60), fill=(255, 0, 0), stroke=(255, 0, 0))
        first, second = BytesIO(), BytesIO()
        image.save(first)
        copied.save(second)
        assert first.getvalue() != second.getvalue()
        assert copied.reset() is copied
        reset = BytesIO()
        copied.save(reset)
        assert reset.getvalue() == first.getvalue()


@pytest.mark.parametrize("force_mediabox", [False, True])
@pytest.mark.parametrize("resolution", [72, 144])
def test_image_cropbox_and_mediabox_sizes(
    text_pdf_bytes: bytes,
    force_mediabox: bool,
    resolution: int,
) -> None:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import RectangleObject

    writer = PdfWriter()
    page = writer.add_page(PdfReader(BytesIO(text_pdf_bytes)).pages[0])
    page.cropbox = RectangleObject((10, 20, 180, 170))
    source = BytesIO()
    writer.write(source)
    snapshots = []
    for library in (reference, compat):
        with library.open(BytesIO(source.getvalue())) as pdf:
            image = pdf.pages[0].to_image(resolution=resolution, force_mediabox=force_mediabox)
            snapshots.append(image.original.size)
    assert snapshots[1] == snapshots[0]

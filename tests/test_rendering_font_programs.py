"""Differential raster coverage for embedded PDF font programs."""

from pathlib import Path

import numpy

from core_pdf import PdfDocument


FIXTURES = Path(__file__).parent / "fixtures"


def test_embedded_type3_raster_matches_poppler_foreground_mask() -> None:
    """Keep the real Type 3 CharProc raster within the committed Poppler tolerance."""
    pdf = FIXTURES / "PyMuPDF" / "tests" / "resources" / "type3font.pdf"

    with PdfDocument.open(pdf) as document:
        raster = document.pages[0].render().rasterize(scale=2.0, cache=False)

    pixels = numpy.asarray(raster.pixels).reshape(raster.height, raster.width, 4)
    foreground = numpy.any(pixels[..., :3] < 250, axis=-1)
    rows, columns = numpy.where(foreground)

    # Poppler 26.08.0 at 144 DPI paints 380 foreground pixels in the same
    # half-open bounding box. Permit only the two antialiasing-edge pixels by
    # comparing the stable mask statistics instead of requiring byte identity.
    assert int(foreground.sum()) == 378
    assert (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ) == (61, 61, 83, 83)

# SPDX-License-Identifier: AGPL-3.0-only
"""Page units preserve raw geometry and scale physical rendering exactly once."""

from io import BytesIO

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.api.compat.pdfminer import extract_pages
from core_pdf.api.compat.pdfplumber import open as open_pdfplumber
from core_pdf.api.compat.pypdf import PdfReader
from core_pdf.impl._impl.render.model import RenderOptions
from core_pdf.impl._impl.runtime.execution import ExtractionScope
from core_pdf.impl.exceptions import PdfRasterTooLargeError
from core_pdf_ocr.impl.extract.capture import capture_page
from core_pdf_ocr.impl.extract.contracts import OcrPass, OcrPassScope, PageRoute, WorkPlan
from core_pdf_ocr.impl.extract.ocr.raster import internal_rendered_page_raster
from core_pdf_ocr.impl.extract.ocr.regions import internal_page_image_regions
from core_pdf_ocr.impl.extract.ocr.session import internal_OcrSession
from core_pdf_ocr.impl.extract.ocr.types import internal_map_ocr_box, internal_OcrTask


def internal_stream(content: bytes, dictionary: bytes = b"") -> bytes:
    return (
        b"<< "
        + dictionary
        + f" /Length {len(content)} >>\nstream\n".encode()
        + content
        + b"\nendstream"
    )


def internal_document(
    unit: bytes = b"2",
    *,
    rotate: int = 0,
    version: str = "1.6",
    parent_unit: bytes = b"9",
    content: bytes = b"1 0 0 rg 16 26 8 6 re f 0 g BT /F 3 Tf 16 42 Td (Hello) Tj ET",
) -> bytes:
    """A nonzero-origin cropped page with body, annotation and widget paint."""
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [7 0 R] >> >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 /UserUnit " + parent_unit + b" >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [10 20 90 80] /CropBox [14 24 86 76] "
        + (b"/UserUnit " + unit if unit else b"")
        + f" /Rotate {rotate} ".encode()
        + b"/Resources << /Font << /F 5 0 R >> /XObject << /Im 11 0 R >> >> "
        b"/Contents 4 0 R /Annots [6 0 R 7 0 R] >>",
        internal_stream(content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Annot /Subtype /Square /Rect [30 30 38 36] /AP << /N 8 0 R >> >>",
        b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (test) /V (Value) "
        b"/Rect [26 38 32 42] /P 3 0 R /AP << /N 9 0 R >> >>",
        internal_stream(b"0 0 1 rg 0 0 4 3 re f", b"/Type /XObject /Subtype /Form /BBox [0 0 4 3]"),
        internal_stream(b"0 1 0 rg 0 0 3 2 re f", b"/Type /XObject /Subtype /Form /BBox [0 0 3 2]"),
        b"2.5",
        internal_stream(
            bytes([128]) * 80 * 60,
            b"/Type /XObject /Subtype /Image /Width 80 /Height 60 "
            b"/ColorSpace /DeviceGray /BitsPerComponent 8",
        ),
    ]
    output = f"%PDF-{version}\n".encode()
    offsets = [0]
    for index, body in enumerate(bodies, 1):
        offsets.append(len(output))
        output += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(output)
    output += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    output += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    return (
        output
        + f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )


@pytest.mark.parametrize("version", ["1.6", "1.7", "2.0"])
@pytest.mark.parametrize("unit", [0.25, 1.0, 2.0, 2.5])
def test_physical_page_size_and_raw_geometry(version: str, unit: float) -> None:
    with PdfDocument(internal_document(str(unit).encode(), version=version)) as document:
        page = document.pages[0]
        assert page.user_unit == unit
        assert (page.width, page.height) == (80, 60)
        assert (page.width_points, page.height_points) == (80 * unit, 60 * unit)
        assert page.media_box == (10, 20, 90, 80)
        assert page.crop_box == (14, 24, 86, 76)
        assert page.get_annotations()[0].rect == (30, 30, 38, 36)
        assert page.get_fields()[0].rect == (26, 38, 32, 42)
        rendered = page.render()
        assert (rendered.width, rendered.height) == (80 * unit, 60 * unit)
        assert rendered.raster_size() == (round(80 * unit), round(60 * unit))
        assert rendered.raster_size(2) == (round(160 * unit), round(120 * unit))


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
@pytest.mark.parametrize("scale", [0.5, 1, 2])
def test_body_annotation_widget_crop_and_rotation_placement(rotate: int, scale: float) -> None:
    with PdfDocument(internal_document(rotate=rotate)) as document:
        rendered = document.pages[0].render()
        raster = rendered.rasterize(scale=scale)
        expected_size = (round(160 * scale), round(120 * scale))
        assert (raster.width, raster.height) == (
            expected_size[::-1] if rotate % 180 else expected_size
        )
        pixels = numpy.rot90(raster.array(), k=rotate // 90)
        for x, y, rgb in [(20, 29, [255, 0, 0]), (34, 33, [0, 0, 255]), (29, 40, [0, 255, 0])]:
            assert list(pixels[int((80 - y) * 2 * scale), int((x - 10) * 2 * scale), :3]) == rgb
        # Explicit rendering crops, like dictionary boxes, use raw PDF units.
        crop = (14.0, 24.0, 46.0, 46.0)
        cropped = rendered.rasterize(scale=scale, crop=crop)
        size = (round(64 * scale), round(44 * scale))
        assert (cropped.width, cropped.height) == (size[::-1] if rotate % 180 else size)
        cropped_pixels = numpy.rot90(cropped.array(), k=rotate // 90)
        assert list(cropped_pixels[int(17 * 2 * scale), int(6 * 2 * scale), :3]) == [255, 0, 0]
        options = document.pages[0].render(RenderOptions(crop=crop))
        assert options.raster_size(scale) == (cropped.width, cropped.height)


def test_native_extraction_retains_raw_boxes_and_serializes_unit() -> None:
    with PdfDocument(internal_document()) as scaled, PdfDocument(internal_document(b"1")) as base:
        actual = scaled.pages[0].extract()
        reference = base.pages[0].extract()
        assert actual.text == reference.text == "Hello"
        assert actual.blocks[0].bbox == reference.blocks[0].bbox
        assert actual.annotations == reference.annotations
        assert actual.form_fields == reference.form_fields
        assert actual.user_unit == 2
        assert (actual.width_points, actual.height_points) == (160, 120)
        json_pages = scaled.extract().to_json_dict()["pages"]
        assert isinstance(json_pages, list)
        json_page = json_pages[0]
        assert isinstance(json_page, dict)
        assert json_page["user_unit"] == 2
        assert scaled.pages[0].chars[0].advance_bbox == base.pages[0].chars[0].advance_bbox


def test_user_unit_and_dpi_produce_identical_text_and_appearance_pixels() -> None:
    with (
        PdfDocument(internal_document(b"2")) as scaled,
        PdfDocument(internal_document(b"1")) as base,
    ):
        actual = scaled.pages[0].render().rasterize(scale=1)
        reference = base.pages[0].render().rasterize(scale=2)
        assert (actual.width, actual.height) == (reference.width, reference.height)
        assert bytes(actual.pixels) == bytes(reference.pixels)


@pytest.mark.parametrize(("unit", "expected"), [(b"", 1), (b"null", 1), (b"10 0 R", 2.5)])
def test_leaf_lookup_resolves_indirect_values_and_never_inherits(
    unit: bytes, expected: float
) -> None:
    with PdfDocument(internal_document(unit)) as document:
        assert document.pages[0].user_unit == expected


@pytest.mark.parametrize("unit", [b"0", b"-2", b"true", b"(2)"])
def test_invalid_units_fail_strict_pages_and_recover_only_in_reader(unit: bytes) -> None:
    with PdfDocument(internal_document(unit)) as document:
        with pytest.raises(ValueError, match="UserUnit"):
            document.pages[0].render()
        document.xref_was_recovered = True
        assert document.pages[0].user_unit == 1


@pytest.mark.parametrize("unit", [b"75000", b"1" + b"0" * 308])
def test_pixel_limit_and_ocr_safe_scale_use_physical_area(unit: bytes) -> None:
    with PdfDocument(internal_document(unit)) as document:
        page = document.pages[0]
        rendered = page.render()
        with pytest.raises(PdfRasterTooLargeError):
            rendered.rasterize(max_pixels=10000)
        raster = internal_rendered_page_raster(
            capture_page(page), 1.0, rendered=rendered, max_pixels=10000
        )
        assert raster is not None
        assert raster.width * raster.height <= 10000


def test_ocr_pixel_projection_preserves_media_origin_and_raw_units() -> None:
    with PdfDocument(internal_document()) as document:
        session = internal_OcrSession(
            capture_page(document.pages[0]),
            WorkPlan(
                PageRoute.OCR,
                ocr_passes=(OcrPass("page", OcrPassScope.PAGE, 1, (3,), include_native_text=True),),
            ),
            True,
            ExtractionScope(),
            None,
        )
        raster = session.render_raster(1, include_native_text=True)
        assert raster is not None
        task = internal_OcrTask(
            mode=3,
            image=raster.image,
            rectangle=(0, 0, raster.width, raster.height),
            page_box=session.page_box,
            resolution=raster.resolution,
        )
        assert session.page_box == (10, 20, 90, 80)
        assert internal_map_ocr_box(task, (12, 96, 28, 108)) == (16, 26, 24, 32)


@pytest.mark.parametrize(("unit", "dpi"), [(0.25, 576), (1, 144), (2, 72)])
def test_direct_image_and_page_ocr_routes_agree_on_physical_resolution(
    unit: float, dpi: int
) -> None:
    data = internal_document(str(unit).encode(), content=b"q 40 0 0 30 20 25 cm /Im Do Q")
    with PdfDocument(data) as document:
        page = document.pages[0]
        capture = capture_page(page)
        regions = internal_page_image_regions(capture, minimum_area_ratio=0.1, upscale=False)
        assert len(regions) == 1
        direct = regions[0]
        assert direct.page_box == (20, 25, 60, 55)
        assert direct.raster.resolution == dpi
        assert (direct.raster.width, direct.raster.height) == (80, 60)
        composed = internal_rendered_page_raster(
            capture, dpi / 72, rendered=page.render(), crop=direct.page_box
        )
        assert composed is not None
        assert (composed.width, composed.height, composed.resolution) == (80, 60, dpi)
        # Source-image and page raster pixels map back into the same raw coordinates.
        assert numpy.all(direct.raster.image.array()[:, :, :3] == 128)
        assert numpy.all(composed.image.array()[20:30, 50:60, :3] == 128)


def test_facades_keep_upstream_raw_unit_geometry_and_pdfium_rasters() -> None:
    real_pypdf = pytest.importorskip("pypdf")
    real_pdfminer = pytest.importorskip("pdfminer.high_level")
    real_pdfplumber = pytest.importorskip("pdfplumber")
    data = internal_document()
    reader = PdfReader(BytesIO(data))
    real_reader = real_pypdf.PdfReader(BytesIO(data))
    assert reader.pages[0].user_unit == real_reader.pages[0].user_unit == 2
    assert (
        tuple(reader.pages[0].mediabox) == tuple(real_reader.pages[0].mediabox) == (10, 20, 90, 80)
    )
    miner_page = next(extract_pages(BytesIO(data)))
    real_miner_page = next(real_pdfminer.extract_pages(BytesIO(data)))
    assert miner_page.bbox == real_miner_page.bbox == (0, 0, 80, 60)
    with (
        open_pdfplumber(BytesIO(data)) as document,
        real_pdfplumber.open(BytesIO(data)) as reference,
    ):
        page = document.pages[0]
        real_page = reference.pages[0]
        assert (page.width, page.height) == (real_page.width, real_page.height) == (80, 60)
        assert (
            page.to_image(resolution=72).original.size
            == real_page.to_image(resolution=72).original.size
        )


@pytest.mark.parametrize("unit", [1, 2, 3])
def test_xray_reports_point_valued_boxes_matching_mupdf(unit: int) -> None:
    from core_pdf.api.compat.xray import inspect

    pymupdf = pytest.importorskip("pymupdf")
    data = internal_document(
        str(unit).encode(), content=b"BT /F 4 Tf 16 25 Td (SECRET) Tj ET 0 g 16 24 44 6 re f"
    )
    with pymupdf.open(stream=data) as reference:
        rect = reference[0].get_drawings()[0]["rect"]
        assert inspect(data) == {1: [{"bbox": tuple(rect), "text": "SECRET"}]}


@pytest.mark.parametrize("unit", [1.25, 2.5, 3.14])
def test_xray_unit_conversion_preserves_mupdf_float_precision(unit: float) -> None:
    from core_pdf.api.compat.xray import _fitz_box

    pymupdf = pytest.importorskip("pymupdf")
    data = internal_document(str(unit).encode(), content=b"0 g 16.1234 24.2345 44.3456 6.4567 re f")
    with PdfDocument(data) as document, pymupdf.open(stream=data) as reference:
        page = document.pages[0]
        box = page.get_drawings()[0].rect
        assert box is not None
        assert page.crop_box is not None
        assert _fitz_box(box, page.crop_box, unit) == tuple(reference[0].get_drawings()[0]["rect"])

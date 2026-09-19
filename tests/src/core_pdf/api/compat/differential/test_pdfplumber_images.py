from io import BytesIO
from pathlib import Path
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


@pytest.mark.parametrize(
    ("method", "shape", "points"),
    [
        ("draw_rect", (20, 20, 80, 60), ((50, 20), (50, 40), (10, 10))),
        (
            "draw_rect",
            {"x0": 20, "top": 20, "x1": 80, "bottom": 60},
            ((50, 20), (50, 40), (10, 10)),
        ),
        ("draw_line", ((20, 20), (80, 20), (80, 60)), ((50, 20), (80, 40), (50, 40))),
        ("draw_line", {"pts": ((20, 20), (80, 20), (80, 60))}, ((50, 20), (80, 40), (50, 40))),
        ("draw_circle", (50, 40), ((60, 40), (50, 30), (50, 40))),
        (
            "draw_circle",
            {"x0": 40, "top": 30, "x1": 60, "bottom": 50},
            ((60, 40), (50, 30), (50, 40)),
        ),
    ],
)
def test_annotations_preserve_outline_geometry(
    text_pdf_bytes: bytes,
    method: str,
    shape: Any,
    points: tuple[tuple[int, int], ...],
) -> None:
    snapshots = []
    for library in (reference, compat):
        with library.open(BytesIO(text_pdf_bytes)) as pdf:
            image = pdf.pages[0].to_image()
            options: dict[str, Any] = {"stroke": "blue"}
            if method != "draw_line":
                options["fill"] = None
            if method == "draw_circle":
                options["radius"] = 10
            assert getattr(image, method)(shape, **options) is image
            output = BytesIO()
            image.save(output, format="PNG", quantize=False)
            with Image.open(BytesIO(output.getvalue())) as saved:
                rgb = saved.convert("RGB")
                snapshots.append([rgb.getpixel(point) for point in points])
    assert snapshots[0][0] == (0, 0, 255)
    assert snapshots[0][-1] == (255, 255, 255)
    assert snapshots[1] == snapshots[0]


@pytest.mark.parametrize("channels", [3, 4])
def test_annotation_batches_share_canvas_and_save_to_path(
    text_pdf_bytes: bytes,
    tmp_path: Path,
    channels: int,
) -> None:
    from types import SimpleNamespace

    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        page = pdf.pages[0]
        raster = SimpleNamespace(
            data=bytes([255]) * 200 * 200 * channels,
            width=200,
            height=200,
            channels=channels,
            dpi=72,
        )
        image = compat.PageImage(page, raster)
        assert image.draw_vlines((10, 20), stroke="blue") is image
        assert image.draw_hlines((10, 20), stroke="blue") is image
        assert image.draw_circles(((50, 50),), radius=5, stroke="blue") is image
        assert image.draw_rects(((80, 30, 100, 50),), stroke="blue") is image
        assert (
            image.draw_words(({"x0": 110, "top": 30, "x1": 130, "bottom": 50},), stroke="blue")
            is image
        )
        target = tmp_path / "annotations.png"
        image.save(target)
        assert target.read_bytes() == image._repr_png_()
        with Image.open(target) as saved:
            rgb = saved.convert("RGB")
            for point in ((10, 5), (5, 20), (55, 50), (90, 30), (120, 30)):
                assert rgb.getpixel(point) == (0, 0, 255)
            assert rgb.getpixel((150, 150)) == (255, 255, 255)


def test_outline_and_table_debug_helpers_draw_requested_regions(text_pdf_bytes: bytes) -> None:
    from types import SimpleNamespace

    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        page = pdf.pages[0]
        explicit = [{"x0": 20, "top": 20, "x1": 40, "bottom": 40}]
        image = page.to_image()
        assert image.outline_words(explicit, stroke="blue") is image
        assert image.outline_chars(explicit, stroke="blue") is image
        table = SimpleNamespace(cells=[None, (60, 20, 80, 40)])
        assert image.debug_table(table, stroke="blue") is image
        with Image.open(BytesIO(image._repr_png_())) as saved:
            rgb = saved.convert("RGB")
            assert rgb.getpixel((30, 20)) == (0, 0, 255)
            assert rgb.getpixel((70, 20)) == (0, 0, 255)
        assert page.to_image().outline_words()._repr_png_() != page.to_image()._repr_png_()
        assert page.to_image().outline_chars()._repr_png_() != page.to_image()._repr_png_()


def test_image_save_rejects_unsupported_channel_layout(text_pdf_bytes: bytes) -> None:
    from types import SimpleNamespace

    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        image = compat.PageImage(
            pdf.pages[0], SimpleNamespace(data=b"\xff", width=1, height=1, channels=1, dpi=72)
        )
        with pytest.raises(ValueError, match="RGB or RGBA"):
            image.save(BytesIO())


@pytest.mark.parametrize("stroke", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_rgb_annotation_values_use_byte_intensities(
    text_pdf_bytes: bytes, stroke: tuple[int, int, int]
) -> None:
    for library in (reference, compat):
        with library.open(BytesIO(text_pdf_bytes)) as pdf:
            image = pdf.pages[0].to_image().draw_rect((20, 20, 80, 60), stroke=stroke, fill=None)
            output = BytesIO()
            image.save(output, format="PNG", quantize=False)
            with Image.open(BytesIO(output.getvalue())) as saved:
                assert saved.convert("RGB").getpixel((50, 20)) == stroke


@pytest.mark.parametrize("tables", [False, True])
def test_debug_tablefinder_draws_edges_and_detected_cells(
    text_pdf_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
    tables: bool,
) -> None:
    from types import SimpleNamespace

    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        page = pdf.pages[0]
        settings = {"vertical_strategy": "lines"}
        seen = []

        def finder(options: Any) -> Any:
            seen.append(options)
            return SimpleNamespace(
                edges=[{"x0": 20, "top": 20, "x1": 80, "bottom": 20}],
                tables=[SimpleNamespace(cells=[(20, 40, 80, 60)])] if tables else [],
            )

        monkeypatch.setattr(page, "debug_tablefinder", finder)
        image = page.to_image()
        assert image.debug_tablefinder(settings) is image
        assert seen == [settings]
        with Image.open(BytesIO(image._repr_png_())) as saved:
            rgb = saved.convert("RGB")
            assert rgb.getpixel((50, 20)) != (255, 255, 255)
            assert rgb.getpixel((50, 40)) == ((0, 0, 255) if tables else (255, 255, 255))


def test_show_displays_the_same_annotated_pixels_as_save(text_pdf_bytes, monkeypatch):
    shown = []
    monkeypatch.setattr(Image.Image, "show", lambda image: shown.append(image.copy()))
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        image = pdf.pages[0].crop((10, 20, 110, 170)).to_image()
        image.draw_rect((20, 30, 50, 60), fill="red", stroke="blue")
        output = BytesIO()
        image.save(output)
        assert image.show() is None
        assert len(shown) == 1
        with Image.open(BytesIO(output.getvalue())) as saved:
            assert shown[0].size == saved.size
            assert shown[0].tobytes() == saved.tobytes()
        shown[0].close()


def test_show_propagates_viewer_errors_without_mutating_annotations(text_pdf_bytes, monkeypatch):
    def fail(image):
        raise OSError("viewer failed")

    monkeypatch.setattr(Image.Image, "show", fail)
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        image = pdf.pages[0].to_image().draw_line(((0, 0), (10, 10)))
        before = image._repr_png_()
        with pytest.raises(OSError, match="viewer failed"):
            image.show()
        assert image._repr_png_() == before

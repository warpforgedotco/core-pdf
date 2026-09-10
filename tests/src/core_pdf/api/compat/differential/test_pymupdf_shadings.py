"""Shading image exports compared with the reference structured text device."""

from pathlib import Path

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_geometry import internal_geometry_expected
from .test_pymupdf_text import real_pymupdf

pytestmark = pytest.mark.compat_differential


def internal_shading_pdf(
    shading_type: int,
    coords: str,
    *,
    extend: bool = False,
    content_matrix: str = "1 0 0 1 0 0",
    extra: str = "",
    opacity: float = 1,
    color_space: str = "DeviceRGB",
    function: str = ("<< /FunctionType 2 /Domain [0 1] /C0 [1 0.25 0] /C1 [0 0.75 1] /N 1 >>"),
) -> bytes:
    with real_pymupdf.open() as document:
        page = document.new_page(width=90, height=70)
        shading = document.get_new_xref()
        document.update_object(
            shading,
            f"<< /ShadingType {shading_type} /ColorSpace /{color_space} /Coords [{coords}] "
            f"/Extend [{'true true' if extend else 'false false'}] {extra} "
            f"/Function {function} >>",
        )
        document.xref_set_key(
            page.xref,
            "Resources",
            f"<< /Shading << /Sh {shading} 0 R >> /ExtGState << /GS << /ca {opacity} >> >> >>",
        )
        stream = document.get_new_xref()
        document.update_object(stream, "<< >>")
        document.update_stream(
            stream,
            f"q 7.25 8.5 71.5 53.25 re W n {content_matrix} cm /GS gs /Sh sh Q".encode(),
        )
        page.set_contents(stream)
        return document.tobytes()


def internal_compare_shading(source: bytes, **kwargs: object) -> None:
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        assert actual[0].get_text("dict", **kwargs) == internal_geometry_expected(
            expected[0].get_text("dict", **kwargs)
        )


@pytest.mark.parametrize(
    ("shading_type", "coords"),
    [
        (2, "10 20 70 20"),
        (2, "20 10 20 60"),
        (2, "10 10 75 55"),
        (3, "42 35 0 42 35 29"),
        (3, "30 30 5 49 39 25"),
        (3, "45 35 30 40 30 2"),
    ],
)
@pytest.mark.parametrize("extend", [False, True])
@pytest.mark.parametrize("flags", [4, 199])
def test_shading_raster_payload_and_bounds(
    shading_type: int, coords: str, extend: bool, flags: int
) -> None:
    source = internal_shading_pdf(shading_type, coords, extend=extend)
    internal_compare_shading(source, flags=flags)


@pytest.mark.parametrize(
    "content_matrix", ["0 1 -1 0 80 0", "1 0.15 0.3 1 -9 -8", "0.5 0 0 0.8 20 5"]
)
@pytest.mark.parametrize("clip", [(0, 0, 90, 70), (20.25, 15.75, 60.5, 53.25)])
@pytest.mark.parametrize("extra", ["", "/BBox [15 12 72 59]"])
def test_transformed_shading_and_content_clip(
    content_matrix: str, clip: tuple[float, ...], extra: str
) -> None:
    source = internal_shading_pdf(2, "10 10 75 55", content_matrix=content_matrix, extra=extra)
    internal_compare_shading(source, clip=clip)


@pytest.mark.parametrize("opacity", [0.25, 0.5, 1])
@pytest.mark.parametrize("extra", ["", "/Background [0.25 0.5 0.75]"])
def test_shading_background_and_image_opacity(opacity: float, extra: str) -> None:
    source = internal_shading_pdf(2, "20 20 60 20", opacity=opacity, extra=extra)
    internal_compare_shading(source)


@pytest.mark.parametrize("exponent", [0.5, 2, 3.7])
@pytest.mark.parametrize(
    ("color_space", "start", "end"),
    [
        ("DeviceGray", "0", "1"),
        ("DeviceRGB", "0.1 0.5 0.9", "0.9 0.75 0.2"),
        ("DeviceCMYK", "0 0.2 0.4 0.1", "0.6 0.7 0.1 0"),
    ],
)
def test_shading_function_color_table(
    exponent: float, color_space: str, start: str, end: str
) -> None:
    source = internal_shading_pdf(
        2,
        "10 20 70 20",
        color_space=color_space,
        function=(f"<< /FunctionType 2 /Domain [0 1] /C0 [{start}] /C1 [{end}] /N {exponent} >>"),
    )
    internal_compare_shading(source)


@pytest.mark.parametrize("flags", [4, 199])
@pytest.mark.parametrize("shading_type", [2, 3])
def test_curved_content_clip_uses_control_point_bounds(flags: int, shading_type: int) -> None:
    source = internal_shading_pdf(
        shading_type, "10 10 75 55" if shading_type == 2 else "30 30 5 49 39 25"
    )
    with real_pymupdf.open(stream=source) as document:
        contents = document[0].get_contents()[0]
        document.update_stream(
            contents,
            document.xref_stream(contents).replace(
                b"7.25 8.5 71.5 53.25 re",
                b"10 10 m 40 68 60 -5 80 60 c 10 60 l h",
            ),
        )
        source = document.tobytes()
    internal_compare_shading(source, flags=flags)


@pytest.mark.parametrize("stitched", [False, True])
def test_sampled_shading_rounds_encoded_sample_positions(stitched: bool) -> None:
    source = internal_shading_pdf(2, "10 20 70 20")
    with real_pymupdf.open(stream=source) as document:
        shade = int(document.xref_get_key(document[0].xref, "Resources/Shading/Sh")[1].split()[0])
        function = document.get_new_xref()
        document.update_object(
            function,
            "<< /FunctionType 0 /Domain [0 1] /Range [0 1 0 1 0 1] "
            "/Size [64] /BitsPerSample 8 /Encode [0 63] >>",
        )
        document.update_stream(
            function,
            bytes(component for i in range(64) for component in (216, 79 if i <= 42 else 78, 73)),
        )
        if stitched:
            parent = document.get_new_xref()
            document.update_object(
                parent,
                f"<< /FunctionType 3 /Domain [0 1] /Bounds [] /Encode [0 1] "
                f"/Functions [{function} 0 R] >>",
            )
            function = parent
        document.xref_set_key(shade, "Function", f"{function} 0 R")
        source = document.tobytes()
    internal_compare_shading(source)


@pytest.mark.parametrize("name", ["PDFTriage-p7-p002.pdf", "This_Is_Caltech_2018_p20-20.pdf"])
def test_shading_image_fixture(name: str) -> None:
    path = Path("tests/fixtures/SCORE-Bench/src") / name
    with real_pymupdf.open(path) as expected, compat_pymupdf.open(path) as actual:
        actual_images = [
            block for block in actual[0].get_text("dict")["blocks"] if block["type"] == 1
        ]
        expected_images = [
            block for block in expected[0].get_text("dict")["blocks"] if block["type"] == 1
        ]
        assert actual_images == internal_geometry_expected(expected_images)

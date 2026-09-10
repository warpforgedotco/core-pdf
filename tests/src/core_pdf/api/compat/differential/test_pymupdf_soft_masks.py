import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_geometry import internal_geometry_expected
from .test_pymupdf_structured_text import internal_output
from .test_pymupdf_text import real_pymupdf

pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("flags", [4, 199])
@pytest.mark.parametrize("cleared", [False, True])
@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("translated", [False, True])
def test_soft_mask_form_images(flags: int, cleared: bool, repeated: bool, translated: bool) -> None:
    with real_pymupdf.open() as fixture:
        page = fixture.new_page(width=200, height=200)

        def stream(dictionary: str, content: bytes) -> int:
            xref = fixture.get_new_xref()
            fixture.update_object(xref, dictionary)
            fixture.update_stream(xref, content)
            return xref

        image = stream(
            "<< /Type /XObject /Subtype /Image /Width 3 /Height 2 "
            "/BitsPerComponent 8 /ColorSpace /DeviceGray >>",
            bytes((0, 40, 80, 120, 160, 255)),
        )
        mask = stream(
            "<< /Type /XObject /Subtype /Form /BBox [0 0 200 200] "
            "/Group << /S /Transparency /CS /DeviceGray >> "
            f"/Resources << /XObject << /Im {image} 0 R >> >> >>",
            b"60 0 0 40 20 100 cm /Im Do",
        )
        paint = stream(
            "<< /Type /XObject /Subtype /Form /BBox [30 105 70 135] "
            "/Group << /S /Transparency >> "
            "/Resources << /ExtGState << /Clear << /SMask /None >> >> >> >>",
            b"/Clear gs 20 100 60 40 re f",
        )
        fixture.xref_set_key(
            page.xref,
            "Resources",
            f"<< /XObject << /Paint {paint} 0 R >> /ExtGState << "
            f"/Mask << /SMask << /S /Luminosity /G {mask} 0 R >> >> "
            "/Clear << /SMask /None >> >> >>",
        )
        content = b"q "
        if translated:
            content += b"1 0 0 1 10 15 cm "
        content += b"/Mask gs "
        if cleared:
            content += b"/Clear gs "
        content += b"/Paint Do "
        if repeated:
            content += b"/Paint Do "
        content += b"Q /Paint Do"
        page.set_contents(stream("<< >>", content))
        source = fixture.tobytes()
    assert internal_output(
        compat_pymupdf, source, "rawdict", flags=flags
    ) == internal_geometry_expected(internal_output(real_pymupdf, source, "rawdict", flags=flags))


def internal_mask_paint_source(consumer: str, *, nested: bool = False) -> bytes:
    with real_pymupdf.open() as document:
        page = document.new_page(width=200, height=200)

        def stream(dictionary: str, content: bytes) -> int:
            xref = document.get_new_xref()
            document.update_object(xref, dictionary)
            document.update_stream(xref, content)
            return xref

        image = stream(
            "<< /Type /XObject /Subtype /Image /Width 3 /Height 2 "
            "/BitsPerComponent 8 /ColorSpace /DeviceGray >>",
            bytes((0, 40, 80, 120, 160, 255)),
        )
        image2 = stream(
            "<< /Type /XObject /Subtype /Image /Width 4 /Height 1 "
            "/BitsPerComponent 8 /ColorSpace /DeviceGray >>",
            bytes((255, 180, 120, 60)),
        )
        shade = document.get_new_xref()
        document.update_object(
            shade,
            "<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 200 0] "
            "/Extend [true true] /Function << /FunctionType 2 /Domain [0 1] "
            "/C0 [0 0 0] /C1 [1 1 1] /N 1 >> >>",
        )
        if consumer == "shading-bbox":
            document.xref_set_key(shade, "BBox", "[30 105 70 135]")
        child = stream(
            "<< /Type /XObject /Subtype /Form /BBox [0 0 200 200] "
            "/Group << /S /Transparency /CS /DeviceGray >> "
            f"/Resources << /XObject << /Im {image} 0 R >> >> >>",
            b"60 0 0 40 20 100 cm /Im Do",
        )
        mask_content = b"q 60 0 0 40 20 100 cm /Im Do Q"
        if consumer == "mixed":
            mask_content += b" /Sh sh"
            if nested:
                mask_content += b" /Nested gs"
            mask_content += b" q 40 0 0 20 40 100 cm /Im2 Do Q"
        mask = stream(
            "<< /Type /XObject /Subtype /Form /BBox [0 0 200 200] "
            "/Group << /S /Transparency /CS /DeviceGray >> "
            f"/Resources << /XObject << /Im {image} 0 R /Im2 {image2} 0 R >> "
            f"/Shading << /Sh {shade} 0 R >> /ExtGState << "
            f"/Nested << /SMask << /S /Luminosity /G {child} 0 R >> >> >> >> >>",
            mask_content,
        )
        paint = stream(
            "<< /Type /XObject /Subtype /Form /BBox [30 105 70 135] "
            "/Group << /S /Transparency >> /Resources << >> >>",
            b"20 100 60 40 re f",
        )
        document.xref_set_key(
            page.xref,
            "Resources",
            f"<< /XObject << /Paint {paint} 0 R /Im {image} 0 R >> "
            f"/Shading << /Sh {shade} 0 R >> "
            f"/ExtGState << /Mask << /SMask << /S /Luminosity /G {mask} 0 R >> >> >> >>",
        )
        paint_content = {
            "fill": b"30 105 40 30 re f",
            "clipped-fill": b"35 110 20 15 re W n 30 105 40 30 re f",
            "image": b"40 0 0 30 30 105 cm /Im Do",
            "inline": (b"40 0 0 30 30 105 cm BI /W 1 /H 1 /BPC 8 /CS /G ID \x80 EI"),
            "mixed": b"/Paint Do",
            "shading": b"/Sh sh",
            "clipped-shading": b"35 110 20 15 re W n /Sh sh",
            "curved-shading": b"30 105 m 30 140 70 90 70 135 c 30 135 l h W n /Sh sh",
            "shading-bbox": b"/Sh sh",
        }[consumer]
        page.set_contents(stream("<< >>", b"q /Mask gs " + paint_content + b" Q"))
        return document.tobytes()


@pytest.mark.parametrize("flags", [4, 199])
@pytest.mark.parametrize(
    "consumer",
    [
        "fill",
        "clipped-fill",
        "image",
        "inline",
        "mixed",
        "shading",
        "clipped-shading",
        "curved-shading",
        "shading-bbox",
    ],
)
def test_soft_mask_consumer_clip_and_mixed_paint_order(consumer: str, flags: int) -> None:
    source = internal_mask_paint_source(consumer)
    assert internal_output(
        compat_pymupdf, source, "rawdict", flags=flags
    ) == internal_geometry_expected(internal_output(real_pymupdf, source, "rawdict", flags=flags))


@pytest.mark.parametrize("flags", [4, 199])
def test_nested_soft_mask_images_keep_consumer_order(flags: int) -> None:
    source = internal_mask_paint_source("mixed", nested=True)
    assert internal_output(
        compat_pymupdf, source, "rawdict", flags=flags
    ) == internal_geometry_expected(internal_output(real_pymupdf, source, "rawdict", flags=flags))


@pytest.mark.parametrize(
    "text",
    [
        b"(ABCD) Tj",
        b"(AB) Tj (CD) Tj",
        b"[(A) 0 (B) -1 (CD)] TJ",
        b"(AB) Tj 1 0 0 rg (CD) Tj",
    ],
)
def test_soft_mask_continuous_text_paints_are_captured_once(text: bytes) -> None:
    with real_pymupdf.open(stream=internal_mask_paint_source("fill")) as fixture:
        page = fixture[0]
        fixture.xref_set_key(
            page.xref,
            "Resources/Font",
            "<< /F << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >>",
        )
        fixture.update_stream(
            page.get_contents()[0], b"q /Mask gs BT /F 20 Tf 30 120 Td " + text + b" ET Q"
        )
        source = fixture.tobytes()
    assert internal_output(
        compat_pymupdf, source, "rawdict", flags=4
    ) == internal_geometry_expected(internal_output(real_pymupdf, source, "rawdict", flags=4))

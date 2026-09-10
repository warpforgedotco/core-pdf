import base64
import json
import zlib
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_geometry import internal_geometry_expected
from .test_pymupdf_text import internal_PDFS, real_pymupdf

pytestmark = pytest.mark.compat_differential


def internal_output(module: Any, source: bytes, kind: str, **kwargs: Any) -> Any:
    with module.open(stream=source) as document:
        value = document[0].get_text(kind, **kwargs)
        return json.loads(value) if kind in ("json", "rawjson") else value


@pytest.mark.parametrize("path", internal_PDFS, ids=lambda path: path.name)
@pytest.mark.parametrize("kind", ["dict", "rawdict", "json", "rawjson"])
def test_structured_text_and_image_blocks(path: Path, kind: str) -> None:
    source = path.read_bytes()
    assert internal_output(compat_pymupdf, source, kind) == internal_geometry_expected(
        internal_output(real_pymupdf, source, kind)
    )


@pytest.mark.parametrize(
    "fontname",
    [
        "helv",
        "hebo",
        "heit",
        "hebi",
        "tiro",
        "tibo",
        "tiit",
        "tibi",
        "cour",
        "cobo",
        "coit",
        "cobi",
        "symb",
        "zadb",
    ],
)
def test_standard_font_span_metadata(fontname: str) -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page()
        page.insert_text((50, 50), "First words\nSecond line", fontname=fontname)
        page.insert_text((50, 120), "Different size", fontname=fontname, fontsize=16)
        fixture.select([len(fixture) - 1])
        source = fixture.tobytes()
    assert internal_output(compat_pymupdf, source, "rawdict") == internal_geometry_expected(
        internal_output(real_pymupdf, source, "rawdict")
    )


@pytest.mark.parametrize("render_mode", [0, 1, 2, 3])
@pytest.mark.parametrize("opacity", [1, 0.4])
def test_span_paint_and_opacity(render_mode: int, opacity: float) -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page()
        page.insert_text(
            (50, 50),
            "Paint",
            fontname="hebo",
            color=(0.25, 0.5, 0.75),
            render_mode=render_mode,
            fill_opacity=opacity,
            stroke_opacity=opacity,
        )
        fixture.select([len(fixture) - 1])
        source = fixture.tobytes()
    assert internal_output(compat_pymupdf, source, "rawdict") == internal_geometry_expected(
        internal_output(real_pymupdf, source, "rawdict")
    )


@pytest.mark.parametrize(
    "method", ["extractDICT", "extractRAWDICT", "extractJSON", "extractRAWJSON"]
)
def test_structured_snapshot_survives_document_close(method: str) -> None:
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        document = module.open(internal_PDFS[1])
        page = document[0]
        snapshot = page.get_textpage(flags=199)
        document.close()
        del page
        value = getattr(snapshot, method)()
        results.append(json.loads(value) if "JSON" in method else value)
    assert results[0] == internal_geometry_expected(results[1])


def test_json_serialization_matches_reference() -> None:
    source = internal_PDFS[0].read_bytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        for kind in ("json", "rawjson"):
            assert actual[0].get_text(kind) == expected[0].get_text(kind)


@pytest.mark.parametrize("flags", [4, 199])
@pytest.mark.parametrize("clip", [(240, 80, 541, 390), (250, 100, 300, 200), (0, 0, 500, 792)])
def test_image_clipping_and_block_records(flags: int, clip: tuple[int, ...]) -> None:
    source = internal_PDFS[1].read_bytes()
    with real_pymupdf.open(stream=source) as expected, compat_pymupdf.open(stream=source) as actual:
        for kind in ("dict", "blocks", "words"):
            assert actual[0].get_text(kind, flags=flags, clip=clip) == internal_geometry_expected(
                expected[0].get_text(kind, flags=flags, clip=clip)
            )


def internal_test_pixmap(grayscale: bool) -> Any:
    space = real_pymupdf.csGRAY if grayscale else real_pymupdf.csRGB
    pixmap = real_pymupdf.Pixmap(space, real_pymupdf.IRect(0, 0, 3, 2), False)
    for y in range(2):
        for x in range(3):
            value = 20 + 50 * x + 30 * y
            pixmap.set_pixel(x, y, (value,) if grayscale else (value, 200 - value, 120))
    return pixmap


@pytest.mark.parametrize("flags", [4, 199])
@pytest.mark.parametrize("x", [-100, 30, 70, 71, 250])
@pytest.mark.parametrize("content_clip", [False, True])
@pytest.mark.parametrize("clip", [None, (30, 65, 70, 95), (0, 85, 200, 90)])
def test_clipped_image_records_retain_empty_bounds(
    flags: int, x: int, content_clip: bool, clip: tuple[int, ...] | None
) -> None:
    with real_pymupdf.open() as fixture:
        page = fixture.new_page(width=200, height=200)
        image = fixture.get_new_xref()
        fixture.update_object(
            image,
            "<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
            "/BitsPerComponent 8 /ColorSpace /DeviceGray >>",
        )
        fixture.update_stream(image, b"\x80")
        fixture.xref_set_key(page.xref, "Resources", f"<< /XObject << /Im {image} 0 R >> >>")
        content = b"q "
        if content_clip:
            content += b"30 105 40 30 re W n "
        content += f"40 0 0 20 {x} 100 cm /Im Do Q".encode()
        stream = fixture.get_new_xref()
        fixture.update_object(stream, "<< >>")
        fixture.update_stream(stream, content)
        page.set_contents(stream)
        source = fixture.tobytes()
    assert internal_output(
        compat_pymupdf, source, "rawdict", flags=flags, clip=clip
    ) == internal_geometry_expected(
        internal_output(real_pymupdf, source, "rawdict", flags=flags, clip=clip)
    )


@pytest.mark.parametrize("format", ["png", "jpeg"])
@pytest.mark.parametrize("grayscale", [False, True])
def test_native_image_payload_encoding(format: str, grayscale: bool) -> None:
    pixmap = internal_test_pixmap(grayscale)
    image = pixmap.tobytes(format)
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page()
        page.insert_image(real_pymupdf.Rect(50, 50, 110, 90), stream=image)
        page.insert_text((50, 120), "Caption")
        fixture.select([len(fixture) - 1])
        source = fixture.tobytes()
    assert internal_output(compat_pymupdf, source, "dict") == internal_geometry_expected(
        internal_output(real_pymupdf, source, "dict")
    )


@pytest.mark.parametrize("wrapper", ["array", "ascii85", "flate"])
@pytest.mark.parametrize("grayscale", [False, True])
@pytest.mark.parametrize("format", ["jpeg", "jpx"])
def test_compressed_image_filters_preserve_encoded_payload(
    wrapper: str, grayscale: bool, format: str
) -> None:
    pixmap = internal_test_pixmap(grayscale)
    if format == "jpeg":
        compressed = pixmap.tobytes("jpeg")
    else:
        image_module = pytest.importorskip("PIL.Image")
        output = BytesIO()
        image_module.open(BytesIO(pixmap.tobytes("png"))).save(output, format="JPEG2000")
        compressed = output.getvalue()
    image_filter = "/DCTDecode" if format == "jpeg" else "/JPXDecode"
    with real_pymupdf.open() as fixture:
        page = fixture.new_page(width=200, height=200)
        image_xref = page.insert_image(real_pymupdf.Rect(20, 20, 80, 60), stream=compressed)
        encoded = compressed
        filters = f"[{image_filter}]"
        if wrapper == "ascii85":
            encoded = base64.a85encode(compressed) + b"~>"
            filters = f"[/ASCII85Decode {image_filter}]"
        elif wrapper == "flate":
            encoded = zlib.compress(compressed)
            filters = f"[/FlateDecode {image_filter}]"
        fixture.update_stream(image_xref, encoded, compress=False)
        fixture.xref_set_key(image_xref, "Filter", filters)
        source = fixture.tobytes()
    assert internal_output(compat_pymupdf, source, "rawdict") == internal_geometry_expected(
        internal_output(real_pymupdf, source, "rawdict")
    )


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("stencil", [False, True])
@pytest.mark.parametrize("inverted", [False, True])
@pytest.mark.parametrize("clipped", [False, True])
@pytest.mark.parametrize("flags", [4, 199])
def test_bilevel_image_placement_and_masks(
    inline: bool, stencil: bool, inverted: bool, clipped: bool, flags: int
) -> None:
    samples = b"\x40\xa0"
    properties = "/Width 3 /Height 2 /BitsPerComponent 1 "
    properties += "/ImageMask true " if stencil else "/ColorSpace /DeviceGray "
    properties += "/Decode [1 0]" if inverted else "/Decode [0 1]"
    with real_pymupdf.open() as fixture:
        page = fixture.new_page(width=200, height=200)
        content = b"q "
        if clipped:
            content += b"30 45 40 30 re W n "
        content += b"60 0 0 40 20 40 cm "
        if inline:
            content += b"BI " + properties.encode() + b" ID " + samples + b" EI "
        else:
            image_xref = fixture.get_new_xref()
            fixture.update_object(image_xref, f"<< /Subtype /Image {properties} >>")
            fixture.update_stream(image_xref, samples, compress=False)
            fixture.xref_set_key(
                page.xref, "Resources", f"<< /XObject << /Im {image_xref} 0 R >> >>"
            )
            content += b"/Im Do "
        content += b"Q"
        content_xref = fixture.get_new_xref()
        fixture.update_object(content_xref, "<< >>")
        fixture.update_stream(content_xref, content)
        page.set_contents(content_xref)
        source = fixture.tobytes()
    assert internal_output(
        compat_pymupdf, source, "rawdict", flags=flags
    ) == internal_geometry_expected(internal_output(real_pymupdf, source, "rawdict", flags=flags))


@pytest.mark.parametrize("matte", [(0, 0, 0), (1, 1, 1), (0.25, 0.5, 0.75)])
def test_image_soft_mask_matte_exports(matte: tuple[float, ...]) -> None:
    with real_pymupdf.open() as fixture:
        page = fixture.new_page(width=200, height=200)
        image_xref = page.insert_image(
            real_pymupdf.Rect(20, 20, 80, 60),
            stream=internal_test_pixmap(False).tobytes("png"),
        )
        mask_xref = fixture.get_new_xref()
        fixture.update_object(
            mask_xref,
            "<< /Type /XObject /Subtype /Image /Width 3 /Height 2 "
            "/ColorSpace /DeviceGray /BitsPerComponent 8 "
            f"/Matte [{' '.join(map(str, matte))}] >>",
        )
        fixture.update_stream(mask_xref, bytes((0, 20, 128, 200, 254, 255)))
        fixture.xref_set_key(image_xref, "SMask", f"{mask_xref} 0 R")
        source = fixture.tobytes()
    assert internal_output(compat_pymupdf, source, "rawdict") == internal_geometry_expected(
        internal_output(real_pymupdf, source, "rawdict")
    )


@pytest.mark.parametrize("wrapper", ["array", "ascii85", "flate"])
def test_four_component_image_streams_remain_in_structured_output(wrapper: str) -> None:
    pixmap = real_pymupdf.Pixmap(real_pymupdf.csCMYK, real_pymupdf.IRect(0, 0, 3, 2), False)
    for y in range(2):
        for x in range(3):
            pixmap.set_pixel(x, y, (30 * x, 40 * y, 120, 20))
    with real_pymupdf.open() as fixture:
        page = fixture.new_page(width=200, height=200)
        image_xref = page.insert_image(real_pymupdf.Rect(20, 20, 80, 60), pixmap=pixmap)
        raw = pixmap.samples
        filters = "[]"
        if wrapper == "ascii85":
            raw = base64.a85encode(raw) + b"~>"
            filters = "[/ASCII85Decode]"
        elif wrapper == "flate":
            raw = zlib.compress(raw)
            filters = "[/FlateDecode]"
        fixture.update_stream(image_xref, raw, compress=False)
        fixture.xref_set_key(image_xref, "Filter", filters)
        source = fixture.tobytes()
    keys = ("bbox", "width", "height", "colorspace", "bpc", "transform")
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        blocks = internal_output(module, source, "rawdict")["blocks"]
        results.append([{key: block[key] for key in keys} for block in blocks])
    assert results[0] == internal_geometry_expected(results[1])


@pytest.mark.parametrize("form", [False, True])
@pytest.mark.parametrize("unit", [1, 2])
def test_nested_image_transform_rounding_and_restore(form: bool, unit: int) -> None:
    with real_pymupdf.open() as fixture:
        page = fixture.new_page(width=297.638, height=595.276)
        fixture.xref_set_key(page.xref, "UserUnit", str(unit))
        image = page.insert_image(
            real_pymupdf.Rect(20, 20, 80, 60), stream=internal_test_pixmap(False).tobytes("png")
        )
        paint = b"q -6.0621 0 0 596.4719 5.4285 -0.606 cm /Im Do Q"
        resources = f"<< /XObject << /Im {image} 0 R >> >>"
        if form:
            form_xref = fixture.get_new_xref()
            fixture.update_object(
                form_xref,
                "<< /Type /XObject /Subtype /Form /BBox [-1000 -1000 1000 1000] "
                f"/Matrix [1 0 0 1 143.4393 -0.5792] /Resources {resources} >>",
            )
            fixture.update_stream(form_xref, paint)
            resources = f"<< /XObject << /Im {image} 0 R /Form {form_xref} 0 R >> >>"
            paint = b"/Form Do"
        else:
            paint = b"1 0 0 1 143.4393 -0.5792 cm " + paint
        contents = fixture.get_new_xref()
        fixture.update_object(contents, "<< >>")
        fixture.update_stream(contents, b"q " + paint + b" Q q 60 0 0 40 20 100 cm /Im Do Q")
        page.set_contents(contents)
        fixture.xref_set_key(page.xref, "Resources", resources)
        source = fixture.tobytes()
    assert internal_output(compat_pymupdf, source, "rawdict") == internal_geometry_expected(
        internal_output(real_pymupdf, source, "rawdict")
    )


@pytest.mark.parametrize(
    ("grayscale", "mismatched_profile"), [(False, False), (True, False), (True, True)]
)
def test_icc_image_exports_match_reference(grayscale: bool, mismatched_profile: bool) -> None:
    image_cms = pytest.importorskip("PIL.ImageCms")
    if grayscale and not mismatched_profile:
        profile = Path("tests/fixtures/pikepdf/tests/resources/Gray.icc").read_bytes()
    else:
        profile = image_cms.ImageCmsProfile(image_cms.createProfile("sRGB")).tobytes()
    with real_pymupdf.open() as fixture:
        page = fixture.new_page(width=200, height=200)
        image_xref = page.insert_image(
            real_pymupdf.Rect(20, 20, 80, 60),
            stream=internal_test_pixmap(grayscale).tobytes("png"),
        )
        profile_xref = fixture.get_new_xref()
        fixture.update_object(profile_xref, f"<< /N {1 if grayscale else 3} >>")
        fixture.update_stream(profile_xref, profile)
        fixture.xref_set_key(image_xref, "ColorSpace", f"[/ICCBased {profile_xref} 0 R]")
        source = fixture.tobytes()
    assert internal_output(compat_pymupdf, source, "rawdict") == internal_geometry_expected(
        internal_output(real_pymupdf, source, "rawdict")
    )


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("matrix", [(1, 0, 0, 1, 0, 0), (0.5, 0, 0, 0.5, 10, 20)])
def test_structured_snapshot_transform_and_independent_results(
    sort: bool, matrix: tuple[float, ...]
) -> None:
    results = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(internal_PDFS[1]) as document:
            page = document[0]
            snapshot = page.get_textpage(flags=199, matrix=module.Matrix(matrix))
            first = snapshot.extractRAWDICT(sort=sort)
            first["blocks"].clear()
            results.append(snapshot.extractRAWDICT(sort=sort))
    assert results[0] == internal_geometry_expected(results[1])

import json
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


@pytest.mark.parametrize("format", ["png", "jpeg"])
@pytest.mark.parametrize("grayscale", [False, True])
def test_native_image_payload_encoding(format: str, grayscale: bool) -> None:
    space = real_pymupdf.csGRAY if grayscale else real_pymupdf.csRGB
    pixmap = real_pymupdf.Pixmap(space, real_pymupdf.IRect(0, 0, 3, 2), False)
    for y in range(2):
        for x in range(3):
            value = 20 + 50 * x + 30 * y
            pixmap.set_pixel(x, y, (value,) if grayscale else (value, 200 - value, 120))
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

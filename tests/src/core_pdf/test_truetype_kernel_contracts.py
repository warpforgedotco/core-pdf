"""TrueType outlines from truetype_contours are fontTools' outlines, glyph for glyph.

The fonts are the TrueType programs embedded in fixture pages chosen for
what their glyphs use: composites with offsets, one scale, x and y scales
and two-by-two matrices, composites nested in composites, and simple
glyphs whose left side bearing differs from their xMin. Every glyph is
drawn both ways -- the kernel, and fontTools' glyph set through
DecomposingRecordingPen and recording_to_contours, as fonttools_contours
drew it -- and compared by repr, which keeps each float exactly. Glyphs
the kernel declines are left to fontTools; the hand-built ones below pin
which those are.
"""

import struct
from typing import Any

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl import fonts_font_program as font_program
from core_pdf.impl.fonts_raster_kernel import scale_contours
from core_pdf_cythonized import truetype_contours

FIXTURES = [
    "tests/fixtures/pypdf/resources/fontsampler.pdf",
    "tests/fixtures/PyMuPDF/tests/resources/test_3062.pdf",
    "tests/fixtures/PyMuPDF/tests/resources/test_4716.pdf",
    "tests/fixtures/SCORE-Bench/src/zero-trust-architecture-tech_p009.pdf",
    "tests/fixtures/pdfplumber/tests/pdfs/issue-71-duplicate-chars-2.pdf",
    "tests/fixtures/unstructured/example-docs/pdf/pdf-bad-color-space.pdf",
    "tests/fixtures/x-ray/tests/assets/rect_ordering_2.1.pdf",
    "tests/fixtures/SCORE-Bench/src/CV_RenyuHu_2023p4-4.pdf",
]


def embedded_programs(path: str, monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    programs: list[bytes] = []
    original = font_program.parse_truetype_program

    def captured(data: bytes) -> Any:
        programs.append(data)
        return original(data)

    monkeypatch.setattr(font_program, "parse_truetype_program", captured)
    with open(path, "rb") as handle, PdfDocument(handle.read()) as document:
        for index in range(min(2, len(document.pages))):
            document.pages[index].render().rasterize(scale=0.25)
    monkeypatch.undo()
    return list(dict.fromkeys(programs))


def fonttools_drawn(font: Any, gid: int, scale: float) -> Any:
    try:
        contours = font_program.fonttools_contours(font, gid)
    except Exception:  # noqa: BLE001 -- normalized_glyph_contours draws nothing then
        return ()
    return contours if scale == 1.0 else scale_contours(contours, scale)


@pytest.mark.parametrize("path", FIXTURES)
def test_every_glyph_is_drawn_as_fonttools_draws_it(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    programs = embedded_programs(path, monkeypatch)
    assert programs
    drawn = declined = 0
    for data in programs:
        font = font_program.parse_truetype_program(data)
        access = font_program.FontToolsOutlineAccess(font)
        tables = font_program.truetype_tables(font)
        if tables is None:
            continue
        glyf, loca, lsb, count = tables
        for gid in range(count):
            got = truetype_contours(glyf, loca, lsb, count, gid, access.scale)
            if got is None:
                declined += 1
                continue
            assert repr(got) == repr(fonttools_drawn(font, gid, access.scale)), (path, gid)
            drawn += 1
    assert drawn
    assert declined <= drawn // 100


def simple_glyph(points: list[tuple[int, int, bool]], *, flags_or: int = 0) -> bytes:
    """One contour, long coordinates, no instructions."""
    xs = [x for x, _, _ in points]
    ys = [y for _, y, _ in points]
    data = struct.pack(">hhhhh", 1, min(xs), min(ys), max(xs), max(ys))
    data += struct.pack(">H", len(points) - 1) + struct.pack(">h", 0)
    data += bytes((1 if on else 0) | flags_or for _, _, on in points)
    previous = 0
    for x in xs:
        data += struct.pack(">h", x - previous)
        previous = x
    previous = 0
    for y in ys:
        data += struct.pack(">h", y - previous)
        previous = y
    return data


def call(glyphs: list[bytes], gid: int) -> Any:
    loca = [0]
    for glyph in glyphs:
        loca.append(loca[-1] + len(glyph))
    return truetype_contours(
        b"".join(glyphs),
        numpy.asarray(loca, dtype=numpy.int64),
        numpy.zeros(len(glyphs), dtype=numpy.int64),
        len(glyphs),
        gid,
        1.0,
    )


SQUARE = [(0, 0, True), (100, 0, True), (100, 100, True), (0, 100, True)]


def test_a_square_is_its_corners_closed() -> None:
    assert call([simple_glyph(SQUARE)], 0) == (
        ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)),
    )


def test_an_all_off_curve_contour_draws_nothing() -> None:
    off = [(x, y, False) for x, y, _ in SQUARE]
    assert call([simple_glyph(off)], 0) == ()


@pytest.mark.parametrize(
    "glyphs",
    [
        pytest.param([simple_glyph(SQUARE)[:-3]], id="truncated"),
        pytest.param([simple_glyph(SQUARE, flags_or=0x80)], id="cubic"),
        pytest.param(
            [
                simple_glyph(SQUARE),
                struct.pack(">hhhhh", -1, 0, 0, 1, 1) + struct.pack(">HHBB", 0x0000, 0, 1, 2),
            ],
            id="point-matched",
        ),
        pytest.param(
            [
                simple_glyph(SQUARE),
                struct.pack(">hhhhh", -1, 0, 0, 1, 1) + struct.pack(">HHhh", 0x0003, 9, 0, 0),
            ],
            id="component-past-the-glyphs",
        ),
        pytest.param(
            [struct.pack(">hhhhh", -1, 0, 0, 1, 1) + struct.pack(">HHhh", 0x0003, 0, 0, 0)],
            id="recursive",
        ),
        pytest.param([struct.pack(">hhhhh", -2, 0, 0, 1, 1)], id="negative-contours"),
    ],
)
def test_what_fonttools_treats_otherwise_is_declined(glyphs: list[bytes]) -> None:
    assert call(glyphs, len(glyphs) - 1) is None


def test_a_scaled_component_takes_the_2_14_scale_and_an_unscaled_offset() -> None:
    # getComponentInfo gives (0.5, 0, 0, 0.5, 10, 0), and TransformPen adds
    # the offset after scaling, whatever the scaled-offset flags say.
    half = struct.pack(">h", 8192)
    composite = (
        struct.pack(">hhhhh", -1, 0, 0, 50, 50) + struct.pack(">HHhh", 0x000B, 0, 10, 0) + half
    )
    assert call([simple_glyph(SQUARE), composite], 1) == (
        ((10.0, 0.0), (60.0, 0.0), (60.0, 50.0), (10.0, 50.0), (10.0, 0.0)),
    )

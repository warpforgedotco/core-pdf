import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_geometry import internal_geometry_expected
from .test_pymupdf_text import internal_PDFS, real_pymupdf

pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("scale", [0, 50, 75, 125, 200, -100])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("spacing", [0, 1.5])
def test_horizontal_text_scaling(scale: int, rotation: int, spacing: float) -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page()
        page.insert_text((250, 300), "Scaled text", rotate=rotation)
        for xref in page.get_contents():
            content = fixture.xref_stream(xref).replace(
                b"BT", f"BT {scale} Tz {spacing} Tc".encode()
            )
            fixture.update_stream(xref, content)
        fixture.select([len(fixture) - 1])
        source = fixture.tobytes()
    with (
        real_pymupdf.open(stream=source) as reference,
        compat_pymupdf.open(stream=source) as actual,
    ):
        ref_page, page = reference[0], actual[0]
        for kind in ["text", "words", "blocks", "rawdict"]:
            assert page.get_text(kind) == internal_geometry_expected(ref_page.get_text(kind))
        assert [
            tuple(tuple(p) for p in q) for q in page.search_for("Scaled", quads=True)
        ] == internal_geometry_expected(
            [tuple(tuple(p) for p in q) for q in ref_page.search_for("Scaled", quads=True)]
        )


@pytest.mark.parametrize(
    "transform",
    [
        (1, 0.2, 0.35, 1),
        (0.8, 0.6, -0.6, 0.8),
        (1, 0, 0, 2),
        (2, 0, 0, 1),
        (-1, 0, 0, 1),
        (1, 0, 0.5, 0),
    ],
)
@pytest.mark.parametrize("snapshot_transform", [False, True])
def test_affine_text_geometry_and_font_size(
    transform: tuple[float, ...], snapshot_transform: bool
) -> None:
    with real_pymupdf.open(internal_PDFS[0]) as fixture:
        page = fixture.new_page()
        page.insert_text(
            (250, 300),
            "Affine text",
            morph=(real_pymupdf.Point(250, 300), real_pymupdf.Matrix(*transform, 0, 0)),
        )
        fixture.select([len(fixture) - 1])
        source = fixture.tobytes()
    outputs = []
    for module in (compat_pymupdf, real_pymupdf):
        with module.open(stream=source) as document:
            page = document[0]
            matrix = module.Matrix(1, 0.1, -0.1, 1, 0, 0) if snapshot_transform else None
            snapshot = page.get_textpage(matrix=matrix)
            outputs.append(
                (snapshot.extractText(), snapshot.extractWORDS(), snapshot.extractRAWDICT())
            )
    assert outputs[0] == internal_geometry_expected(outputs[1])


@pytest.mark.parametrize("scale", [0.988, 1.02])
@pytest.mark.parametrize("rotation", [0, 90])
def test_fractional_text_matrix_long_cursor_advances(scale: float, rotation: int) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page(width=2000, height=2000)
        page.insert_text((100, 1800), "Long text", fontsize=9.9626)
        font = page.get_fonts()[0][4]
        basis = f"{scale} 0 0 1" if rotation == 0 else f"0 {scale} -1 0"
        text = "Repeated long text with fractional scaling. " * 6
        document.update_stream(
            page.get_contents()[0],
            f"BT /{font} 9.9626 Tf {basis} 100 200 Tm ({text}) Tj ET".encode(),
        )
        source = document.tobytes()
    with (
        real_pymupdf.open(stream=source) as reference,
        compat_pymupdf.open(stream=source) as actual,
    ):
        ref_page, page = reference[0], actual[0]
        for kind in ("text", "words", "rawdict"):
            assert page.get_text(kind) == internal_geometry_expected(ref_page.get_text(kind))


@pytest.mark.parametrize("rotation", [0, 90])
@pytest.mark.parametrize("page_height", [792, 2000])
@pytest.mark.parametrize("restore", [False, True])
def test_nested_graphics_transforms_preserve_text_origin_precision(
    rotation: int, page_height: int, restore: bool
) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page(width=1000, height=page_height)
        page.insert_text((100, 200), "Transformed text", fontsize=12)
        font = page.get_fonts()[0][4]
        basis = "1 0 0 1" if rotation == 0 else "0 1 -1 0"
        paint = f"BT /{font} 12 Tf {basis} 100 100 Tm (Transformed text) Tj ET".encode()
        content = (
            b"q 1 0 0 1 143.4393 -.5792 cm q .6121 0 0 .9719 5.4285 -.606 cm " + paint + b" Q "
        )
        if restore:
            content += b"BT /" + font.encode() + b" 12 Tf 1 0 0 1 100 200 Tm (Restored text) Tj ET "
        document.update_stream(page.get_contents()[0], content + b"Q")
        source = document.tobytes()
    with (
        real_pymupdf.open(stream=source) as reference,
        compat_pymupdf.open(stream=source) as actual,
    ):
        ref_page, page = reference[0], actual[0]
        for kind in ("text", "words", "blocks", "rawdict"):
            assert page.get_text(kind) == internal_geometry_expected(ref_page.get_text(kind))


@pytest.mark.parametrize("start", [522.48, 500.125])
@pytest.mark.parametrize("shift", [-477, -475.12])
@pytest.mark.parametrize("rotation", [0, 90])
def test_relative_text_move_rounds_before_graphics_transform(
    start: float, shift: float, rotation: int
) -> None:
    with real_pymupdf.open() as document:
        page = document.new_page(width=1000, height=1000)
        page.insert_text((100, 200), "Seed", fontsize=9.6)
        font = page.get_fonts()[0][4]
        basis = "1 0 0 1" if rotation == 0 else "0 1 -1 0"
        document.update_stream(
            page.get_contents()[0],
            (
                f"q .12 0 0 .12 0 0 cm q 8.33333 0 0 8.33333 0 0 cm "
                f"BT /{font} 9.6 Tf {basis} {start} 700.75 Tm (First) Tj "
                f"{shift} 0 Td (Second) Tj 3.12 -10.51 Td (Third) Tj ET Q Q"
            ).encode(),
        )
        source = document.tobytes()
    with (
        real_pymupdf.open(stream=source) as reference,
        compat_pymupdf.open(stream=source) as actual,
    ):
        ref_page, page = reference[0], actual[0]
        for kind in ("text", "words", "blocks", "rawdict"):
            assert page.get_text(kind) == internal_geometry_expected(ref_page.get_text(kind))

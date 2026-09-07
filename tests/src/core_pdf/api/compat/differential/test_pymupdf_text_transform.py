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
